#!/usr/bin/env python3
"""汇总学习策略搜索、模型时序、对照和账本，汇总过程不重新选参。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] in ['complete', 'completed_with_errors']
    records = [json.loads(line) for p in sorted(run.glob('search/*/*/trials.jsonl')) for line in p.read_text().splitlines()]
    rows = [dict(family=r['family'], outer_year=r['outer_year'], id=r['id'], phase=r['phase'], status=r['status'],
                 params_json=json.dumps(r['params'],sort_keys=True), **{'p_'+k:v for k,v in r['params'].items()},
                 **{k:v for k,v in r.get('metrics',{}).items() if k!='yearly'},
                 constant_score_days=r.get('constant_score_days'), no_score_days=r.get('no_score_days'),
                 error=r.get('error','')) for r in records]
    trials = pd.DataFrame(rows)
    trials.to_csv(run / 'all_trials.csv', index=False)
    trials.to_parquet(run / 'all_trials.parquet', index=False)
    counts = trials.groupby(['outer_year','family']).agg(trials=('id','size'), feasible=('feasible','sum'),
                                                       errors=('status',lambda x:int((x!='ok').sum()))).reset_index()
    counts.to_csv(run / 'search_counts.csv',index=False)
    choices = json.loads((run/'all_locked_choices.json').read_text())
    pd.DataFrame([dict(family=c['family'],outer_year=c['outer_year'],locked_at=c['locked_at'],
                       selected_id=c['selected']['id'] if c['selected'] else '',feasible=c['feasible'],
                       params_json=json.dumps(c['selected']['params'],sort_keys=True) if c['selected'] else '{}',
                       neighbor_count=c['selected'].get('neighbor_count',0) if c['selected'] else 0,
                       constant_score_days=c['selected'].get('constant_score_days',0) if c['selected'] else 0)
                  for c in choices]).to_csv(run/'locked_selection_summary.csv',index=False)
    fits=[]
    for path in sorted(run.glob('predictions/*/*/*/fits.json')):
        for item in json.loads(path.read_text()):
            if item['status']=='ok':
                assert pd.Timestamp(item['max_label_end']) < pd.Timestamp(item['fit_date'])
                model=path.parent/'models'/item['model_file']
                assert hashlib.sha256(model.read_bytes()).hexdigest()==item['model_sha256']
            fits.append(dict(context=str(path.parent.relative_to(run)), **{k:v for k,v in item.items() if k not in ['params','weights','importance']},
                             family=item['params']['family'],variant=item['params']['variant']))
    pd.DataFrame(fits).to_csv(run/'fit_audit_summary.csv',index=False)
    # 只列实际评估的同配置切片；无覆盖证据时不插值。
    reversals=[]
    for (family,year),group in trials[trials.status.eq('ok')].groupby(['family','outer_year']):
        for axis in sorted({k for p in group.params_json for k,v in json.loads(p).items() if isinstance(v,(int,float))}):
            buckets={}
            for r in group.itertuples():
                p=json.loads(r.params_json)
                if axis not in p:
                    continue
                fixed=json.dumps({k:v for k,v in p.items() if k!=axis},sort_keys=True)
                buckets.setdefault(fixed,[]).append((p[axis],r.median_ann,r.id))
            for fixed,points in buckets.items():
                points.sort()
                for a,b,c in zip(points,points[1:],points[2:]):
                    if (b[1]-a[1])*(c[1]-b[1]) < -1e-10:
                        reversals.append(dict(family=family,outer_year=year,axis=axis,fixed_params=fixed,
                                               x1=a[0],x2=b[0],x3=c[0],y1=a[1],y2=b[1],y3=c[1]))
    pd.DataFrame(reversals,columns=['family','outer_year','axis','fixed_params','x1','x2','x3','y1','y2','y3']).to_csv(run/'nonmonotonic_samples.csv',index=False)
    if manifest['pilot']:
        print(f'PILOT: trials={len(trials)}, fits={len(fits)}')
        return
    import sys
    import joblib
    import yaml
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    from learning_extensions import infer_bundle
    plan=yaml.safe_load((run/'sources/learning_extensions_v1.yaml').read_text())
    summary=pd.read_csv(run/'outer_summary.csv')
    controls=pd.read_csv(run/'control_summary.csv')
    paired=pd.concat([pd.read_csv(p) for p in sorted(run.glob('paired/*/summary.csv'))],ignore_index=True)
    paired.to_csv(run/'paired_summary.csv',index=False)
    names={'neural':'神经网络MLP','state_experts':'市场状态专家','adaptive_ic':'动态因子权重',
           'trade_filter':'学习型交易过滤','multi_horizon':'多周期联合预测','correlation_experts':'相关性分组专家','market_gate':'市场风险门控'}
    feature=pd.read_parquet(run/'inputs/learning_features.parquet')
    eligible=pd.read_parquet(run/'inputs/eligible.parquet')
    context=pd.read_parquet(run/'inputs/market_context.parquet')
    ends=pd.read_parquet(run/'inputs/label_end_dates.parquet')
    # 逐实际模型分支保存/加载重放，不依赖训练时留在内存中的对象。
    checked=set(); reload_rows=[]
    for path in sorted(run.glob('predictions/*/*/*/fits.json')):
        records=[x for x in json.loads(path.read_text()) if x['status']=='ok']
        if not records:
            continue
        fit=records[0]; p=fit['params']; branch=(p['family'],p['variant'])
        if branch in checked:
            continue
        date=pd.Timestamp(fit['fit_date'])
        raw=feature.xs(date,level='date').reindex(index=eligible.columns,columns=list(plan['features'])).to_numpy()
        missing=~np.isfinite(raw)
        allowed=eligible.loc[date].to_numpy() & ((~missing).sum(axis=1)>=plan['training']['min_features'])
        # 与训练数组保持同样的C布局，避免float32宿主求均值的归约顺序不同。
        ranks=np.ascontiguousarray(np.nan_to_num(raw),dtype=np.float32)
        data=dict(allowed=allowed[None],ranks=ranks[None],
                  x=np.ascontiguousarray(np.concatenate([ranks,missing.astype(np.float32)],axis=1))[None],context=context.loc[[date]].to_numpy())
        bundle=joblib.load(path.parent/'models'/fit['model_file'])
        prediction,risk=infer_bundle(bundle,data,0)
        reference=pd.read_parquet(path.parent/'return_predictions.parquet').loc[date].to_numpy()
        reference_risk=pd.read_parquet(path.parent/'risk_predictions.parquet').loc[date].to_numpy()
        np.testing.assert_allclose(prediction,reference,rtol=1e-12,atol=1e-12,equal_nan=True)
        np.testing.assert_allclose(risk,reference_risk,rtol=1e-12,atol=1e-12,equal_nan=True)
        error=lambda a,b: float(np.nanmax(np.abs(a-b))) if np.isfinite(a-b).any() else 0.
        reload_rows.append(dict(family=branch[0],variant=branch[1],fit_date=fit['fit_date'],
                                return_max_error=error(prediction,reference),risk_max_error=error(risk,reference_risk),passed=True))
        checked.add(branch)
    assert checked=={(f,v) for f,variants in plan['families'].items() for v in variants}
    pd.DataFrame(reload_rows).to_csv(run/'model_roundtrip_checks.csv',index=False)
    # 在相同外层已成熟目标上，分别汇报排名质量与门控概率；不将IC当成交易收益。
    quality=[]
    for choice in choices:
        if choice['selected'] is None:
            continue
        family,year=choice['family'],choice['outer_year']; p=choice['selected']['params']
        paths=list((run/'predictions'/f'outer_{year}'/family).glob('*/return_predictions.parquet'))
        assert len(paths)==1
        forecast=pd.read_parquet(paths[0])
        probability=pd.read_parquet(paths[0].parent/'risk_predictions.parquet')
        returns=pd.read_parquet(run/'inputs'/f"label_returns_{p['horizon']}.parquet").reindex_like(forecast)
        ready=ends[str(p['horizon'])].reindex(forecast.index).le(pd.Timestamp(f'{year}-12-31'))
        returns=returns.where(np.broadcast_to(ready.to_numpy()[:,None],returns.shape))
        valid=forecast.notna() & returns.notna()
        x=forecast.where(valid).rank(axis=1,pct=True);y=returns.where(valid).rank(axis=1,pct=True)
        x=x.sub(x.mean(axis=1),axis=0);y=y.sub(y.mean(axis=1),axis=0)
        denominator=np.sqrt((x*x).sum(axis=1)*(y*y).sum(axis=1))
        ic=((x*y).sum(axis=1)/denominator.where(denominator>0)).where(valid.sum(axis=1)>=10)
        row=dict(family=family,year=year,horizon=p['horizon'],mean_daily_rank_ic=ic.mean(),ic_days=int(ic.notna().sum()),
                 raw_forecast_min=float(forecast.min().min()),raw_forecast_max=float(forecast.max().max()))
        if family in ['trade_filter','market_gate']:
            fit_records=[r for r in json.loads((paths[0].parent/'fits.json').read_text()) if r['status']=='ok']
            base=pd.Series({pd.Timestamp(r['fit_date']):r['positive_rate'] for r in fit_records}).reindex(forecast.index).ffill()
            if family=='trade_filter':
                predicted=probability.where(valid).to_numpy()
                actual=(returns>plan['training']['filter_return_hurdle']).where(valid).to_numpy(dtype=float)
                constant=np.broadcast_to(base.to_numpy()[:,None],actual.shape)
                row['gate_pass_fraction']=float(((probability>=p['probability_cut']) & valid).to_numpy().sum()/max(valid.to_numpy().sum(),1))
            else:
                # 原始宿主预测覆盖全部可选股；目标只在此集合且完整价格标签上求平均。
                target=returns.where(forecast.notna()).mean(axis=1)
                predicted=probability.mean(axis=1).where(target.notna()).to_numpy()
                actual=(target < -p['loss_threshold']).where(target.notna()).to_numpy(dtype=float)
                constant=base.to_numpy()
                row['gate_pass_fraction']=float((predicted[np.isfinite(predicted)]<=p['probability_cut']).mean())
            row.update(probability_brier=float(np.nanmean((predicted-actual)**2)),
                       past_prevalence_brier=float(np.nanmean(np.where(np.isfinite(predicted), (constant-actual)**2,np.nan))),
                       realized_event_rate=float(np.nanmean(actual)),probability_observations=int(np.isfinite(predicted-actual).sum()))
        quality.append(row)
    pd.DataFrame(quality).to_csv(run/'forecast_quality.csv',index=False)
    audits=[json.loads(p.read_text()) for p in run.rglob('audit.json')]
    for a in audits:
        assert all(a[k] for k in ['t_plus_one_pass','official_limits_pass','no_negative_cash'])
        assert max(v for k,v in a.items() if k.endswith('_max_error'))<1e-6
    nav={}; yearly=[]; diagnostics=[]
    for row in summary.itertuples():
        d=pd.read_parquet(run/'outer'/row.family/'daily.parquet')
        nav[row.family]=d.nav
        yearly.extend(dict(family=row.family,**x) for x in json.loads(row.yearly))
        signals=pd.read_parquet(run/'outer'/row.family/'signals.parquet')
        for year,part in signals.groupby(signals.index.year):
            diagnostics.append(dict(family=row.family,year=int(year),signal_dates=len(part),
                                    constant_score_dates=int(part.nunique(axis=1).eq(1).sum()),
                                    no_score_dates=int(part.notna().sum(axis=1).eq(0).sum())))
    pd.DataFrame(diagnostics).to_csv(run/'outer_prediction_diagnostics.csv',index=False)
    pd.DataFrame(yearly).to_csv(run/'outer_yearly.csv',index=False)
    pd.DataFrame(nav).to_csv(run/'outer_nav.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,constrained_layout=True)
    for name,series in nav.items():
        axes[0].plot(series.index,series,label=name)
        peaks=np.maximum.accumulate(np.r_[1.,series.to_numpy()])[1:]
        axes[1].plot(series.index,series.to_numpy()/peaks-1,label=name)
    axes[0].legend(ncol=2);axes[0].set_ylabel('Continuous NAV')
    axes[1].axhline(-.3,color='red',linestyle='--');axes[1].set_ylabel('Drawdown')
    fig.suptitle('Learning extensions, 2021-2023; previously observed engineering sample')
    fig.savefig(run/'outer_nav_drawdown.png',dpi=160);plt.close(fig)
    validation=dict(trials=len(trials),errors=int((trials.status!='ok').sum()),unique_fit_records=len(fits),
                    successful_fits=sum(x['status']=='ok' for x in fits),
                    insufficient_training=sum(x['status']!='ok' for x in fits),label_availability_pass=True,
                    model_hashes_pass=True,independent_accounts=len(audits),trades_checked=sum(x['trades_checked'] for x in audits),
                    cash_max_error=max(x['cash_max_error'] for x in audits),roundtrip_branches=len(reload_rows),
                    real_prediction_prefix_checks=len(json.loads((run/'real_prediction_prefix_checks.json').read_text())),
                    account_prefix_checks=sum(len(json.loads(p.read_text())) for p in run.glob('outer/*/prefix_checks.json')),
                    nonmonotonic_triplets=len(reversals))
    # 所有外层拟合在参数锁定之后，归档模型与输入/源码未变化。
    latest_lock=max(pd.Timestamp(c['locked_at']).timestamp() for c in choices)
    assert all(p.stat().st_mtime>=latest_lock for p in run.glob('predictions/outer_*/*/*/models/*'))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(digest(root.parent/p)==h for p,h in manifest['source_hashes'].items())
    assert all(digest(run/'inputs'/p)==h for p,h in json.loads((run/'input_hashes.json').read_text()).items())
    validation.update(outer_after_all_locks=True,current_source_hashes_pass=True,input_hashes_pass=True)
    (run/'validation_summary.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    prior=root/'outputs/learning_strategy_search/20260920T112021_620509Z/cumulative_search_ledger.csv'
    ledger=pd.read_csv(prior)
    extra=[]
    for path in sorted((root/'outputs/learning_extensions').glob('*/manifest.json')):
        m=json.loads(path.read_text())
        attempts=sum(len(p.read_text().splitlines()) for p in path.parent.glob('search/*/*/trials.jsonl'))
        extra.append(dict(run=str(path.parent.relative_to(root.parent)),role='extension_engineering_pilot' if m['pilot'] else 'formal_extension_search',
                          trials=attempts,status=m['status'],errors=m.get('errors')))
    ledger=pd.concat([ledger,pd.DataFrame(extra)],ignore_index=True)
    ledger.to_csv(run/'cumulative_search_ledger.csv',index=False)
    gate_quality=[x for x in quality if x['family'] in ['trade_filter','market_gate']]
    assert len(gate_quality)==6 and all(x['probability_brier']>x['past_prevalence_brier'] for x in gate_quality)
    lines=['# 第二批学习策略结果（2026-09-20）','',
           '本报告只汇总本次实际运行，不引用历史文档收益。数据、账户与成交约束沿用前一批。',
           f"本轮{len(summary)}类策略中，{int(summary.risk_pass.sum())}类通过30%最大回撤验收，{int((summary.total>0).sum())}类连续账户净收益为正。",
           f"正式运行：`{run.relative_to(root.parent)}`；{len(trials)}组策略参数、{validation['successful_fits']}次成功拟合批次、{validation['errors']}次参数试验错误。一个专家批次包含多个子模型，不能把批次数当成单模型数。",'',
           '## 连续账户（2021–2023）','',
           '起始10万元，年度间不重置资金；下表全部扣除配置费用。回撤列为跌幅绝对值。', '',
           '| 类别 | 累计净收益 | 年化 | 最大回撤 | 通过30%回撤线 | 有参数年度 |',
           '|---|---:|---:|---:|---|---:|']
    for r in summary.itertuples():
        lines.append(f'| {names[r.family]} | {r.total:.2%} | {r.ann:.2%} | {abs(r.mdd):.2%} | {"是" if r.risk_pass else "否"} | {r.selected_years}/3 |')
    lines+=['', '## 交易与资金使用', '', '| 类别 | 平均股票仓位 | 交易笔数 | 费用合计 |', '|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {names[r.family]} | {r.mean_invested:.1%} | {r.n_trades} | {r.total_fees:.2f}元 |')
    missing=[f"{names[c['family']]}：{c['outer_year']}" for c in choices if c['selected'] is None]
    lines+=['', '无可行参数折：'+('；'.join(missing) if missing else '无')+'。']
    lines+=['','回撤通过不等于有收益、优于基准或可实盘。无内层可行参数的年度按约定持现金，退出仍受次日成交及涨跌停约束；空仓年份不能视为选股成功。', '',
            '## 同参数门控对照','',
            '| 账户 | 累计净收益 | 最大回撤 | 平均股票仓位 | 交易笔数 |', '|---|---:|---:|---:|---:|']
    for r in summary[summary.family.isin(['trade_filter','market_gate'])].itertuples():
        lines.append(f'| {names[r.family]} | {r.total:.2%} | {abs(r.mdd):.2%} | {r.mean_invested:.1%} | {r.n_trades} |')
    for r in paired.itertuples():
        lines.append(f'| {r.strategy} | {r.total:.2%} | {abs(r.mdd):.2%} | {r.mean_invested:.1%} | {r.n_trades} |')
    lines+=['','取消门控对照保留同一年度频率、持股数及无可行参数时现金分支。static_half为预设5万元交易子账户加5万元不计息现金；它不是按外层结果精确匹配仓位。门控与纯仓位作用不能仅凭主表区分。', '',
            '两种门控在2021/2022/2023各年的概率Brier误差均高于用训练期事件率作常数预测的基准，本轮没有显示概率预测增益（这不是显著性检验）。门控类别的排名IC来自固定宿主原始分数，不能解释为门控学习器的选股技能。', '',
            '## 过程与验证','',
            f"- 参数预算：每类每折16组，共7类×3折；联合全局/局部搜索，不假设单调性。实际同轴三点反转记录数{len(reversals)}；未出现反转可能只是切片覆盖不足，不能推断单调。",
            f"- 已登记累计策略试验{int(ledger.trials.sum())}组（含历次规则、学习及工程试跑），不能当成独立显著性检验。",
            f"- {validation['independent_accounts']}个账户独立台账核验，共{validation['trades_checked']}笔交易；资金最大重放误差{validation['cash_max_error']:.3g}元；T+1、官方涨跌停和现金约束通过。",
            f"- {validation['real_prediction_prefix_checks']}类真实数据预测截断重放、{validation['account_prefix_checks']}次连续账户截断重放、{len(reload_rows)}个实际分支的模型保存加载重放通过。",
            '- 全部已成熟标签时序、模型文件哈希、运行源码及输入哈希通过；外层模型生成晚于所有参数锁定。',
            '- 新增模型与原学习、规则、成交相关测试合计60项通过；静态半资金对照的合并现金与资产恒等式通过。', '',
            '## 范围与可解释性','',
            '本轮神经网络是MLP；状态专家是KMeans而非HMM；交易过滤使用固定规则宿主，不是已交叉拟合的两层学习元模型；多周期为多输出Ridge。市场门控预测股票等权未来负收益事件，不直接预测真实账户最大回撤。',
            '动态IC与第一批简单滚动均值/ICIR不同，测试时间递推估计；相关性分组仅用每次拟合前收益。所有模型复用同一23项因子和原执行引擎。',
            '93股工程样本有全期质量隔离偏差，2021–2023亦已观察。只支持当前实现、参数预算与历史窗口内的结论，不可泛化到整个方法类别或未观察市场。',
            'CNN、HMM、GP、学习型持仓/退出与完整元标签堆叠仍未在本轮实现；海外关系及分钟强化学习不在当前冻结样本范围。', '',
            '## 文件','',
            '- `all_trials.csv/parquet`、`search_counts.csv`：全部候选、预算与可行数量。',
            '- `locked_selection_summary.csv`：逐年度锁定参数；`predictions/`：原始预测、模型及训练记录。',
            '- `forecast_quality.csv`：价格排名IC；两种门控的Brier误差及训练期事件率基准（仅诊断，不再选参）。',
            '- `outer/`、`paired/`、`controls/`：逐日账户、信号、交易、事件、持仓及独立台账审核。',
            '- `outer_yearly.csv`、`outer_nav_drawdown.png`、`validation_summary.json`、`cumulative_search_ledger.csv`。','']
    lines += ['## 复现入口', '', '从项目根目录运行：', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python datahub/scripts/frozen_research.py run --release datahub/releases/selection_v1_20260920 alpha_lab/scripts/run_learning_extensions.py --workers 7',
              f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python alpha_lab/scripts/report_learning_extensions.py --run {run.relative_to(root.parent)}', '```', '',
              '首条命令会创建新运行目录；第二条汇总本次已完成目录，新运行时相应替换路径。`--pilot`只做工程试跑，不生成完整外层结果。', '']
    (run/'REPORT.md').write_text('\n'.join(lines))
    doc=root/'docs/LEARNING_EXTENSIONS_RESULTS_20260920.md'
    doc.write_text('\n'.join(lines)+f'\n[完整运行目录](../outputs/learning_extensions/{run.name}/REPORT.md)\n')
    reporting=dict(script_sha256=digest(Path(__file__)),input_run=str(run),does_not_select_parameters=True)
    shutil.copy2(Path(__file__),run/'sources'/Path(__file__).name)
    (run/'reporting_manifest.json').write_text(json.dumps(reporting,indent=2))
    print(json.dumps(validation,ensure_ascii=False,indent=2))
    print(summary[['family','total','mdd','risk_pass','selected_years']].to_string(index=False))


if __name__=='__main__':
    main()
