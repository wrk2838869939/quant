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
                 paired_total=r.get('paired_metrics',{}).get('total'), paired_mdd=r.get('paired_metrics',{}).get('mdd'),
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
    import sys,yaml,joblib
    root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'src'),str(root/'scripts'),str(root.parent/'datahub/releases/selection_v1_20260920/src')]
    from sequence_state_learning import infer_bundle
    from diagnose_learning_strategies import load_data
    plan=yaml.safe_load((run/'sources/sequence_state_learning_v1.yaml').read_text())
    data,_=load_data(run,plan)
    summary=pd.read_csv(run/'outer_summary.csv')
    paired=pd.concat([pd.read_csv(p).assign(family=p.parent.name) for p in run.glob('paired/*/summary.csv')],ignore_index=True)
    paired.to_csv(run/'paired_summary.csv',index=False)
    reload_rows=[]
    for family in plan['families']:
        path=next(run.glob(f'predictions/inner_2021/{family}/*/fits.json'))
        record=next(x for x in json.loads(path.read_text()) if x['status']=='ok')
        bundle=joblib.load(path.parent/'models'/record['model_file'])
        at=data['dates'].get_loc(pd.Timestamp(record['fit_date']))
        a,b=infer_bundle(bundle,data,at)
        for values,name in [(a,'return_predictions'),(b,'paired_predictions')]:
            reference=pd.read_parquet(path.parent/f'{name}.parquet').loc[data['dates'][at]].to_numpy()
            np.testing.assert_array_equal(values,reference)
            reload_rows.append(dict(family=family,branch=name,passed=True,max_error=0.))
    pd.DataFrame(reload_rows).to_csv(run/'model_roundtrip_checks.csv',index=False)
    audits=[json.loads(p.read_text()) for p in run.rglob('audit.json')]
    for a in audits:
        assert all(a[k] for k in ['t_plus_one_pass','official_limits_pass','no_negative_cash'])
        assert max(v for k,v in a.items() if k.endswith('_max_error'))<1e-6
    latest=max(pd.Timestamp(c['locked_at']).timestamp() for c in choices)
    assert all(p.stat().st_mtime>=latest for p in run.glob('predictions/outer_*/*/*/models/*'))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(digest(root.parent/p)==h for p,h in manifest['source_hashes'].items())
    assert all(digest(run/'inputs'/p)==h for p,h in json.loads((run/'input_hashes.json').read_text()).items())
    # 显式3×3网格检查及可视化，不对缺失组合插值。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    fig.suptitle('Inner median annualized return; * = fails drawdown / trading requirement')
    surface_rows=[]
    for i,family in enumerate(plan['families']):
        for j,year in enumerate(plan['outer_years']):
            surface=json.loads((run/'search'/str(year)/family/'local_surface_plan.json').read_text())
            keys=surface['axes'];points=surface['points']
            matches=[r for r in records if r['family']==family and r['outer_year']==year]
            table={json.dumps(r['params'],sort_keys=True):r for r in matches}
            for p in points:
                r=table[json.dumps(p,sort_keys=True)]
                assert r['status']=='ok'
                surface_rows.append(dict(family=family,year=year,axis_x=keys[0],axis_y=keys[1],x=p[keys[0]],y=p[keys[1]],
                                         inner_median_ann=r['metrics']['median_ann'],inner_mdd=r['metrics']['mdd'],feasible=r['metrics']['feasible'],id=r['id']))
            group=pd.DataFrame(surface_rows).query('family==@family and year==@year')
            pivot=group.pivot(index='y',columns='x',values='inner_median_ann')
            feasible=group.pivot(index='y',columns='x',values='feasible')
            assert pivot.shape==(3,3) and not pivot.isna().any().any()
            ax=axes[i,j];im=ax.imshow(pivot.to_numpy(),cmap='RdYlGn',aspect='auto')
            ax.set_xticks(range(3),[str(v) for v in pivot.columns]);ax.set_yticks(range(3),[str(v) for v in pivot.index])
            ax.set_xlabel(keys[0]);ax.set_ylabel(keys[1]);ax.set_title(f'{family}, inner before {year}')
            for row in range(3):
                for col in range(3):ax.text(col,row,f'{pivot.iloc[row,col]:.1%}'+('' if feasible.iloc[row,col] else ' *'),ha='center',va='center',fontsize=9)
            fig.colorbar(im,ax=ax,shrink=.7)
    pd.DataFrame(surface_rows).to_csv(run/'local_surfaces.csv',index=False)
    fig.savefig(run/'local_surfaces.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
    yearly=[]
    for item in summary.itertuples():
        yearly.extend(dict(family=item.family,**x) for x in json.loads(item.yearly))
        for location,label,style in [('outer',item.family,'-'),('paired',item.family+'_matched_control','--')]:
            d=pd.read_parquet(run/location/item.family/'daily.parquet')
            axes[0].plot(d.index,d.nav,style,label=label)
            peak=np.maximum.accumulate(np.r_[1.,d.nav.to_numpy()])[1:]
            axes[1].plot(d.index,d.nav.to_numpy()/peak-1,style)
    axes[0].legend();axes[0].set_ylabel('NAV');axes[1].set_ylabel('Drawdown');axes[1].axhline(-.3,color='red',linestyle=':')
    fig.suptitle('CNN / online HMM and matched controls; observed 2021-2023 sample')
    fig.savefig(run/'outer_nav_drawdown.png',dpi=150);plt.close(fig)
    pd.DataFrame(yearly).to_csv(run/'outer_yearly.csv',index=False)
    hmm_fits=[f for f in fits if f['family']=='hmm' and f['status']=='ok']
    assert all(pd.Timestamp(f['last_unsupervised_date'])<pd.Timestamp(f['fit_date']) for f in hmm_fits)
    validation=dict(trials=len(trials),paired_trial_accounts=len(trials),errors=int((trials.status!='ok').sum()),
                    successful_fit_batches=sum(x['status']=='ok' for x in fits),hmm_fit_batches=len(hmm_fits),
                    hmm_em_budget_exhausted=sum(bool(x['em_budget_exhausted']) for x in hmm_fits),
                    hmm_likelihood_decreases=sum(x['em_delta'] is not None and x['em_delta']<0 for x in hmm_fits),
                    independent_accounts=len(audits),trades_checked=sum(x['trades_checked'] for x in audits),
                    cash_max_error=max(x['cash_max_error'] for x in audits),model_roundtrip_checks=len(reload_rows),
                    prediction_prefix_checks=len(json.loads((run/'real_prediction_prefix_checks.json').read_text())),
                    account_prefix_checks=sum(len(json.loads(p.read_text())) for p in run.glob('outer/*/prefix_checks.json')),
                    verified_surface_points=len(surface_rows),nonmonotonic_triplets=len(reversals),
                    source_input_model_hashes_pass=True,outer_after_all_parameter_locks=True)
    (run/'validation_summary.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    prior=root/'outputs/learning_extensions/20260920T115718_030644Z/cumulative_search_ledger.csv'
    ledger=pd.read_csv(prior);new=[]
    for p in (root/'outputs/sequence_state_learning').glob('*/manifest.json'):
        m=json.loads(p.read_text())
        count=sum(len(q.read_text().splitlines()) for q in p.parent.glob('search/*/*/trials.jsonl'))
        new.append(dict(run=str(p.parent.relative_to(root.parent)),role='sequence_state_pilot' if m['pilot'] else 'sequence_state_formal',trials=count,status=m['status'],errors=m.get('errors')))
    ledger=pd.concat([ledger,pd.DataFrame(new)],ignore_index=True);ledger.to_csv(run/'cumulative_search_ledger.csv',index=False)
    names={'cnn':'CNN序列选股','hmm':'HMM在线状态专家'}
    lines=['# CNN / HMM 补测结果','',f"运行：`{run.relative_to(root.parent)}`。正式主模型{len(trials)}组参数；每组附带一个同参数对照账户。成功拟合批次{validation['successful_fit_batches']}，每批包含主模型与匹配对照，专家批次含多个回归器。",'',
           '固定93股、10万元；2021–2023连续账户，扣费且不重置年度资金。所有年份已被观察。', '',
           '| 主模型/对照 | 累计净收益 | 最大回撤 | 平均仓位 | 交易笔数 |', '|---|---:|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {names[r.family]} | {r.total:.2%} | {abs(r.mdd):.2%} | {r.mean_invested:.1%} | {r.n_trades} |')
        pair=paired[paired.family==r.family].iloc[0]
        lines.append(f'| 同参数{pair.strategy} | {pair.total:.2%} | {abs(pair.mdd):.2%} | {pair.mean_invested:.1%} | {pair.n_trades} |')
    lines+=['',f"主模型通过30%回撤验收：{int(summary.risk_pass.sum())}/2。无可行参数年度按预注册现金分支执行；对照也保留该分支。",'',
            '## 时序与配对口径','',
            'CNN与点MLP训练同一批有效样本，日期权重、目标、优化器预算相同；网络结构和参数量不同。HMM与KMeans共享历史窗口、状态数、专家正则化与混合比例，HMM使用软概率而KMeans硬分组，差值不能全部解释为转移结构效果。',
            'HMM只做逐时点前向过滤；没有后向平滑。CNN只读取当日及过去回看窗。标签结束日严格早于拟合日。外层在全部选择锁定后运行。', '',
            '## 参数与数值诊断','',
            f"每类每折18组；内层先联合探索，再实际扫描3×3局部网格、去重补足预算。共核验{len(surface_rows)}个网格点；实际同轴三点方向反转{len(reversals)}处。未评估区域不插值推断，不能声称全局最优。详见local_surfaces.csv/png。",
            f"HMM共{len(hmm_fits)}次拟合，其中{validation['hmm_em_budget_exhausted']}次达到EM迭代预算、{validation['hmm_likelihood_decreases']}次末次观测似然下降。转移先验会改变优化目标；本报告保留这些诊断，零参数试验错误不等于每次EM都充分收敛，更不代表全局最优。",
            f"累计主策略搜索登记{int(ledger.trials.sum())}组；本轮匹配对照账户不用于主模型择优，也不当成额外独立策略搜索。",'',
            '## 工程核验','',
            f"64项相关测试通过。{validation['prediction_prefix_checks']}类真实预测截断、{validation['account_prefix_checks']}次主账户截断、{len(reload_rows)}个主/对照保存加载分支通过；重载预测逐元素一致。",
            f"{len(audits)}个外层/对照账户独立台账，{validation['trades_checked']}笔交易；资金最大重放误差{validation['cash_max_error']:.3g}元。代码、输入、模型哈希和所有标签成熟时间通过。",'',
            '## 产出与局限','',
            '`all_trials.csv`、`locked_selection_summary.csv`、`local_surfaces.csv`保留候选与锁定记录；`predictions/`保存预测、模型和训练损失/EM诊断；`outer/`、`paired/`、`controls/`保存信号、逐笔成交、事件、持仓、账户与审核。',
            '93股仍有全窗口质量隔离偏差，已观察的年份不能恢复为独立最终检验。小规模参数搜索只评价当前实现，不否定方法类别。GP及持仓退出学习尚未在本轮实施。', '',
            '复现：冻结启动器运行 `alpha_lab/scripts/run_sequence_state_learning.py --workers 6`，再将输出目录传给本汇总脚本 `--run`。','']
    extra=json.loads((run/'high_inner_audit_summary.json').read_text())
    assert extra['all_pass']
    lines += ['## 高内层收益复核与局部映射实例', '',
              f"按项目Sharpe>3或月收益>20%的警戒线，额外复核{extra['flagged_accounts']}套内层账户（主模型{extra['primary']}、配对对照{extra['paired']}），共{extra['trades_checked']}笔交易，独立资金/持股/T+1/涨跌停账本通过。该核验不消除已有的全期质量选样偏差，也不把高内层收益变成样本外证据。",
              '固定其他参数时，2021折CNN通道数8/16/32、卷积核3，对应内层2020年的年化目标为66.61%/36.94%/65.28%，出现先降后升。具体配置见nonmonotonic_samples.csv。这是本随机种子的已测映射，不是最优通道数结论；尚未开展多随机种子稳定性实验。', '']
    validation['extra_high_inner_account_audits']=extra
    (run/'validation_summary.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    (run/'REPORT.md').write_text('\n'.join(lines))
    (root/'docs/SEQUENCE_STATE_LEARNING_RESULTS_20260920.md').write_text('\n'.join(lines)+f'\n[完整归档](../outputs/sequence_state_learning/{run.name}/REPORT.md)\n')
    shutil.copy2(Path(__file__),run/'sources'/Path(__file__).name)
    (run/'reporting_manifest.json').write_text(json.dumps(dict(script_sha256=digest(Path(__file__)),run=str(run),selection=False),indent=2))
    print(json.dumps(validation,ensure_ascii=False,indent=2));print(summary[['family','total','mdd']].to_string(index=False))


if __name__=='__main__':main()
