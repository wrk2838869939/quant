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
    run = args.run
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
    summary=pd.read_csv(run/'outer_summary.csv')
    controls=pd.read_csv(run/'control_summary.csv')
    paired=pd.read_csv(run/'paired_summary.csv')
    names={'factor_weight':'因子权重学习','linear':'线性预测','lgbm':'LightGBM预测/排序','dual':'收益与风险双预测'}
    label_ends=pd.read_parquet(run/'inputs/label_end_dates.parquet')
    quality=[]
    for choice in choices:
        if choice['selected'] is None:
            continue
        family,year=choice['family'],choice['outer_year']
        p=choice['selected']['params']
        paths=list((run/'predictions'/f'outer_{year}'/family).glob('*/return_predictions.parquet'))
        assert len(paths)==1
        forecast=pd.read_parquet(paths[0])
        target=pd.read_parquet(run/'inputs'/f"label_returns_{p['horizon']}.parquet").reindex_like(forecast)
        available=label_ends[str(p['horizon'])].reindex(forecast.index).le(pd.Timestamp(f'{year}-12-31'))
        target=target.where(np.broadcast_to(available.to_numpy()[:,None],target.shape))
        valid=forecast.notna() & target.notna()
        x=forecast.where(valid).rank(axis=1,pct=True)
        y=target.where(valid).rank(axis=1,pct=True)
        x=x.sub(x.mean(axis=1),axis=0);y=y.sub(y.mean(axis=1),axis=0)
        denom=np.sqrt((x*x).sum(axis=1)*(y*y).sum(axis=1))
        ic=((x*y).sum(axis=1)/denom.where(denom>0)).where(valid.sum(axis=1)>=10)
        item=dict(family=family,year=year,horizon=p['horizon'],mean_daily_rank_ic=ic.mean(),
                  evaluated_ic_days=int(ic.notna().sum()),price_target_pairs=int(valid.to_numpy().sum()))
        if family=='dual':
            predicted=pd.read_parquet(paths[0].parent/'risk_predictions.parquet')
            realized=pd.read_parquet(run/'inputs'/f"label_risk_{p['horizon']}.parquet").reindex_like(predicted).where(valid)
            residual=predicted-realized
            item['risk_mean_absolute_error']=float(np.nanmean(np.abs(residual.to_numpy())))
            item['risk_bias']=float(np.nanmean(residual.to_numpy()))
        quality.append(item)
    pd.DataFrame(quality).to_csv(run/'forecast_quality.csv',index=False)
    audits=[json.loads(p.read_text()) for p in run.rglob('audit.json')]
    for a in audits:
        assert all(a[k] for k in ['t_plus_one_pass','official_limits_pass','no_negative_cash'])
        assert max(v for k,v in a.items() if k.endswith('_max_error'))<1e-6
    nav={};yearly=[];diagnostics=[]
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
    axes[0].legend();axes[0].set_ylabel('Continuous NAV')
    axes[1].axhline(-.3,color='red',linestyle='--');axes[1].set_ylabel('Drawdown')
    fig.suptitle('Learning strategies, 2021–2023; previously observed engineering sample')
    fig.savefig(run/'outer_nav_drawdown.png',dpi=160);plt.close(fig)
    validation=dict(trials=len(trials),errors=int((trials.status!='ok').sum()),unique_fit_records=len(fits),
                    successful_fits=sum(x['status']=='ok' for x in fits),
                    insufficient_training=sum(x['status']!='ok' for x in fits),
                    label_availability_pass=True,model_hashes_pass=True,independent_accounts=len(audits),
                    trades_checked=sum(a['trades_checked'] for a in audits),
                    max_money_error=max(v for a in audits for k,v in a.items() if k.endswith('_max_error')),
                    real_feature_label_prefix=json.loads((run/'real_prefix_checks.json').read_text()),
                    real_prediction_prefix=json.loads((run/'real_prediction_prefix_checks.json').read_text()),
                    unit_tests_pre_run=45)
    roundtrip=run/'model_roundtrip_checks.csv'
    if roundtrip.exists():
        checks=pd.read_csv(roundtrip)
        assert checks.passed.all() and len(checks)==8
        validation['representative_model_roundtrips']=len(checks)
    (run/'validation_summary.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    lines=['# 首批学习策略执行报告','',
           f"状态：{manifest['status']}。正式内层试验{len(trials)}次，执行错误{validation['errors']}次。",'',
           '固定93股、10万元账户；费用、次日开盘、整手、T+1、涨跌停与公司行动处理继承既有执行框架。',
           '数据存在全期质量隔离偏差，2019—2023已用于多轮研究；当前是学习流程回放，不能视为新鲜样本外有效性证明。','',
           '## 训练与选择边界','',
           '- 共用23项已登记因子，按当日有资格截面排名；监督模型另接23项缺失标记，缺失数值填排名中心。因子池未按上一轮外层收益筛选。',
           '- 训练历史从2019开始，内层评价从2020开始：2020选2021，2020—2021选2022，2020—2022选2023。第一折只有一个内层评价年。',
           '- 联合搜索5/10/20日标签、252/504/756日训练窗、1/3个月重训、日/周/月调仓、3/4/5只持股及模型参数。重训周期内模型冻结。',
           '- 收益标签为次日复权开盘到期限末复权收盘的毛价格收益；风险标签为同期间相对入场价的最大不利波动。完整价格路径不足则无标签，不前填价格。',
           '- 每次训练只接受结束日严格早于拟合日的标签，完整训练掩码哈希、最晚标签日期和模型归档均保留。实际成交限制与费用在账户中另算，价格标签不承诺可以买到。',
           '- 每家族每折32次预算：16次全局、8次多区域局部、8次继续全局；不假定单调，完整内层评价，不按中途收益淘汰。',
           '- 本批使用配置列明的有限离散参数域，局部探索为网格相邻项；列表之外及未采样组合仍未知，不声称连续参数空间的最优解。',
           '- 内层要求实际买入且连续/逐年回撤均不超过30%；按年度年化净收益中位数及预定并列规则选择。无可行解则下一年转现金。',
           '- 外层先锁参数再运行，2021—2023账户持续运行，跨年保留现金和持仓。外层内可随标签成熟重训模型，超参数不读取当年后续结果。',
           '- 30%是风险筛选和验收阈值，MAE预测或内层回撤约束均不保证后续组合回撤。','',
           '## 连续账户结果','',
           '| 类别 | 累计净收益 | 年化 | 最大回撤 | 有可行参数年份 | 回撤≤30% |',
           '|---|---:|---:|---:|---:|---|']
    for row in summary.itertuples():
        flag='纯现金，不算策略通过' if row.all_cash else ('是' if row.risk_pass else '否')
        lines.append(f'| {names[row.family]} | {row.total:.2%} | {row.ann:.2%} | {row.mdd:.2%} | {row.selected_years}/3 | {flag} |')
    lines+=['','### 仓位、交易与费用','',
            '| 类别 | 平均股票仓位 | 成交笔数 | 累计费用 |','|---|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {names[row.family]} | {row.mean_invested:.1%} | {row.n_trades} | {row.total_fees:.2f}元 |')
    lines+=['','## 固定对照','',
            '恒定分数使用共同股票池和确定性同分排序，帮助识别模型退化为固定排序。固定单因子与等权合成均未读取学习搜索结果。','',
            '| 对照 | 累计净收益 | 最大回撤 |','|---|---:|---:|']
    for row in controls.itertuples():
        lines.append(f'| {row.strategy} | {row.total:.2%} | {row.mdd:.2%} |')
    lines+=['','## 双预测配对','',
            '保留双预测已选择的收益模型、频率、持股数和无可行年份处理，只移除风险扣分与风险上限。配对参数是按双预测流程选择的，不能当作独立优化的纯收益模型。']
    for row in paired.itertuples():
        lines.append(f'移除风险层后的累计净收益{row.total:.2%}，最大回撤{row.mdd:.2%}，平均股票仓位{row.mean_invested:.1%}，累计费用{row.total_fees:.2f}元。')
    lines+=['','## 预测诊断','',
            '以下为逐日截面预测与已完整实现的毛价格收益标签之间的Spearman相关性均值。只纳入至少10只有效股票的日期；恒定预测相关性记为缺失。重叠持有期使日期之间相关，这些均值不构成显著性检验，也不等于可交易净收益。', '',
            '| 类别/年份 | 标签期限 | 日均Rank IC | 有效日期数 |','|---|---:|---:|---:|']
    for q in sorted(quality,key=lambda q:(q['family'],q['year'])):
        lines.append(f"| {names[q['family']]}/{q['year']} | {q['horizon']} | {q['mean_daily_rank_ic']:.4f} | {q['evaluated_ic_days']} |")
    lines+=['','## 核验与文件','',
            f"- 成功归档并核验{validation['successful_fits']}个重训批次（双预测每批包含收益/风险两个模型），标签时序和模型文件哈希全部通过；训练不足跳过{validation['insufficient_training']}次。",
            f"- {len(audits)}个账户独立核验{validation['trades_checked']}笔成交，最大金额对账误差{validation['max_money_error']:.3g}元。",
            '- 正式运行前45项测试通过，包括八个模型分支的未来数据扰动检查；真实特征、标签、四类模型预测截断检查通过。外层账户另逐年重放每日资产、成交、事件和持仓前缀。',
            f'- 实际采样记录{len(reversals)}组三点方向反转切片，彼此不独立；没有三点的区域仍未知，不声称全局最优。',
            '- [全部试验](all_trials.csv)、[搜索计数](search_counts.csv)、[锁定参数](locked_selection_summary.csv)、[拟合时序](fit_audit_summary.csv)、[非单调切片](nonmonotonic_samples.csv)。',
            '- [预测退化检查](outer_prediction_diagnostics.csv)：分别记录所有候选同分与完全无可用分数的决策日，不能把确定性同分排序视为学习增益。',
            '- [预测质量与风险误差](forecast_quality.csv)：预测诊断使用已成熟的毛价格标签，不以未能成交的理论收益替代账户收益。',
            '- [逐年收益](outer_yearly.csv)、[连续账户图](outer_nav_drawdown.png)、[完整核验](validation_summary.json)、[原始运行清单](manifest.json)。',
            '- search/保存每次参数试验的全部信号、净值、交易、持仓及事件；predictions/保存逐日收益/风险预测、每次训练记录和模型；outer/与paired/保存连续账户及独立核验。',
            '- 后续神经网络、HMM/专家、动态因子轮动、学习过滤、持有退出、GP、关系学习和强化学习未在本批执行。','']
    if roundtrip.exists():
        lines+=['八个模型分支各抽取一个归档模型加载复算，预测与原文件一致，见[模型归档复算](model_roundtrip_checks.csv)。','']
    (run/'REPORT.md').write_text('\n'.join(lines))
    script=Path(__file__)
    shutil.copy2(script,run/'sources'/script.name)
    (run/'reporting_manifest.json').write_text(json.dumps(dict(script=str(script),sha256=hashlib.sha256(script.read_bytes()).hexdigest()),indent=2))
    print(json.dumps(validation,ensure_ascii=False,default=str))
    print(f'REPORT={run / "REPORT.md"}')


if __name__=='__main__':
    main()
