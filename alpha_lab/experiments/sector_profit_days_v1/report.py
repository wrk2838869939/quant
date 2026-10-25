"""比较新旧年度模型，所有模型预测完成后才读取评价标签。"""
from pathlib import Path
import hashlib
import json
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import yaml
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path('/root/autodl-tmp');HERE=Path(__file__).resolve().parent
cfg=yaml.safe_load((HERE/'config.yaml').read_text());OUT=ROOT/cfg['output'];BASE=ROOT/cfg['source_run']
sys.path.insert(0,str(HERE))
from model import FuzzyNetwork,predict_network

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save_json(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str))

assert json.loads((OUT/'models_complete.json').read_text())['status']=='complete'
newpred=pd.read_parquet(OUT/'predictions_BEFORE_EVALUATION.parquet')
oldpred=pd.read_parquet(BASE/'predictions_BEFORE_EVALUATION.parquet')
oldpred['variant']='original_return';oldpred['model_year']=oldpred.date.dt.year
labels=pd.read_parquet(OUT/'research_inputs_WITH_NEW_LABELS.parquet')
pred=pd.concat([oldpred[newpred.columns],newpred],ignore_index=True)
data=pred.merge(labels[['event_id','target','complete','terminal_return','positive_days','negative_days','flat_days']],on='event_id',validate='many_to_one')
data['year']=data.date.dt.year
data.to_csv(OUT/'all_event_comparison.csv',index=False)
known=data[data.complete].copy()
baseline=known[known.variant.eq('original_return')].copy()
calendar=pd.DatetimeIndex(sorted(pd.read_parquet(ROOT/cfg['dataset']/'daily.parquet',columns=['date']).date.unique()))
dates=sorted(baseline.date.unique());positions=calendar.get_indexer(dates)
cluster_map=dict(zip(dates,np.cumsum(np.r_[0,np.diff(positions)>cfg['bootstrap']['cluster_gap']])))
known['cluster']=known.date.map(cluster_map)
names={'original_return':'旧模型：20日终点收益','label_only_scale_0p1':'主实验：仅换盈亏天数标签','bounded_scale_1':'缩放敏感性：新标签/1.0'}

def stats(g):
    passed=g[g.selected];rejected=g[~g.selected]
    pos=g.target.gt(0);sel=g.selected
    tp=int((pos&sel).sum());fp=int((~pos&sel).sum());fn=int((pos&~sel).sum());tn=int((~pos&~sel).sum())
    return dict(n=len(g),passed=len(passed),rejected=len(rejected),passed_s=passed.target.mean(),rejected_s=rejected.target.mean(),
                pass_reject_gap=passed.target.mean()-rejected.target.mean(),all_s=g.target.mean(),
                passed_profit_days=passed.positive_days.mean(),passed_s_median=passed.target.median(),
                passed_terminal=passed.terminal_return.mean(),passed_terminal_q10=passed.terminal_return.quantile(.1),
                rejected_terminal=rejected.terminal_return.mean(),precision=tp/(tp+fp) if tp+fp else np.nan,
                recall=tp/(tp+fn) if tp+fn else np.nan,accuracy=(tp+tn)/len(g),TP=tp,FP=fp,TN=tn,FN=fn)

rows=[]
for period,subset in [('2019–2026',known),('2019–2025',known[known.year.lt(2026)])]+[(str(y),g) for y,g in known.groupby('year')]:
    for variant,g in subset.groupby('variant',sort=False):
        allg=data[data.variant.eq(variant)]
        if period=='2019–2025':allg=allg[allg.year.lt(2026)]
        elif period.isdigit():allg=allg[allg.year.eq(int(period))]
        rows.append(dict(period=period,variant=variant,total=len(allg),passed_all=int(allg.selected.sum()),
                         excluded=int((~allg.complete).sum()),passed_excluded=int((allg.selected&~allg.complete).sum()),**stats(g)))
results=pd.DataFrame(rows);results.to_csv(OUT/'metrics_by_year.csv',index=False)

# 各模型用同一事件行情簇做配对重采样，保留选中样本数变化。
rng=np.random.default_rng(cfg['bootstrap']['seed']);intervals=[]
for period,subset in [('2019–2026',known),('2019–2025',known[known.year.lt(2026)])]:
    pieces=[g for _,g in subset.groupby('cluster')]
    draws=[]
    for _ in range(cfg['bootstrap']['repetitions']):
        sample=pd.concat([pieces[i] for i in rng.integers(0,len(pieces),len(pieces))],ignore_index=True)
        values={v:stats(g) for v,g in sample.groupby('variant')}
        for v in names:
            draws.append(dict(variant=v,passed_s=values[v]['passed_s'],gap=values[v]['pass_reject_gap'],
                              improvement=values[v]['passed_s']-values['original_return']['passed_s'],
                              terminal_change=values[v]['passed_terminal']-values['original_return']['passed_terminal']))
    bootstrap=pd.DataFrame(draws)
    observed=results[results.period.eq(period)].set_index('variant')
    for v in names:
        estimates=dict(passed_s=observed.loc[v,'passed_s'],gap=observed.loc[v,'pass_reject_gap'],
                       improvement=observed.loc[v,'passed_s']-observed.loc['original_return','passed_s'],
                       terminal_change=observed.loc[v,'passed_terminal']-observed.loc['original_return','passed_terminal'])
        for metric,estimate in estimates.items():
            values=bootstrap[bootstrap.variant.eq(v)][metric].dropna()
            intervals.append(dict(period=period,variant=v,metric=metric,estimate=estimate,low=values.quantile(.025),
                                  high=values.quantile(.975),clusters=len(pieces),replicates=len(values)))
intervals=pd.DataFrame(intervals);intervals.to_csv(OUT/'paired_cluster_intervals.csv',index=False)

# 入选变化按同一事件逐条匹配；缺标签仍计数量，不参与均值。
change=[]
original=data[data.variant.eq('original_return')].set_index('event_id')
for variant in names:
    if variant=='original_return':continue
    g=data[data.variant.eq(variant)].set_index('event_id')
    for event_id,r in g.iterrows():
        was=bool(original.loc[event_id,'selected']);now=bool(r.selected)
        status='都通过' if was and now else '新增通过' if now else '转为拒绝' if was else '都拒绝'
        change.append(dict(variant=variant,event_id=event_id,date=r.date,year=r.year,sector_code=r.sector_code,
                           status=status,old_selected=was,new_selected=now,complete=r.complete,target=r.target,terminal_return=r.terminal_return))
changes=pd.DataFrame(change);changes.to_csv(OUT/'selection_changes.csv',index=False)
change_summary=changes.groupby(['variant','status']).agg(events=('event_id','size'),complete=('complete','sum'),mean_s=('target','mean'),mean_terminal=('terminal_return','mean')).reset_index()
change_summary.to_csv(OUT/'selection_change_summary.csv',index=False)

# 缺失标签边界，假设未知S属于[-1,1]，不补值。
boundrows=[]
for variant,g in data.groupby('variant'):
    for selected in [True,False]:
        q=g[g.selected.eq(selected)];missing=int((~q.complete).sum());total=q.target.sum()
        boundrows.append(dict(variant=variant,group='通过' if selected else '拒绝',events=len(q),unknown=missing,
                              low=(total-missing)/len(q),high=(total+missing)/len(q)))
bounds=pd.DataFrame(boundrows);bounds.to_csv(OUT/'missing_bounds.csv',index=False)

fits=json.loads((OUT/'fits.json').read_text());fit_table=pd.DataFrame([dict(variant=f['variant'],year=f['year'],train=len(f['train_ids']),fit=len(f['fit_ids']),validation=len(f['valid_ids']),
    rules=f['rules'],learning_rate=f['learning_rate'],epochs=f['epochs'],threshold=f['threshold'],last_label=f['train_last_label'],target_scale=f['target_scale']) for f in fits])
fit_table.to_csv(OUT/'annual_training_summary.csv',index=False)

def saved_score(variant,feature_frame):
    folder=OUT/variant;mcfg=yaml.safe_load((folder/'model_config.yaml').read_text());values=[]
    for seed in cfg['model']['ensemble_seeds']:
        saved=torch.load(folder/'models'/f'2026_{seed}.pt',map_location='cpu',weights_only=False)
        net=FuzzyNetwork.__new__(FuzzyNetwork);torch.nn.Module.__init__(net)
        for key,value in saved['state_dict'].items():net.register_parameter(key,torch.nn.Parameter(torch.empty_like(value)))
        net.width_floor=mcfg['fnn']['width_floor'];net.load_state_dict(saved['state_dict']);net.eval()
        x=feature_frame[mcfg['features']].to_numpy(float)
        x=np.clip((np.where(np.isfinite(x),x,saved['median'])-saved['mean'])/saved['scale'],-mcfg['feature_clip'],mcfg['feature_clip']).astype('float32')
        values.append(predict_network(net,x)*mcfg['target_scale'])
    scores=np.mean(values,axis=0);threshold=saved['audit']['threshold']
    return scores,threshold

latest=ROOT/'alpha_lab/outputs/strategy_2026_opportunity_scan/20261020_eod'
fresh_features=pd.read_parquet(latest/'interval_sector_features.parquet')
fresh_events=pd.read_csv(latest/'today_first_confirmations_asof_model.csv')
live=[]
for variant in names:
    if variant=='original_return':
        score=float(fresh_events.score.iloc[0]);threshold=float(fresh_events.threshold.iloc[0])
    else:
        values,threshold=saved_score(variant,fresh_features);score=float(values[0])
    live.append(dict(variant=variant,sector='养殖业',date='2026-10-20',score=score,threshold=threshold,selected=score>=threshold,diagnostic_only=True))
    hotel=data[(data.variant==variant)&(data.date==pd.Timestamp('2026-09-17'))&(data.sector_code=='801219.SI')].iloc[0]
    live.append(dict(variant=variant,sector='酒店餐饮（原首次事件）',date='2026-09-17',score=hotel.score,threshold=hotel.threshold,selected=hotel.selected,diagnostic_only=True))
live=pd.DataFrame(live);live.to_csv(OUT/'recent_event_diagnostics.csv',index=False)

font=FontProperties(fname=str(ROOT/'fonts/SimHei.ttf'));plt.rcParams['axes.unicode_minus']=False
colors={'original_return':'#627d98','label_only_scale_0p1':'#217964','bounded_scale_1':'#c58c3e'}
fig,axs=plt.subplots(1,2,figsize=(13,4.8));years=sorted(known.year.unique());x=np.arange(len(years))
for i,(variant,label) in enumerate(names.items()):
    r=results[(results.variant==variant)&results.period.isin([str(y) for y in years])].set_index('period').loc[[str(y) for y in years]]
    axs[0].bar(x+(i-1)*.25,r.passed_s,width=.25,label=label,color=colors[variant])
    axs[1].plot(x,r.passed,'o-',label=label,color=colors[variant])
for ax in axs:
    ax.set_xticks(x,years);ax.grid(axis='y',alpha=.2);ax.legend(prop=font,fontsize=9)
axs[0].axhline(0,color='#444',lw=.8);axs[0].set_ylim(-1,1);axs[0].set_title('通过组盈亏天数指标均值（完整标签）',fontproperties=font)
axs[1].set_title('通过事件数量（完整标签）',fontproperties=font)
fig.tight_layout();fig.savefig(OUT/'yearly_comparison.png',dpi=160);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for ax,metric,title in zip(axs,['improvement','terminal_change'],['通过组S均值相对旧模型变化','通过组终点平均收益相对旧模型变化']):
    rows=intervals[(intervals.period=='2019–2026')&(intervals.metric==metric)&(intervals.variant!='original_return')]
    for i,r in enumerate(rows.itertuples()):
        ax.plot([r.low,r.high],[i,i],color=colors[r.variant],lw=3);ax.scatter(r.estimate,i,color=colors[r.variant],s=50)
    ax.axvline(0,color='#555',ls='--');ax.set_yticks(range(len(rows)),[names[v] for v in rows.variant],fontproperties=font)
    ax.set_title(title+'\n配对行情簇95%区间',fontproperties=font);ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'comparison_intervals.png',dpi=160);plt.close(fig)

def table(r):return r.to_markdown(index=False)
def show_results(frame):
    r=frame[['period','variant','n','passed','passed_excluded','passed_s','rejected_s','passed_profit_days','passed_terminal','precision','recall']].copy()
    r['variant']=r.variant.map(names)
    for key in ['passed_s','rejected_s']:r[key]=r[key].map(lambda v:f'{v:+.3f}')
    r['passed_profit_days']=r.passed_profit_days.map(lambda v:f'{v:.2f}')
    for key in ['passed_terminal','precision','recall']:r[key]=r[key].map(lambda v:f'{v:.2%}')
    r.columns=['期间','模型','完整事件','完整通过','通过但无标签','通过组S','拒绝组S','通过组盈利天数','通过组终点收益','S>0精确率','S>0召回率']
    return table(r)

aggregate=results[results.period.isin(['2019–2026','2019–2025','2026'])]
ci_show=intervals[intervals.metric.isin(['improvement','gap'])&(intervals.variant!='original_return')].copy();ci_show['variant']=ci_show.variant.map(names)
for key in ['estimate','low','high']:ci_show[key]=ci_show[key].map(lambda v:f'{v:+.4f}')
live_show=live.copy();live_show['variant']=live_show.variant.map(names)
fit_show=fit_table.copy();fit_show['variant']=fit_show.variant.map(names)
main=results[(results.period=='2019–2026')&(results.variant=='label_only_scale_0p1')].iloc[0]
old=results[(results.period=='2019–2026')&(results.variant=='original_return')].iloc[0]
improvement=intervals[(intervals.period=='2019–2026')&(intervals.variant=='label_only_scale_0p1')&(intervals.metric=='improvement')].iloc[0]
yearly_table=show_results(results[results.period.isin([str(y) for y in years])])
text=f'''# 年度板块模型：盈亏天数标签重训实验

2026-10-21。研究实验，未替换正式模型。沿用原331个事件及13项特征，重做2019—2026年度训练；最终对同一256个年度预测事件比较，其中226个有完整20日标签。原数据工程和模型权重未改动。

**主实验通过组S均值{main.passed_s:+.3f}，旧模型为{old.passed_s:+.3f}，差{improvement.estimate:+.4f}；配对行情簇95%区间[{improvement.low:+.4f}, {improvement.high:+.4f}]。** 这份报告只评价新旧模型，不根据结果改选股规则或下单。

本次可评价样本没有显示换标签带来改善：主实验通过组平均盈利天数下降，且通过组S均值低于自身拒绝组；预设缩放检查组也没有改善。主实验全样本的重采样差值区间上界非常接近0，剔除2026年的历史期区间仍跨0，不将其解释成稳定的显著劣势。新模型通过但缺标签的事件更多，必须保留缺失偏差。这也不构成旧模型已被证明有效的证据。

## 标签和实验口径

新标签S=(未来20日相对确认日收盘盈利天数−亏损天数)/20；第1—20个交易日，平价记0，范围[-1,1]。必须全路径有效且同一指数连续段；缺失不缩短窗口、不补零。原标签是20日终点收益，新标签衡量盈利持续性而不衡量盈亏幅度。

主实验只替换标签，target_scale保留0.1，即网络学习S/0.1，输出乘回0.1。另在训练前固定scale=1.0敏感性，网络直接学习S。SmoothL1损失、权重衰减和优化对标签尺度敏感，因此两组可能不同；不能把尺度差异伪装成纯标签收益。没有按测试结果选择上线版本。

两组都沿用原年度时间推进：最多此前8年、标签在年初前20交易日截止之前成熟；内层按日期末25%验证，同样隔离标签；仅历史内层选4/8规则、学习率0.003/0.01及早停轮数。3种子集成；同日事件降权、历史半衰期1095日。

**每个新模型的门槛由该年度历史训练预测中位数确定，不能套用旧收益分数门槛0.003289。** 新输出是S的回归评分，不是收益率或胜率；FNN不是有界输出层，本实验不事后截断预测。

## 新旧模型比较

{show_results(aggregate)}

通过组终点收益是未扣费的自算板块指数观察。S>0精确率指通过事件中“盈利天数多于亏损天数”的比例；召回率仅针对本次形态事件池中S>0的事件。不能与上一份“终点收益>0”的分类指标混淆。

![逐年新旧比较](yearly_comparison.png)

{yearly_table}

## 配对比较与入选变化

同一行情簇同时重采样新旧模型预测，区间保留同日板块及相近事件相关性。improvement为新模型通过组S均值减旧模型通过组S均值；gap为该模型通过组减拒绝组。各模型通过数量不相同，本实验未强行在测试集调门槛配平数量。

{table(ci_show)}

![配对区间](comparison_intervals.png)

{table(change_summary.round(4))}

## 最近事件：仅作新旧模型诊断

{table(live_show.round(6))}

酒店餐饮行对应9月17日首次事件，不是今天新买点；养殖业行对应10月20日首次确认。二者完整20日标签均尚未成熟，新评分没有利用它们未来走势。上述仅为研究模型诊断，不改变原扫描结果。

## 年度拟合记录

{table(fit_show.round(6))}

所有年度新旧训练/拟合/验证事件ID完全一致。2026模型训练最晚标签仍为2025-11-18；文件实际于2026-10-21训练，不能称当年初真实存在。

## 缺失、偏差和边界

原256条预测有30条无完整评价标签，其中29条路径缺失或连续段中断、1条9月17日酒店餐饮未成熟。2025年37条中25条无法评价；结果不代表完整全年。通过但无标签数量已列入表，不删除候选数。按未知S取[-1,1]得到各组全样本均值数学边界：

{table(bounds.round(4))}

本次是在查看旧模型表现之后改变目标的研究，使用今天的历史供应商数据，存在开发与历史版本偏差；即使某组改善，也不是全新前瞻检验。2026仅16个成熟事件。盈利持续性不限制最大亏损或终点损失，必须与收益幅度并看。

个股模型没有重训。新板块模型会改变个股模型历史训练事件分布，不能直接组合后声称全流程已验证。本次不改正式策略配置、生产扫描或数据构建代码。

## 核验与复现

- 331个事件的新旧标签完整性和结束日期逐条一致，288个完整标签的旧终点收益重算一致；226个可评分完整事件与上一轮评价一致。
- 所有年度训练计划在当年及未来标签、特征扰动后不变；2026主实验全量输入与删除未来输入实际重训后的预测逐值一致。
- 16组年度拟合、48份种子模型均保存；权重重载后的预测与训练结束预测一致。预测先落盘再合并评价标签。
- 原模型与数据工程哈希未改变；[实验前固定协议](PREREG.md)、[标签审计](label_audit.json)、[时序审计](timing_plan_audit.json)、[拟合记录](fits.json)均保留。

[逐事件新旧比较](all_event_comparison.csv) · [入选变化](selection_changes.csv) · [逐年指标](metrics_by_year.csv) · [近期事件评分](recent_event_diagnostics.csv) · [配对区间](paired_cluster_intervals.csv)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python alpha_lab/experiments/sector_profit_days_v1/train.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python alpha_lab/experiments/sector_profit_days_v1/report.py
```
'''
(OUT/'REPORT.md').write_text(text)
style='body{font:16px/1.7 sans-serif;max-width:1550px;margin:30px auto;padding:0 24px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd5dc;padding:6px}img{max-width:100%}pre{white-space:pre-wrap}a{color:#246b97}'
(OUT/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>盈亏天数标签年度重训</title><style>'+style+'</style><body>'+markdown.markdown(text,extensions=['tables','fenced_code'])+'</body></html>')
checks=json.loads((OUT/'original_hashes_before.json').read_text())
for name,h in checks.items():assert digest(ROOT/name)==h,name
assert len(baseline)==226
previous=pd.read_csv(ROOT/'alpha_lab/outputs/sector_profit_days_audit/20261021/event_metrics.csv').set_index('event_id')
np.testing.assert_allclose(labels.set_index('event_id').reindex(previous.index).target,previous.profit_days_score,atol=1e-12,equal_nan=True)
save_json(OUT/'verification.json',dict(status='passed',original_hashes_unchanged=True,annual_fits=len(fits),saved_models=len(list(OUT.glob('*/models/*.pt'))),predictions_before_evaluation=True,complete_events_per_model=226,common_prediction_events=256,new_targets_match_previous_independent_audit=True))
shutil.copy2(HERE/'report.py',OUT/'sources/report.py')
artifacts={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
save_json(OUT/'manifest.json',dict(status='complete',experiment=cfg,artifacts=artifacts))
nav=ROOT/'alpha_lab/docs/current_strategy_20261013/SECTOR_PROFIT_DAYS_TRAINING_20261021.md'
nav.write_text('# 盈亏天数标签：年度板块模型重训\n\n[完整报告与图表](../../outputs/sector_profit_days_training/20261021/REPORT.html) · [逐事件比较](../../outputs/sector_profit_days_training/20261021/all_event_comparison.csv)\n\n两组预设实验均为研究模型，未替换正式策略；原模型与数据工程保持不变。\n')
print(show_results(aggregate));print(live.to_string(index=False));print(ci_show.to_string(index=False));print('REPORT_COMPLETE',OUT/'REPORT.html')
