"""重算报告统计，不训练模型，不追加历史研究截止日之后的标签。"""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
MODEL=ROOT/'alpha_lab/outputs/sector_fuzzy_random_halves/20260923T111853_561924Z'
RET=ROOT/'alpha_lab/outputs/sector_confirmation_returns_multiscale/20260923T081158_848823Z'
PANEL=ROOT/'alpha_lab/data/sector_datasets/sw_l2_v1_20260921T073401_889301Z'
for folder in [MODEL,RET,PANEL]:
 m=json.loads((folder/'manifest.json').read_text())
 for name,value in m['artifacts'].items():
  assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==value,str(folder/name)
labels=pd.read_parquet(RET/'sector_event_outcomes.parquet')
panel=pd.read_parquet(PANEL/'daily.parquet')
calendar=pd.DatetimeIndex(sorted(panel.date.unique()))
wide=panel.pivot(index='date',columns='sector_code',values='index_close').reindex(calendar)
segments=panel.pivot(index='date',columns='sector_code',values='segment_id').reindex(calendar)
checked=0
for row in labels[labels.complete].itertuples():
 i=calendar.get_loc(row.date); h=int(row.horizon)
 prices=wide[row.sector_code].iloc[i:i+h+1].to_numpy()
 assert len(prices)==h+1 and np.isfinite(prices).all()
 assert segments[row.sector_code].iloc[i:i+h+1].eq(row.segment_id).all()
 assert calendar[i+h]==row.end_date
 returns=prices[1:]/prices[0]-1
 np.testing.assert_allclose([returns[-1],returns.max(),returns.min()],[row.end_return,row.max_return,row.min_return],rtol=1e-10,atol=1e-12)
 checked+=1
pred=pd.read_parquet(MODEL/'test_predictions_before_evaluation.parquet')
pred=pred[pred.model.eq('fnn') & pred.retention.eq(.5)]
assert pred.event_id.is_unique and len(pred)==166
samples=pred.merge(labels,on=['event_id','date','sector_code'],validate='one_to_many')
samples.to_csv(OUT/'verified_test_samples.csv',index=False)
summary=[]
for h,g in samples.groupby('horizon'):
 g=g[g.complete]
 for kind,frame in [('全部测试事件',g),('FNN筛选事件',g[g.selected])]:
  x=frame.end_return
  summary.append(dict(horizon=int(h),group=kind,n=len(frame),mean=x.mean(),median=x.median(),q10=x.quantile(.1),q90=x.quantile(.9),positive=x.gt(0).mean(),mean_path_max=frame.max_return.mean(),mean_path_min=frame.min_return.mean(),sample_min=x.min(),sample_max=x.max()))
metrics=pd.DataFrame(summary);metrics.to_csv(OUT/'verified_return_metrics.csv',index=False)
saved=pd.read_csv(MODEL/'selection_summary.csv')
for h in [3,10,20]:
 row=saved[(saved['round']=='combined') & saved.model.eq('fnn') & saved.retention.eq(.5) & saved.horizon.eq(h)].iloc[0]
 b=metrics[(metrics.horizon==h)&metrics.group.eq('全部测试事件')].iloc[0]
 q=metrics[(metrics.horizon==h)&metrics.group.eq('FNN筛选事件')].iloc[0]
 np.testing.assert_allclose([b['mean'],q['mean'],q['median'],q.q10,q.mean_path_max,q.mean_path_min],[row.base_mean,row['mean'],row['median'],row.q10,row.max_return_mean,row.min_return_mean],rtol=1e-10)
thresholds=pred.groupby('round').threshold.first().to_dict()
result=dict(status='complete',historical_cutoff=str(calendar.max().date()),checked_complete_outcome_windows=checked,test_events=len(pred),selected_test_events=int(pred.selected.sum()),thresholds=thresholds,statistics_match_saved_report=True)
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False));print(metrics.round(6).to_string(index=False))
# 原始行情簇上重算描述性区间，固定模型与阈值，不重新训练。
g=pd.read_parquet(MODEL/'test_outcomes_WITH_FUTURE.parquet')
g=g[g.model.eq('fnn') & g.retention.eq(.5) & g.horizon.eq(20) & g.complete]
groups=[x.index.to_numpy() for _,x in g.groupby('cluster_id')]
rng=np.random.default_rng(20260923);res=[]
values,selection=g.end_return.to_dict(),g.selected.to_dict()
for _ in range(2000):
 ids=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
 base=np.array([values[i] for i in ids]);chosen=base[[selection[i] for i in ids]]
 if len(chosen):res.append([chosen.mean()-base.mean(),np.median(chosen)-np.median(base),np.quantile(chosen,.1)-np.quantile(base,.1)])
ci=np.quantile(res,[.025,.975],axis=0)
original=saved[(saved['round']=='combined')&saved.model.eq('fnn')&saved.retention.eq(.5)&saved.horizon.eq(20)].iloc[0]
intervals={}
for j,key in enumerate(['mean','median','q10']):
 np.testing.assert_allclose(ci[:,j],[original[f'ci_{key}_low'],original[f'ci_{key}_high']],atol=1e-12)
 intervals[key]=ci[:,j].tolist()
(OUT/'verified_intervals.json').write_text(json.dumps(dict(clusters=len(groups),repetitions=2000,intervals=intervals),indent=2)+'\n')
print('BOOTSTRAP',intervals)
