"""按原事件首次日期的市场簇作描述性重采样，保留缺失路径压力情景。"""
from pathlib import Path
import pandas as pd,numpy as np,json
OUT=Path(__file__).resolve().parent;ROOT=Path('/root/autodl-tmp')
d=pd.read_csv(OUT/'sector_price_observations_NOT_TRADABLE.csv',parse_dates=['date','first_date'])
panel=pd.read_parquet(ROOT/'alpha_lab/data/sector_live_datasets/sw_l2_v1_20261019T143622_793293Z/daily.parquet',columns=['date']);cal=pd.DatetimeIndex(sorted(panel.date.unique()))
e=pd.read_parquet(OUT/'original_events_with_annual_scores.parquet').sort_values('date')
unique=sorted(e.date.unique());position=cal.get_indexer(unique);cluster=np.cumsum(np.r_[True,np.diff(position)>20]);mapping=dict(zip(unique,cluster))
e['cluster']=e.date.map(mapping);clusters=e.set_index('merged_event_id').cluster
d['cluster']=d.event_id.map(clusters);assert d.cluster.notna().all()
rows=[]
for key,g in d.groupby(['sample','period','original_selected','horizon']):
 valid=g[g.complete];block=valid.groupby('cluster').return_value.agg(['sum','count'])
 rng=np.random.default_rng(20261013)
 if len(block)>1:
  indices=rng.integers(0,len(block),size=(2000,len(block)))
  estimates=block['sum'].to_numpy()[indices].sum(axis=1)/block['count'].to_numpy()[indices].sum(axis=1)
  lo,hi=np.quantile(estimates,[.025,.975])
 else:lo=hi=np.nan
 complete=len(valid);missing=len(g)-complete
 rows.append(dict(zip(['sample','period','original_selected','horizon'],key))|dict(events=len(g),complete=complete,missing=missing,clusters=len(block),mean=valid.return_value.mean(),cluster_ci_low=lo,cluster_ci_high=hi,
  mean_if_missing_zero=valid.return_value.sum()/len(g),mean_if_missing_minus20pct=(valid.return_value.sum()-.2*missing)/len(g)))
result=pd.DataFrame(rows);result.to_csv(OUT/'uncertainty_NOT_TRADABLE.csv',index=False)
e[['merged_event_id','date','cluster']].to_csv(OUT/'fixed_event_clusters.csv',index=False)
# 确认没有将另一个独立市场事件之后的状态混入旧事件。
full=pd.read_parquet(ROOT/'alpha_lab/outputs/sector_live_scan/20261019T144154_853953Z/all_confirmations.parquet').sort_values(['sector_code','segment_id','date'])
full['next_event_date']=full.groupby(['sector_code','segment_id']).date.shift(-1)
r=pd.read_csv(OUT/'all_reconfirmation_starts.csv',parse_dates=['date']).merge(full[['merged_event_id','next_event_date']],left_on='event_id',right_on='merged_event_id',validate='many_to_one')
assert not (r.next_event_date.notna() & r.date.ge(r.next_event_date)).any()
print(result[(result.period=='2019-2025')&~result.original_selected].to_string(index=False))
(OUT/'uncertainty_checks.json').write_text(json.dumps(dict(status='passed',cluster_rule='原首次确认日期相邻间隔超过20个交易日才分新簇；全原事件统一定簇',bootstrap_repetitions=2000,seed=20261013,no_old_event_repeats_after_new_independent_event=True,interval_is_descriptive_not_independent_test=True),ensure_ascii=False,indent=2))
