"""原有形态函数的重复确认诊断，不更新数据或重新训练模型。"""
from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np,yaml
ROOT=Path('/root/autodl-tmp');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'alpha_lab/src'))
from sector_shape_events import identify_events
from sector_shape_expansion import stage_events
from sector_dataset import digest
DATA=ROOT/'alpha_lab/data/sector_live_datasets/sw_l2_v1_20261019T143622_793293Z'
SCAN=ROOT/'alpha_lab/outputs/sector_live_scan/20261019T144154_853953Z'
SOURCE=ROOT/'alpha_lab/outputs/sector_multiscale_events/20260923T080028_978085Z'
PRED=ROOT/'alpha_lab/outputs/sector_asof_replay_v1/20261013T072855_087132Z'
for run in [SCAN,PRED]:
 m=json.loads((run/'manifest.json').read_text())
 for name,h in m['artifacts'].items():assert digest(run/name)==h,(run,name)
base=yaml.safe_load((SOURCE/'sector_shape_events_v1.yaml').read_text())
exp=yaml.safe_load((SOURCE/'sector_shape_expansion_v1.yaml').read_text())
multi=yaml.safe_load((SOURCE/'sector_multiscale_events_v1.yaml').read_text())
panel=pd.read_parquet(DATA/'daily.parquet');cal=pd.DatetimeIndex(sorted(panel.date.unique()))
events=pd.read_parquet(SCAN/'all_confirmations.parquet')
preds=pd.read_parquet(PRED/'predictions_BEFORE_EVALUATION.parquet').rename(columns={'event_id':'merged_event_id','date':'prediction_date'})
events=events.merge(preds[['merged_event_id','prediction_date','score','threshold','selected']],on='merged_event_id',how='inner',validate='one_to_one')
assert events.date.eq(events.prediction_date).all()
events.to_parquet(OUT/'original_events_with_annual_scores.parquet',index=False)
all_repeats=[];hotel_timeline=None;current_source=pd.read_csv(SCAN/'watch_today.csv');current_source=current_source[current_source.sector_code.eq('801219.SI')].iloc[0]
for window in multi['platform_windows']:
 target=events[events.window.eq(window)]
 if target.empty:continue
 sectors=target.sector_code.unique()
 obs,_=identify_events(panel[panel.sector_code.isin(sectors)],{**base,**exp['variants']['expanded'],'platform_window':window})
 staged=stage_events(obs,exp['confirmation_memory_days'])
 # 每个年度已评分首次事件必须能从原状态机准确复原。
 for e in target.itertuples():
  g=staged[(staged.sector_code==e.sector_code)&(staged.event_id==e.event_id)&(staged.segment_id==e.segment_id)&(staged.peak_date==e.peak_date)].sort_values('date').copy()
  start=g[g.date.eq(e.date)];assert len(start)==1 and start.confirmed.iloc[0]
  g=g[g.date.ge(e.date)].copy()
  idx=cal.get_indexer(g.date);assert (idx>=0).all()
  prev_false=(~g.confirmed.shift(1,fill_value=True))
  contiguous=np.r_[False,np.diff(idx)==1]
  repeats=g[g.confirmed & prev_false & contiguous & g.date.gt(e.date)]
  for i,r in enumerate(repeats.itertuples()):
   all_repeats.append(dict(event_id=e.merged_event_id,source_event_id=e.event_id,sector_code=e.sector_code,segment_id=e.segment_id,
    window=int(window),first_date=e.date,peak_date=e.peak_date,date=r.date,age_sessions=int(cal.get_loc(r.date)-cal.get_loc(e.date)),
    repetition=i+1,original_score=e.score,original_threshold=e.threshold,original_selected=bool(e.selected),index_close=r.index_close))
  if e.merged_event_id=='801219.SI_s5_20260917':
   hotel_timeline=g.copy();hotel_timeline['age_sessions']=cal.get_indexer(g.date)-cal.get_loc(e.date)
   hotel_timeline['reconfirmation_start']=g.index.isin(repeats.index)
   hotel_timeline.to_csv(OUT/'hotel_daily_states.csv',index=False)
   last=g.iloc[-1]
   for field in ['confirmed','slowed','support_recent','near_reference']:assert bool(last[field])==bool(current_source[field])
   np.testing.assert_allclose(last.index_close,current_source.index_close,rtol=1e-12)
 print('WINDOW',window,'events',len(target),'repeats_total',len(all_repeats),flush=True)
repeats=pd.DataFrame(all_repeats).sort_values(['event_id','date'])
repeats.to_csv(OUT/'all_reconfirmation_starts.csv',index=False)
first=repeats.groupby('event_id',sort=False).head(1).copy();first['sample']='first_repeat'
late=repeats[repeats.age_sessions.ge(10)].groupby('event_id',sort=False).head(1).copy();late['sample']='first_repeat_at_least_10_sessions'
samples=pd.concat([first,late],ignore_index=True)
lookup={key:g.set_index('date') for key,g in panel.groupby('sector_code')}
rows=[]
for r in samples.itertuples():
 pos=cal.get_loc(r.date);g=lookup[r.sector_code]
 for h in [3,10,20]:
  record=r._asdict();record.pop('Index',None)
  record.update(horizon=h,entry_date=pd.NaT,end_date=pd.NaT,complete=False,return_value=np.nan,minimum_close_return=np.nan,missing_reason='end_of_data')
  if pos+1+h<len(cal):
   dates=cal[pos+1:pos+2+h];path=g.reindex(dates)
   record.update(entry_date=dates[0],end_date=dates[-1])
   valid=path.index_valid.fillna(False)&path.segment_id.eq(r.segment_id)&path.index_close.gt(0)
   if valid.all():
    prices=path.index_close.to_numpy();record.update(complete=True,return_value=prices[-1]/prices[0]-1,minimum_close_return=np.min(prices[1:]/prices[0]-1),missing_reason='')
   else:record['missing_reason']='invalid_quote_or_segment'
  rows.append(record)
outcomes=pd.DataFrame(rows);outcomes['period']=np.where(pd.to_datetime(outcomes.date).dt.year.le(2025),'2019-2025','2026_monitor')
outcomes.to_csv(OUT/'sector_price_observations_NOT_TRADABLE.csv',index=False)
summ=[]
for key,g in outcomes.groupby(['sample','period','original_selected','horizon']):
 good=g[g.complete];summ.append(dict(zip(['sample','period','original_selected','horizon'],key))|dict(events=len(g),complete=len(good),incomplete=len(g)-len(good),mean=good.return_value.mean(),median=good.return_value.median(),positive=(good.return_value>0).mean() if len(good) else np.nan,q10=good.return_value.quantile(.1),minimum=good.return_value.min()))
pd.DataFrame(summ).to_csv(OUT/'sector_summary_NOT_TRADABLE.csv',index=False)
yearly=outcomes[outcomes.complete].copy();yearly['year']=pd.to_datetime(yearly.date).dt.year
yearly.groupby(['sample','year','original_selected','horizon']).return_value.agg(['count','mean','median','min']).reset_index().to_csv(OUT/'yearly_NOT_TRADABLE.csv',index=False)
(OUT/'replay_checks.json').write_text(json.dumps(dict(status='passed',original_annual_events=len(events),reconfirmation_starts=len(repeats),distinct_reconfirmed_events=repeats.event_id.nunique(),latest_hotel_state_matches_original_scan=True,prereg_sha256=digest(OUT/'PREREG.md')),ensure_ascii=False,indent=2))
print(pd.DataFrame(summ).to_string(index=False),flush=True)
print('HOTEL_TIMELINE',hotel_timeline[['date','age_sessions','index_return','confirmed','reconfirmation_start','slowed','support_recent']].to_string(index=False),flush=True)
