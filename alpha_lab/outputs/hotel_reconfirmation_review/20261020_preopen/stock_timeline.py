"""同轮17个交易日的个股状态复原；按当日全体合格成员排名。"""
from pathlib import Path
import sys,importlib.util,json
import pandas as pd,numpy as np
ROOT=Path('/root/autodl-tmp');OUT=Path(__file__).resolve().parent
sys.argv=[sys.argv[0],'--scan',str(ROOT/'alpha_lab/outputs/sector_live_scan/20261019T144154_853953Z')]
spec=importlib.util.spec_from_file_location('existing_eval',ROOT/'alpha_lab/outputs/strategy_2026_opportunity_scan/20261019_eod/evaluate_existing_models.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=ROOT/'alpha_lab/outputs/hotel_stock_review/20261019_eod';raw=pd.read_parquet(source/'stock_history.parquet')
hub=m.DataHub();dates=hub.calendar.cn_days(start=str(raw.date.min().date()),end='2026-10-19');codes=sorted(raw.ts_code.unique())
wide={f:raw.pivot(index='date',columns='ts_code',values=f).reindex(index=dates,columns=codes) for f in ['hfq_close','turnover_rate','close','vol_share']}
rules=m.yaml.safe_load((m.SECTOR/'sources/left_side_trend_v2.yaml').read_text());cfg=m.yaml.safe_load((m.STOCK/'config.yaml').read_text())
shape=m.stock_shape(wide,rules['stock'],rules['stock']);ret=wide['hfq_close'].pct_change(fill_method=None);p=rules['stock']
price=(shape['early_return'].le(-p['minimum_early_fall'])&shape['slowdown'].le(p['slowdown_ratio'])&shape['late_return'].le(p['maximum_recent_gain'])&ret.ge(-p['maximum_current_daily_fall']))
strength=wide['hfq_close']/wide['hfq_close'].shift(60)-1;vol=ret.rolling(20).std()
nh=pd.read_parquet(source/'fresh_namechange.parquet')
for key in ['start_date','end_date','ann_date']:nh[key]=pd.to_datetime(nh[key],format='%Y%m%d',errors='coerce')
master=pd.read_parquet(hub.cfg.meta_dir/'stock_basic.parquet').set_index('ts_code').reindex(codes)
listed=pd.to_datetime(master.list_date,format='%Y%m%d',errors='coerce');delisted=pd.to_datetime(master.delist_date,format='%Y%m%d',errors='coerce')
ds=ROOT/'alpha_lab/data/sector_live_datasets/sw_l2_v1_20261019T143622_793293Z';membership=pd.read_parquet(ds/'membership_daily.parquet');sector_index=json.loads((ds/'sector_codes.json').read_text()).index('801219.SI')
weights=np.load(m.STOCK/'models/2026.npz');rows=[];diagnostics=[]
for date in dates[dates>='2026-09-17']:
 members=membership.columns[membership.loc[date].eq(sector_index)].tolist();assert sorted(members)==codes
 normal=m.announced_normal_names(nh,date,codes)
 qual=normal & listed.le(date)&(delisted.isna()|delisted.gt(date))&wide['close'].loc[date].gt(0)&wide['vol_share'].loc[date].gt(0)
 trend=m.trend_at_date(wide['hfq_close'],date,rules['trend'])
 frame=pd.DataFrame([dict(event_id=f'OBSERVATION_{date:%Y%m%d}',date=date,ts_code=c,name=master.at[c,'name'],qualified=bool(qual[c]),trend=bool(trend[c]),price=bool(price.at[date,c]),strength=strength.at[date,c],volatility=vol.at[date,c],close=wide['close'].at[date,c],daily_return=ret.at[date,c],**{f:shape[f].at[date,c] for f in ['expansion','contraction','slowdown','early_return','late_return']}) for c in codes])
 diagnostics.append(frame.copy());frame=frame[frame.qualified].copy().reset_index(drop=True)
 feats,names=m.cross_section_features(frame,cfg['model']);assert names==weights['features'].tolist()
 frame['model_score']=feats[names].to_numpy()@weights['coef'];frame['observation_candidate']=False
 frame.loc[m.select_positive_top(frame,'model_score',cfg),'observation_candidate']=True
 rows.append(frame)
allrows=pd.concat(rows,ignore_index=True);pd.concat(diagnostics).to_csv(OUT/'stock_daily_qualification.csv',index=False)
allrows.to_csv(OUT/'stock_daily_scores.csv',index=False)
chosen=allrows[allrows.ts_code.isin(['000721.SZ','605108.SH'])];chosen.to_csv(OUT/'two_stocks_daily.csv',index=False)
prior=pd.read_csv(source/'observation_candidates.csv').set_index('ts_code')
latest=allrows[allrows.date.eq(pd.Timestamp('2026-10-19')) & allrows.observation_candidate].set_index('ts_code')
assert set(prior.index)==set(latest.index)
np.testing.assert_allclose(prior.model_score,latest.loc[prior.index,'model_score'],rtol=1e-10,atol=1e-12)
for date in ['2026-09-17','2026-09-30']:
 truncated={k:v.loc[:date] for k,v in wide.items()}
 alt=m.stock_shape(truncated,rules['stock'],rules['stock'])
 for field in ['early_return','late_return','slowdown','expansion','contraction']:
  np.testing.assert_allclose(alt[field].loc[date],shape[field].loc[date],rtol=1e-12,atol=1e-12,equal_nan=True)
for filename in ['pipeline_hashes_before.json','model_hashes_before.json']:
 for name,h in json.loads((ROOT/'alpha_lab/outputs/strategy_2026_opportunity_scan/20261019_eod'/filename).read_text()).items():assert m.digest(ROOT/name)==h,name
(OUT/'stock_checks.json').write_text(json.dumps(dict(status='passed',dates=allrows.date.nunique(),membership_checked_each_date=True,latest_candidates_and_scores_equal_previous_review=True,feature_prefix_checks=['2026-09-17','2026-09-30'],unchanged_programs_and_annual_models=True),ensure_ascii=False,indent=2))
print(chosen[['date','name','close','price','slowdown','model_score','observation_candidate']].to_string(index=False),flush=True)
