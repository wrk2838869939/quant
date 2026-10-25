"""养殖业20日个股独立观察，不作为原策略首次确认入场信号。"""
from pathlib import Path
import sys,importlib.util,json
import pandas as pd
ROOT=Path('/root/autodl-tmp');OUT=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--scan',type=Path,required=True)
SCAN=parser.parse_args().scan.resolve()
sys.argv=[sys.argv[0],'--scan',str(SCAN)]
spec=importlib.util.spec_from_file_location('existing_evaluation',ROOT/'alpha_lab/outputs/strategy_2026_opportunity_scan/20261020_eod/evaluate_existing_models.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT
manifest=json.loads((SCAN/'manifest.json').read_text());ds=m.SectorDataset(manifest['dataset']);hub=m.DataHub()
assert hub.cfg.root.resolve()==(ROOT/'datahub/releases/selection_live_20261020_eod').resolve()
asof=pd.Timestamp('2026-10-20')
selected=pd.DataFrame([dict(event_id='OBSERVATION_ONLY_801017_20261020',date=asof,sector_code='801017.SI')])
d=m.stock_at_date(selected,asof,ds,hub)
d.to_csv(OUT/'stock_scores_old_name_history.csv',index=False)
meta=pd.read_parquet(hub.cfg.meta_dir/'stock_basic.parquet').set_index('ts_code')
allrows=pd.read_csv(OUT/'all_member_qualification_20261020.csv')
codes=allrows.ts_code.tolist();dates=hub.calendar.cn_days(end='2026-10-20')[-400:]
raw=hub.cn.daily(codes=codes,start=str(dates[0].date()),end='2026-10-20',fields=['hfq_close','turnover_rate','close','vol_share','amount_yuan'])
raw.to_parquet(OUT/'stock_history.parquet',index=False)
wide={f:raw.pivot(index='date',columns='ts_code',values=f).reindex(index=dates,columns=codes) for f in ['hfq_close','turnover_rate','close','vol_share']}
rules=m.yaml.safe_load((m.SECTOR/'sources/left_side_trend_v2.yaml').read_text());sh=m.stock_shape(wide,rules['stock'],rules['stock'])
ret=wide['hfq_close'].pct_change(fill_method=None)
info=pd.DataFrame(index=codes)
info['name']=meta.name.reindex(codes)
info['close']=wide['close'].loc[asof];info['day_return']=ret.loc[asof]
info['early_return']=sh['early_return'].loc[asof];info['late_return']=sh['late_return'].loc[asof]
info['amount_yuan']=raw[raw.date.eq(asof)].set_index('ts_code').amount_yuan
info['ret5']=(wide['hfq_close']/wide['hfq_close'].shift(5)-1).loc[asof]
info['ret20']=(wide['hfq_close']/wide['hfq_close'].shift(20)-1).loc[asof]
info.index.name='ts_code';info=info.reset_index()
allrows=allrows.merge(info,on='ts_code',validate='one_to_one')
allrows.to_csv(OUT/'all_members_detailed.csv',index=False)
print(allrows.merge(d[['ts_code','model_score','ridge_price5']],on='ts_code',how='left').to_string(index=False),flush=True)
