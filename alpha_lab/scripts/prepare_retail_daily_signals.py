#!/usr/bin/env python3
"""冻结恢复入场可用的每日信号；只做PIT资格过滤，不做资金/股价过滤。"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_gp_mining import build_wide
from synthesis_and_test import build_all_factors
from portfolio import board_of
from prepare_monthly_audit import fetch_execution_data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);a=ap.parse_args();out=a.prepared
    cfg=yaml.safe_load((out/'config.yaml').read_text());end=pd.Timestamp(cfg['valid'][1])
    if end>=pd.Timestamp('2024-01-01'):raise ValueError('禁止TEST')
    parent=ROOT/cfg['execution_prepared'];pc=yaml.safe_load((parent/'config.yaml').read_text())
    # 这是执行前的协议补充，不覆盖最初登记，不根据VALID资金结果选型。
    (out/'daily_refresh_addendum.json').write_text(json.dumps(dict(reentry_uses_fresh_daily_signal=True,
        forecast_model_unchanged=True,monthly_selection_unchanged=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    cols=dict(ret1='pct_chg',vol='vol_share',amt='amount_yuan',to='turnover_rate',high='high',low='low',close='close',hclose='hfq_close',buy_lg='buy_lg_amount_yuan',sell_lg='sell_lg_amount_yuan',pe='pe_ttm',pb='pb')
    w=build_wide(cols,end_date=end);w['ret1']/=100
    w['vwap_dev']=w['amt']/w['vol']/w['close']-1;w['lg_share']=(w['buy_lg']+w['sell_lg'])/(w['amt']+1e-6)
    fac=build_all_factors(w,cfg['horizon']);fac['ep']=1/w['pe'].where(w['pe']>0);fac['bp']=1/w['pb'].where(w['pb']>0)
    dates=w['close'].loc[slice(*cfg['valid'])].index
    common=np.logical_and.reduce([np.isfinite(f.loc[dates].to_numpy()) for f in fac.values()])
    signal=fac['lit:amihud'].loc[dates].where(common)
    original=pd.read_parquet(ROOT/cfg['raw_signals'])
    pd.testing.assert_frame_equal(signal.loc[original.index],original,check_dtype=False,atol=1e-6,rtol=1e-6)
    basics=pd.read_parquet(ROOT/'data/raw/stock_basic_all.parquet').set_index('ts_code')
    listing=pd.to_datetime(basics.list_date).reindex(signal.columns);delisting=pd.to_datetime(basics.delist_date).reindex(signal.columns)
    names=pd.read_parquet(ROOT/'data/raw/namechange_all.parquet');names=names[names.name.str.contains('ST',case=False,na=False)].copy()
    names['start']=pd.to_datetime(names.start_date);names['end']=pd.to_datetime(names.end_date)
    starts=pd.Series({c:pd.Timestamp(pc['portfolio']['boards'][board_of(c)]['start']) for c in signal.columns})
    arr=signal.to_numpy(copy=True);codes=set()
    for i,date in enumerate(dates):
        eligible=(date-listing).dt.days.ge(pc['portfolio']['min_listing_calendar_days'])&(delisting.isna()|delisting.gt(date))&starts.le(date)
        st=names.loc[names.start.le(date)&(names.end.isna()|names.end.ge(date)),'ts_code']
        eligible.loc[eligible.index.isin(st)]=False
        arr[i,~eligible.to_numpy()]=np.nan
    signal=pd.DataFrame(arr,index=dates,columns=signal.columns)
    model=pd.read_csv(out/'frozen_return_model.csv',index_col='bucket').expected_return
    rank=signal.rank(axis=1,ascending=False,method='first')
    bucket=np.digitize(rank.to_numpy(),cfg['rank_edges'][1:],right=True)
    pred=np.full(bucket.shape,np.nan)
    for b,mu in model.items():pred[(bucket==b)&signal.notna().to_numpy()]=mu
    forecasts=pd.DataFrame(pred,index=dates,columns=signal.columns)
    signal.to_parquet(out/'daily_signals.parquet');forecasts.to_parquet(out/'daily_forecasts.parquet')
    signal.loc[original.index].to_parquet(out/'monthly_signals.parquet');forecasts.loc[original.index].to_parquet(out/'monthly_forecasts.parquet')
    for _,row in signal.iterrows():codes.update(row.dropna().nlargest(cfg['candidate_pool']).index)
    pd.DataFrame({'ts_code':sorted(codes)}).to_csv(out/'execution_codes.csv',index=False)
    (out/'execution_config.yaml').write_text((parent/'config.yaml').read_text())
    print(f'每日候选冻结，执行股票{len(codes)}只；这是候选并集，不是同时持仓',flush=True)
    cache=fetch_execution_data(codes,tuple(map(pd.Timestamp,cfg['valid'])),pc)
    (out/'execution_cache.txt').write_text(str(cache))
    inputs=[out/f for f in ['daily_signals.parquet','daily_forecasts.parquet','monthly_signals.parquet','monthly_forecasts.parquet','execution_codes.csv']]
    (out/'daily_refresh_manifest.json').write_text(json.dumps(dict(status='complete',raw_monthly_reproduced=True,
        price_budget_filter=False,hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2))


if __name__=='__main__':main()
