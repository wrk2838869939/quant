"""离线全A规模工程检验；次日行情全部合成，禁止作为成交或收益证据。"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from closing_projection import project,proxy_receipt
from execution_market_feed import file_hash
from monthly_risk_live import read_index_close
from shadow_inputs import build_bundle
from shadow_planner import canonical,make_plan,ShadowError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['previous','basis','index-history','model','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    previous=json.loads((a.previous/'manifest.json').read_text())
    for name,digest in previous['output_sha256'].items():
        if file_hash(a.previous/name)!=digest:raise ShadowError('封存来源改变')
    date='2026-09-15'
    cfg=yaml.safe_load((ROOT/'configs/monthly_risk_close_day15.yaml').read_text())
    basis=pd.read_parquet(a.basis)
    history=pd.read_parquet(a.previous/'feature_panel.parquet')
    prior_date=history.date.max();market=pd.read_parquet(a.previous/'market.parquet')
    actual=market.loc[market.date.eq(prior_date)].set_index('ts_code')
    b=basis.set_index('ts_code');idx=read_index_close(a.index_history)
    names=pd.read_parquet(a.previous/'basics.parquet').set_index('ts_code')
    suspended=sorted(set(b.index)-set(actual.index))
    q=pd.DataFrame([dict(ts_code=c,name='SYNTHETIC',pre_close=float(b.at[c,'close']),
        close=float(b.at[c,'close'])*1.001,vol_share=float(actual.at[c,'vol_share']),
        timestamp=date+'T14:30:30+08:00') for c in sorted(actual.index)]+[
            dict(ts_code='000300.SH',name='SYNTHETIC INDEX',pre_close=float(idx.iloc[-1]),
                close=float(idx.iloc[-1])*1.001,vol_share=1.,timestamp=date+'T14:30:30+08:00')])
    receipt=proxy_receipt(q,date,date+'T14:31:00+08:00','SYNTHETIC engineering fixture; not historical quotes')
    receipt['synthetic']=True
    calendar=json.loads((a.previous/'calendar.json').read_text())
    raw,index,marks=project(history,idx,basis,q,receipt,calendar,suspended)
    projection_elapsed=time.monotonic()-started
    bundle=build_bundle(raw,marks,names.reset_index(),pd.read_parquet(a.previous/'st.parquet'),
        pd.read_parquet(a.previous/'actions.parquet'),date,calendar,cfg,json.loads(a.model.read_text()),[],receipt,index)
    snapshot=dict(account_id='SYNTHETIC_NONTRADABLE',asof=date,phase='intraday',timestamp=date+'T14:32:00+08:00',
        source='SYNTHETIC',cash=80000.,equity=80000.,receivables=0.,frozen_cash=0.,open_orders=[],positions={})
    blocked=False
    try:make_plan(snapshot,bundle,calendar,cfg)
    except ShadowError as e:blocked='合成盘中观察' in str(e)
    if not blocked:raise AssertionError('合成快照没有被执行入口阻断')
    prior=history[history.date.eq(prior_date)].set_index('ts_code')
    cap=b.loc[actual.index,'close']*b.loc[actual.index,'total_share']
    cap_difference=(cap-prior.loc[actual.index,'total_mv_yuan']).abs()/prior.loc[actual.index,'total_mv_yuan']
    prior_reprice=b.loc[actual.index,'total_mv_yuan']
    rel=((prior_reprice-prior.loc[actual.index,'total_mv_yuan']).abs()/prior.loc[actual.index,'total_mv_yuan']).max()
    if not np.isfinite(rel) or rel>1e-10:raise AssertionError('未保留供应商前日市值口径')
    turnover=actual.vol_share/b.loc[actual.index,'float_share']*100
    turnover_error=(turnover-prior.loc[actual.index,'turnover_rate']).abs().max()
    result=dict(synthetic_only=True,performance_backtest=False,live_orders=False,
        real_historical_through=str(prior_date.date()),synthetic_decision_date=date,
        historical_rows=len(history),basis_securities=len(basis),synthetic_traded_securities=len(marks),
        synthetic_suspended_securities=len(suspended),full_universe=bundle['provenance']['full_universe'],
        prior_eod_market_cap_max_relative_error=float(rel),
        naive_shares_times_price_max_relative_difference=float(cap_difference.max()),
        naive_market_cap_differences_gt_1e5=cap_difference[cap_difference>1e-5].to_dict(),
        prior_turnover_max_absolute_error=float(turnover_error),
        projection_seconds=projection_elapsed,total_seconds=time.monotonic()-started,
        synthetic_plan_blocked=blocked,
        limitations=['没有真实14:30历史行情','没有真实收盘竞价成交回报','未检验真实Sina全A实时覆盖率'])
    (a.output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (a.output/'receipt.json').write_text(canonical(receipt))
    (a.output/'manifest.json').write_text(canonical(dict(input_sha256={str(f):file_hash(f) for f in
        [a.previous/'manifest.json',a.basis,a.index_history,a.model]},
        output_sha256={f.name:file_hash(f) for f in a.output.iterdir()},synthetic_only=True)))
    print(canonical(result))


if __name__=='__main__':main()
