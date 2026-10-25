"""准备下一交易日盘前股本基准；不把状态模板当作当天停牌确认。"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from execution_market_feed import file_hash
from shadow_planner import canonical, ShadowError, validate_calendar


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['observation','market-stage','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--archive',type=Path,default=ROOT/'data/panel/daily_panel.parquet')
    a=p.parse_args()
    manifest=json.loads((a.observation/'manifest.json').read_text())
    for name,digest in manifest['output_sha256'].items():
        if file_hash(a.observation/name)!=digest:raise ShadowError('收盘观察包发生变化')
    bundle=json.loads((a.observation/'bundle.json').read_text());date=bundle['asof']
    if not bundle['eod_complete']:raise ShadowError('股本基准必须来自真实前日EOD包')
    calendar=json.loads((a.observation/'calendar.json').read_text())
    execution=validate_calendar(calendar,date)
    receipt=json.loads((a.market_stage/'fetch_receipt.json').read_text())
    requests=[r for r in receipt['requests'] if r['api']=='daily_basic' and r['trade_date']==date.replace('-','') and r['status']=='downloaded']
    if len(requests)!=1:raise ShadowError('缺指定前日股本原件')
    path=a.market_stage/requests[0]['file']
    if file_hash(path)!=requests[0]['sha256']:raise ShadowError('股本原件改变')
    basic=pd.read_parquet(path)
    if not basic.trade_date.astype(str).eq(date.replace('-','')).all():raise ShadowError('股本原件日期不同')
    feature=pd.read_parquet(a.observation/'feature_panel.parquet')
    reference=feature.loc[feature.date.eq(date),['date','ts_code','hfq_close','total_mv_yuan']]
    basis=reference.merge(basic[['ts_code','close','total_share','float_share']],on='ts_code',validate='one_to_one')
    master=pd.read_parquet(a.observation/'basics.parquet')
    universe=set(master.loc[master.list_date.le(execution)&(master.delist_date.isna()|master.delist_date.gt(execution)),'ts_code'])
    missing=universe-set(basis.ts_code)
    sources=[path,a.observation/'manifest.json']
    if missing:
        # 前日未交易股票仍保留在全集，不能把缺日线解释成已退市。
        archived=pd.read_parquet(a.archive,columns=['date','ts_code','close','hfq_close','total_mv_yuan','total_share','float_share'],
            filters=[('ts_code','in',sorted(missing)),('date','<=',pd.Timestamp(date))])
        latest=archived.sort_values('date').groupby('ts_code',sort=False).tail(1)
        basis=pd.concat([basis,latest],ignore_index=True)
        sources.append(a.archive)
    if universe-set(basis.ts_code):raise ShadowError('仍缺证券股本基准：'+','.join(sorted(universe-set(basis.ts_code))))
    basis=basis.loc[basis.ts_code.isin(universe)].sort_values('ts_code').reset_index(drop=True)
    # Tushare原始总/流通股本为万股，出口统一为股。
    basis[['total_share','float_share']]*=10000
    if basis.isna().any().any():raise ShadowError('股本/复权基准未知')
    a.output.mkdir(parents=True,exist_ok=False)
    basis.to_parquet(a.output/'basis.parquet',index=False)
    status=dict(asof=execution,timestamp=None,complete=False,source=None,evidence=None,
        universe=sorted(universe),suspended=[],
        note='必须按当天官方/券商状态补齐证券变更和停牌清单；空数组不是已确认无停牌。')
    (a.output/'status_template.json').write_text(canonical(status))
    (a.output/'coverage.json').write_text(canonical(dict(asof=date,execution_date=execution,
        securities=len(basis),prior_session_unquoted=sorted(missing),
        stale_basis_dates={r.ts_code:str(r.date.date()) for r in basis.itertuples() if r.date<pd.Timestamp(date)},
        requires_current_status=True,live_orders=False)))
    (a.output/'manifest.json').write_text(canonical(dict(input_sha256={str(x):file_hash(x) for x in sources},
        output_sha256={f.name:file_hash(f) for f in a.output.iterdir()},live_orders=False)))
    print((a.output/'coverage.json').read_text())


if __name__=='__main__':main()
