"""核对隔离下载包并输出只追加增量，不修改任何旧面板或封存信号。"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from execution_market_feed import normalize_day,file_hash,require_current_metadata
from shadow_planner import ShadowError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--anchor',type=Path,default=ROOT/'data/raw/by_date/daily/20260908.parquet')
    args=parser.parse_args();out=args.stage/'validated';out.mkdir()
    receipt=json.loads((args.stage/'fetch_receipt.json').read_text())
    files={}
    for item in receipt['requests']:
        if item['status']!='downloaded':raise ShadowError('存在下载失败，不发布部分包')
        path=args.stage/item['file']
        if file_hash(path)!=item['sha256']:raise ShadowError('下载文件被修改')
        files[(item['trade_date'],item['api'])]=path
    checks,features,markets,gaps=[],[],[],[]
    previous=set(pd.read_parquet(args.anchor).ts_code)
    for day in sorted({day for day,_ in files}):
        date=pd.Timestamp(day).strftime('%Y-%m-%d')
        inputs=[pd.read_parquet(files[(day,api)]) for api in ['daily','daily_basic','adj_factor']]
        feature,market,check=normalize_day(*inputs,date)
        current=set(market.ts_code)
        gaps.extend(dict(date=date,ts_code=c,status='absence_unexplained_requires_status_source') for c in sorted(previous-current))
        check.update(new_quote_codes=sorted(current-previous),missing_previous_quote_codes=sorted(previous-current))
        previous=current;features.append(feature);markets.append(market);checks.append(check)
    if not checks:raise ShadowError('没有新的完整交易日')
    pd.concat(features).to_parquet(out/'feature_increment.parquet',index=False)
    pd.concat(markets).to_parquet(out/'market_increment.parquet',index=False)
    pd.DataFrame(gaps,columns=['date','ts_code','status']).to_csv(out/'unexplained_absences.csv',index=False)
    package=dict(schema_version=1,asof=checks[-1]['date'],source=receipt['source'],price_checks_passed=True,
        readiness='price_only_not_trade_ready',checks=checks,unexplained_absences=len(gaps),metadata={},
        input_sha256={str(p.resolve()):file_hash(p) for p in files.values()},
        artifacts={p.name:file_hash(p) for p in out.iterdir()},source_sha256=file_hash(__file__),
        module_sha256=file_hash(ROOT/'src/execution_market_feed.py'),live_orders=False)
    try:require_current_metadata(package,package['asof'])
    except ShadowError as exc:package['planning_blocker']=str(exc)
    (out/'package.json').write_text(json.dumps(package,ensure_ascii=False,indent=2))
    print(json.dumps(dict(asof=package['asof'],rows=sum(c['rows'] for c in checks),
        unexplained_absences=len(gaps),planning_blocker=package.get('planning_blocker')),ensure_ascii=False))


if __name__=='__main__':main()
