"""隔离获取执行元数据，记录实际获取时刻；不把今天下载伪装成历史可见。"""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from execution_market_feed import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--price-date',default='2026-09-11')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    bundle=json.loads(args.bundle.read_text())
    scope=set(bundle['signals'])
    if args.snapshot:scope.update(json.loads(args.snapshot.read_text())['positions'])
    now=pd.Timestamp.now(tz='Asia/Shanghai');today=now.strftime('%Y%m%d')
    requests=[]
    for status in ['L','D','P','G','UN']:
        requests.append(('stock_basic',dict(list_status=status,fields='ts_code,symbol,name,market,exchange,list_status,list_date,delist_date'),6000))
    for exchange in ['SSE','SZSE']:
        requests.append(('trade_cal',dict(exchange=exchange,start_date=now.strftime('%Y0101'),end_date=now.strftime('%Y1231')),1000))
    dates=sorted({bundle['signal_date'].replace('-',''),args.price_date.replace('-',''),today})
    for day in dates:requests.append(('stock_st',dict(trade_date=day),1000))
    requests.extend([('namechange',dict(start_date=bundle['signal_date'].replace('-',''),end_date=today),1000),
        ('stk_limit',dict(trade_date=today),6000),('suspend_d',dict(trade_date=today),6000),
        ('dividend',dict(ex_date=today),2000)])
    for code in sorted(scope):requests.append(('dividend',dict(ts_code=code),2000))
    receipt=dict(schema_version=1,source='configured Tushare provider',price_date=args.price_date,
        signal_date=bundle['signal_date'],observation_date=now.strftime('%Y-%m-%d'),scope=sorted(scope),
        account_scope_supplied=args.snapshot is not None,bundle_sha256=file_hash(args.bundle),
        snapshot_sha256=file_hash(args.snapshot) if args.snapshot else None,requests=[],live_orders=False)
    for index,(api,params,cap) in enumerate(requests):
        item=dict(api=api,params=params,requested_at=datetime.now(timezone.utc).isoformat(),row_cap=cap)
        try:
            with open('/dev/null','w') as sink,redirect_stdout(sink),redirect_stderr(sink):
                from ts_client import get_pro
                frame=get_pro().query(api,**params)
            if frame is None:raise ValueError('no response')
            path=args.output/f'{index:03d}_{api}.parquet';frame.to_parquet(path,index=False)
            item.update(file=path.name,rows=len(frame),sha256=file_hash(path),
                status='possibly_truncated' if len(frame)>=cap else 'downloaded',
                received_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:item.update(status='failed',error_type=type(exc).__name__)
        receipt['requests'].append(item)
        (args.output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        print(index,api,item['status'],item.get('rows',''),flush=True)


if __name__=='__main__':main()
