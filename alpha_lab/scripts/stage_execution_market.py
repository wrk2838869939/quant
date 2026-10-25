"""隔离下载最近已收盘交易日；不覆盖历史面板，不输出凭据或远端异常正文。"""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--after',default='2026-09-08')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    now=pd.Timestamp.now(tz='Asia/Shanghai')
    calendar=pd.read_parquet(ROOT/'data/raw/calendar.parquet')
    dates=pd.to_datetime(calendar.loc[calendar.is_open.astype(int).eq(1),'cal_date'],format='%Y%m%d')
    latest=now.normalize().tz_localize(None)
    if now.hour<18:latest-=pd.Timedelta(days=1)
    dates=dates[(dates>pd.Timestamp(args.after))&(dates<=latest)]
    rows=[]
    for date in dates:
        for api in ['daily','daily_basic','adj_factor']:
            item=dict(api=api,trade_date=date.strftime('%Y%m%d'),requested_at=datetime.now(timezone.utc).isoformat())
            try:
                with open('/dev/null','w') as sink,redirect_stdout(sink),redirect_stderr(sink):
                    from ts_client import get_pro
                    frame=get_pro().query(api,trade_date=item['trade_date'])
                if frame is None or frame.empty:raise ValueError('empty response')
                path=args.output/f'{api}_{item["trade_date"]}.parquet'
                frame.to_parquet(path,index=False)
                item.update(status='downloaded',rows=len(frame),file=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except Exception as exc:
                item.update(status='failed',error_type=type(exc).__name__)
            rows.append(item)
            (args.output/'fetch_receipt.json').write_text(json.dumps(dict(source='configured Tushare provider',
                latest_completed_session=str(dates.max().date()) if len(dates) else None,requests=rows,
                validation_pending=True,live_orders=False),ensure_ascii=False,indent=2))
            print(api,item['trade_date'],item['status'],item.get('rows',''),flush=True)


if __name__=='__main__':main()
