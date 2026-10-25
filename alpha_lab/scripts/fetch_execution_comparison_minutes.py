"""隔离下载全A历史30分钟线；两进程限流、断点回执，不读取/打印认证信息。"""
from contextlib import redirect_stdout,redirect_stderr
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json,sys,time

import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from execution_market_feed import file_hash


def fetch(task):
    folder=Path(task['folder']);path=folder/(task['name']+'.parquet');receipt=path.with_suffix('.json')
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get('status')=='downloaded' and saved.get('rows',0)>0 and file_hash(path)==saved['sha256']:
            return saved
    if receipt.exists():
        import shutil
        archive=folder/'attempts';archive.mkdir(exist_ok=True)
        suffix=file_hash(receipt)[:12]
        shutil.copy2(receipt,archive/(receipt.stem+'_'+suffix+'.json'))
        if path.exists():shutil.copy2(path,archive/(path.stem+'_'+suffix+'.parquet'))
    result=dict(task,requested_at=pd.Timestamp.now(tz='UTC').isoformat())
    for attempt in range(2):
        try:
            with open('/dev/null','w') as sink,redirect_stdout(sink),redirect_stderr(sink):
                from ts_client import get_pro
                frame=get_pro().query(task['api'],**task['params'])
            if frame is None or frame.empty or len(frame)>=8000:raise ValueError('truncated_or_missing_response')
            if len(frame) and (not {'ts_code','trade_time','close','vol'}.issubset(frame)
                    or not set(frame.ts_code)<=set(task['params']['ts_code'].split(','))):
                raise ValueError('invalid_schema_or_codes')
            frame.to_parquet(path,index=False)
            result.update(status='downloaded',rows=len(frame),sha256=file_hash(path),
                received_at=pd.Timestamp.now(tz='UTC').isoformat(),attempts=attempt+1)
            break
        except Exception as exc:
            result.update(status='failed',error_type=type(exc).__name__,attempts=attempt+1)
            time.sleep(.5)
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',default='2026-01-05');p.add_argument('--end',default='2026-09-08')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    feature=ROOT/'outputs/monthly_manual_day15/observation_final/feature_panel.parquet'
    raw=pd.read_parquet(feature,columns=['date','ts_code'])
    active=raw.loc[raw.date.between(a.start,a.end)];codes=sorted(active.ts_code.unique())
    dates=sorted(active.date.unique());jobs=[]
    stock=a.output/'stocks30';stock.mkdir(exist_ok=True)
    batch_size=max(1,min(20,7999//(len(dates)*9)))
    for i in range(0,len(codes),batch_size):
        jobs.append(dict(name=f'batch_{i//batch_size:04d}',folder=str(stock),api='stk_mins',params=dict(ts_code=','.join(codes[i:i+batch_size]),
            freq='30min',start_date=a.start+' 09:00:00',end_date=a.end+' 15:00:00')))
    index=a.output/'index30';index.mkdir(exist_ok=True)
    for date in dates:
        iso=str(pd.Timestamp(date).date())
        jobs.append(dict(name=iso,folder=str(index),api='idx_mins',params=dict(ts_code='000300.SH',freq='30min',
            start_date=iso+' 09:00:00',end_date=iso+' 15:00:00')))
    protocol=dict(start=a.start,end=a.end,codes=codes,sessions=[str(pd.Timestamp(d).date()) for d in dates],
        jobs=len(jobs),workers=2,feature_sha256=file_hash(feature),live_orders=False)
    old=a.output/'protocol.json'
    if old.exists() and json.loads(old.read_text())!=protocol:raise ValueError('不能更改已有下载协议')
    old.write_text(json.dumps(protocol,ensure_ascii=False,indent=2));done=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(fetch,j) for j in jobs]):
            done.append(f.result())
            if len(done)%25==0 or done[-1]['status']=='failed':
                print(json.dumps(dict(completed=len(done),total=len(jobs),failed=sum(x['status']=='failed' for x in done),
                    elapsed_seconds=round(time.monotonic()-started))),flush=True)
    (a.output/'download_summary.json').write_text(json.dumps(dict(jobs=len(jobs),
        downloaded=sum(x['status']=='downloaded' for x in done),failed=[x['name'] for x in done if x['status']=='failed'],
        elapsed_seconds=time.monotonic()-started,live_orders=False),indent=2))


if __name__=='__main__':main()
