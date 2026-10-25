"""候选全集的开市前5分钟及收盘竞价分钟；不按最终成交名单筛数据。"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fetch_execution_comparison_minutes import fetch
from execution_market_feed import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    spec=json.loads((run/'prepared/verification.json').read_text());codes=spec['codes']
    dates=json.loads((run/'minutes/protocol.json').read_text())['sessions']
    folder=run/'execution_raw';folder.mkdir(exist_ok=True);jobs=[]
    for date in dates:
        for label,start,end in [('morning','09:30:00','09:35:00'),('closing','14:59:00','15:00:00')]:
            jobs.append(dict(name=date+'_'+label,folder=str(folder),api='stk_mins',params=dict(ts_code=','.join(codes),freq='1min',
                start_date=date+' '+start,end_date=date+' '+end)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=[]
        for future in as_completed([pool.submit(fetch,j) for j in jobs]):
            result=future.result();results.append(result)
            if len(results)%20==0:print('quotes',len(results),'/',len(jobs),flush=True)
    if any(x['status']!='downloaded' for x in results):raise ValueError('存在分钟下载失败')
    frames=[]
    for item in results:
        path=folder/(item['name']+'.parquet')
        if file_hash(path)!=item['sha256']:raise ValueError('执行原件改变')
        x=pd.read_parquet(path)
        if len(x):frames.append(x)
    x=pd.concat(frames,ignore_index=True);x['timestamp']=pd.to_datetime(x.trade_time);x['date']=x.timestamp.dt.normalize()
    if x.duplicated(['ts_code','timestamp']).any():raise ValueError('执行报价重复')
    x['time']=x.timestamp.dt.strftime('%H:%M:%S')
    for name,at,price in [('open_first','09:31:00','open'),('open_1m','09:31:00','close'),
                           ('open_5m','09:35:00','close'),('close_auction','15:00:00','close')]:
        q=x[x.time.eq(at)].copy();q['price']=q[price];q['volume']=q.vol;q['target_time']=at
        q[['date','ts_code','price','volume','timestamp','target_time']].to_parquet(run/'prepared'/(name+'_quotes.parquet'),index=False)
    (folder/'completion.json').write_text(json.dumps(dict(jobs=len(jobs),codes=codes,source='configured Tushare archived minute bars',
        vintage='downloaded now; historical publication timing not independently certified',
        receipts={p.name:file_hash(p) for p in folder.glob('*.json') if p.name!='completion.json'},live_orders=False),indent=2))


if __name__=='__main__':main()
