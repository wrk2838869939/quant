"""隔离增量获取沪深300，保存原始响应、历史散列和实际获取时刻。"""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import argparse,json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from execution_market_feed import file_hash
from shadow_planner import ShadowError


def main():
    p=argparse.ArgumentParser();p.add_argument('--asof',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--history',type=Path,default=ROOT/'data/raw/index_daily.parquet');a=p.parse_args()
    now=pd.Timestamp.now(tz='Asia/Shanghai')
    if now < pd.Timestamp(a.asof,tz='Asia/Shanghai')+pd.Timedelta(hours=18):raise ShadowError('指数增量须等该日收盘后')
    a.output.mkdir(parents=True,exist_ok=False)
    oldpath=a.history;old=pd.read_parquet(oldpath);old=old[old.ts_code.eq('000300.SH')].copy()
    old['date']=pd.to_datetime(old['date'] if 'date' in old else old.trade_date);old=old[old.date<=pd.Timestamp(a.asof)]
    calendar_path=ROOT/'data/raw/calendar.parquet'
    calendar=pd.read_parquet(calendar_path)
    sessions=pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open.astype(int).eq(1),'cal_date'].astype(str)))
    days=sessions[(sessions>old.date.max())&(sessions<=pd.Timestamp(a.asof))]
    frames=[];requests=[]
    # 按交易日精确查询；区间接口曾混入休市日，不能把它静默过滤后发布。
    for day in days:
        with open('/dev/null','w') as sink,redirect_stdout(sink),redirect_stderr(sink):
            from ts_client import get_pro
            frame=get_pro().query('index_daily',ts_code='000300.SH',trade_date=day.strftime('%Y%m%d'))
        if frame is None or len(frame)!=1 or not frame.trade_date.astype(str).eq(day.strftime('%Y%m%d')).all() or set(frame.ts_code)!={'000300.SH'}:
            raise ShadowError('指数逐日返回缺失、串日或代码不符')
        rawpath=a.output/f'raw_{day:%Y%m%d}.parquet';frame.to_parquet(rawpath,index=False)
        requests.append(dict(date=str(day.date()),received_at=pd.Timestamp.now(tz='UTC').isoformat(),sha256=file_hash(rawpath)))
        frames.append(frame)
    if not frames:raise ShadowError('没有新交易日')
    fresh=pd.concat(frames,ignore_index=True);fresh['date']=pd.to_datetime(fresh.trade_date)
    combined=pd.concat([old,fresh],ignore_index=True).sort_values('date')
    expected=sessions[(sessions>=combined.date.min())&(sessions<=pd.Timestamp(a.asof))]
    if not pd.DatetimeIndex(combined.date).equals(expected) or set(combined.ts_code)!={'000300.SH'} or combined.date.max()!=pd.Timestamp(a.asof):raise ShadowError('指数日期与交易日历不符')
    previous=combined.set_index('date').close.shift(1)
    if not (abs(fresh.set_index('date').pre_close-previous.loc[fresh.date])<.001).all():raise ShadowError('指数昨收与上一交易日不一致')
    path=a.output/'index_close.parquet';combined[['date','ts_code','close']].to_parquet(path,index=False)
    receipt=dict(code='000300.SH',asof=a.asof,complete=True,source='configured Tushare provider',downloaded_at=pd.Timestamp.now(tz='UTC').isoformat(),
        sha256=file_hash(path),historical_path=str(oldpath),historical_sha256=file_hash(oldpath),requests=requests,calendar_sha256=file_hash(calendar_path),query_mode='exact_trading_date',prior_range_response_quarantined=True)
    (a.output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print(json.dumps(dict(asof=a.asof,new_rows=len(fresh),rows=len(combined))))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        # 远端异常文本可能携带连接凭据，只报告异常类型。
        print(json.dumps(dict(status='failed',error_type=type(exc).__name__)));sys.exit(2)
