"""实际退出事件的可知状态及独立标记收益；不等同于组合现金收益。"""
import numpy as np
import pandas as pd

def state_features(prices,trend,observable):
    for f in [trend,observable]:
        if not prices.index.equals(f.index) or not prices.columns.equals(f.columns):raise ValueError('状态矩阵未对齐')
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:raise ValueError('价格日期须唯一递增')
    p=prices.where(prices>0);ret=p.pct_change(fill_method=None);ma=p.rolling(120,min_periods=120).mean()
    known=observable.astype(int).rolling(20,min_periods=20).sum()
    weak=(observable&~trend).astype(int).rolling(20,min_periods=20).sum().where(known.eq(20))/20
    return dict(ret20=p/p.shift(20)-1,ret60=p/p.shift(60)-1,ret120=p/p.shift(120)-1,
        ma120_gap=p/ma-1,ma120_slope20=ma/ma.shift(20)-1,
        drawdown120=p/p.rolling(120,min_periods=120).max()-1,
        rebound20=p/p.rolling(20,min_periods=20).min()-1,weak_fraction20=weak,
        volatility20=ret.rolling(20,min_periods=20).std())

def nonoverlapping_events(frame,calendar,distance):
    """按当时日期保留每股最早事件；间隔严格大于H，未来标签区间不相交。"""
    if type(distance) is not int or distance<1:raise ValueError('间隔须正整数')
    out=frame.sort_values(['date','ts_code'],kind='stable').copy();last={};keep=[]
    position={d:i for i,d in enumerate(calendar)}
    for row in out.itertuples():
        i=position[row.date];accept=row.ts_code not in last or i-last[row.ts_code]>distance
        keep.append(accept)
        if accept:last[row.ts_code]=i
    out['nonoverlap_primary']=keep
    return out

def forward_marks(prices,date,code,horizon):
    """决策后下一交易日收盘至H交易日后收盘，缺失/尾部不补值。"""
    i=prices.index.get_loc(date);start=i+1;end=start+horizon
    if end>=len(prices):return dict(ret=np.nan,mae=np.nan,mfe=np.nan,label_start=pd.NaT,label_end=pd.NaT,missing=True,right_censored=True)
    path=prices.iloc[start:end+1][code]
    invalid=path.isna().any() or path.le(0).any()
    result=dict(label_start=prices.index[start],label_end=prices.index[end],missing=bool(invalid),right_censored=False)
    if invalid:return dict(**result,ret=np.nan,mae=np.nan,mfe=np.nan)
    relative=path/path.iloc[0]-1
    return dict(**result,ret=float(relative.iloc[-1]),mae=float(relative.min()),mfe=float(relative.max()))
