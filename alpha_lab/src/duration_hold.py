"""退出证据按交易日计量，复核日只决定何时执行。"""
import pandas as pd

def duration_holds(prices,trend,observable,reviews,spec):
    for f in [trend,observable]:
        if not f.index.equals(prices.index) or not f.columns.equals(prices.columns):raise ValueError('矩阵未对齐')
    window=spec['window'];usual=spec['usual_weak_days'];broken_days=spec['broken_weak_days'];long=spec['long_window']
    if not all(type(v) is int for v in [window,usual,broken_days,long]) or not 0<broken_days<=usual<=window or long<window:raise ValueError('持续性参数非法')
    weak=observable & ~trend
    counts=weak.astype(int).rolling(window,min_periods=window).sum()
    coverage=observable.astype(int).rolling(window,min_periods=window).sum()
    clean=prices.where(prices>0)
    broken=(clean/clean.shift(long)-1).le(0)&clean.le(clean.rolling(long,min_periods=long).mean())
    threshold=pd.DataFrame(usual,index=prices.index,columns=prices.columns).mask(broken,broken_days)
    exits=weak & counts.ge(threshold) & coverage.eq(window)
    return (~exits).astype(float).where(observable).loc[reviews]
