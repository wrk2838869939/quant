"""趋势入场的已知特征、事后路径标签及耐心持有的信号生命周期。"""
import numpy as np
import pandas as pd


def entry_features(close, size, turnover, medium=60, long=120):
    """所有滚动量截至当日；连续性变量仅受文献启发，不声称逐式复现。"""
    price=close.where(close>0);ret=price.pct_change(fill_method=None)
    logret=np.log(price/price.shift(1));positive=logret.clip(lower=0)
    sign=ret.gt(0).astype(float)-ret.lt(0).astype(float)
    sign=sign.where(ret.notna())
    return dict(mom60=price/price.shift(medium)-1,mom120=price/price.shift(long)-1,
        continuity=sign.rolling(long,min_periods=long).mean(),
        jump_share=positive.rolling(long,min_periods=long).max()/positive.rolling(long,min_periods=long).sum().replace(0,np.nan),
        extension=price/price.rolling(medium,min_periods=medium).mean()-1,
        volatility=ret.rolling(medium,min_periods=medium).std(),
        drawdown=price/price.rolling(long,min_periods=long).max()-1,
        log_size=np.log(size.where(size>0)),turnover=turnover.rolling(medium,min_periods=medium).mean())


def marked_paths(entry, closes, terminal, delist_offset, spec):
    """T+1开盘到T+H+1开盘的标记价格代理，缺失保留、退市减记。

    未知退出用最后已知价格估值；这不是可成交退出。停牌缺口不得删除候选。
    输入delist_offset为路径中的归零步(1..H+1)，未退市为无穷。
    """
    entry=np.asarray(entry,float);closes=np.asarray(closes,float);terminal=np.asarray(terminal,float)
    n=len(entry);h=len(closes)
    if closes.shape!=(h,n) or terminal.shape!=(n,) or np.asarray(delist_offset).shape!=(n,):raise ValueError('路径形状不一致')
    entry_ok=np.isfinite(entry)&(entry>0)
    path=np.ones((h+2,n));missing=np.zeros(n,int)
    terminated=np.zeros(n,bool)
    raw=np.vstack([closes,terminal])
    for k,prices in enumerate(raw,1):
        quoted=np.isfinite(prices)&(prices>0)
        missing+=~quoted
        path[k]=path[k-1]
        active = entry_ok & quoted & ~terminated
        path[k,active]=prices[active]/entry[active]
        terminated |= np.asarray(delist_offset)<=k
        path[k,terminated&entry_ok]=0.
    final=path[-1]-1;peak=path.max(axis=0)-1;mae=path.min(axis=0)-1
    peaks=np.maximum.accumulate(path,axis=0)
    dd=(path/peaks-1).min(axis=0)
    checkpoints=np.array([max(1,h//3),max(1,2*h//3),h+1])
    positive=(path[checkpoints]>1).sum(axis=0)
    sustained=entry_ok&(final>0)&(mae>=-spec['max_adverse'])&(positive>=spec['positive_checkpoints'])
    roundtrip=entry_ok&(peak>=spec['peak_for_roundtrip'])&(final<=0)
    slow_weak=entry_ok&(final<=0)&~roundtrip
    state=np.select([~entry_ok,sustained,roundtrip,slow_weak],['no_entry','sustained','roundtrip','weak'],default='other')
    return pd.DataFrame(dict(ret=final,mae=mae,mfe=peak,mdd=dd,giveback=peak-final,
        late_return=path[-1]-path[min(5,h+1)],sustained=sustained,roundtrip=roundtrip,state=state,
        entry_missing=~entry_ok,missing_marks=missing,exit_marked=~(np.isfinite(terminal)&(terminal>0)),delisted=terminated&entry_ok))


def patient_memberships(snapshots, policy, max_positions, exit_confirmations, min_entry_score=-np.inf):
    """只模拟信号名单，持有期间不按排名替换；缺少价格/成交验证不可叫真实持仓。"""
    held={};rows=[]
    for date,g in snapshots.groupby('date',sort=True):
        indexed=g.set_index('ts_code');valid=set(g.loc[g.trend,'ts_code'])
        for code in list(held):
            if code not in indexed.index:
                del held[code]
                continue
            held[code]=0 if code in valid else held[code]+1
            if held[code]>=exit_confirmations:del held[code]
        candidates=g[g.trend & g[policy].gt(min_entry_score)].sort_values([policy,'ts_code'],ascending=[False,True])
        for code in candidates.ts_code:
            if len(held)>=max_positions:break
            held.setdefault(code,0)
        rows.extend(dict(date=date,ts_code=code,policy=policy,failed_reviews=count) for code,count in held.items())
    return pd.DataFrame(rows,columns=['date','ts_code','policy','failed_reviews'])
