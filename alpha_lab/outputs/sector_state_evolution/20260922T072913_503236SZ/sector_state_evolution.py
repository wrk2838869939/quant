"""固定起点事件的状态跟踪；不因后来改善/失效而筛掉事件。"""
import numpy as np
import pandas as pd


def state_features(panel,cfg):
    """连续段内计算收盘可知的广度与跌速，不跨缺口前填。"""
    pieces=[]
    for _,part in panel[panel.index_valid].groupby(['sector_code','segment_id']):
        part=part.sort_values('date').copy()
        breadth=part.up_share.where(part.breadth_members/part.known_members>=cfg['minimum_observation_coverage'])
        daily=part.index_close.pct_change(fill_method=None)
        loss=(-daily).clip(lower=0)
        k,j=cfg['recent_days'],cfg['preceding_days']
        part['breadth_change']=breadth.rolling(k).mean()-breadth.shift(k).rolling(j).mean()
        part['loss_recent']=loss.rolling(k).mean()
        part['loss_previous']=loss.shift(k).rolling(j).mean()
        part['slowdown_ratio']=part.loss_recent/part.loss_previous.where(part.loss_previous>0)
        part['breadth_improving']=part.breadth_change>0
        part['slowing']=part.slowdown_ratio<=cfg['slowdown_ratio']
        known=part.breadth_change.notna()&part.loss_recent.notna()&part.loss_previous.notna()
        positive=part.loss_previous>0
        part['state']=np.select([
            ~known,known&~positive,
            known&positive&part.breadth_improving&part.slowing,
            known&positive&part.breadth_improving&~part.slowing,
            known&positive&~part.breadth_improving&part.slowing],
            ['状态数据不足','此前无下跌','广度改善且跌速放缓','仅广度改善','仅跌速放缓'],
            default='广度未改善且未减速')
        pieces.append(part)
    return pd.concat(pieces,ignore_index=True)


def build_paths(panel,origins,observations,calendar,cfg):
    """按全市场交易日定位0至10日，原形态失效仍留在起点队列内。"""
    features=state_features(panel,cfg).set_index(['sector_code','date'])
    lookup=observations.set_index(['event_id','date'])
    rows=[]
    for r in origins.itertuples():
        origin_i=calendar.get_loc(r.date)
        for offset in range(cfg['tracking_sessions']+1):
            target_i=origin_i+offset
            if target_i>=len(calendar):
                continue
            date=calendar[target_i]
            key=(r.sector_code,date)
            same=key in features.index and int(features.loc[key,'segment_id'])==r.segment_id
            record=dict(event_id=r.event_id,sector_code=r.sector_code,segment_id=r.segment_id,
                        origin_date=r.date,date=date,offset=offset,available=bool(same),origin_price=r.index_close,
                        origin_peak=r.peak_price,index_close=np.nan,breadth_change=np.nan,slowdown_ratio=np.nan,
                        loss_previous=np.nan,loss_recent=np.nan,breadth_improving=False,slowing=False,
                        state='状态数据不足',price_change_from_origin=np.nan,episode_active=False,
                        still_pullback=False,complete_candidate=False)
            if same:
                x=features.loc[key]
                for k in ['index_close','breadth_change','slowdown_ratio','loss_previous','loss_recent','breadth_improving','slowing','state']:
                    record[k]=x[k]
                record['price_change_from_origin']=x.index_close/r.index_close-1
                if (r.event_id,date) in lookup.index:
                    old=lookup.loc[(r.event_id,date)]
                    record.update(episode_active=True,still_pullback=bool(old.pullback_gate),complete_candidate=bool(old.candidate))
            rows.append(record)
    return pd.DataFrame(rows)


def forward_labels(panel,paths,calendar,benchmark,cfg):
    """只从实际观察日之后计算结果，缺口/截尾为空；不返回到状态特征。"""
    prices=panel[panel.index_valid].set_index(['sector_code','date'])
    records=[]
    for r in paths[paths.offset.isin(cfg['landmarks'])].itertuples():
        start=calendar.get_loc(r.date)
        for horizon in cfg['horizons']:
            future=calendar[start+1:start+horizon+1]
            key=pd.MultiIndex.from_product([[r.sector_code],future],names=['sector_code','date'])
            part=prices.reindex(key)
            full=(r.available and len(future)==horizon and part.segment_id.eq(r.segment_id).all() and part.index_close.notna().all())
            end=future[-1] if len(future)==horizon else pd.NaT
            raw=float(part.index_close.iloc[-1]/r.index_close-1) if full else np.nan
            market=float(benchmark.loc[end]/benchmark.loc[r.date]-1) if full and pd.notna(benchmark.get(end)) and pd.notna(benchmark.get(r.date)) else np.nan
            records.append(dict(event_id=r.event_id,offset=r.offset,date=r.date,horizon=horizon,complete=bool(full),
                                end_date=end,close_return=raw,market_return=market,market_excess=raw-market,
                                min_close_return=float(part.index_close.min()/r.index_close-1) if full else np.nan))
    return pd.DataFrame(records)
