"""固定候选的可知特征、日期等权与行情簇重采样。"""
import numpy as np
import pandas as pd


def build_features(panel, signals, cfg):
    """板块连续段内按交易日计算辅助字段；只使用信号日及之前观测。"""
    extras=[]
    for (_, _),part in panel[panel.index_valid].groupby(['sector_code','segment_id']):
        part=part.sort_values('date').copy()
        breadth=part.up_share.where(part.breadth_members/part.known_members>=cfg['minimum_observation_coverage'])
        turnover=part.turnover_pct.where(part.turnover_members/part.known_members>=cfg['minimum_observation_coverage'])
        k,j=cfg['recent_days'],cfg['preceding_days']
        part['breadth_recent']=breadth.rolling(k).mean()
        part['breadth_previous']=breadth.shift(k).rolling(j).mean()
        part['breadth_change']=part.breadth_recent-part.breadth_previous
        part['turnover_recent']=turnover.rolling(k).mean()
        part['turnover_previous']=turnover.shift(k).rolling(j).mean()
        part['turnover_contraction']=part.turnover_recent/part.turnover_previous.where(part.turnover_previous>0)
        extras.append(part[['date','sector_code','segment_id','breadth_recent','breadth_previous','breadth_change','turnover_recent','turnover_previous','turnover_contraction']])
    extra=pd.concat(extras,ignore_index=True)
    out=signals.merge(extra,on=['date','sector_code','segment_id'],how='left',validate='one_to_one')
    out['drawdown_depth']=-out.drawdown
    out['decline_speed']=-np.log(out.index_close/out.peak_price)/out.pullback_days
    out['slowdown_ratio']=out.recent_loss_mean/out.previous_loss_mean.where(out.previous_loss_mean>0)
    out['price_zone']=np.select([out.index_close<out.platform_low,out.index_close<=out.platform_high,out.index_close<=out.reference_high],cfg['price_zones'][:3],default=cfg['price_zones'][3])
    for key,edges in cfg['bins'].items():
        out[key+'_group']=pd.cut(out[key],bins=edges,labels=cfg['labels'][key],right=False).astype('object').fillna('缺失或超出分组范围')
    out['price_zone_group']=out.price_zone
    return out


def assign_clusters(dates, calendar, gap):
    """将距离不超过 gap 个交易日的信号日期连为一簇，仅供事后推断。"""
    dates=pd.DatetimeIndex(sorted(pd.unique(dates)))
    positions=calendar.get_indexer(dates)
    if (positions<0).any():raise ValueError('信号日期不在交易日历')
    ids=np.cumsum(np.r_[True,np.diff(positions)>gap])-1
    return pd.DataFrame({'date':dates,'cluster_id':ids})


def date_equal(frame, value):
    """先同日平均，再对日期等权；同一天股票多不会加大该日权重。"""
    return frame.groupby('date')[value].mean().dropna()


def summarize_group(frame, value, cfg, seed):
    """按完整行情簇重采样日期，保留簇内全部日期；小样本不报区间。"""
    per_date=frame.groupby(['date','cluster_id'])[value].mean().dropna().reset_index()
    clusters=per_date.cluster_id.unique()
    row=dict(events=int(frame[value].notna().sum()),dates=len(per_date),clusters=len(clusters),
             mean=float(per_date[value].mean()),median=float(per_date[value].median()),
             positive_dates=int(per_date[value].gt(0).sum()),ci_low=np.nan,ci_high=np.nan)
    if len(clusters)>=cfg['minimum_ci_clusters']:
        groups=per_date.groupby('cluster_id')[value].agg(['sum','count'])
        rng=np.random.default_rng(seed)
        draws=rng.integers(0,len(groups),size=(cfg['bootstrap_repetitions'],len(groups)))
        means=groups['sum'].to_numpy()[draws].sum(axis=1)/groups['count'].to_numpy()[draws].sum(axis=1)
        row['ci_low'],row['ci_high']=np.quantile(means,[.025,.975]).tolist()
    return row
