"""只用截至决策收盘的市场特征，生成固定规则与恢复确认状态。"""
import numpy as np
import pandas as pd


def market_risk_features(panel, cfg):
    """输入已按PIT资格过滤的主板日线；返回非可交易市场代理及风险状态。

    涨跌幅用面板百分数收益，避免用未复权收盘跨除权计算广度。
    趋势基于逐股收益累积的价格代理；市场代理是每日有效股票的平均收益，
    仅用于风险信号，不能作为可交易指数业绩。新股/停牌缺失日不虚构收益。
    """
    if panel.duplicated(['date','ts_code']).any():
        raise ValueError('市场风险面板有重复键')
    x=panel.sort_values(['ts_code','date']).copy()
    x['ret']=x.pct_chg/100.
    if not np.isfinite(x.ret).all() or (x.ret<=-1).any():
        raise ValueError('市场收益缺失或越界')
    x['path']=x.groupby('ts_code').ret.transform(lambda s: (1+s).cumprod())
    x['ma']=x.groupby('ts_code').path.transform(lambda s:s.rolling(cfg['trend_days']).mean())
    x['above']=np.where(x.ma.notna(),(x.path>x.ma).astype(float),np.nan)
    g=x.groupby('date')
    d=g.agg(market_ret=('ret','mean'),up_ratio=('ret',lambda s:(s>0).mean()),
            above_ma_ratio=('above','mean'),n_market=('ts_code','size'),n_ma=('ma','count'))
    if (d.n_market<cfg['min_market_names']).any():
        raise ValueError('市场覆盖不足，不能默认低风险')
    d['trend_return']=(1+d.market_ret).rolling(cfg['trend_days']).apply(np.prod,raw=True)-1
    d['short_return']=(1+d.market_ret).rolling(cfg['short_return_days']).apply(np.prod,raw=True)-1
    d['median_return']=d.market_ret.rolling(cfg['median_days']).median()
    d['volatility']=d.market_ret.rolling(cfg['volatility_days']).std()
    # 阈值排除当日；全样本分位数会让未来改变历史决策。
    d['past_vol_quantile']=d.volatility.shift(1).rolling(cfg['quantile_days']).quantile(cfg['extreme_quantile'])
    return d


def fixed_risk_plan(features, rule, cfg):
    """当日风险成立则请求退出；连续无风险K日后恢复，不用未来价格。"""
    fields={'legacy_v14':['median_return','up_ratio'],
            'breadth_trend':['trend_return','above_ma_ratio'],
            'extreme_downside':['volatility','past_vol_quantile','short_return']}
    if rule not in fields:
        raise ValueError('未注册的风险规则')
    f=features[fields[rule]]
    valid=np.isfinite(f).all(axis=1)
    if rule=='legacy_v14':
        risk=(f.median_return<cfg['legacy_median_exit'])&(f.up_ratio<cfg['legacy_breadth_exit'])
    elif rule=='breadth_trend':
        risk=(f.trend_return<0)&(f.above_ma_ratio<cfg['breadth_exit'])
    else:
        risk=(f.volatility>f.past_vol_quantile)&(f.short_return<0)
    confirmation=cfg['recovery_confirmation_days']
    if int(confirmation)!=confirmation or confirmation<1:
        raise ValueError('恢复确认日数须为正整数')
    allowed, streak = 0, 0
    rows=[]
    for date in f.index:
        if not valid.loc[date]:
            allowed,streak=0,0
            reason='missing_features'
        elif risk.loc[date]:
            allowed,streak=0,0
            reason=rule
        else:
            streak+=1
            if streak>=confirmation:
                allowed=1
            reason='normal' if allowed else 'recovery_confirmation'
        rows.append(dict(date=date,allowed=allowed,raw_risk=bool(risk.loc[date]),
                         valid=bool(valid.loc[date]),safe_streak=streak,reason=reason))
    return pd.DataFrame(rows).set_index('date')
