"""市场风险证据：仅用当日及此前输入，目标从次日开盘开始。"""
import numpy as np
import pandas as pd


FAMILIES = {
    'price': ['ret20', 'drawdown60'],
    'breadth': ['ret20', 'drawdown60', 'below_ma20', 'breadth_change5', 'decliners5'],
    'volatility': ['ret20', 'drawdown60', 'vol20', 'vol_shock'],
    'correlation': ['ret20', 'drawdown60', 'stock_index_corr20'],
    'liquidity': ['ret20', 'drawdown60', 'amount_relative', 'illiquidity_relative'],
}
FAMILIES['joint'] = list(dict.fromkeys(x for cols in FAMILIES.values() for x in cols))


def market_features(close, amount, index_close):
    """历史当日完整价格样本聚合；不使用当前成分股名单。"""
    if not close.index.equals(amount.index) or not close.columns.equals(amount.columns):
        raise ValueError('价格和成交额轴必须一致')
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ValueError('日期必须有序且唯一')
    ret = close.div(close.shift(1)) - 1
    ready = close.notna().cumsum().ge(120) & close.rolling(60, min_periods=60).count().eq(60)
    ready &= close.notna() & close.gt(0)
    ma20 = close.rolling(20, min_periods=20).mean()
    breadth = close.lt(ma20).where(ready).mean(axis=1)
    decline = ret.lt(0).where(ready).mean(axis=1)
    index_close = index_close.reindex(close.index)
    mret = index_close / index_close.shift(1) - 1
    corr = ret.rolling(20, min_periods=20).corr(mret).where(ready).median(axis=1)
    relative_amount = amount / amount.rolling(20, min_periods=20).median()
    illiquidity = ret.abs().div(amount.where(amount.gt(0)))
    relative_illiq = illiquidity / illiquidity.rolling(60, min_periods=60).median()
    result = pd.DataFrame(dict(
        ret20=index_close/index_close.shift(20)-1,
        drawdown60=1-index_close/index_close.rolling(60, min_periods=60).max(),
        below_ma20=breadth, breadth_change5=breadth-breadth.shift(5),
        decliners5=decline.rolling(5, min_periods=5).mean(),
        vol20=mret.rolling(20, min_periods=20).std(),
        vol_shock=mret.rolling(5, min_periods=5).std()/mret.rolling(20, min_periods=20).std(),
        stock_index_corr20=corr,
        amount_relative=relative_amount.where(ready & amount.gt(0)).median(axis=1),
        illiquidity_relative=relative_illiq.where(ready).median(axis=1),
        price_coverage=ready.sum(axis=1),
        liquidity_coverage=(ready & relative_illiq.notna() & amount.gt(0)).sum(axis=1),
    ))
    return result.replace([np.inf, -np.inf], np.nan)


def forward_outcomes(open_price, close_price, horizon=20, loss_threshold=.05):
    """次日开盘至第H日收盘最差收益，以及第H+1日开盘退出的固定期限收益。"""
    if not open_price.index.equals(close_price.index):
        raise ValueError('标签价格日期必须一致')
    entry = open_price.shift(-1)
    paths = pd.concat({j:close_price.shift(-j)/entry-1 for j in range(1,horizon+1)}, axis=1)
    worst = paths.min(axis=1).where(paths.notna().all(axis=1))
    terminal = open_price.shift(-(horizon+1))/entry-1
    end = pd.Series(open_price.index, index=open_price.index).shift(-(horizon+1))
    result = pd.DataFrame(dict(worst=worst, terminal=terminal, label_end=end))
    result['loss_event'] = worst.le(-loss_threshold).astype(float).where(worst.notna())
    result['cash_benefit_proxy'] = -terminal - .004  # 固定双边40bp往返成本代理，不是账户成交。
    return result


def matured_weekly_rows(features, outcomes, start, end):
    """周首决策、最长目标已成熟，边界不按行数猜测。"""
    weekly = features.index.to_series().groupby(features.index.to_period('W')).first()
    all_features = FAMILIES['joint']
    valid = features[all_features].notna().all(axis=1) & features.price_coverage.ge(1000)
    valid &= outcomes[['worst','terminal']].notna().all(axis=1)
    valid &= features.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end))
    valid &= outcomes.label_end.le(pd.Timestamp(end))
    return features.index[valid & features.index.isin(weekly)]
