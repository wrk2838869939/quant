"""长期趋势假设和低频复核日历；本模块不把趋势分数当预期收益。"""
import numpy as np
import pandas as pd


def review_dates(calendar, frequency):
    """每周/月首个交易日收盘复核；在任一历史截断点不会倒填月末信号。"""
    calendar=pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('交易日历必须唯一且递增')
    if frequency not in ['weekly','monthly']:
        raise ValueError('只允许周频或月频复核')
    periods=calendar.to_period('W-SUN' if frequency=='weekly' else 'M')
    return calendar[~periods.duplicated()]


def trend_state(adjusted_close, spec):
    """仅用已知价格：60/120日均上涨且高于长期均线作为待检验条件。"""
    medium,long=spec['medium_window'],spec['long_window']
    if not 1<=medium<long or not np.isfinite(spec['min_return']):
        raise ValueError('趋势窗口或门槛非法')
    close=adjusted_close.where(adjusted_close>0)
    ret_medium=close/close.shift(medium)-1
    ret_long=close/close.shift(long)-1
    ma=close.rolling(long,min_periods=long).mean()
    eligible=ret_medium.gt(spec['min_return'])&ret_long.gt(spec['min_return'])
    if spec['require_above_long_ma']:eligible &= close.gt(ma)
    # 相对强弱只是排序假设，尚未通过条件收益与可成交验证。
    score=(ret_medium.rank(axis=1,pct=True)+ret_long.rank(axis=1,pct=True))/2
    return score.where(eligible),eligible
