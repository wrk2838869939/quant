"""月度执行下的趋势退出确认；日线只提供过去持续性，不增加交易日。"""
import numpy as np
import pandas as pd


def build_hold_variants(trend, observable, reviews, spec):
    """构造确认快慢对照；未知打断月度计数，日线缺失不能支持提前退出。"""
    if not trend.index.equals(observable.index) or not trend.columns.equals(observable.columns):
        raise ValueError('趋势与可观测矩阵不对齐')
    if not trend.index.is_monotonic_increasing or trend.index.has_duplicates:
        raise ValueError('日历必须唯一且递增')
    reviews = pd.DatetimeIndex(reviews)
    if not reviews.is_monotonic_increasing or reviews.has_duplicates or not reviews.isin(trend.index).all():
        raise ValueError('复核日必须来自原日历且唯一递增')
    window = spec['persistence_window']
    if type(window) is not int or window < 1:
        raise ValueError('持续性窗口必须为正整数')
    if not observable.isin([False, True]).all().all():
        raise ValueError('可观测性必须明确为布尔值')
    known = observable.astype(bool)
    if not (trend.isin([False, True]) | ~known).all().all():
        raise ValueError('可观测位置必须有明确趋势状态')
    weak = known & ~trend.astype(bool)
    count = weak.astype(int).rolling(window, min_periods=window).sum().loc[reviews]
    coverage = known.astype(int).rolling(window, min_periods=window).sum().loc[reviews]
    monthly_known = known.loc[reviews]
    monthly_weak = weak.loc[reviews]
    streak = np.zeros(len(trend.columns), dtype=int)
    streaks = []
    for date in reviews:
        streak = np.where(monthly_weak.loc[date].to_numpy(), streak + 1, 0)
        streaks.append(streak.copy())
    streaks = pd.DataFrame(streaks, index=reviews, columns=trend.columns)
    variants = {}
    for name, rule in spec['variants'].items():
        confirmations = rule['monthly_confirmations']
        threshold = rule.get('persistent_weak_days')
        if type(confirmations) is not int or confirmations < 1:
            raise ValueError('月度确认数非法')
        exit_now = streaks.ge(confirmations)
        if threshold is not None:
            if type(threshold) is not int or not 1 <= threshold <= window:
                raise ValueError('日线持续性门槛非法')
            exit_now |= monthly_weak & coverage.eq(window) & count.ge(threshold)
        variants[name] = (~exit_now).astype(float).where(monthly_known)
    evidence = dict(monthly_observable=monthly_known, monthly_weak=monthly_weak,
                    monthly_failure_streak=streaks, daily_weak_count=count, daily_known_count=coverage)
    return variants, evidence
