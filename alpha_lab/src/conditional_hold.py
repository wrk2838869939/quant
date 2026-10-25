"""原趋势内的条件确认速度；分治状态不是未来收益标签。"""
import numpy as np
import pandas as pd


def build_conditional_holds(prices, trend, observable, reviews, spec):
    """长期两项均好时多等一次；均弱且持续时提前，其他保留原确认。"""
    from hold_sensitivity import build_hold_variants
    if not prices.index.equals(trend.index) or not prices.columns.equals(trend.columns):
        raise ValueError('价格与趋势矩阵不对齐')
    for key in ['medium_window','long_window','baseline_confirmations','patient_confirmations',
                'persistence_window','persistent_weak_days']:
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError('窗口和确认数必须正整数')
    if (spec['medium_window'] >= spec['long_window'] or
        spec['patient_confirmations'] <= spec['baseline_confirmations'] or
        spec['persistent_weak_days'] > spec['persistence_window']):
        raise ValueError('窗口/确认阈值顺序非法')
    original, evidence = build_hold_variants(trend, observable, reviews, dict(
        persistence_window=spec['persistence_window'],variants=dict(
            baseline=dict(monthly_confirmations=spec['baseline_confirmations']),
            persist10=dict(monthly_confirmations=spec['baseline_confirmations'],
                           persistent_weak_days=spec['persistent_weak_days']))))
    clean = prices.where(prices > 0)
    long_return = clean / clean.shift(spec['long_window']) - 1
    average = clean.rolling(spec['long_window'], min_periods=spec['long_window']).mean()
    # 比较运算沿用原trend_state，不用比值gap替代严格close>MA；gap只作描述。
    above = clean.gt(average)
    known = evidence['monthly_observable']
    weak = evidence['monthly_weak']
    lr = long_return.loc[reviews]
    ma = average.loc[reviews]
    price = clean.loc[reviews]
    if ((lr.isna() | ma.isna() | price.isna()) & known).any().any():
        raise ValueError('可观测位置缺少长期价格证据')
    intact = known & lr.gt(0) & above.loc[reviews]
    broken = known & lr.le(0) & price.le(ma)
    persistent = (weak & evidence['daily_known_count'].eq(spec['persistence_window'])
                  & evidence['daily_weak_count'].ge(spec['persistent_weak_days']))
    streak = evidence['monthly_failure_streak']
    variants = {}
    for name, rule in spec['variants'].items():
        if rule['accelerate'] not in ['none','all','broken'] or type(rule['patient']) is not bool:
            raise ValueError('分治政策非法')
        threshold = pd.DataFrame(spec['baseline_confirmations'],index=known.index,columns=known.columns)
        if rule['patient']:
            threshold = threshold.mask(intact & weak, spec['patient_confirmations'])
        exits = streak.ge(threshold)
        if rule['accelerate'] == 'all':
            exits |= persistent
        elif rule['accelerate'] == 'broken':
            exits |= persistent & broken
        variants[name] = (~exits).astype(float).where(known)
    if 'baseline' in variants:
        pd.testing.assert_frame_equal(variants['baseline'], original['baseline'], check_exact=True)
    if 'persist10' in variants:
        pd.testing.assert_frame_equal(variants['persist10'], original['persist10'], check_exact=True)
    states = pd.DataFrame('unknown', index=known.index, columns=known.columns)
    states = states.mask(known & ~weak, 'strong').mask(weak, 'mixed')
    states = states.mask(weak & intact, 'long_intact_weak').mask(weak & broken, 'long_broken_weak')
    evidence.update(long_intact=intact,long_broken=broken,weak_regime=states,long_return=lr,
                    long_ma=ma,long_ma_gap=price/ma-1)
    return variants,evidence
