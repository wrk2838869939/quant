"""把结构触发、价值区观察与短期确认分开；只使用当时已知数据。"""
import pandas as pd


def stage_events(observations, memory_days):
    """在同一事件和峰值内保留短期证据，新峰值出现后重新累计。

    观察池要求已先涨后落并出现快速下跌；接近同轮价格参照后触发。
    确认时仍要求处于回落和参考区，减速必须是当日可见。
    """
    result = observations.sort_values(['event_id', 'date']).copy()
    result['watch'] = result.pullback_gate & result.fast_fall_seen
    result['trigger'] = result.watch & result.near_reference
    groups = result.groupby(['event_id', 'peak_date'], sort=False)
    result['trigger_recent'] = groups.trigger.transform(
        lambda x: x.rolling(memory_days, min_periods=1).max()).astype(bool)
    result['support_recent'] = groups.support_broken.transform(
        lambda x: x.rolling(memory_days, min_periods=1).max()).astype(bool)
    result['confirmed'] = (result.pullback_gate & result.near_reference &
                           result.trigger_recent & result.support_recent & result.slowed)
    for key in ['watch', 'trigger', 'confirmed']:
        count = result.groupby('event_id')[key].cumsum()
        result['first_' + key] = result[key] & count.eq(1)
    return result.reset_index(drop=True)
