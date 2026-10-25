"""板块首次确认的入场资格及指定日期个股趋势；只依赖当时信息。"""
import numpy as np
import pandas as pd


def entry_window_rows(event, dates, source_states, representative_confirmed):
    """固定事件资格失效后不可复活；日期由市场交易日历提供。"""
    alive = True
    rows = []
    for age, date in enumerate(dates):
        state = source_states.loc[date] if date in source_states.index else None
        reason = ''
        if state is None:
            reason = 'missing_source_state'
        elif (state.event_id != event.event_id or state.segment_id != event.segment_id
              or state.peak_date != event.peak_date):
            reason = 'episode_segment_or_peak_changed'
        elif not bool(state.pullback_gate and state.near_reference):
            reason = 'structure_or_reference_invalid'
        valid = not reason
        alive &= valid
        rows.append(dict(event_id=event.merged_event_id, source_event_id=event.event_id,
                         sector_code=event.sector_code, window=int(event.window),
                         confirmation_date=event.date, date=date, age_sessions=age,
                         legacy_today=bool(representative_confirmed.get((date, event.sector_code), False)),
                         same_event_today=bool(valid and state.confirmed), persistent_5d=alive,
                         invalid_reason=reason if reason else '' if alive else 'previously_invalidated'))
    return rows


def trend_at_date(close, date, cfg):
    """仅计算给定日期，与trend_pullback相同的峰值、趋势及回落条件。

    完整历史仅用于已发生峰值处的指标；输出不会读取date之后的行。
    """
    history = close.loc[:date]
    result = pd.Series(False, index=close.columns)
    if len(history) <= cfg['peak_lookback']:
        return result
    c = history.to_numpy(float)
    last = len(c) - 1
    peaks = c[last - cfg['peak_lookback']:last]
    valid = np.isfinite(peaks).all(axis=0) & (peaks > 0).all(axis=0) & np.isfinite(c[last])
    avg = history.rolling(cfg['average_window']).mean().to_numpy()
    for j in np.flatnonzero(valid):
        p = last - 1 - int(np.argmax(peaks[::-1, j]))
        if p < max(cfg['advance_lookback'], cfg['historical_window'], cfg['average_slope_days']):
            continue
        prior = c[p - cfg['advance_lookback']:p, j]
        if not np.isfinite(prior).all():
            continue
        a = p - 1 - int(np.argmin(prior[::-1]))
        if c[a, j] <= 0:
            continue
        historical = (c[p, j] / c[p - cfg['historical_window'], j] - 1 >= cfg['minimum_historical_return']
                      and c[p, j] > avg[p, j]
                      and avg[p, j] / avg[p - cfg['average_slope_days'], j] - 1 > 0)
        ordered = (cfg['minimum_pullback_days'] <= last - p <= cfg['maximum_pullback_days']
                   and p - a >= cfg['minimum_advance_days']
                   and c[p, j] / c[a, j] - 1 >= cfg['stock_minimum_advance']
                   and c[last, j] / c[p, j] - 1 <= -cfg['stock_minimum_pullback'])
        result.iloc[j] = historical and ordered and (
            not cfg['require_current_above_average'] or c[last, j] > avg[last, j])
    return result
