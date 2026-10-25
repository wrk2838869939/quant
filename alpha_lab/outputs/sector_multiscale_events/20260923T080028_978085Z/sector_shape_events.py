"""平台—上涨—回落事件的逐日状态机；未来观察结果不进入识别。"""
import numpy as np
import pandas as pd


def identify_events(panel, cfg):
    """逐个连续有效指数段扫描；平台与上涨初期参照绑定同一事件。

    平台使用突破前完整窗口，参照带仅在上涨初期扩展。历史趋势在
    当时已观察到的峰值处评价，不要求回落当日重新站上均线。
    """
    observations, episodes = [], []
    for sector, stock in panel.groupby('sector_code', sort=True):
        stock = stock.sort_values('date').reset_index(drop=True)
        for segment, part in stock[stock.index_valid].groupby('segment_id', sort=True):
            part = part.reset_index(drop=True)
            if not segment or part.empty:
                continue
            c = part.index_close
            daily_return = c.pct_change(fill_method=None)
            avg = c.rolling(cfg['trend_average_window']).mean()
            trend_ret = c / c.shift(cfg['trend_window']) - 1
            slope = avg / avg.shift(cfg['trend_slope_days']) - 1
            trend = (trend_ret >= cfg['trend_minimum_return']) & (c > avg) & (slope > 0)
            loss = (-daily_return).clip(lower=0)
            late = loss.rolling(cfg['recent_slow_days']).mean()
            early = loss.shift(cfg['recent_slow_days']).rolling(cfg['previous_slow_days']).mean()
            fast = c / c.shift(cfg['fast_fall_window']) - 1
            turn_base = part.turnover_pct.shift(1).rolling(cfg['turnover_baseline_window']).mean()
            state = None

            def finish(reason, at_i):
                if state is None:
                    return
                episodes.append({**state, 'end_date': part.date.iloc[at_i],
                                 'end_reason': reason, 'end_price': c.iloc[at_i],
                                 'right_censored': reason == 'dataset_end'})

            for i, row in part.iterrows():
                price = float(c.iloc[i])
                if state is not None:
                    age = i-state['breakout_i']
                    stale_unconfirmed = not state['rally_confirmed'] and age > cfg['maximum_unconfirmed_days']
                    stale_pullback = state['rally_confirmed'] and i-state['peak_i'] > cfg['maximum_pullback_days']
                    if age > cfg['maximum_episode_days'] or stale_unconfirmed or stale_pullback:
                        finish('unconfirmed_timeout' if stale_unconfirmed else 'pullback_timeout' if stale_pullback else 'episode_timeout', i)
                        state = None
                if state is None and i >= cfg['platform_window']:
                    base = c.iloc[i-cfg['platform_window']:i]
                    low, high = float(base.min()), float(base.max())
                    flat = high/low-1 <= cfg['platform_max_range']
                    flat &= abs(base.iloc[-1]/base.iloc[0]-1) <= cfg['platform_max_abs_return']
                    if flat and price >= high*(1+cfg['breakout_margin']):
                        start_i = i-cfg['platform_window']
                        state = dict(event_id=f'{sector}_s{int(segment)}_{row.date:%Y%m%d}',
                                     sector_code=sector, classification_version=row.classification_version,
                                     segment_id=int(segment), platform_start=part.date.iloc[start_i],
                                     platform_end=part.date.iloc[i-1], platform_start_i=start_i,
                                     platform_low=low, platform_high=high,
                                     platform_median=float(base.median()),
                                     platform_total_cap=float(part.total_mv_yuan.iloc[start_i:i].median()),
                                     platform_cap_members=float(part.total_mv_yuan_members.iloc[start_i:i].median()),
                                     breakout_date=row.date, breakout_i=i, reference_high=price,
                                     reference_ready=False, peak_date=row.date, peak_i=i, peak_price=price,
                                     trend_at_peak=bool(trend.iloc[i]), trend_return_at_peak=trend_ret.iloc[i],
                                     trend_slope_at_peak=slope.iloc[i], support_at_peak=float(base.tail(cfg['support_window']).min()),
                                     rally_confirmed=False, rally_date=pd.NaT, first_pullback_date=pd.NaT,
                                     first_candidate_date=pd.NaT, fast_fall_seen=False,
                                     added_members_since_platform=int(part.added_members.iloc[start_i:i+1].sum()),
                                     removed_members_since_platform=int(part.removed_members.iloc[start_i:i+1].sum()))
                elif state is not None:
                    state['added_members_since_platform'] += int(row.added_members)
                    state['removed_members_since_platform'] += int(row.removed_members)
                if state is None:
                    continue
                age = i-state['breakout_i']
                if age < cfg['early_rise_days']:
                    state['reference_high'] = max(state['reference_high'], price)
                state['reference_ready'] = age >= cfg['early_rise_days']-1
                if price > state['peak_price']:
                    state.update(peak_price=price, peak_i=i, peak_date=row.date,
                                 trend_at_peak=bool(trend.iloc[i]), trend_return_at_peak=trend_ret.iloc[i],
                                 trend_slope_at_peak=slope.iloc[i],
                                 support_at_peak=float(c.iloc[max(0,i-cfg['support_window']):i].min()),
                                 fast_fall_seen=False)
                rise = state['peak_price']/state['platform_high']-1
                if not state['rally_confirmed'] and rise >= cfg['minimum_rally'] and state['peak_i']-state['breakout_i'] >= cfg['minimum_rally_days']:
                    state['rally_confirmed'] = True
                    state['rally_date'] = row.date
                pullback_age = i-state['peak_i']
                drawdown = price/state['peak_price']-1
                post_peak_fall = price/c.iloc[max(state['peak_i'], i-cfg['fast_fall_window'])]-1
                if pullback_age > 0 and post_peak_fall <= -cfg['minimum_fast_fall']:
                    state['fast_fall_seen'] = True
                pullback = bool(state['rally_confirmed'] and state['trend_at_peak'] and
                                cfg['minimum_pullback_days'] <= pullback_age <= cfg['maximum_pullback_days'] and
                                drawdown <= -cfg['minimum_pullback'])
                slowed = bool(early.iloc[i] > 0 and late.iloc[i] <= early.iloc[i]*cfg['slowdown_ratio'])
                support_broken = price < state['support_at_peak']
                near_reference = bool(state['reference_ready'] and price <= state['reference_high']*(1+cfg['reference_upper_tolerance']) and
                                      price >= state['platform_low']*(1-cfg['reference_lower_tolerance']))
                candidate = bool(pullback and state['fast_fall_seen'] and slowed and support_broken and near_reference)
                first_pullback = pullback and pd.isna(state['first_pullback_date'])
                first_candidate = candidate and pd.isna(state['first_candidate_date'])
                if first_pullback:
                    state['first_pullback_date'] = row.date
                if first_candidate:
                    state['first_candidate_date'] = row.date
                observations.append({**row.to_dict(), **state, 'date': row.date,
                                     'episode_age': age, 'rally_return': rise, 'drawdown': drawdown,
                                     'pullback_days': pullback_age, 'return_10d': fast.iloc[i],
                                     'post_peak_fast_return': post_peak_fall,
                                     'previous_loss_mean': early.iloc[i], 'recent_loss_mean': late.iloc[i],
                                     'slowed': slowed, 'support_broken': support_broken,
                                     'near_reference': near_reference, 'reference_distance': price/state['reference_high']-1,
                                     'observed_cap_vs_platform': row.total_mv_yuan/state['platform_total_cap']-1 if state['platform_total_cap'] > 0 else np.nan,
                                     'turnover_vs_previous20': row.turnover_pct/turn_base.iloc[i] if turn_base.iloc[i] > 0 else np.nan,
                                     'pullback_gate': pullback, 'candidate': candidate,
                                     'first_pullback': first_pullback, 'first_candidate': first_candidate,
                                     'stage': 'candidate' if candidate else 'pullback' if pullback else 'rally' if state['rally_confirmed'] else 'breakout'})
            if state is not None:
                is_last = part.date.iloc[-1] == panel.date.max()
                finish('dataset_end' if is_last else 'segment_end', len(part)-1)
    return pd.DataFrame(observations), pd.DataFrame(episodes)


def evaluate_outcomes(panel, observations, horizons):
    """独立的事后观察表；不进入事件状态机，不作为策略收益。"""
    records = []
    for (sector, segment), part in panel[panel.index_valid].groupby(['sector_code', 'segment_id']):
        part = part.sort_values('date').reset_index(drop=True)
        positions = {d:i for i,d in enumerate(part.date)}
        signals = observations[(observations.sector_code == sector) & (observations.segment_id == segment) &
                               (observations.first_pullback | observations.first_candidate)]
        for row in signals.itertuples():
            for trigger in ['first_pullback', 'first_candidate']:
                if not getattr(row, trigger):
                    continue
                i = positions[row.date]
                for horizon in horizons:
                    complete = i+horizon < len(part)
                    future = part.index_close.iloc[i+1:i+horizon+1] if complete else pd.Series(dtype=float)
                    records.append(dict(event_id=row.event_id, sector_code=sector, signal_date=row.date,
                                        trigger=trigger, horizon=horizon, complete=complete,
                                        close_return=float(future.iloc[-1]/row.index_close-1) if complete else np.nan,
                                        max_close_return=float(future.max()/row.index_close-1) if complete else np.nan,
                                        min_close_return=float(future.min()/row.index_close-1) if complete else np.nan,
                                        end_date=part.date.iloc[i+horizon] if complete else pd.NaT))
    return pd.DataFrame(records, columns=['event_id','sector_code','signal_date','trigger','horizon','complete',
                                         'close_return','max_close_return','min_close_return','end_date'])
