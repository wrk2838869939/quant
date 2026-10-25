"""反例驱动：纯下跌、很久以前涨过、缺少一层趋势均不能代替本轮上涨回落。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from left_side_trend import trend_pullback, filter_scores


def config():
    return dict(historical_window=60, minimum_historical_return=.05,
                average_window=30, average_slope_days=10, peak_lookback=40,
                advance_lookback=30, minimum_advance_days=5,
                minimum_pullback_days=2, maximum_pullback_days=15,
                require_current_above_average=False)


def rising_then_falling():
    return np.r_[np.linspace(60, 80, 80), np.linspace(81, 120, 30), [115, 108, 98, 92, 90]]


def test_requires_ordered_rise_and_fall_and_allows_price_below_average():
    c = rising_then_falling()
    frame = pd.DataFrame({'rally': c, 'decline': np.linspace(180, 90, len(c))},
                         index=pd.bdate_range('2020-01-01', periods=len(c)))
    f = trend_pullback(frame, config(), .15, .04)
    assert f['passes'].iloc[-1]['rally'] == 1
    assert frame.rally.iloc[-1] < frame.rally.rolling(30).mean().iloc[-1]
    assert f['passes'].iloc[-1]['decline'] == 0
    assert f['anchor_i'].iloc[-1]['rally'] < f['peak_i'].iloc[-1]['rally'] < len(c) - 1


def test_stale_rally_does_not_qualify_as_recent_pullback():
    c = np.r_[rising_then_falling(), np.linspace(89, 70, 25)]
    frame = pd.DataFrame({'x': c})
    f = trend_pullback(frame, config(), .15, .04)
    assert f['pullback_days'].iloc[-1, 0] > config()['maximum_pullback_days']
    assert f['passes'].iloc[-1, 0] == 0


def test_future_prices_cannot_rewrite_old_peaks_or_trends():
    a = rising_then_falling()
    c = pd.DataFrame({'x': np.r_[a, [200, 10, 300]]})
    full = trend_pullback(c, config(), .15, .04)
    short = trend_pullback(c.iloc[:len(a)], config(), .15, .04)
    for k in short:
        pd.testing.assert_frame_equal(short[k], full[k].iloc[:len(a)], check_exact=True)


def test_both_layers_are_necessary():
    base = pd.DataFrame([[1., 2., 3.]])
    stock = {'passes': pd.DataFrame([[1., 0., 1.]])}
    sector = {'passes': pd.DataFrame([[0., 1., 1.]])}
    f = filter_scores(base, stock, sector)
    assert f.notna().iloc[0].tolist() == [False, False, True]


def test_gap_inside_swing_is_unknown():
    c = pd.DataFrame({'x': rising_then_falling()})
    c.iloc[-10, 0] = np.nan
    f = trend_pullback(c, config(), .15, .04)
    assert np.isnan(f['passes'].iloc[-1, 0])
