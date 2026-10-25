"""经典策略的日期边界和入选方向，使用手工可核对的输入。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classic_smoke import decision_grid, make_signals


def test_monthly_grid_uses_month_end_and_no_final_day_order():
    days = pd.bdate_range("2023-01-30", "2023-03-02")
    assert list(decision_grid(days, "monthly")) == list(pd.to_datetime(
        ["2023-01-30", "2023-01-31", "2023-02-28"]))


def test_direction_and_eligibility():
    days = pd.bdate_range("2023-01-02", periods=3)
    eligible = pd.DataFrame([[True, True, False]] * 3, index=days, columns=["A", "B", "C"])
    def frame(values):
        return pd.DataFrame([values] * 3, index=days, columns=eligible.columns)
    features = dict(ret20=frame([.1, -.1, .3]), ma20=frame([12., 8., 12.]),
                    ma60=frame([10., 10., 10.]), hfq_close=frame([13., 9., 13.]),
                    macd_dif=frame([2., -1., 2.]), macd_dea=frame([1., -2., 1.]))
    cfg = dict(momentum_window=20, ma_fast=20, ma_slow=60, random_seed=7)
    for algorithm in ["momentum", "macd", "moving_average"]:
        score = make_signals(features, eligible, cfg, algorithm, "daily")
        assert score.A.notna().all() and score[["B", "C"]].isna().all().all()
        assert days[-1] not in score.index


def test_random_prefix_stable_and_unknown_never_selected():
    days = pd.bdate_range("2023-01-02", periods=80)
    eligible = pd.DataFrame(True, index=days, columns=["A", "B"])
    eligible.loc[days[10], "B"] = False
    cfg = dict(random_seed=7)
    full = make_signals({}, eligible, cfg, "random", "daily")
    short = make_signals({}, eligible.iloc[:40], cfg, "random", "daily")
    pd.testing.assert_frame_equal(short, full.loc[short.index])
    assert np.isnan(full.loc[days[10], "B"])
