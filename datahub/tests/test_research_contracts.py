"""研究接口边界回归：独立数据根、查询稳定性、空结果和时间边界。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datahub import DataHub
from datahub import indicators, io_utils


@pytest.fixture
def tiny_hub(tmp_path):
    days = pd.bdate_range("2023-01-02", periods=280)
    # 有长暂停区间，用于区分交易所日历与个股观测窗口。
    dates = days.delete(slice(100, 160))
    price = 20 + np.sin(np.arange(len(dates)) / 5) + np.arange(len(dates)) * 0.03
    frame = pd.DataFrame(dict(ts_code="000001.SZ", date=dates, open=price,
                              high=price + 1, low=price - 1, close=price,
                              hfq_open=price * 2, hfq_high=(price + 1) * 2,
                              hfq_low=(price - 1) * 2, hfq_close=price * 2,
                              board="sz_main"))
    for year, part in frame.groupby(frame.date.dt.year):
        p = tmp_path / f"datasets/cn_daily/year={year}"
        p.mkdir(parents=True)
        part.to_parquet(p / "data.parquet", index=False)
    c = tmp_path / "datasets/calendar"
    c.mkdir()
    pd.DataFrame(dict(cal_date=days, is_open=1)).to_parquet(c / "cn_calendar.parquet")
    return DataHub(tmp_path), frame


def test_daily_default_and_unfiltered(tiny_hub):
    hub, frame = tiny_hub
    pd.testing.assert_frame_equal(hub.cn.daily()[frame.columns], frame)


def test_adjust_preserves_explicit_hfq_column(tiny_hub):
    hub, frame = tiny_hub
    got = hub.cn.daily("000001.SZ", fields=["close", "hfq_close"], adjust="hfq")
    assert got.columns.tolist() == ["ts_code", "date", "close", "hfq_close"]
    np.testing.assert_array_equal(got.close, frame.hfq_close)
    np.testing.assert_array_equal(got.hfq_close, frame.hfq_close)


def test_separate_data_roots(tiny_hub, tmp_path):
    hub, frame = tiny_hub
    hub.cn.daily("000001.SZ", fields=["close"])
    other = tmp_path / "other"
    p = other / "datasets/cn_daily/year=2023"
    p.mkdir(parents=True)
    frame.iloc[:1].assign(close=123).to_parquet(p / "data.parquet", index=False)
    assert DataHub(other).cn.daily("000001.SZ", fields=["close"]).close.iloc[0] == 123


@pytest.mark.parametrize("spec", ["ma20", "ema12", "macd", "rsi14", "atr14"])
def test_feature_query_start_invariance(tiny_hub, spec):
    hub, frame = tiny_hub
    start = frame.date.iloc[105]
    end = frame.date.iloc[125]
    full = hub.cn.features("000001.SZ", end=end, specs=[spec])
    short = hub.cn.features("000001.SZ", start=start, end=end, specs=[spec])
    pd.testing.assert_frame_equal(short, full.loc[full.date.ge(start)].reset_index(drop=True),
                                  rtol=1e-12, atol=1e-12)


def test_empty_feature_schema(tiny_hub):
    hub, _ = tiny_hub
    f = hub.cn.features([], specs=["macd", "ma20"])
    assert f.empty
    assert set(f.columns) == {"ts_code", "date", "macd_dif", "macd_dea", "macd_hist", "ma20"}


def test_no_future_dependency(tiny_hub):
    hub, frame = tiny_hub
    specs = ["ma20", "ema12", "macd", "rsi14", "atr14", "ret5", "boll20_2"]
    cut = frame.date.iloc[100]
    a = hub.cn.features("000001.SZ", end=cut, specs=specs)
    b = hub.cn.features("000001.SZ", specs=specs)
    pd.testing.assert_frame_equal(a, b.loc[b.date.le(cut)].reset_index(drop=True))


def test_calendar_overflow_rejected(tiny_hub):
    hub, _ = tiny_hub
    last = hub.calendar.cn_days()[-1]
    with pytest.raises(ValueError):
        hub.calendar.shift_sessions_cn(last, 1)
    with pytest.raises(ValueError):
        hub.calendar.shift_sessions_cn("1990-01-01", 0)


def test_ohlc_feature_rejects_nonclose_price(tiny_hub):
    hub, _ = tiny_hub
    with pytest.raises(ValueError):
        hub.cn.features("000001.SZ", specs=["atr14"], price="open")


def test_returns_do_not_fill_missing_prices():
    df = pd.DataFrame(dict(ts_code="X", date=pd.bdate_range("2024-01-01", periods=3),
                           close=[100., np.nan, 110.]))
    got = indicators.compute_features(df, ["ret1"], price="close")
    assert got.ret1.isna().all()


def test_default_features_include_codes_missing_from_master(tiny_hub):
    hub, frame = tiny_hub
    got = hub.cn.features(start=frame.date.iloc[-5], specs=["ma20"])
    assert len(got) == 5 and got.ma20.notna().all()


def test_cache_preserves_requested_column_order(tiny_hub):
    hub, _ = tiny_hub
    hub.cn.features("000001.SZ", specs=["ma20", "ret5"], cache=True)
    cached = hub.cn.features("000001.SZ", specs=["ret5", "ma20"], cache=True)
    direct = hub.cn.features("000001.SZ", specs=["ret5", "ma20"])
    pd.testing.assert_frame_equal(cached, direct)


def test_pit_conflicts_and_unknown_st(tiny_hub):
    hub, _ = tiny_hub
    hub.meta._cache = {
        "sw_l1_member_hist": pd.DataFrame(dict(
            con_code=["X", "X"], index_code=["A", "B"],
            in_date=["20200101", "20210101"], out_date=["20991231", "20991231"])),
        "sw_l1_classify": pd.DataFrame(dict(index_code=["A", "B"], industry_name=["甲", "乙"])),
        "namechange": pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
    }
    assert hub.meta.sector_of("X", "2020-06-01") == "甲"
    with pytest.raises(ValueError, match="冲突"):
        hub.meta.sector_of("X", "2022-01-01")
    with pytest.raises(ValueError, match="冲突"):
        hub.meta.sector_members(index_code="A", date="2022-01-01")
    assert hub.meta.is_st("X", "2022-01-01") is None


def test_us_empty_selection(tiny_hub):
    hub, _ = tiny_hub
    hub.cfg.us_daily.parent.mkdir()
    pd.DataFrame(dict(ticker=["ABC"], date=[pd.Timestamp("2024-01-02")],
                      asset_type=["stock"], close=[100.])).to_parquet(hub.cfg.us_daily)
    assert hub.us.features([], specs=["ma20"]).empty
    assert hub.us.features("UNKNOWN", specs=["ma20"]).columns.tolist() == ["date", "ticker", "ma20"]
