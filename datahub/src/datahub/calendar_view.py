"""交易日历视图 (中/美)。"""
from __future__ import annotations

import pandas as pd

from .config import HubConfig


def _cn_open_days(cfg: HubConfig) -> pd.DatetimeIndex:
    df = pd.read_parquet(cfg.calendar_dir / "cn_calendar.parquet")
    df = df[pd.to_numeric(df["is_open"], errors="coerce") == 1]
    return pd.DatetimeIndex(pd.to_datetime(df["cal_date"]).sort_values().unique())


def cn_shift_sessions(cfg: HubConfig, date, n: int) -> pd.Timestamp:
    """从某日期出发平移 n 个交易日 (n<0 向前), 模块级 (供并行 worker 使用)。"""
    days = _cn_open_days(cfg)
    d = pd.Timestamp(date).normalize()
    if len(days) == 0 or d < days[0] or d > days[-1]:
        raise ValueError(f"日期 {d.date()} 超出可用交易日历范围")
    idx = days.searchsorted(d, side="right") - 1
    target = idx + n
    if target < 0 or target >= len(days):
        raise ValueError(f"平移 {n} 个交易日超出可用日历范围")
    return days[target]


class CalendarView:
    def __init__(self, hub):
        self._hub = hub

    # ---------------- A股 ----------------
    def _cn_cal(self) -> pd.DataFrame:
        cfg = self._hub.cfg
        if not hasattr(self, "_cn_cache"):
            df = pd.read_parquet(cfg.calendar_dir / "cn_calendar.parquet")
            df = df[df["is_open"] == 1]
            self._cn_cache = df.sort_values("cal_date").reset_index(drop=True)
        return self._cn_cache

    def cn_days(self, start=None, end=None) -> pd.DatetimeIndex:
        """A股交易日 (SSE 日历)。"""
        s = self._cn_cal()["cal_date"]
        m = pd.Series(True, index=s.index)
        if start is not None:
            m &= s >= pd.Timestamp(start)
        if end is not None:
            m &= s <= pd.Timestamp(end)
        return pd.DatetimeIndex(s[m].values)

    def cn_sessions(self, n: int, end=None) -> pd.DatetimeIndex:
        """截至 end (含) 的最近 n 个交易日。"""
        s = self._cn_cal()["cal_date"]
        if end is not None:
            s = s[s <= pd.Timestamp(end)]
        return pd.DatetimeIndex(s.tail(n).values)

    def shift_sessions_cn(self, date, n: int) -> pd.Timestamp:
        """从某日期出发平移 n 个交易日 (n<0 向前)。"""
        return cn_shift_sessions(self._hub.cfg, date, n)

    def is_cn_trading_day(self, date) -> bool:
        d = pd.Timestamp(date).normalize()
        days = self.cn_days(d, d)
        return len(days) > 0

    # ---------------- 美股 ----------------
    def _us_cal(self) -> pd.DatetimeIndex:
        if not hasattr(self, "_us_cache"):
            df = pd.read_parquet(self._hub.cfg.calendar_dir / "us_calendar.parquet")
            self._us_cache = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        return self._us_cache

    def us_days(self, start=None, end=None) -> pd.DatetimeIndex:
        """美股交易日 (由美上市资产日期并集推得)。"""
        days = self._us_cal()
        m = pd.Series(True, index=days)
        if start is not None:
            m &= days >= pd.Timestamp(start)
        if end is not None:
            m &= days <= pd.Timestamp(end)
        return pd.DatetimeIndex(days[m.values])
