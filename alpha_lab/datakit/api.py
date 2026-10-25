# -*- coding: utf-8 -*-
"""datakit 运行时 API — 策略回测的唯一数据入口。

设计:
  - load(y0, y1, codes=None) → (market, basics, st, limits, actions): 与
    audit_monthly_portfolio.load_inputs 同签名, 但零网络、来自构建期验证过的产物,
    隔离注册表代码默认不可见。
  - attrs(dates) → 匹配/归因用截面快照 (log_mv/log_amount20/industry/board)
  - investable(dates, capital, top_n) → 可投资掩码 (复用 investable_signals)
  - month_ends / index_close 常用件
纪律: 窗口 ≤ meta.window[1] (TEST 硬闸)。
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
DK = ROOT / "data/datakit"

_cache: dict = {}


def meta() -> dict:
    if "meta" not in _cache:
        _cache["meta"] = json.loads((DK / "meta.json").read_text())
    return _cache["meta"]


def _check_window(y0, y1):
    end_cap = pd.Timestamp(meta()["window"][1])
    y0, y1 = pd.Timestamp(y0), pd.Timestamp(y1)
    if y1 > end_cap or y0 < pd.Timestamp(meta()["window"][0]):
        raise ValueError(f"窗口 [{y0.date()}, {y1.date()}] 超出数据集覆盖 "
                         f"{meta()['window']} (TEST 纪律)")
    return y0, y1


def exclusions() -> set:
    if "exc" not in _cache:
        _cache["exc"] = set(pd.read_parquet(DK / "excluded.parquet").ts_code)
    return _cache["exc"]


def _core() -> pd.DataFrame:
    if "core" not in _cache:
        _cache["core"] = pd.read_parquet(DK / "market_core.parquet")
    return _cache["core"]


def load(y0, y1, codes=None, include_excluded=False):
    """引擎五件套 (market, basics, st, limits, actions), 零网络。

    market 列与 run_backtest 期望一致 (含 industry/limit_pct 等附加列无害)。
    """
    y0, y1 = _check_window(y0, y1)
    df = _core()
    m = (df.date >= y0) & (df.date <= y1)
    if codes is not None:
        m &= df.ts_code.isin(set(codes))
    market = df[m].copy()
    if not include_excluded:
        exc = exclusions()
        market = market[~market.ts_code.isin(exc)]
    basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
    for c in ("list_date", "delist_date"):
        basics[c] = pd.to_datetime(basics[c], errors="coerce")
    names = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    names["start"] = pd.to_datetime(names.start_date, errors="coerce")
    names["end"] = pd.to_datetime(names.end_date, errors="coerce")
    st = names[names.name.str.contains("ST", case=False, na=False)][
        ["ts_code", "start", "end"]]
    limits = pd.read_parquet(DK / "limits.parquet")
    limits = limits[(limits.date >= y0) & (limits.date <= y1)]
    actions = pd.read_parquet(DK / "actions.parquet")
    actions = actions[(actions.ex_date >= y0) & (actions.ex_date <= y1)]
    if not include_excluded:
        keepc = set(market.ts_code)
        limits = limits[limits.ts_code.isin(keepc)]
        actions = actions[actions.ts_code.isin(keepc)]
    return market, basics, st, limits, actions


def investable(dates, capital, top_n, price_multiplier=1.05) -> pd.DataFrame:
    """可投资掩码 (bool 宽表, index=dates)。复用 investable_signals, 排除隔离代码。"""
    from investable import investable_signals
    dates = [pd.Timestamp(d) for d in dates]
    y0, y1 = _check_window(min(dates), max(dates))
    df = _core()
    sub = df[df.date.isin(set(dates))]
    close = sub.pivot(index="date", columns="ts_code", values="close").reindex(index=dates)
    volume = sub.pivot(index="date", columns="ts_code", values="vol_share").reindex(index=dates)
    sig = pd.DataFrame(1.0, index=dates, columns=close.columns)
    basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
    for c in ("list_date", "delist_date"):
        basics[c] = pd.to_datetime(basics[c], errors="coerce")
    names = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    names["start"] = pd.to_datetime(names.start_date, errors="coerce")
    names["end"] = pd.to_datetime(names.end_date, errors="coerce")
    st = names[names.name.str.contains("ST", case=False, na=False)][
        ["ts_code", "start", "end"]]
    cfg = dict(yaml.safe_load((ROOT / "configs/monthly_portfolio_audit.yaml"))["portfolio"],
               capital=capital, top_n=top_n)
    _, allowed = investable_signals(sig, close, volume, basics, st, cfg, price_multiplier)
    return allowed & pd.DataFrame(True, index=dates, columns=close.columns)


def attrs(dates) -> pd.DataFrame:
    """截面快照: date, ts_code, log_mv, log_amount20, industry, board, close。"""
    dates = sorted(pd.Timestamp(d) for d in dates)
    y0, y1 = _check_window(dates[0], dates[-1])
    df = _core()
    sub = df[df.date.isin(set(dates))][
        ["date", "ts_code", "circ_mv_yuan", "amount_yuan", "industry", "board", "close"]]
    full = df[(df.date >= y0 - pd.Timedelta(days=40)) & (df.date <= dates[-1])]
    amt = full.pivot_table(index="date", columns="ts_code", values="amount_yuan",
                           aggfunc="last").sort_index()
    amt20 = amt.where(amt > 0).rolling(20, min_periods=20).mean()
    sub = sub.sort_values(["date", "ts_code"])
    lookup = pd.MultiIndex.from_frame(sub[["date", "ts_code"]])
    sub["log_mv"] = np.log(sub.circ_mv_yuan.where(sub.circ_mv_yuan > 0))
    sub["log_amount20"] = np.log(amt20.stack()).reindex(lookup).to_numpy()
    return sub


def month_ends(y0, y1) -> list:
    y0, y1 = _check_window(y0, y1)
    df = _core()
    d = pd.DatetimeIndex(np.sort(df.date.unique()))
    d = d[(d >= y0) & (d <= y1)]
    return list(pd.Series(d, index=d).groupby([d.year, d.month]).last())


def index_close(code: str = "000300.SH") -> pd.Series:
    idx = pd.read_parquet(DK / "index_daily.parquet")
    s = idx[idx.ts_code == code].sort_values("date").set_index("date")["close"]
    return s.loc[:pd.Timestamp(meta()["window"][1])]
