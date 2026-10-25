#!/usr/bin/env python3
"""Phase 0 面板拼接: by_date/{4接口} → 统一 schema 的全A长表 (2014-至今)。

输出:
  data/panel/daily_panel.parquet        全量长表 (~1400万行)
  docs/DATA_SCHEMA.md                   schema 文档 (单位/口径/派生列定义)

统一口径 (避免历史单位断档教训):
  - vol_share:   成交量(股)   = tushare vol(手) × 100
  - amount_yuan: 成交额(元)   = tushare amount(千元) × 1000
  - circ_mv_yuan/total_mv_yuan: (万元)×1e4
  - moneyflow *_amount_yuan / *_vol_share 同理
  - hfq_open/high/low/close: 原始价 × adj_factor (后复权)
  - date: datetime64; code: 6位字符串; ts_code: 带交易所后缀
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
ROOT = Path("/root/autodl-tmp/alpha_lab")
RAW = ROOT / "data/raw"
PANEL = ROOT / "data/panel"

DROP_DB = ["close"]  # daily_basic 的 close 与 daily 重复, 前缀处理


def load_by_date(api: str) -> pd.DataFrame:
    d = RAW / "by_date" / api
    files = sorted(d.glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # 空文件只有表头
    df = df[df["trade_date"].notna()] if "trade_date" in df else df
    return df


def main():
    PANEL.mkdir(parents=True, exist_ok=True)
    print("读 daily...", flush=True)
    daily = load_by_date("daily")
    print(f"  {len(daily):,} 行, {daily['ts_code'].nunique()} 只, "
          f"{daily['trade_date'].min()} ~ {daily['trade_date'].max()}")
    daily = daily.drop_duplicates(["ts_code", "trade_date"])

    print("读 adj_factor...", flush=True)
    adj = load_by_date("adj_factor").drop_duplicates(["ts_code", "trade_date"])
    print(f"  {len(adj):,} 行")

    print("读 daily_basic...", flush=True)
    db = load_by_date("daily_basic")
    db = db.drop(columns=[c for c in DROP_DB if c in db], errors="ignore")
    db = db.drop_duplicates(["ts_code", "trade_date"])
    print(f"  {len(db):,} 行, cols={len(db.columns)}")

    print("读 moneyflow...", flush=True)
    mf = load_by_date("moneyflow").drop_duplicates(["ts_code", "trade_date"])
    print(f"  {len(mf):,} 行")

    print("合并...", flush=True)
    df = daily.merge(adj[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    df = df.merge(db, on=["ts_code", "trade_date"], how="left")
    df = df.merge(mf, on=["ts_code", "trade_date"], how="left")

    # ---- 统一口径与派生列 ----
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["code"] = df["ts_code"].str[:6]

    for c in ["open", "high", "low", "close"]:
        df[f"hfq_{c}"] = df[c] * df["adj_factor"]

    df["vol_share"] = df["vol"] * 100.0          # 手 → 股
    df["amount_yuan"] = df["amount"] * 1000.0    # 千元 → 元
    for c in ["circ_mv", "total_mv"]:
        if c in df:
            df[f"{c}_yuan"] = df[c] * 1e4        # 万元 → 元
    for c in [c for c in mf.columns if c.endswith("_amount") and c != "trade_date"]:
        df[f"{c}_yuan"] = df[c] * 1e4
    for c in [c for c in mf.columns if c.endswith("_vol")]:
        df[f"{c}_share"] = df[c] * 100.0

    df = df.sort_values(["ts_code", "date"]).reset_index(drop=True)

    out = PANEL / "daily_panel.parquet"
    print(f"落盘 {out}...", flush=True)
    df.to_parquet(out, index=False)
    print(f"完成: {len(df):,} 行 × {len(df.columns)} 列, {df['ts_code'].nunique()} 只, "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")

    # 股票数按年统计
    g = df.groupby(df["date"].dt.year)["ts_code"].nunique()
    print("年度股票数:\n", g.to_string())


if __name__ == "__main__":
    main()
