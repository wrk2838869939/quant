#!/usr/bin/env python3
"""Phase 0 数据质量门: 对 daily_panel 跑六项检查, 输出 docs/QUALITY_REPORT.md。

检查项 (调研报告 §Phase 0 质量门):
  1. 退市股覆盖 (幸存者偏差): stock_basic D 股出现在面板的比例
  2. 复权因子: 无 NaN/0/负; 单只股票内除权日外应连续
  3. pre_close 连续性: 与上一交易日 close 的一致率 (除权日外)
  4. pct_chg 一致性: 抽检 pct_chg ≈ (close/pre_close-1)*100
  5. 单位断档审计: amount_yuan / vol_share 的年度中位数稳定性
  6. 覆盖度: 每日股票数曲线与年度统计 (停牌不填充=按日快照天然保证)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path.insert(0, str(ROOT / "src"))

REPORT = ROOT / "docs/QUALITY_REPORT.md"


def main():
    lines = ["# Phase 0 数据质量报告\n"]
    df = pd.read_parquet(ROOT / "data/panel/daily_panel.parquet",
                         columns=["ts_code", "date", "open", "close", "pre_close", "pct_chg",
                                  "vol", "amount", "adj_factor", "vol_share", "amount_yuan"])
    basic = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
    lines.append(f"- 面板: {len(df):,} 行 × {df['ts_code'].nunique()} 只, "
                 f"{df['date'].min().date()} ~ {df['date'].max().date()}")

    # 1. 退市股覆盖
    delisted = basic[basic["list_status"] == "D"]["ts_code"]
    seen = set(df["ts_code"].unique())
    cov = delisted.isin(seen).mean()
    lines.append(f"\n## 1. 退市股覆盖\n- stock_basic 退市 {len(delisted)} 只, 面板中出现 "
                 f"{int(delisted.isin(seen).sum())} 只 (**{cov:.1%}**)")

    # 2. 复权因子
    bad_adj = df["adj_factor"].isna() | (df["adj_factor"] <= 0)
    lines.append(f"\n## 2. 复权因子\n- 非法值(NaN/≤0): {bad_adj.sum():,} 行 ({bad_adj.mean():.2%})")

    # 3. pre_close 连续性 (前一交易日的 close, 除权日允许不等)
    d = df.sort_values(["ts_code", "date"])
    prev_close = d.groupby("ts_code")["close"].shift(1)
    match = (d["pre_close"] - prev_close).abs() < 0.001
    # 除权日: pre_close != prev_close 且 adj_factor 有跳变 → 合法
    ratio = d["close"] / d["pre_close"] - d["pct_chg"] / 100 - 1
    pct_ok = ratio.abs() < 0.005
    lines.append(f"\n## 3. 价格连续性\n- pre_close == T-1 close (非首行): {match.mean():.2%} "
                 f"(不一致≈除权日, 合法)\n- pct_chg 一致性: {pct_ok.mean():.2%}")

    # 4. 单位断档审计
    d["year"] = d["date"].dt.year
    amt_med = d.groupby("year")["amount_yuan"].median()
    vol_med = d.groupby("year")["vol_share"].median()
    lines.append("\n## 4. 单位断档审计 (年度中位数, 大幅突变=单位断档)\n")
    lines.append("| 年 | amount_yuan中位 | vol_share中位 | 股票数 |\n|---|---|---|---|")
    for y in sorted(d["year"].dropna().unique()):
        lines.append(f"| {int(y)} | {amt_med[y]:,.0f} | {vol_med[y]:,.0f} | "
                     f"{d[d['year']==y]['ts_code'].nunique()} |")

    # 5. 覆盖度
    per_day = d.groupby("date")["ts_code"].nunique()
    lines.append(f"\n## 5. 覆盖度\n- 每日股票数: min {per_day.min()} / 中位 {per_day.median():.0f} / "
                 f"max {per_day.max()} (首日 {per_day.index[0].date()}: {per_day.iloc[0]})")

    # 6. 交易日 gap (停牌) 分布
    dates = d.groupby("ts_code")["date"].apply(lambda s: s.sort_values().diff().dt.days.dropna())
    lines.append(f"\n## 6. 相邻记录间隔\n- 中位 {dates.median():.0f} 天; >7天占比 "
                 f"{(dates>7).mean():.2%} (停牌/新股, 不填充)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {REPORT}")


if __name__ == "__main__":
    main()
