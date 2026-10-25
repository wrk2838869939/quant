#!/usr/bin/env python3
"""实际持仓的事后风格暴露、行业覆盖、换手和执行缺口；不用于选股。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neutralize import attach_industry_pit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    args = ap.parse_args()
    output = args.prepared
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    lo, hi = map(pd.Timestamp, cfg["valid"])
    if hi >= pd.Timestamp("2024-01-01") or lo > hi:
        raise ValueError("归因日期不得跨入TEST")
    codes = pd.read_csv(output / "execution_codes.csv").ts_code.tolist()
    market = pd.read_parquet(ROOT / "data/panel/daily_panel.parquet",
        columns=["date", "ts_code", "circ_mv_yuan", "amount_yuan", "turnover_rate"],
        filters=[("date", ">=", lo), ("date", "<=", hi), ("ts_code", "in", codes)])
    sw = ROOT / "data/raw/sw_l1_member_hist.parquet"
    styles = attach_industry_pit(market, pd.read_parquet(sw))
    rows, industry_rows, skipped = [], [], []
    for name in pd.read_csv(output / "summary.csv").strategy:
        daily = pd.read_csv(output / f"{name}_cost1_daily.csv", index_col="date", parse_dates=True)
        h = pd.read_csv(output / f"{name}_cost1_holdings.csv", parse_dates=["date"])
        trades = pd.read_csv(output / f"{name}_cost1_trades.csv", parse_dates=["date"])
        events = pd.read_csv(output / f"{name}_cost1_events.csv")
        h = h.merge(styles, on=["date", "ts_code"], how="left", validate="one_to_one")
        h["equity_weight"] = h.value / h.date.map(daily.equity)
        h["stock_weight"] = h.value / h.groupby("date").value.transform("sum")
        known = h.sw_l1.notna()
        # 按完整日期重新索引，不能丢掉零覆盖日来抬高覆盖率。
        coverage = h.loc[known].groupby("date").equity_weight.sum().reindex(daily.index, fill_value=0.)
        stock_weight = daily.stock_value / daily.equity
        item = dict(strategy=name, mean_stock_weight=stock_weight.mean(),
                    industry_known_equity_weight=coverage.mean(),
                    industry_known_stock_fraction=(coverage / stock_weight.replace(0, np.nan)).mean())
        for field in ["circ_mv_yuan", "amount_yuan", "turnover_rate"]:
            valid = np.isfinite(h[field]) & (h[field] > 0)
            denom = h.loc[valid].groupby("date").stock_weight.sum()
            weighted = (np.log(h.loc[valid, field]) * h.loc[valid, "stock_weight"]).groupby(h.loc[valid, "date"]).sum()
            item[f"stock_weighted_geomean_{field}"] = float(np.exp((weighted / denom).mean()))
            item[f"coverage_{field}"] = float(denom.reindex(daily.index, fill_value=0).mean())
        h["industry"] = h.sw_l1.fillna("__unknown__")
        group = h.groupby(["date", "industry"]).equity_weight.sum().unstack(fill_value=0).reindex(daily.index, fill_value=0)
        for industry, values in group.items():
            industry_rows.append(dict(strategy=name, industry=industry,
                                       mean_equity_weight=values.mean(), max_equity_weight=values.max()))
        # 买卖成交额之和/2/前一日净资产，累计后按观察年数年化。
        previous = daily.equity.shift(1).fillna(cfg["portfolio"]["capital"])
        turnover = trades.groupby("date").gross.sum().reindex(daily.index, fill_value=0) / (2 * previous)
        item["annualized_half_gross_turnover"] = turnover.sum() * cfg["portfolio"]["annual_days"] / len(daily)
        item["fees_yuan"] = trades.fee.sum()
        item["max_single_equity_weight"] = h.equity_weight.max()
        rows.append(item)
        failed = events[events.kind.isin(["buy_skipped", "sell_deferred"])]
        for (kind, reason), group in failed.groupby(["kind", "reason"]):
            skipped.append(dict(strategy=name, kind=kind, reason=reason, event_count=len(group)))
    pd.DataFrame(rows).to_csv(output / "actual_style_exposures.csv", index=False)
    pd.DataFrame(industry_rows).to_csv(output / "actual_industry_weights.csv", index=False)
    pd.DataFrame(skipped).to_csv(output / "execution_shortfalls.csv", index=False)
    manifest = dict(status="complete", purpose="ex_post_descriptive_holdings_not_risk_model",
        test_market_data_used=False, fresh_holdout=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        industry_input_sha256=hashlib.sha256(sw.read_bytes()).hexdigest())
    (output / "holdings_report_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
