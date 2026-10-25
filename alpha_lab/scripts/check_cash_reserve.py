#!/usr/bin/env python3
"""真实缩小交易账户，再合并固定现金储备；重新计算手数，不按净值曲线事后缩放。"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from portfolio import run_backtest, daily_metrics
from audit_monthly_portfolio import load_inputs, reconcile
from improve_monthly_portfolio import digest


def add_cash_reserve(result, reserve, annual_days):
    """合并不计息的独立现金，初始金额固定，后续不再平衡为恒定比例。"""
    active = result.summary["capital"]
    total = active + reserve
    if reserve < 0 or not pd.notna(reserve) or not 0 < total < float("inf"):
        raise ValueError("储备现金及总资产无效")
    reconcile(result, active)
    result.daily["cash"] += reserve
    result.daily["equity"] += reserve
    metrics, result.daily["ret"] = daily_metrics(result.daily.equity, total, annual_days)
    result.daily["nav"] = result.daily.equity / total
    result.summary.update(metrics, capital=total, active_capital=active, reserve_cash=reserve,
        mean_invested=float((result.daily.stock_value / result.daily.equity).mean()))
    reconcile(result, total)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/cash_reserve_check.yaml")
    args = ap.parse_args()
    params = yaml.safe_load(args.config.read_text())
    declaration = json.loads((args.prepared / "cash_reserve_preregistered.json").read_text())
    if digest(args.config) != declaration["config_sha256"]:
        raise ValueError("现金比例配置与事前登记不符")
    output = args.prepared / "cash_reserve"
    output.mkdir(exist_ok=False)
    (output / "config.yaml").write_text(args.config.read_text())
    cfg = yaml.safe_load((args.prepared / "config.yaml").read_text())
    market, basics, st, limits, actions = load_inputs(args.prepared, cfg)
    corporate = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())
    total_capital = cfg["portfolio"]["capital"]
    if not 0 <= params["cash_fraction"] < 1:
        raise ValueError("现金比例必须位于[0,1)")
    reserve = total_capital * params["cash_fraction"]
    summaries, years = [], []
    for multiplier in params["cost_multipliers"]:
        portfolio = dict(cfg["portfolio"], capital=total_capital - reserve)
        for key in ("buy_cost", "sell_cost", "min_fee"):
            portfolio[key] *= multiplier
        result = run_backtest(pd.read_parquet(args.prepared / params["signal_file"]), market, basics,
                              st, limits, actions, *cfg["valid"], portfolio, corporate.get("rights_issues", []))
        result = add_cash_reserve(result, reserve, portfolio["annual_days"])
        summary = dict(strategy="tail_half_cash", cost_multiplier=multiplier, **result.summary,
                       **reconcile(result, total_capital))
        summaries.append(summary)
        previous = total_capital
        for year, daily in result.daily.groupby(result.daily.index.year):
            metrics, _ = daily_metrics(daily.equity, previous, portfolio["annual_days"])
            years.append(dict(year=year, cost_multiplier=multiplier, **metrics))
            previous = daily.equity.iloc[-1]
        for name, frame in [("daily", result.daily), ("trades", result.trades),
                            ("events", result.events), ("holdings", result.holdings)]:
            frame.to_csv(output / f"cost{multiplier:g}_{name}.csv", index=name == "daily")
        print(json.dumps(summary), flush=True)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(years).to_csv(output / "yearly.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(dict(status="complete", fresh_holdout=False,
        script_sha256=digest(Path(__file__)), config_sha256=digest(args.config),
        signal_sha256=digest(args.prepared / params["signal_file"])), indent=2))


if __name__ == "__main__":
    main()
