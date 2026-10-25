#!/usr/bin/env python3
"""固定可投资股票池对照：按信号日预算筛选，翌日成交仍严格记账。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from investable import investable_signals
from ledger import ExperimentLedger
from portfolio import run_backtest, daily_metrics
from prepare_monthly_audit import fetch_execution_data
from audit_monthly_portfolio import load_inputs, reconcile
from check_cash_reserve import add_cash_reserve
from check_portfolio_improvement import asset_pnl
from improve_monthly_portfolio import digest


def prepare(config):
    experiment = yaml.safe_load(config.read_text())
    parent = ROOT / experiment["parent_prepared"]
    cfg = yaml.safe_load((parent / "config.yaml").read_text())
    if pd.Timestamp(cfg["valid"][1]) >= pd.Timestamp("2024-01-01"):
        raise ValueError("禁止读取 TEST")
    output = ROOT / experiment["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.yaml").write_text(config.read_text())
    (output / "config.yaml").write_text((parent / "config.yaml").read_text())
    print(f"事前可投资股票池对照 {output}", flush=True)
    sources = [Path(__file__), ROOT / "src/investable.py", ROOT / "src/portfolio.py",
               ROOT / "scripts/check_cash_reserve.py", ROOT / "scripts/check_portfolio_improvement.py",
               ROOT / "scripts/audit_monthly_portfolio.py", ROOT / "scripts/prepare_monthly_audit.py",
               ROOT / "configs/corporate_action_components.yaml"]
    manifest = dict(status="preparing", test_market_data_used=False, fresh_holdout=False,
                    parent=str(parent), parent_signal_hash=digest(parent / experiment["signal_file"]),
                    source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ledger = ExperimentLedger()
    for variant in experiment["variants"]:
        ledger.log(variant["name"], "signal_day_investability", "VALID reused development",
                   np.nan, np.nan, np.nan, np.nan, params=dict(output=str(output), **variant),
                   verdict="PREREGISTERED", note="固定1.05价格预算；全资金/半资金各一组，费用压力不选参数")
    signals = pd.read_parquet(parent / experiment["signal_file"])
    if signals.index.max() >= pd.Timestamp("2024-01-01") or signals.index.min() < pd.Timestamp(cfg["valid"][0]):
        raise ValueError("信号超出开发窗口")
    market = pd.read_parquet(ROOT / "data/panel/daily_panel.parquet",
        columns=["date", "ts_code", "close", "vol_share"], filters=[("date", "in", signals.index.tolist())])
    close = market.pivot(index="date", columns="ts_code", values="close").reindex(index=signals.index, columns=signals.columns)
    volume = market.pivot(index="date", columns="ts_code", values="vol_share").reindex(index=signals.index, columns=signals.columns)
    basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
    for col in ("list_date", "delist_date"):
        basics[col] = pd.to_datetime(basics[col], errors="coerce")
    names = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    st = names[names.name.str.contains("ST", case=False, na=False)].copy()
    st["start"] = pd.to_datetime(st.start_date, errors="coerce")
    st["end"] = pd.to_datetime(st.end_date, errors="coerce")
    codes, coverage = set(), []
    for variant in experiment["variants"]:
        p = dict(cfg["portfolio"], capital=cfg["portfolio"]["capital"] * variant["active_fraction"])
        plan, eligible = investable_signals(signals, close, volume, basics, st, p, experiment["price_multiplier"])
        name = variant["name"]
        plan.to_parquet(output / f"{name}_signals.parquet")
        for date, row in plan.iterrows():
            codes.update(row.dropna().sort_values(ascending=False, kind="stable").head(p["top_n"]).index)
            coverage.append(dict(strategy=name, date=date, original_count=signals.loc[date].notna().sum(),
                                 feasible_count=row.notna().sum(), nominal_slot_budget=p["capital"] / p["top_n"]))
    pd.DataFrame(coverage).to_csv(output / "universe_coverage.csv", index=False)
    pd.DataFrame({"ts_code": sorted(codes)}).to_csv(output / "execution_codes.csv", index=False)
    manifest.update(status="signals_frozen", n_codes=len(codes),
        input_hashes={p.name: digest(p) for p in output.glob("*_signals.parquet")},
        config_hashes={name: digest(output / name) for name in ("config.yaml", "experiment.yaml", "execution_codes.csv")})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"计划冻结，执行代码 {len(codes)} 只", flush=True)
    return output


def evaluate(output):
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    experiment = yaml.safe_load((output / "experiment.yaml").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] == "complete":
        raise ValueError("已完成的结果不能覆盖")
    for name, expected in manifest["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"冻结后源码变化: {name}")
    for group in (manifest["input_hashes"], manifest["config_hashes"]):
        for name, expected in group.items():
            if digest(output / name) != expected:
                raise ValueError(f"冻结产物变化: {name}")
    codes = pd.read_csv(output / "execution_codes.csv").ts_code.tolist()
    cache = fetch_execution_data(codes, tuple(map(pd.Timestamp, cfg["valid"])), cfg)
    (output / "execution_cache.txt").write_text(str(cache))
    results, years = [], []
    try:
        market, basics, st, limits, actions = load_inputs(output, cfg)
        corporate = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())
        for variant in experiment["variants"]:
            for multiplier in experiment["cost_multipliers"]:
                total = cfg["portfolio"]["capital"]
                p = dict(cfg["portfolio"], capital=total * variant["active_fraction"])
                for key in ("buy_cost", "sell_cost", "min_fee"):
                    p[key] *= multiplier
                name = variant["name"]
                result = run_backtest(pd.read_parquet(output / f"{name}_signals.parquet"), market, basics,
                    st, limits, actions, *cfg["valid"], p, corporate.get("rights_issues", []))
                result = add_cash_reserve(result, total - p["capital"], p["annual_days"])
                tag = f"{name}_cost{multiplier:g}"
                for suffix, frame in [("daily", result.daily), ("trades", result.trades),
                                      ("events", result.events), ("holdings", result.holdings)]:
                    frame.to_csv(output / f"{tag}_{suffix}.csv", index=suffix == "daily")
                pnl = asset_pnl(result.trades, result.events, result.holdings, result.daily)
                if not np.isclose(pnl.sum(), result.daily.equity.iloc[-1] - total, atol=1e-6, rtol=0):
                    raise ValueError("逐股利润不平")
                pnl.rename("pnl_yuan").to_csv(output / f"{tag}_asset_pnl.csv")
                summary = dict(strategy=name, cost_multiplier=multiplier, **result.summary, **reconcile(result, total))
                results.append(summary)
                pd.DataFrame(results).to_csv(output / "summary.csv", index=False)
                print(json.dumps(summary), flush=True)
                previous = total
                for year, daily in result.daily.groupby(result.daily.index.year):
                    metrics, _ = daily_metrics(daily.equity, previous, p["annual_days"])
                    years.append(dict(strategy=name, year=year, cost_multiplier=multiplier, **metrics))
                    previous = daily.equity.iloc[-1]
        pd.DataFrame(years).to_csv(output / "yearly.csv", index=False)
        hashes = [dict(path=str(cache / api / f"{code}.parquet"), sha256=digest(cache / api / f"{code}.parquet"))
                  for code in codes for api in ("dividend", "stk_limit")]
        pd.DataFrame(hashes).to_csv(output / "execution_input_hashes.csv", index=False)
        manifest.update(status="complete", results=results)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/investability_followup.yaml")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    evaluate(args.resume or prepare(args.config))


if __name__ == "__main__":
    main()
