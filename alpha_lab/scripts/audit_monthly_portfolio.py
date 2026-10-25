#!/usr/bin/env python3
"""读取固定信号及执行数据，运行 VALID 日频记账审计；不扫描参数。"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio import run_backtest, normalize_actions, AccountingError
from ledger import ExperimentLedger


def reconcile(result, capital):
    """仅从成交/事件流水重建现金和股数，与引擎状态独立对账。"""
    cashflow = pd.Series(0., index=result.daily.index)
    shareflow = []
    for row in result.trades.itertuples():
        sign = -1 if row.side == "buy" else 1
        cashflow.loc[row.date] += sign * row.gross - row.fee
        shareflow.append(dict(date=row.date, ts_code=row.ts_code, shares=-sign * row.shares))
    if not result.events.empty:
        for row in result.events.to_dict("records"):
            if row["kind"] == "dividend_paid":
                cashflow.loc[row["date"]] += row["amount"]
            if row["kind"] == "delist_writeoff":
                cashflow.loc[row["date"]] += row["cash_recovery"]
                shareflow.append(dict(date=row["date"], ts_code=row["ts_code"], shares=-row["shares"]))
            if row["kind"] == "bonus_shares":
                shareflow.append(dict(date=row["date"], ts_code=row["ts_code"], shares=row["shares"]))
    cash_error = float((capital + cashflow.cumsum() - result.daily.cash).abs().max())
    share_error = 0.
    if shareflow:
        flow = pd.DataFrame(shareflow).pivot_table(index="date", columns="ts_code", values="shares", aggfunc="sum")
        expected = flow.reindex(result.daily.index).fillna(0).cumsum()
        if result.holdings.empty:
            actual = expected * 0
        else:
            actual = result.holdings.pivot(index="date", columns="ts_code", values="shares").reindex(
                index=expected.index, columns=expected.columns).fillna(0)
        share_error = float((expected - actual).abs().max().max())
    if cash_error > 1e-6 or share_error != 0:
        raise AccountingError(f"独立流水对账失败：cash={cash_error}, shares={share_error}")
    return dict(cash_max_error=cash_error, shares_max_error=share_error)


def legacy_comparison(signals, market, basics, st, cfg):
    """同一冻结信号下复现旧错误口径，仅供归因，不作为策略结果。"""
    spec = importlib.util.spec_from_file_location("legacy_v3", ROOT / "legacy/backtest_weekly_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    panels = {c: market.pivot(index="date", columns="ts_code", values=c).sort_index()
              for c in ("open", "close", "pre_close")}
    idx = {d: i for i, d in enumerate(panels["open"].index)}
    rebal = [d for d in signals.index if d in idx and idx[d] + 1 < len(idx)]
    p = cfg["portfolio"]
    summary, _ = module.run_backtest(signals, panels["open"], panels["close"], panels["pre_close"],
        idx, rebal, p["top_n"], st, basics.set_index("ts_code").list_date.to_dict(),
        p["buy_cost"], p["sell_cost"], p["band_mult"], cfg["horizon"])
    return summary


def load_inputs(prepared, cfg):
    codes = pd.read_csv(prepared / "execution_codes.csv").ts_code.tolist()
    start, end = map(pd.Timestamp, cfg["valid"])
    if end >= pd.Timestamp("2024-01-01"):
        raise ValueError("本审计不允许访问 TEST 行情")
    market = pd.read_parquet(ROOT / "data/panel/daily_panel.parquet",
        columns=["ts_code", "date", "open", "close", "pre_close", "adj_factor", "vol_share"],
        filters=[("date", ">=", start), ("date", "<=", end), ("ts_code", "in", codes)])
    basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
    for col in ("list_date", "delist_date"):
        basics[col] = pd.to_datetime(basics[col], errors="coerce")
    names = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    names["start"] = pd.to_datetime(names.start_date, errors="coerce")
    names["end"] = pd.to_datetime(names.end_date, errors="coerce")
    st = names[names.name.str.contains("ST", case=False, na=False)][["ts_code", "start", "end"]]
    cache = Path((prepared / "execution_cache.txt").read_text().strip())
    actions = pd.concat([pd.read_parquet(cache / "dividend" / f"{code}.parquet") for code in codes], ignore_index=True)
    limits = pd.concat([pd.read_parquet(cache / "stk_limit" / f"{code}.parquet") for code in codes], ignore_index=True)
    limits["date"] = pd.to_datetime(limits.trade_date, errors="raise")
    for col in ("up_limit", "down_limit"):
        limits[col] = pd.to_numeric(limits[col], errors="coerce")
    components = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())
    normalized = normalize_actions(actions, components["components"],
                                   components.get("reference_adjustments", []),
                                   components.get("source_corrections", []))
    # 审计记录写入实验目录；原始接口缓存始终不修改。
    (prepared / "corporate_action_source_corrections.json").write_text(json.dumps(
        {"raw_cache_modified": False, "applied": normalized.attrs.get("source_corrections", [])},
        ensure_ascii=False, indent=2))
    return market, basics, st, limits, normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load((args.prepared / "config.yaml").read_text())
    output = args.prepared / ("ledger_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    sources = [ROOT / "src/portfolio.py", Path(__file__), ROOT / "legacy/backtest_weekly_v3.py",
               ROOT / "configs/corporate_action_components.yaml"]
    manifest = dict(status="running", prepared=str(args.prepared), valid=cfg["valid"],
                    test_market_data_used=False, fresh_holdout=False,
                    source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    manifest["signal_config_hashes"] = {name: hashlib.sha256((args.prepared / name).read_bytes()).hexdigest()
        for name in ("signals.parquet", "benchmark_signals.parquet", "weights.csv", "config.yaml")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"审计目录 {output}", flush=True)
    ledger = ExperimentLedger()
    try:
        market, basics, st, limits, actions = load_inputs(args.prepared, cfg)
        action_cfg = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())
        reports, comparisons = [], []
        for name, file in [("monthly_synth", "signals.parquet"), ("large_cap_50", "benchmark_signals.parquet")]:
            signals = pd.read_parquet(args.prepared / file)
            result = run_backtest(signals, market, basics, st, limits, actions,
                                  *cfg["valid"], cfg["portfolio"], action_cfg.get("rights_issues", []))
            for suffix, frame in [("daily", result.daily), ("trades", result.trades),
                                  ("events", result.events), ("holdings", result.holdings)]:
                frame.to_csv(output / f"{name}_{suffix}.csv", index=suffix == "daily")
            summary = dict(strategy=name, **result.summary)
            summary.update(reconcile(result, cfg["portfolio"]["capital"]))
            reports.append(summary)
            old = legacy_comparison(signals, market, basics, st, cfg)
            comparisons.append(dict(strategy=name, interpretation="旧错误口径，禁止用于收益依据", **old))
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            ledger.log(name, "portfolio_daily_nav", "VALID 2022-2023 accounting audit",
                np.nan, np.nan, np.nan, summary["ann"], params={"output": str(output)},
                verdict="ACCOUNTING_REVIEW", note="固定方案；不是新OOS；仍有成交与税费模型假设")
        pd.DataFrame(reports).to_csv(output / "summary.csv", index=False)
        pd.DataFrame(comparisons).to_csv(output / "legacy_invalid_comparison.csv", index=False)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
