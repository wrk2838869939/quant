#!/usr/bin/env python3
"""全方案费用压力、配对区块重采样和持仓归因；只读已使用的开发窗口。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from audit_monthly_portfolio import load_inputs, reconcile
from neutralize import attach_industry_pit
from portfolio import run_backtest


def asset_pnl(trades, events, holdings, daily):
    """净利润逐股分解；应收在除权确认，避免把到账重复计入收益。"""
    t = trades.copy()
    t["flow"] = np.where(t.side.eq("sell"), t.gross, -t.gross) - t.fee
    pnl = t.groupby("ts_code").flow.sum()
    for kind, col in [("dividend_receivable", "amount"), ("delist_writeoff", "cash_recovery")]:
        rows = events[events.kind.eq(kind)]
        if len(rows):
            pnl = pnl.add(rows.groupby("ts_code")[col].sum(), fill_value=0)
    last = holdings[holdings.date.eq(daily.index[-1])].set_index("ts_code").value
    return pnl.add(last, fill_value=0).sort_values()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/portfolio_improvement_checks.yaml")
    args = ap.parse_args()
    output = args.prepared
    checks = output / "checks"
    checks.mkdir(exist_ok=False)
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    params = yaml.safe_load(args.config.read_text())
    experiment = yaml.safe_load((output / "experiment.yaml").read_text())
    (checks / "config.yaml").write_text(args.config.read_text())
    market, basics, st, limits, actions = load_inputs(output, cfg)
    corporate = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())
    # 持仓风格只作事后归因，不参与打分或筛选。
    start, end = map(pd.Timestamp, cfg["valid"])
    codes = pd.read_csv(output / "execution_codes.csv").ts_code.tolist()
    styles = pd.read_parquet(ROOT / "data/panel/daily_panel.parquet",
        columns=["date", "ts_code", "circ_mv_yuan", "pb", "turnover_rate"],
        filters=[("date", ">=", start), ("date", "<=", end), ("ts_code", "in", codes)])
    styles = attach_industry_pit(styles, pd.read_parquet(ROOT / "data/raw/sw_l1_member_hist.parquet"))
    baseline = pd.read_csv(output / "baseline_daily.csv", index_col="date", parse_dates=True)
    rng = np.random.default_rng(params["bootstrap_seed"])
    n = len(baseline)
    starts = rng.integers(0, n, size=(params["bootstrap_samples"], int(np.ceil(n / params["bootstrap_block_days"]))))
    indices = (starts[:, :, None] + np.arange(params["bootstrap_block_days"])) % n
    indices = indices.reshape(params["bootstrap_samples"], -1)[:, :n]
    stresses, exposures, intervals = [], [], []
    for variant in experiment["variants"]:
        name = variant["name"]
        p = dict(cfg["portfolio"], band_mult=variant["band_mult"])
        for key in ("buy_cost", "sell_cost", "min_fee"):
            p[key] *= params["cost_multiplier"]
        result = run_backtest(pd.read_parquet(output / f'{variant["blend"]}_signals.parquet'),
                              market, basics, st, limits, actions, *cfg["valid"], p,
                              corporate.get("rights_issues", []))
        stress = dict(strategy=name, **result.summary, **reconcile(result, p["capital"]))
        stresses.append(stress)
        result.daily.to_csv(checks / f"{name}_stress_daily.csv")
        result.trades.to_csv(checks / f"{name}_stress_trades.csv", index=False)
        result.events.to_csv(checks / f"{name}_stress_events.csv", index=False)
        result.holdings.to_csv(checks / f"{name}_stress_holdings.csv", index=False)
        print(json.dumps(stress), flush=True)
        daily = pd.read_csv(output / f"{name}_daily.csv", index_col="date", parse_dates=True)
        h = pd.read_csv(output / f"{name}_holdings.csv", parse_dates=["date"])
        trades = pd.read_csv(output / f"{name}_trades.csv")
        events = pd.read_csv(output / f"{name}_events.csv")
        pnl = asset_pnl(trades, events, h, daily)
        if not np.isclose(pnl.sum(), daily.equity.iloc[-1] - p["capital"], atol=1e-6, rtol=0):
            raise ValueError("逐股利润与总净利润不一致")
        pnl.rename("pnl_yuan").to_csv(checks / f"{name}_asset_pnl.csv")
        delta = daily.ret.to_numpy() - baseline.ret.to_numpy()
        sampled = delta[indices].mean(axis=1) * p["annual_days"]
        low, high = np.quantile(sampled, params["bootstrap_quantiles"])
        intervals.append(dict(strategy=name, annualized_mean_daily_delta=delta.mean() * p["annual_days"],
                              descriptive_low=low, descriptive_high=high, fresh_holdout=False))
        h = h.merge(styles, on=["date", "ts_code"], how="left", validate="one_to_one")
        h["weight"] = h.value / h.groupby("date").value.transform("sum")
        h["equity_weight"] = h.value / h.date.map(daily.equity)
        row = dict(strategy=name, max_single_equity_weight=h.equity_weight.max(),
                   industry_coverage=float(h.loc[h.sw_l1.notna()].groupby("date").weight.sum().mean()))
        for field in ("circ_mv_yuan", "pb", "turnover_rate"):
            valid = h[field].notna() & (h[field] > 0)
            # 显式按字段覆盖度归一化，缺失不能当零；市值用几何均值以限制极大值支配。
            values = np.log(h.loc[valid, field]) if field == "circ_mv_yuan" else h.loc[valid, field]
            numerator = (values * h.loc[valid, "weight"]).groupby(h.loc[valid, "date"]).sum()
            denominator = h.loc[valid].groupby("date").weight.sum()
            mean = (numerator / denominator).mean()
            row[field] = float(np.exp(mean) if field == "circ_mv_yuan" else mean)
            row[field + "_coverage"] = float(denominator.mean())
        sector = h.groupby(["date", "sw_l1"], dropna=False).equity_weight.sum()
        row["mean_largest_sector_weight"] = float(sector.groupby(level=0).max().mean())
        row["worst_asset_pnl"] = float(pnl.min())
        row["best_asset_pnl"] = float(pnl.max())
        exposures.append(row)
    pd.DataFrame(stresses).to_csv(checks / "stress_summary.csv", index=False)
    pd.DataFrame(exposures).to_csv(checks / "exposures.csv", index=False)
    pd.DataFrame(intervals).to_csv(checks / "paired_bootstrap.csv", index=False)
    (checks / "manifest.json").write_text(json.dumps(dict(status="complete", fresh_holdout=False,
        interpretation="固定压力情景和已使用样本的描述性区间，不是新的策略试验或显著性认证",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
