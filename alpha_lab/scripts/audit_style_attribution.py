#!/usr/bin/env python3
"""固定匹配对照的纯多头日频账本归因；研究读取硬限定到2023年底。"""
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
from ledger import ExperimentLedger, deflated_sharpe_report
from neutralize import attach_industry_pit, neutralize
from portfolio import board_of
from style_matching import match_controls
from improve_monthly_portfolio import digest
from run_investability_followup import evaluate


def prepare(config):
    experiment = yaml.safe_load(config.read_text())
    parent = ROOT / experiment["parent_prepared"]
    cfg = yaml.safe_load((parent / "config.yaml").read_text())
    end = pd.Timestamp(cfg["valid"][1])
    start = pd.Timestamp(experiment["lookback_start"])
    if end >= pd.Timestamp("2024-01-01") or start >= end:
        raise ValueError("禁止读取TEST或使用无效窗口")
    signals = pd.read_parquet(parent / experiment["signal_file"])
    if (signals.index.max() > end or signals.index.min() < pd.Timestamp(cfg["valid"][0])
            or signals.index.has_duplicates or signals.columns.has_duplicates):
        raise ValueError("冻结信号日期或代码无效")
    output = ROOT / experiment["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"风格归因目录 {output}", flush=True)
    (output / "requested_experiment.yaml").write_text(config.read_text())
    (output / "config.yaml").write_text((parent / "config.yaml").read_text())
    controls = experiment["variants"]
    experiment["matching_variants"] = controls
    experiment["variants"] = [dict(name=name, active_fraction=experiment["active_fraction"])
        for name in ["candidate"] + [v["name"] for v in controls]]
    (output / "experiment.yaml").write_text(yaml.safe_dump(experiment, sort_keys=False))
    sources = [Path(__file__), ROOT / "src/style_matching.py", ROOT / "src/neutralize.py",
        ROOT / "src/ledger.py", ROOT / "src/portfolio.py", ROOT / "scripts/run_investability_followup.py",
        ROOT / "scripts/check_cash_reserve.py", ROOT / "scripts/check_portfolio_improvement.py",
        ROOT / "scripts/audit_monthly_portfolio.py", ROOT / "scripts/prepare_monthly_audit.py",
        ROOT / "configs/corporate_action_components.yaml"]
    manifest = dict(status="preparing", test_market_data_used=False, fresh_holdout=False,
        interpretation="matched descriptive controls; no new alpha selection; unknown industries explicit",
        parent_signal_hash=digest(parent / experiment["signal_file"]),
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ledger = ExperimentLedger()
    for variant in experiment["variants"]:
        ledger.log(variant["name"], "matched_style_control", "2022-2023 reused development",
            np.nan, np.nan, np.nan, np.nan, params=dict(output=str(output), **variant),
            verdict="PREREGISTERED", experiment_id=str(output), parent_experiment=str(parent),
            source_hashes=manifest["source_hashes"], input_hashes={"signal": manifest["parent_signal_hash"]},
            selection_note="行业/板块固定，依次匹配规模与成交额；所有对照均保留，不选冠军")
    panel = ROOT / "data/panel/daily_panel.parquet"
    panel_stat = panel.stat()
    market = pd.read_parquet(panel,
        columns=["date", "ts_code", "circ_mv_yuan", "amount_yuan", "turnover_rate", "pct_chg", "close"],
        filters=[("date", ">=", start), ("date", "<=", end)])
    market = market.sort_values(["ts_code", "date"])
    amount = market.pivot(index="date", columns="ts_code", values="amount_yuan").sort_index()
    returns = market.pivot(index="date", columns="ts_code", values="pct_chg").reindex_like(amount) / 100
    lookback = experiment["lookback_days"]
    amount20 = amount.where(amount > 0).rolling(lookback, min_periods=lookback).mean().loc[signals.index]
    rv20 = returns.rolling(lookback, min_periods=lookback).std().loc[signals.index]
    snapshot = market[market.date.isin(signals.index)].copy()
    snapshot["log_mv"] = np.log(snapshot.circ_mv_yuan.where(snapshot.circ_mv_yuan > 0))
    lookup = snapshot.set_index(["date", "ts_code"]).index
    snapshot["log_amount20"] = np.log(amount20.stack()).reindex(lookup).to_numpy()
    snapshot["rv20"] = rv20.stack().reindex(lookup).to_numpy()
    sw_path = ROOT / "data/raw/sw_l1_member_hist.parquet"
    snapshot = attach_industry_pit(snapshot, pd.read_parquet(sw_path)).rename(columns={"sw_l1": "industry"})
    snapshot["board"] = snapshot.ts_code.map(board_of)
    snapshot = snapshot.set_index(["date", "ts_code"]).sort_index()
    snapshot["score"] = signals.stack().reindex(snapshot.index)
    # 统计和匹配共享已经冻结的信号日可投资池。
    eligible = snapshot[snapshot.score.notna()].copy()
    eligible.to_parquet(output / "signal_day_exposures.parquet")
    _, neutralization = neutralize(eligible.reset_index(), "score", "industry", "log_mv",
        min_samples=experiment["min_neutralization_samples"], return_diagnostics=True)
    neutralization.to_csv(output / "neutralization_status.csv", index=False)
    plans = {v["name"]: pd.DataFrame(np.nan, index=signals.index, columns=signals.columns) for v in controls}
    matches, coverage, exposures = [], [], []
    codes = set()
    for date, row in signals.iterrows():
        selected = row.dropna().sort_values(ascending=False, kind="stable").head(cfg["portfolio"]["top_n"]).index.tolist()
        universe = eligible.loc[date]
        codes.update(selected)
        coverage.append(dict(date=date, n_eligible=len(universe), n_targets=len(selected),
            target_industry_coverage=universe.loc[selected, "industry"].notna().mean(),
            universe_industry_coverage=universe.industry.notna().mean()))
        groups = {"candidate": selected}
        for variant in controls:
            matched = match_controls(universe, selected, variant["features"], experiment["seed"])
            matched.insert(0, "date", date)
            matched.insert(0, "strategy", variant["name"])
            matches.append(matched)
            names = matched.control.dropna().tolist()
            plans[variant["name"]].loc[date, names] = np.arange(len(names), 0, -1, dtype=float)
            groups[variant["name"]] = names
            codes.update(names)
        for name, names in groups.items():
            frame = universe.loc[names]
            for feature in ["log_mv", "log_amount20", "rv20", "close", "score"]:
                exposures.append(dict(date=date, strategy=name, feature=feature,
                    mean=frame[feature].mean(), n_valid=frame[feature].notna().sum(), n_target=len(names)))
    signals.to_parquet(output / "candidate_signals.parquet")
    for name, frame in plans.items():
        frame.to_parquet(output / f"{name}_signals.parquet")
    pd.concat(matches, ignore_index=True).to_csv(output / "matched_pairs.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(exposures).to_csv(output / "target_exposures.csv", index=False)
    pd.DataFrame(dict(ts_code=sorted(codes))).to_csv(output / "execution_codes.csv", index=False)
    manifest.update(status="signals_frozen", n_codes=len(codes),
        input_hashes={p.name: digest(p) for p in output.glob("*_signals.parquet")},
        config_hashes={name: digest(output / name) for name in
            ["config.yaml", "experiment.yaml", "execution_codes.csv", "signal_day_exposures.parquet", "matched_pairs.csv"]},
        static_input_hashes={str(sw_path): digest(sw_path)},
        panel_metadata=dict(path=str(panel), bytes=panel_stat.st_size, mtime_ns=panel_stat.st_mtime_ns,
                            note="原始大面板记录元数据；参与匹配的派生快照另有内容SHA256"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"固定匹配完成：{len(codes)}只执行代码；未按后续收益调整匹配", flush=True)
    return output


def report(output):
    experiment = yaml.safe_load((output / "experiment.yaml").read_text())
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    summary = pd.read_csv(output / "summary.csv")
    baseline = pd.read_csv(output / "candidate_cost1_daily.csv", index_col="date", parse_dates=True)
    original = pd.read_csv(ROOT / experiment["parent_prepared"] / "tail_feasible_half_cost1_daily.csv",
                           index_col="date", parse_dates=True)
    pd.testing.assert_frame_equal(baseline, original, rtol=1e-10, atol=1e-7)
    pairs = pd.read_csv(output / "matched_pairs.csv")
    exposure = pd.read_csv(output / "target_exposures.csv")
    rng = np.random.default_rng(experiment["seed"])
    n = len(baseline)
    block = experiment["bootstrap_block_days"]
    starts = rng.integers(0, n, size=(experiment["bootstrap_samples"], int(np.ceil(n / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(len(starts), -1)[:, :n]
    contrasts, statistics = [], []
    for name in summary.strategy:
        d = pd.read_csv(output / f"{name}_cost1_daily.csv", index_col="date", parse_dates=True)
        if not d.index.equals(baseline.index):
            raise ValueError("对照收益日期不一致")
        delta = baseline.ret.to_numpy() - d.ret.to_numpy()
        lo, hi = np.quantile(delta[indices].mean(axis=1)*cfg["portfolio"]["annual_days"],
                             experiment["bootstrap_quantiles"])
        contrasts.append(dict(control=name, candidate_minus_control_annual_mean=
            delta.mean()*cfg["portfolio"]["annual_days"], descriptive_low=lo, descriptive_high=hi,
            fresh_holdout=False))
        statistics.append(dict(strategy=name, **deflated_sharpe_report(
            d.ret.mean()/d.ret.std(ddof=1), len(d), rets=d.ret.to_numpy())))
    pd.DataFrame(contrasts).to_csv(output / "paired_contrasts.csv", index=False)
    (output / "dsr_status.json").write_text(json.dumps(statistics, indent=2))
    # 利润按每只股票初次实际买入前的PIT行业分组，避免用最后行业覆盖历史。
    snapshot = pd.read_parquet(output / "signal_day_exposures.parquet")
    attribution = []
    for name in summary.strategy:
        trades = pd.read_csv(output / f"{name}_cost1_trades.csv", parse_dates=["signal_date", "date"])
        first = trades[trades.side == "buy"].sort_values("date").drop_duplicates("ts_code")
        pnl = pd.read_csv(output / f"{name}_cost1_asset_pnl.csv").set_index("ts_code")
        key = pd.MultiIndex.from_frame(first[["signal_date", "ts_code"]]).set_names(["date", "ts_code"])
        first["industry"] = snapshot.industry.reindex(key).to_numpy()
        first["industry"] = first.industry.fillna("__unknown__")
        joined = pnl.join(first.set_index("ts_code")[["industry"]])
        joined["industry"] = joined.industry.fillna("__unknown__")
        for industry, frame in joined.groupby("industry"):
            attribution.append(dict(strategy=name, industry_at_first_buy=industry,
                                    pnl_yuan=frame.pnl_yuan.sum(), n_stocks=len(frame)))
    pd.DataFrame(attribution).to_csv(output / "industry_pnl.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name in summary.strategy:
        d = pd.read_csv(output / f"{name}_cost1_daily.csv", index_col="date", parse_dates=True)
        axes[0].plot(d.index, d.nav, label=name)
        axes[1].plot(d.index, d.stock_value/d.equity, label=name)
    axes[0].set(title="Matched controls: reused 2022-2023 development data", ylabel="Daily NAV")
    axes[1].set(ylabel="Stock / total equity")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "matched_comparison.png", dpi=150)
    plt.close(fig)
    print(summary[["strategy", "ann", "mdd", "sharpe", "mean_invested"]].to_string(index=False))
    print("行业覆盖", pd.read_csv(output / "coverage.csv").target_industry_coverage.mean())
    print("匹配率", pairs.groupby("strategy").status.apply(lambda s: (s == "matched").mean()).to_dict())
    print(exposure.pivot_table(index="strategy", columns="feature", values="mean").to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/style_attribution.yaml")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--report-only", type=Path)
    args = ap.parse_args()
    if args.report_only:
        report(args.report_only)
        return
    output = args.resume or prepare(args.config)
    if not args.prepare_only:
        evaluate(output)
        report(output)


if __name__ == "__main__":
    main()
