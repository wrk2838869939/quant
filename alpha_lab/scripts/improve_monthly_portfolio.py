#!/usr/bin/env python3
"""固定五组机制对照：TRAIN 诊断、冻结信号、VALID 真实账本；禁止读取 TEST。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from factor_blend import blend_factors
from labels import purged_dates
from ledger import ExperimentLedger
from portfolio import run_backtest, daily_metrics
from prepare_monthly_audit import fetch_execution_data
from audit_monthly_portfolio import load_inputs, reconcile
from run_gp_mining import build_wide
from synthesis_and_test import build_all_factors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(factors, blends, labels, weights, eligible, output, quantiles, top_n):
    """次日开盘到第 h+1 日开盘的含复权诊断，绝不当作可执行收益。"""
    scores = {name: f * np.sign(weights[name]) for name, f in factors.items()}
    scores.update(blends)
    rows, correlations = [], []
    for date in labels.index:
        f = pd.DataFrame({name: value.loc[date] for name, value in scores.items()})
        y = labels.loc[date].replace([np.inf, -np.inf], np.nan)
        for name, values in f.items():
            ranked = values[eligible.loc[date]].replace([np.inf, -np.inf], np.nan).dropna()
            # 先定排序与组别，再检查远期标签，避免用未来可见性改选股集合。
            order = ranked.sort_values(ascending=False, kind="stable")
            group = np.ceil(ranked.rank(pct=True, method="average") * quantiles)
            benchmark = y.reindex(ranked.index).mean()
            for q in range(1, quantiles + 1):
                selected = ranked.index[group == q]
                observed = y.reindex(selected)
                rows.append(dict(date=date, factor=name, bucket=f"Q{q:02d}", n_selected=len(selected),
                                 n_labels=observed.notna().sum(), mean_return=observed.mean(),
                                 excess=observed.mean() - benchmark))
            observed = y.reindex(order.head(top_n).index)
            rows.append(dict(date=date, factor=name, bucket=f"top{top_n}", n_selected=len(observed),
                             n_labels=observed.notna().sum(), mean_return=observed.mean(),
                             excess=observed.mean() - benchmark))
        corr = f.loc[eligible.loc[date]].corr(method="spearman")
        correlations.append(corr)
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "train_buckets_by_date.csv", index=False)
    raw.groupby(["factor", "bucket"])[["mean_return", "excess", "n_selected", "n_labels"]].mean().to_csv(
        output / "train_buckets.csv")
    raw["year"] = pd.to_datetime(raw.date).dt.year
    raw.groupby(["year", "factor", "bucket"])[["mean_return", "excess", "n_labels"]].mean().to_csv(
        output / "train_buckets_by_year.csv")
    pd.concat(correlations).groupby(level=0).mean().to_csv(output / "train_rank_correlations.csv")


def prepare(config):
    experiment = yaml.safe_load(config.read_text())
    baseline = ROOT / experiment["baseline_prepared"]
    cfg = yaml.safe_load((baseline / "config.yaml").read_text())
    lo, hi = map(pd.Timestamp, cfg["valid"])
    if (hi >= pd.Timestamp("2024-01-01") or pd.Timestamp(cfg["train"][1]) >= lo
            or pd.Timestamp(cfg["train"][0]) < pd.Timestamp("2014-01-01")
            or pd.Timestamp(cfg["train"][1]) >= pd.Timestamp("2022-01-01")):
        raise ValueError("禁止访问 TEST，TRAIN/VALID 必须隔离")
    output = ROOT / experiment["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.yaml").write_text(config.read_text())
    (output / "config.yaml").write_text((baseline / "config.yaml").read_text())
    (output / "weights.csv").write_text((baseline / "weights.csv").read_text())
    sources = [Path(__file__), ROOT / "src/factor_blend.py", ROOT / "src/portfolio.py",
               ROOT / "scripts/synthesis_and_test.py", ROOT / "scripts/run_gp_mining.py",
               ROOT / "scripts/prepare_monthly_audit.py", ROOT / "scripts/audit_monthly_portfolio.py",
               ROOT / "src/expr_engine.py", ROOT / "src/labels.py", ROOT / "src/rank_ic.py",
               ROOT / "configs/corporate_action_components.yaml"]
    manifest = dict(status="preparing", test_market_data_used=False, fresh_holdout=False,
                    selection_scope="2022-2023 is reused development data; historical pool already selected",
                    source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
                    experiment_hash=digest(config), baseline_signal_hash=digest(baseline / "signals.parquet"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"对照目录 {output}", flush=True)
    ledger = ExperimentLedger()
    for variant in experiment["variants"]:
        ledger.log(variant["name"], "portfolio_improvement", "TRAIN diagnosis; VALID development",
                   np.nan, np.nan, np.nan, np.nan, params=dict(output=str(output), **variant),
                   verdict="PREREGISTERED", note="五组固定机制对照；失败也保留；TEST 冻结")
    cols = dict(ret1="pct_chg", vol="vol_share", amt="amount_yuan", to="turnover_rate",
                high="high", low="low", close="close", hclose="hfq_close", hopen="hfq_open",
                buy_lg="buy_lg_amount_yuan", sell_lg="sell_lg_amount_yuan", pe="pe_ttm", pb="pb")
    wide = build_wide(cols, end_date=hi)
    wide["ret1"] /= 100
    wide["vwap_dev"] = wide["amt"] / wide["vol"] / wide["close"] - 1
    wide["lg_share"] = (wide["buy_lg"] + wide["sell_lg"]) / (wide["amt"] + 1e-6)
    factors = build_all_factors(wide, horizon=cfg["horizon"])
    factors["lit:ep"] = 1 / wide["pe"].where(wide["pe"] > 0)
    factors["lit:bp"] = 1 / wide["pb"].where(wide["pb"] > 0)
    train = purged_dates(wide["close"].loc[slice(*cfg["train"])].index,
                         cfg["horizon"] + 1, cfg["embargo"])[::cfg["horizon"]]
    valid = pd.read_parquet(baseline / "signals.parquet").index
    dates = train.union(valid)
    factors = {name: frame.loc[dates] for name, frame in factors.items()}
    weights = pd.read_csv(output / "weights.csv", index_col=0).weight.to_dict()
    blends = {method: blend_factors(factors, weights, method, experiment["families"])
              for method in dict.fromkeys(v["blend"] for v in experiment["variants"])}
    pd.testing.assert_frame_equal(blends["z_icir"].loc[valid], pd.read_parquet(baseline / "signals.parquet"))
    codes = set()
    for method, frame in blends.items():
        frame.loc[valid].to_parquet(output / f"{method}_signals.parquet")
        for _, row in frame.loc[valid].iterrows():
            codes.update(row.dropna().sort_values(ascending=False, kind="stable").head(cfg["portfolio"]["top_n"]).index)
    pd.DataFrame({"ts_code": sorted(codes)}).to_csv(output / "execution_codes.csv", index=False)
    # 诊断的时点筛选只用信号日的上市天数与名称历史；真实回放仍由账本统一过滤。
    basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet").set_index("ts_code")
    listings = pd.to_datetime(basics.list_date, errors="coerce").reindex(wide["close"].columns)
    names = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    names = names[names.name.str.contains("ST", case=False, na=False)].copy()
    names["start"] = pd.to_datetime(names.start_date, errors="coerce")
    names["end"] = pd.to_datetime(names.end_date, errors="coerce")
    eligible = wide["close"].loc[train].notna()
    for date in train:
        active = set(names.loc[(names.start <= date) & (names.end.isna() | (names.end >= date)), "ts_code"])
        eligible.loc[date] &= ((date - listings).dt.days >= cfg["portfolio"]["min_listing_calendar_days"])
        eligible.loc[date, eligible.columns.isin(active)] = False
    labels = (wide["hopen"].shift(-cfg["horizon"] - 1) / wide["hopen"].shift(-1) - 1).loc[train]
    diagnose({k: v.loc[train] for k, v in factors.items()}, {k: v.loc[train] for k, v in blends.items()},
             labels, weights, eligible, output, experiment["diagnostic_quantiles"], cfg["portfolio"]["top_n"])
    manifest.update(status="signals_frozen", n_codes=len(codes), diagnostic_last_signal=str(train[-1]),
                    signal_hashes={p.name: digest(p) for p in output.glob("*_signals.parquet")},
                    config_hashes={name: digest(output / name) for name in
                                   ("config.yaml", "experiment.yaml", "weights.csv", "execution_codes.csv")})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"TRAIN诊断与信号已固定，候选 {len(codes)} 只", flush=True)
    return output


def evaluate(output):
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    experiment = yaml.safe_load((output / "experiment.yaml").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] == "complete":
        raise ValueError("已完成的对照目录不可重复覆盖")
    for name, expected in manifest["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"冻结后源码发生变化: {name}；需新实验目录")
    for name, expected in manifest["signal_hashes"].items():
        if digest(output / name) != expected:
            raise ValueError(f"冻结信号被更改: {name}")
    for name, expected in manifest["config_hashes"].items():
        if digest(output / name) != expected:
            raise ValueError(f"冻结配置被更改: {name}")
    codes = pd.read_csv(output / "execution_codes.csv").ts_code.tolist()
    cache = fetch_execution_data(codes, tuple(map(pd.Timestamp, cfg["valid"])), cfg)
    (output / "execution_cache.txt").write_text(str(cache))
    print("开始真实现金账本对照", flush=True)
    market, basics, st, limits, actions = load_inputs(output, cfg)
    action_cfg = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())
    results, years, navs = [], [], {}
    try:
        for variant in experiment["variants"]:
            p = dict(cfg["portfolio"], band_mult=variant["band_mult"])
            signals = pd.read_parquet(output / f'{variant["blend"]}_signals.parquet')
            result = run_backtest(signals, market, basics, st, limits, actions, *cfg["valid"], p,
                                  action_cfg.get("rights_issues", []))
            name = variant["name"]
            for suffix, frame in [("daily", result.daily), ("trades", result.trades),
                                  ("events", result.events), ("holdings", result.holdings)]:
                frame.to_csv(output / f"{name}_{suffix}.csv", index=suffix == "daily")
            summary = dict(strategy=name, **result.summary, **reconcile(result, p["capital"]))
            results.append(summary)
            pd.DataFrame(results).to_csv(output / "summary.csv", index=False)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            previous = p["capital"]
            for year, daily in result.daily.groupby(result.daily.index.year):
                metrics, _ = daily_metrics(daily.equity, previous, p["annual_days"])
                years.append(dict(strategy=name, year=year, **metrics))
                previous = daily.equity.iloc[-1]
            navs[name] = result.daily.nav
        pd.DataFrame(years).to_csv(output / "yearly.csv", index=False)
        # 登记结果不会伪装成新样本外，保留全部方案而非仅最优者。
        manifest.update(status="complete", results=results)
        panel_stat = (ROOT / "data/panel/daily_panel.parquet").stat()
        manifest["panel"] = dict(size=panel_stat.st_size, mtime_ns=panel_stat.st_mtime_ns)
        execution_hashes = [dict(path=str(cache / api / f"{code}.parquet"),
                                 sha256=digest(cache / api / f"{code}.parquet"))
                            for code in codes for api in ("dividend", "stk_limit")]
        pd.DataFrame(execution_hashes).to_csv(output / "execution_input_hashes.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for name, nav in navs.items():
            axes[0].plot(nav.index, nav, label=name)
            axes[1].plot(nav.index, nav / np.maximum.accumulate(np.r_[1., nav])[1:] - 1, label=name)
        axes[0].set(title="2022-2023 development comparison (not fresh OOS)", ylabel="Daily NAV")
        axes[1].set(ylabel="Drawdown")
        axes[0].legend()
        for ax in axes:
            ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output / "comparison.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def replay(parent, reason):
    """记账修复后另建回放目录；策略/信号完全冻结，旧失败记录原样保留。"""
    manifest = json.loads((parent / "manifest.json").read_text())
    experiment = yaml.safe_load((parent / "experiment.yaml").read_text())
    output = ROOT / experiment["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for hashes in (manifest["signal_hashes"], manifest["config_hashes"]):
        for name, expected in hashes.items():
            if digest(parent / name) != expected:
                raise ValueError(f"原实验冻结产物已改变: {name}")
            shutil.copy2(parent / name, output / name)
    for p in parent.glob("train_*.csv"):
        shutil.copy2(p, output / p.name)
    manifest.update(status="signals_frozen", replay_parent=str(parent), replay_reason=reason,
                    parent_manifest_sha256=digest(parent / "manifest.json"),
                    source_hashes={name: digest(ROOT / name) for name in manifest["source_hashes"]})
    for key in ("error", "error_type", "results"):
        manifest.pop(key, None)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"冻结信号回放目录 {output}", flush=True)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/portfolio_improvement.yaml")
    ap.add_argument("--resume", type=Path, help="复用冻结信号补下载或回放，不重新生成方案")
    ap.add_argument("--replay", type=Path, help="记账代码修复后另建目录，保留原实验")
    ap.add_argument("--reason", help="记录回放原因，不得修改策略参数或重新选因子")
    args = ap.parse_args()
    if args.replay and (args.resume or not args.reason):
        ap.error("--replay 需同时提供 --reason，且不能与 --resume 同用")
    output = replay(args.replay, args.reason) if args.replay else (args.resume or prepare(args.config))
    evaluate(output)


if __name__ == "__main__":
    main()
