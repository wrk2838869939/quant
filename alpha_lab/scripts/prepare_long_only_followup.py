#!/usr/bin/env python3
"""按 TRAIN 的纯多头前端收益选一个信号，固定200日趋势规则作正交对照。"""
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
from labels import purged_dates
from ledger import ExperimentLedger
from run_gp_mining import build_wide
from synthesis_and_test import build_all_factors
from improve_monthly_portfolio import evaluate, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/long_only_followup.yaml")
    args = ap.parse_args()
    experiment = yaml.safe_load(args.config.read_text())
    baseline = ROOT / experiment["baseline_prepared"]
    cfg = yaml.safe_load((baseline / "config.yaml").read_text())
    if pd.Timestamp(cfg["train"][1]) >= pd.Timestamp("2022-01-01") or pd.Timestamp(cfg["valid"][1]) >= pd.Timestamp("2024-01-01"):
        raise ValueError("禁止跨越 TRAIN 或访问 TEST")
    output = ROOT / experiment["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"纯多头目标对照 {output}", flush=True)
    (output / "experiment.yaml").write_text(args.config.read_text())
    for name in ("config.yaml", "weights.csv"):
        (output / name).write_text((baseline / name).read_text())
    ledger = ExperimentLedger()
    for variant in experiment["variants"]:
        ledger.log(variant["name"], "long_only_followup", "TRAIN selection; reused VALID development",
                   np.nan, np.nan, np.nan, np.nan, params=dict(output=str(output), **variant),
                   verdict="PREREGISTERED", note="固定早晚TRAIN均值为正的前50超额筛选；单一200日趋势，无阈值扫描")
    sources = [Path(__file__), ROOT / "scripts/improve_monthly_portfolio.py", ROOT / "scripts/run_gp_mining.py",
               ROOT / "scripts/synthesis_and_test.py", ROOT / "scripts/prepare_monthly_audit.py",
               ROOT / "scripts/audit_monthly_portfolio.py", ROOT / "src/portfolio.py", ROOT / "src/labels.py",
               ROOT / "src/expr_engine.py", ROOT / "configs/corporate_action_components.yaml"]
    manifest = dict(status="preparing", test_market_data_used=False, fresh_holdout=False,
                    source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources},
                    experiment_hash=digest(args.config), baseline_signal_hash=digest(baseline / "signals.parquet"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    cols = dict(ret1="pct_chg", vol="vol_share", amt="amount_yuan", to="turnover_rate", high="high", low="low",
                close="close", hclose="hfq_close", hopen="hfq_open", buy_lg="buy_lg_amount_yuan",
                sell_lg="sell_lg_amount_yuan", pe="pe_ttm", pb="pb")
    wide = build_wide(cols, end_date=cfg["valid"][1])
    wide["ret1"] /= 100
    wide["vwap_dev"] = wide["amt"] / wide["vol"] / wide["close"] - 1
    wide["lg_share"] = (wide["buy_lg"] + wide["sell_lg"]) / (wide["amt"] + 1e-6)
    factors = build_all_factors(wide, cfg["horizon"])
    factors["lit:ep"] = 1 / wide["pe"].where(wide["pe"] > 0)
    factors["lit:bp"] = 1 / wide["pb"].where(wide["pb"] > 0)
    train = purged_dates(wide["close"].loc[slice(*cfg["train"])].index,
                         cfg["horizon"] + 1, cfg["embargo"])[::cfg["horizon"]]
    original = pd.read_parquet(baseline / "signals.parquet")
    dates = train.union(original.index)
    weights = pd.read_csv(output / "weights.csv", index_col=0).weight.to_dict()
    factors = {name: frame.loc[dates].replace([np.inf, -np.inf], np.nan) * np.sign(weights[name])
               for name, frame in factors.items()}
    common = np.logical_and.reduce([frame.notna().to_numpy() for frame in factors.values()])
    common = pd.DataFrame(common, index=dates, columns=original.columns)
    # 同一完整股票池才能比较信号；未来停牌只影响目标收益，不能提前排除候选。
    basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet").set_index("ts_code")
    listings = pd.to_datetime(basics.list_date, errors="coerce").reindex(original.columns)
    delistings = pd.to_datetime(basics.delist_date, errors="coerce").reindex(original.columns)
    names = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    names = names[names.name.str.contains("ST", case=False, na=False)].copy()
    names["start"] = pd.to_datetime(names.start_date, errors="coerce")
    names["end"] = pd.to_datetime(names.end_date, errors="coerce")
    entry = wide["hopen"].shift(-1).loc[train]
    exit_open = wide["hopen"].shift(-cfg["horizon"] - 1).loc[train]
    last_close = wide["hclose"].ffill().shift(-cfg["horizon"]).loc[train]
    labels = exit_open.fillna(last_close).div(entry) - 1
    # T+1 没报价不买，以现金收益0计；到期停牌先按最后收盘估值，摘牌按0回收。
    labels = labels.where(entry.notna(), 0.)
    observations = []
    for date in train:
        end_date = wide["hopen"].index[wide["hopen"].index.get_loc(date) + cfg["horizon"] + 1]
        labels.loc[date, delistings.notna() & (delistings <= end_date) & entry.loc[date].notna()] = -1.
        eligible = common.loc[date] & ((date - listings).dt.days >= cfg["portfolio"]["min_listing_calendar_days"])
        st = set(names.loc[(names.start <= date) & (names.end.isna() | (names.end >= date)), "ts_code"])
        eligible.loc[eligible.index.isin(st)] = False
        if eligible.sum() < experiment["selection"]["min_cross_section"]:
            continue
        y = labels.loc[date]
        benchmark = y[eligible].mean()
        for name, frame in factors.items():
            selected = frame.loc[date, eligible].sort_values(ascending=False, kind="stable").head(cfg["portfolio"]["top_n"]).index
            observations.append(dict(date=date, year=date.year, factor=name,
                mean_return=y[selected].mean(), excess=y[selected].mean() - benchmark,
                n_entry_missing=entry.loc[date, selected].isna().sum(),
                n_exit_marked=exit_open.loc[date, selected].isna().sum(), n_labels=y[selected].notna().sum()))
    obs = pd.DataFrame(observations)
    obs.to_csv(output / "train_tail_observations.csv", index=False)
    selection = obs.groupby("factor").excess.mean().to_frame("mean_excess")
    for period in ("early", "late"):
        start, end = experiment["selection"][period]
        selection[period] = obs[obs.year.between(start, end)].groupby("factor").excess.mean()
    selection["eligible"] = (selection[["early", "late"]] > experiment["selection"]["min_mean_excess"]).all(axis=1)
    selection.sort_values("mean_excess", ascending=False).to_csv(output / "train_selection.csv")
    candidates = selection[selection.eligible].sort_values("mean_excess", ascending=False)
    if candidates.empty:
        manifest.update(status="no_train_candidate", reason="早晚TRAIN均为正的固定门槛无候选通过")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print("无TRAIN候选通过，不进行VALID选择", flush=True)
        return
    chosen = candidates.index[0]
    print(f"TRAIN 选定 {chosen}: {candidates.iloc[0].to_dict()}", flush=True)
    index = pd.read_parquet(ROOT / "data/raw/index_daily.parquet",
        columns=["ts_code", "trade_date", "close"],
        filters=[("ts_code", "==", experiment["trend"]["index"]),
                 ("trade_date", "<=", pd.Timestamp(cfg["valid"][1]).strftime("%Y%m%d"))])
    index["date"] = pd.to_datetime(index.trade_date)
    index = index.set_index("date").sort_index()
    if index.index.has_duplicates:
        raise ValueError("趋势指数日期重复")
    lookback = experiment["trend"]["lookback"]
    index["ma"] = index.close.rolling(lookback, min_periods=lookback).mean()
    gate = (index.close > index.ma).reindex(original.index, fill_value=False)
    index.loc[original.index, ["close", "ma"]].assign(risk_on=gate).to_csv(output / "trend_decisions.csv")
    tail = factors[chosen].loc[original.index].where(original.notna())
    blends = dict(baseline=original, baseline_trend=original.where(gate, axis=0),
                  tail_signal=tail, tail_trend=tail.where(gate, axis=0))
    codes = set()
    for name, frame in blends.items():
        frame.to_parquet(output / f"{name}_signals.parquet")
        for _, row in frame.iterrows():
            codes.update(row.dropna().sort_values(ascending=False, kind="stable").head(cfg["portfolio"]["top_n"]).index)
    pd.DataFrame({"ts_code": sorted(codes)}).to_csv(output / "execution_codes.csv", index=False)
    manifest.update(status="signals_frozen", chosen_factor=chosen, n_codes=len(codes),
        trend_signal_on=int(gate.sum()), trend_signal_total=len(gate),
        signal_hashes={p.name: digest(p) for p in output.glob("*_signals.parquet")},
        config_hashes={name: digest(output / name) for name in ("config.yaml", "experiment.yaml", "weights.csv", "execution_codes.csv")})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    del wide, factors, labels, entry, exit_open, last_close
    evaluate(output)


if __name__ == "__main__":
    main()
