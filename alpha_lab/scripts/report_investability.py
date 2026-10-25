#!/usr/bin/env python3
"""汇总已完成账本，绘制收益/回撤/仓位，并保留板块贡献与描述性不确定性。"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio import board_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, required=True)
    args = ap.parse_args()
    output = args.prepared
    experiment = yaml.safe_load((output / "experiment.yaml").read_text())
    parent = ROOT / experiment["parent_prepared"]
    cfg = yaml.safe_load((output / "config.yaml").read_text())
    params = yaml.safe_load((ROOT / "configs/portfolio_improvement_checks.yaml").read_text())
    inputs = dict(baseline=(parent, "baseline"), tail_unfiltered=(parent, "tail_signal"))
    inputs.update({v["name"]: (output, v["name"] + "_cost1") for v in experiment["variants"]})
    daily = {name: pd.read_csv(path / f"{prefix}_daily.csv", index_col="date", parse_dates=True)
             for name, (path, prefix) in inputs.items()}
    n = len(daily["baseline"])
    rng = np.random.default_rng(params["bootstrap_seed"])
    block = params["bootstrap_block_days"]
    starts = rng.integers(0, n, size=(params["bootstrap_samples"], int(np.ceil(n / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(params["bootstrap_samples"], -1)[:, :n]
    intervals, boards, concentration = [], [], []
    for name, (path, prefix) in inputs.items():
        d = daily[name]
        if not d.index.equals(daily["baseline"].index):
            raise ValueError("净值日期不一致")
        delta = d.ret.to_numpy() - daily["baseline"].ret.to_numpy()
        sample = delta[indices].mean(axis=1) * cfg["portfolio"]["annual_days"]
        low, high = np.quantile(sample, params["bootstrap_quantiles"])
        intervals.append(dict(strategy=name, annualized_mean_daily_delta=delta.mean() * cfg["portfolio"]["annual_days"],
                              descriptive_low=low, descriptive_high=high, fresh_holdout=False))
        h = pd.read_csv(path / f"{prefix}_holdings.csv", parse_dates=["date"])
        h["board"] = h.ts_code.map(board_of)
        h["weight"] = h.value / h.date.map(d.equity)
        exposure = h.groupby(["date", "board"]).weight.sum().unstack().reindex(d.index).fillna(0).mean()
        pnl_path = (path / f"{prefix}_asset_pnl.csv") if path == output else (path / "checks" / f"{prefix}_asset_pnl.csv")
        pnl = pd.read_csv(pnl_path)
        pnl["board"] = pnl.ts_code.map(board_of)
        contribution = pnl.groupby("board").pnl_yuan.sum()
        for board in sorted(set(exposure.index) | set(contribution.index)):
            boards.append(dict(strategy=name, board=board, mean_equity_weight=exposure.get(board, 0.),
                               pnl_yuan=contribution.get(board, 0.)))
        concentration.append(dict(strategy=name, max_single_equity_weight=h.weight.max(),
                                   largest_asset_pnl=pnl.pnl_yuan.max(), worst_asset_pnl=pnl.pnl_yuan.min()))
    pd.DataFrame(intervals).to_csv(output / "paired_bootstrap.csv", index=False)
    pd.DataFrame(boards).to_csv(output / "board_attribution.csv", index=False)
    pd.DataFrame(concentration).to_csv(output / "concentration.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for name, d in daily.items():
        axes[0].plot(d.index, d.nav, label=name)
        axes[1].plot(d.index, d.nav / np.maximum.accumulate(np.r_[1., d.nav])[1:] - 1)
        axes[2].plot(d.index, d.stock_value / d.equity)
    axes[0].set(title="2022-2023 development comparison (not fresh OOS)", ylabel="Daily NAV")
    axes[1].set(ylabel="Drawdown")
    axes[2].set(ylabel="Stock / total equity", ylim=(0, 1.05))
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
