"""用同账户路径与全部起点离散度展示稳定性与收益代价。"""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run; out = run / 'analysis'
    policies = ['baseline', 'smooth20', 'rank_buffer', 'combined']
    labels = ['Original', '20-day persistence', 'Rank buffer', 'Both']
    colors = ['#555555', '#2374ab', '#dd8b27', '#248f69']
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)
    for policy, label, color in zip(policies, labels, colors):
        d = pd.read_parquet(run / 'accounts' / f'cold_01_{policy}' / 'daily.parquet')
        axes[0].plot(d.index, (d.equity / 80000 - 1) * 100, label=label, color=color)
    axes[0].set_title('Same CNY 80,000 account: 2024 cash start, original month-start reviews')
    axes[0].set_ylabel('Net return (%)'); axes[0].legend(); axes[0].grid(alpha=.2)
    ranges = pd.read_csv(out / 'startup_ranges.csv')
    for i, (policy, label, color) in enumerate(zip(policies, labels, colors)):
        values = ranges.loc[ranges.policy.eq(policy), 'return_range'] * 100
        axes[1].scatter(np.full(len(values), i) + np.linspace(-.13, .13, len(values)), values, color=color, alpha=.7)
        axes[1].plot([i-.2, i+.2], [values.median()]*2, color='black', linewidth=2)
    axes[1].set_xticks(range(4), labels); axes[1].set_ylabel('Date return range (pp)')
    axes[1].set_title('All 15 startup windows: range across day 8/15/22; black line = median')
    axes[1].grid(axis='y', alpha=.2)
    pairs = pd.read_csv(out / 'paired_results.csv')
    for i, (policy, label, color) in enumerate(zip(policies[1:], labels[1:], colors[1:])):
        values = pairs.loc[pairs.kind.eq('long') & pairs.policy.eq(policy), 'ann_delta'] * 100
        axes[2].scatter(np.full(len(values), i) + np.linspace(-.13, .13, len(values)), values, color=color)
        axes[2].plot([i-.2, i+.2], [values.median()]*2, color='black', linewidth=2)
    axes[2].axhline(0, color='gray', linestyle='--'); axes[2].axhline(-2, color='red', linestyle=':', alpha=.7)
    axes[2].set_xticks(range(3), labels[1:]); axes[2].set_ylabel('Annual return delta (pp)')
    axes[2].set_title('All 8 long-window paired accounts; red line = predeclared -2 pp median tolerance')
    axes[2].grid(axis='y', alpha=.2)
    fig.savefig(out / 'stability_tradeoffs.png', dpi=160)
    fig.savefig(out / 'stability_tradeoffs.pdf')


if __name__ == '__main__': main()
