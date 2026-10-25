"""输出首次入场日期的同窗净值与跨窗口敏感性图。"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    out = run / 'analysis'
    capital = json.loads((run / 'manifest.json').read_text())['config']['capital']
    detail = json.loads((out / 'verification.json').read_text())['largest_fresh_spread']
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for offset in [8, 15, 22]:
        name = f"{detail['cohort']}_{offset:02d}_fresh_once"
        daily = pd.read_parquet(run / 'accounts' / name / 'daily.parquet')
        axes[0].plot(daily.index, (daily.equity / capital - 1) * 100, label=f'First review: day {offset}')
    wait = pd.read_parquet(run / 'accounts' / (detail['cohort'] + '_wait') / 'daily.parquet')
    axes[0].plot(wait.index, (wait.equity / capital - 1) * 100, '--', color='gray', label='Wait for next month')
    axes[0].set_title(f"Most sensitive fresh-start window: {detail['cohort']} (diagnostic example)")
    axes[0].set_ylabel('Net return (%)'); axes[0].legend(loc='upper left'); axes[0].grid(alpha=.2)
    spreads = pd.read_csv(out / 'within_cohort_spreads.csv').pivot(index='cohort', columns='policy', values='total_range') * 100
    spreads = spreads.rename(columns={'fresh_once': 'Fresh start, then original calendar',
                                      'stale_once': 'Old list, then original calendar',
                                      'shifted_calendar': 'Shift entire review calendar'})
    spreads.plot.bar(ax=axes[1], width=.8)
    axes[1].set_ylabel('Return range across day 8/15/22 (pp)')
    axes[1].set_title('All 15 windows; each account spans the same 244 sessions and starts with CNY 80,000')
    axes[1].set_xlabel('Start cohort (overlapping windows, not independent samples)')
    axes[1].grid(axis='y', alpha=.2); axes[1].legend(fontsize=8)
    fig.savefig(out / 'timing_paths.png', dpi=160)
    fig.savefig(out / 'timing_paths.pdf')


if __name__ == '__main__':
    main()
