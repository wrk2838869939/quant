#!/usr/bin/env python3
"""输出现金动作的固定对照图与持仓过程表，不选择阈值。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    folder = run / 'cash_execution'
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['completed_cases'] != 16:
        raise ValueError('16个现金案例必须完整')
    summary = pd.read_csv(folder / 'execution_summary.csv')
    baseline = summary[summary.name.str.startswith('base_')]
    baseline.to_csv(run / 'cash_capital_comparison.csv', index=False)
    cases = [(folder, 'base_50000_none', 'No market gate'),
        (folder, 'base_50000_entry_only', 'Block new entries'),
        (folder, 'base_50000_liquidate', 'Liquidate'),
        (folder, 'fixed_half_50000_none', 'Fixed CNY 25,000 reserve')]
    confirm = run / 'cash_confirm'
    if (confirm / 'manifest.json').exists() and json.loads((confirm / 'manifest.json').read_text())['status'] == 'complete':
        cases.append((confirm, 'base_50000_entry_only', 'Adaptive: market + stock failure'))
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    process = []
    for case_root, case, label in cases:
        daily = pd.read_csv(case_root / case / 'daily.csv', index_col=0, parse_dates=True)
        periods = pd.read_csv(case_root / case / 'holding_periods.csv')
        local_summary = pd.read_csv(case_root / 'execution_summary.csv')
        metrics = local_summary.set_index('name').loc[case]
        axes[0].plot(daily.index, daily.equity / 50000, label=label)
        axes[1].plot(daily.index, 100*(daily.equity / daily.equity.cummax().clip(lower=50000)-1))
        axes[2].plot(daily.index, 100*(1-daily.cash / daily.equity))
        process.append(dict(case=f'{case_root.name}/{case}', closed_episodes=int((~periods.open_at_end).sum()),
            censored_episodes=int(periods.open_at_end.sum()),
            closed_median_days=periods.loc[~periods.open_at_end, 'holding_trading_days'].median(),
            cumulative_turnover=metrics.cumulative_two_sided_turnover,
            fees_yuan=metrics.total_fees, mean_exposure=metrics.mean_invested,
            cash_days_including_initial=metrics.n_cash_days))
    episodes = pd.read_csv(folder / 'base_50000_liquidate/alarm_episodes.csv')
    for episode in episodes.itertuples():
        for axis in axes:
            axis.axvspan(pd.Timestamp(episode.signal_date), pd.Timestamp(episode.recovery_execution_date),
                color='grey', alpha=.12)
    axes[0].set_title('Frozen cash actions / CNY 50,000 / reused 2022-2023 development')
    axes[0].set_ylabel('Account NAV'); axes[0].legend(fontsize=9, ncol=2)
    axes[1].set_ylabel('Drawdown (%)'); axes[2].set_ylabel('Stock value / equity (%)')
    for axis in axes:
        axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(run / 'cash_action_paths.png', dpi=150); plt.close(figure)
    pd.DataFrame(process).to_csv(run / 'cash_process_comparison.csv', index=False)


if __name__ == '__main__':
    main()
