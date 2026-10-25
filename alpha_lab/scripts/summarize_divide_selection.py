#!/usr/bin/env python3
"""汇总固定分治实验；只读取本轮产物，不访问行情或挑选新参数。"""
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
    execution = args.run / 'execution'
    manifest = json.loads((execution / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['completed_cases'] != 20:
        raise ValueError('必须先完成全部20个固定案例')
    summary = pd.read_csv(execution / 'execution_summary.csv')
    base = summary[~summary.reverse_input].copy()
    base.to_csv(args.run / 'capital_comparison.csv', index=False)
    rows = []
    phases = []
    board_rows = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for case in base[base.capital.eq(50000)].itertuples():
        folder = execution / case.name
        daily = pd.read_csv(folder / 'daily.csv', index_col=0, parse_dates=True)
        periods = pd.read_csv(folder / 'holding_periods.csv')
        episodes = pd.read_csv(folder / 'entry_phase_episodes.csv')
        selections = pd.read_csv(folder / 'selection_trace.csv')
        coverage = pd.read_csv(execution / 'entry_coverage.csv')
        coverage = coverage[coverage.policy.eq(case.policy)]
        rows.append(dict(policy=case.policy, actual_entries=len(episodes),
            median_closed_holding_days=periods.loc[~periods.open_at_end, 'holding_trading_days'].median(),
            unknown_episodes=int(episodes.unknown_hold_reviews.gt(0).sum()),
            unknown_hold_reviews=int(episodes.unknown_hold_reviews.sum()),
            min_admitted=int(coverage.admitted.min()),
            selection_reasons=json.dumps(selections.reason.value_counts().to_dict(), sort_keys=True)))
        phase = pd.read_csv(folder / 'entry_phase_attribution.csv')
        phase.insert(0, 'policy', case.policy)
        phases.append(phase)
        # 仅做实际路径贡献分解；不能当作限制账户权限后的反事实回测。
        def board(code):
            if code.endswith('.BJ'):
                return 'Beijing'
            if code.startswith('688'):
                return 'STAR'
            if code.startswith(('300', '301')):
                return 'ChiNext'
            return 'Main'
        episodes['board'] = episodes.ts_code.map(board)
        for name, group in episodes.groupby('board'):
            board_rows.append(dict(policy=case.policy, board=name, entries=len(group),
                stocks=group.ts_code.nunique(), pnl_yuan=group.pnl_yuan.sum(),
                entry_gross=group.entry_gross.sum()))
        nav = daily.equity / case.capital
        peak = daily.equity.cummax().clip(lower=case.capital)
        axes[0].plot(daily.index, nav, label=case.policy)
        axes[1].plot(daily.index, 100 * (daily.equity / peak - 1))
        axes[2].plot(daily.index, 100 * daily.cash / daily.equity)
    axes[0].set_ylabel('Account NAV')
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].set_title('Monthly, at most 5 stocks, CNY 50,000 — reused development data')
    axes[1].set_ylabel('Drawdown (%)')
    axes[2].set_ylabel('Cash / equity (%)')
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(args.run / 'execution_paths.png', dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.run / 'process_comparison.csv', index=False)
    pd.concat(phases, ignore_index=True).to_csv(args.run / 'actual_phase_pnl.csv', index=False)
    pd.DataFrame(board_rows).to_csv(args.run / 'board_pnl_attribution.csv', index=False)


if __name__ == '__main__':
    main()
