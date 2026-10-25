#!/usr/bin/env python3
"""汇总预登记对称阶段实验，拒绝不完整矩阵，不选择最优参数。"""
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
    args = parser.parse_args();out = args.run;execution = out/'execution'
    manifest = json.loads((execution/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('真实执行尚未完整完成')
    summary = pd.read_csv(execution/'execution_summary.csv')
    if len(summary) != 21:
        raise ValueError('必须保留全部21个预登记案例')
    base = summary[summary.name.str.startswith('base_')]
    base.to_csv(out/'capital_comparison.csv', index=False)
    processes, phases, months = [], [], []
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for case in base[base.capital.eq(50000)].itertuples():
        folder = execution/case.name
        daily = pd.read_csv(folder/'daily.csv', index_col=0, parse_dates=True)
        periods = pd.read_csv(folder/'holding_periods.csv')
        exits = pd.read_csv(folder/'exit_reasons.csv')
        last_october = daily.loc[:'2023-10-31'].equity.iloc[-1]
        processes.append(dict(policy=case.policy,
            closed_episodes=int((~periods.open_at_end).sum()), censored_episodes=int(periods.open_at_end.sum()),
            closed_median_days=periods.loc[~periods.open_at_end,'holding_trading_days'].median(),
            exit_reasons=json.dumps(exits.reason.value_counts().to_dict(), sort_keys=True),
            october_2023_total=last_october/case.capital-1,
            final_two_months_pnl=daily.equity.iloc[-1]-last_october))
        phase = pd.read_csv(folder/'entry_phase_attribution.csv');phase.insert(0,'policy',case.policy)
        phases.append(phase)
        monthly = daily.equity.resample(pd.offsets.MonthEnd()).last()
        monthly_return = monthly.pct_change();monthly_return.iloc[0] = monthly.iloc[0]/case.capital-1
        months.extend(dict(policy=case.policy,date=date,ret=ret) for date,ret in monthly_return.items())
        axes[0].plot(daily.index,daily.equity/case.capital,label=case.policy)
        peak = daily.equity.cummax().clip(lower=case.capital)
        axes[1].plot(daily.index,100*(daily.equity/peak-1))
        axes[2].plot(daily.index,100*daily.cash/daily.equity)
    axes[0].set_title('Mainboard execution / mainboard training — reused development, CNY 50,000')
    axes[0].set_ylabel('Account NAV');axes[0].legend(fontsize=9,ncol=2)
    axes[1].set_ylabel('Drawdown (%)');axes[2].set_ylabel('Cash / equity (%)')
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'execution_paths.png',dpi=150);plt.close(fig)
    pd.DataFrame(processes).to_csv(out/'process_comparison.csv',index=False)
    pd.concat(phases,ignore_index=True).to_csv(out/'actual_phase_pnl.csv',index=False)
    pd.DataFrame(months).to_csv(out/'monthly_returns.csv',index=False)
    comparisons = []
    for policy in base.policy.unique():
        b = base[base.capital.eq(50000)&base.policy.eq(policy)].iloc[0]
        t = summary[summary.name.str.startswith('transfer_')&summary.policy.eq('transfer_'+policy)].iloc[0]
        comparisons.append(dict(policy=policy, mainboard_training_total=b.total,
            all_a_training_total=t.total, mainboard_training_mdd=b.mdd, all_a_training_mdd=t.mdd))
    pd.DataFrame(comparisons).to_csv(out/'training_transfer_comparison.csv',index=False)


if __name__ == '__main__':main()
