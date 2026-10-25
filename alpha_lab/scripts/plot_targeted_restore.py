"""绘制原五席位故障窗口及完整账户路径，不挑选新的最优窗口。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
    for suffix, label in [('control', 'Without restoration'), ('restore', 'Targeted restoration')]:
        daily = pd.read_parquet(run / f'accounts/cold_01_staged_cost0_monthly_{suffix}/daily.parquet')
        window = daily.loc['2024-10-08':'2024-10-25']
        axes[0].plot(window.index, window.stock_value / window.equity, marker='o', label=label)
        axes[1].plot(daily.index, daily.nav, label=label)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    axes[0].set(title='Original failure window: all five positions remain occupied', ylabel='Stock exposure')
    axes[1].set(title='Same account, full observed history: monthly restoration', ylabel='NAV (initial = 1)')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('CNY 80,000 | Calendar 01 | Base fees included | Historical simulation')
    for suffix in ['png', 'pdf']:
        fig.savefig(run / f'analysis/original_failure_process.{suffix}', dpi=160)


if __name__ == '__main__':
    main()
