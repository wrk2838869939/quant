"""导出同一固定执行契约下的净值与回撤，不将账本曲线标作实盘收益。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    workspace = Path(__file__).resolve().parents[2]
    font_manager.fontManager.addfont(str(workspace / 'fonts/SimHei.ttf'))
    plt.rcParams.update({'font.family':'SimHei', 'axes.unicode_minus':False,
                         'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex='col')
    for col, (scenario, title) in enumerate([('continued','2022年起连续账户'),('cold','2024年空仓启动')]):
        for variant, color, label in [('baseline','#2563eb','baseline'),('hybrid','#d97706','Hybrid')]:
            daily = pd.read_csv(run / 'execution' / f'{scenario}_{variant}' / 'daily.csv',
                                parse_dates=['date']).set_index('date')
            values = daily.equity.to_numpy()
            drawdown = values / np.maximum.accumulate(np.r_[50000., values])[1:] - 1
            axes[0,col].plot(daily.index, values/50000., color=color, label=label, linewidth=1.6)
            axes[1,col].plot(daily.index, drawdown, color=color, linewidth=1.2)
        axes[0,col].set_title(title + ' · 初始5万元')
        axes[0,col].set_ylabel('净值（初始资金=1）')
        axes[0,col].legend(loc='upper left', frameon=False)
        axes[1,col].set_ylabel('相对历史峰值回撤')
        axes[1,col].yaxis.set_major_formatter(ticker.PercentFormatter(1))
        # 显式日期与标签配对，避免运行环境的时区转换将年初标签显示为上一年。
        ticks = list(pd.date_range(daily.index[0].normalize().replace(month=1, day=1),
                                   daily.index[-1], freq='YS' if scenario == 'continued' else '6MS'))
        ticks[0] = daily.index[0]
        axes[1,col].set_xticks(ticks, labels=[t.strftime('%Y' if scenario == 'continued' else '%Y-%m') for t in ticks])
        axes[1,col].set_xlim(daily.index[0], daily.index[-1])
        axes[1,col].tick_params(axis='x', labelrotation=25)
        for row in range(2):
            axes[row,col].grid(alpha=.18)
    fig.suptitle('前晚锁定日限价订单：固定2%买价缓冲下的研究回放', fontsize=15)
    fig.text(.5,.015,'截至2026-09-08；已观察历史；日线穿价与足量成交仍是模型假设，未获实盘撮合认证。',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(run / 'causal_equity_drawdown.png', dpi=180)
    fig.savefig(run / 'causal_equity_drawdown.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
