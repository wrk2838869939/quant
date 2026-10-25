"""固定月初账户过程与全部配对指标；不选最好启动日。"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run
    protocol=json.loads((run/'protocol.json').read_text())
    paths=[('Original rank',Path(protocol['cash_parent'])/'accounts/cold_01_rank_cost0'),
        ('Old cash rule',run/'accounts/cold_01_staged_cost0_fast_restore_control'),
        ('Long-trend confirmation',run/'accounts/cold_01_staged_cost0_fast_restore_trend_break')]
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
    for label,folder in paths:
        daily=pd.read_parquet(folder/'daily.parquet')
        axes[0].plot(daily.index,daily.nav,label=label)
        axes[1].plot(daily.index,daily.stock_value/daily.equity,label=label,alpha=.8)
    axes[0].set(ylabel='NAV (initial = 1)',title='CNY 80,000 | Start 2024 | Calendar 01 | Base fees included')
    axes[1].set(ylabel='Stock exposure')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    for ax in axes:
        ax.legend();ax.grid(alpha=.2)
    for suffix in ['png','pdf']:
        fig.savefig(run/f'analysis/account_process.{suffix}',dpi=160)
    plt.close(fig)
    data=pd.read_csv(run/'analysis/paired.csv');data=data.loc[data.cost_bps.eq(0)&data.label.ne('trend_half')]
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    for label,g in data.groupby('label'):
        ax.scatter(g.ann_delta*100,g.mdd_delta*100,label=label,s=50,alpha=.8)
    ax.axvline(0,color='grey',lw=1);ax.axhline(0,color='grey',lw=1)
    ax.axvline(-2,color='grey',ls='--',lw=1)
    ax.set(xlabel='Annual return difference vs rank (percentage points)',
           ylabel='Maximum drawdown improvement (percentage points)',
           title='All eight matched accounts per rule | Observed historical windows')
    ax.legend();ax.grid(alpha=.2)
    for suffix in ['png','pdf']:
        fig.savefig(run/f'analysis/all_account_tradeoffs.{suffix}',dpi=160)


if __name__=='__main__':main()
