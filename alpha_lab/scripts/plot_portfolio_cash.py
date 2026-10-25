"""展示风险解除后现金未自动回到旧仓的真实账户路径。"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;out=run/'analysis'
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    for label,title in [('rank','Rank buffer'),('staged','Staged cash'),('fixed_half','Fixed 50% cap')]:
        d=pd.read_parquet(run/'accounts'/f'cold_01_{label}_cost0'/'daily.parquet')
        axes[0].plot(d.index,d.equity/80000,label=title)
        axes[1].plot(d.index,d.stock_value/d.equity,label=title,alpha=.8)
    axes[0].set(ylabel='NAV',title='80,000 CNY, January 2024 start; reused historical window')
    axes[1].set(ylabel='Stock exposure',ylim=(0,1.05))
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'cash_tradeoff.png',dpi=180);fig.savefig(out/'cash_tradeoff.pdf');plt.close(fig)
    folder=run/'accounts/cold_01_staged_cost0'
    d=pd.read_parquet(folder/'daily.parquet').loc['2024-09-01':'2024-10-28']
    e=pd.read_parquet(folder/'events.parquet')
    state=e.loc[e.kind.eq('portfolio_cash_state')].set_index('date').cap.reindex(d.index)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    axes[0].plot(d.index,d.stock_value/d.equity,label='Actual stock exposure')
    axes[0].step(d.index,state,where='post',label='Allowed exposure cap',linestyle='--')
    axes[0].axvline(pd.Timestamp('2024-10-08'),color='gray',linestyle=':')
    axes[0].set(ylabel='Fraction of account equity',title='Risk cleared on October 8; five small holdings still occupy all slots')
    axes[1].step(d.index,d.n_hold,where='post',label='Occupied stock slots',color='tab:green')
    axes[1].set(ylabel='Stock count',ylim=(0,5.5))
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'recovery_capacity.png',dpi=180);fig.savefig(out/'recovery_capacity.pdf');plt.close(fig)


if __name__=='__main__':main()
