"""风险识别、现金效用与实际仓位预算结果并列，避免只画最终收益。"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;x=pd.read_csv(run/'metrics.csv')
    names=['price','breadth','volatility','correlation','liquidity','joint','joint_utility']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for split,label,offset in [('validation','2019-2021 selection',-.17),('observed_late','2022-2026 observed',.17)]:
        rows=x.loc[x.split.eq(split)].set_index('model').loc[names]
        positions=[i+offset for i in range(len(names))]
        axes[0].barh(positions,rows.auc,height=.32,label=label)
        axes[1].barh(positions,rows.mean_cash_benefit*100,height=.32,label=label)
    for ax in axes:
        ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].axvline(.5,color='black',lw=1)
    axes[0].set(xlabel='AUC: future 20-session loss risk',title='Risk ranking is not cash profitability')
    axes[1].axvline(0,color='black',lw=1)
    axes[1].set(xlabel='Mean fixed-horizon cash benefit (%)',title='40 bp round-trip cost proxy included')
    axes[1].legend(loc='lower right')
    for suffix in ['png','pdf']:
        fig.savefig(run/f'risk_vs_cash.{suffix}',dpi=160)


if __name__=='__main__':main()
