#!/usr/bin/env python3
"""固定诊断图：校准曲线与同资金真实路径。"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);args=p.parse_args();out=args.run
    d=pd.read_csv(out/'rank_calibration_summary.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(d.bin+1,d.prediction*100,'o-',label='Predicted 60-day mean')
    axes[0].plot(d.bin+1,d.realized*100,'o-',label='Realized opportunity mean')
    axes[0].set_ylabel('Return (%)');axes[0].legend(fontsize=8)
    axes[1].plot(d.bin+1,d.mae*100,'o-',color='tab:red',label='Adverse move from entry')
    axes[1].set_ylabel('Mean adverse move (%)');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Predicted-return decile (low to high)');ax.grid(alpha=.25)
    fig.suptitle('2019-2021 reused development: risk ordering is clearer than return ordering')
    fig.tight_layout();fig.savefig(out/'calibration_and_risk.png',dpi=170);plt.close(fig)
    if not (out/'risk_overlay/summary.csv').exists():return
    policies=['low_turnover','full_point','full_confidence']
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True)
    for row,policy in enumerate(policies):
        paths={'patient':out/'execution'/f'base_50000_monthly_{policy}',
               'market gate':out/'risk_overlay'/f'market_gate_{policy}',
               'half cash':out/'risk_overlay'/f'fixed_half_cash_{policy}'}
        for label,folder in paths.items():
            d=pd.read_csv(folder/'daily.csv',index_col=0,parse_dates=True)
            axes[row,0].plot(d.index,d.equity/50000,label=label)
            peak=d.equity.cummax().clip(lower=50000)
            axes[row,1].plot(d.index,(d.equity/peak-1)*100)
            axes[row,2].plot(d.index,d.stock_value/d.equity*100)
        axes[row,0].set_ylabel(policy);axes[row,0].legend(fontsize=8)
        for col in range(3):axes[row,col].grid(alpha=.2)
    for ax,label in zip(axes[0],['NAV','Drawdown (%)','Stock exposure (%)']):ax.set_title(label)
    fig.suptitle('CNY 50,000, <=5 stocks, monthly review: 2022-2023 reused development')
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/'patient_execution_paths.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
