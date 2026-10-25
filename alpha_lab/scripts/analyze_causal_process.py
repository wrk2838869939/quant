"""按年度与逐股现金流定位新执行下Hybrid增益，不把集中利润当成独立alpha。"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_causal_execution import save
from portfolio import daily_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = {}
    for scenario in ['cold', 'continued']:
        base = pd.read_csv(run/'execution'/f'{scenario}_baseline'/'daily.csv',
                           parse_dates=['date']).set_index('date')
        hybrid = pd.read_csv(run/'execution'/f'{scenario}_hybrid'/'daily.csv',
                             parse_dates=['date']).set_index('date')
        delta = hybrid.equity-base.equity
        previous, annual = 0., []
        for year, values in delta.groupby(delta.index.year):
            end = float(values.iloc[-1])
            annual.append(dict(year=int(year), incremental_pnl=end-previous,
                               cumulative_equity_difference=end))
            previous = end
        details = dict(yearly_incremental_pnl=annual, final_difference=float(delta.iloc[-1]),
                       fraction_from_2026=float(annual[-1]['incremental_pnl']/delta.iloc[-1]))
        pnl = pd.read_csv(run/'execution'/f'{scenario}_hybrid'/'asset_pnl.csv').set_index('ts_code').pnl_yuan
        bank = pnl.reindex(['601288.SH','601398.SH','601988.SH','600000.SH','601998.SH'],fill_value=0).sum()
        details.update(five_bank_net_pnl=float(bank), account_net_pnl=float(pnl.sum()),
                       five_bank_fraction=float(bank/pnl.sum()))
        peaks = np.maximum.accumulate(np.r_[50000.,hybrid.equity.to_numpy()])[1:]
        drawdown = hybrid.equity.to_numpy()/peaks-1
        trough = int(np.argmin(drawdown))
        earlier = hybrid.iloc[:trough+1]
        peak_date = earlier.equity.idxmax()
        recovered = hybrid[(hybrid.index>hybrid.index[trough]) & hybrid.equity.ge(earlier.equity.max())]
        details['max_drawdown_episode'] = dict(peak=str(peak_date.date()),
            trough=str(hybrid.index[trough].date()), recovered=str(recovered.index[0].date()) if len(recovered) else None)
        trades = pd.read_csv(run/'execution'/f'{scenario}_hybrid'/'trades.csv')
        details.update(trade_days=int(trades.date.nunique()), buys=int(trades.side.eq('buy').sum()),
                       sells=int(trades.side.eq('sell').sum()))
        if scenario=='continued':
            details['old_2022_2023'] = {label:daily_metrics(frame.loc[:'2023-12-31'].equity,50000.,244)[0]
                                      for label,frame in [('baseline',base),('hybrid',hybrid)]}
        summary[scenario] = details
    save(run/'process_findings.json', summary)


if __name__ == '__main__':
    main()
