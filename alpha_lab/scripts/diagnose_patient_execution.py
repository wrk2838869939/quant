#!/usr/bin/env python3
"""执行路径事后归因：买入排名、首次名单分歧、现金规则事件，禁止策略选型。"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
    out=root/'execution';summary=pd.read_csv(out/'execution_summary.csv');buys=[];holding_rows=[];differences=[]
    for case in summary.to_dict('records'):
        folder=out/case['name'];t=pd.read_csv(folder/'trades.csv',parse_dates=['date','signal_date'])
        h=pd.read_csv(folder/'holding_episodes.csv');closed=h[~h.open_at_end]
        holding_rows.append(dict(name=case['name'],episodes=len(h),closed=len(closed),
            median_completed_days=closed.holding_trading_days.median(),
            median_all_observed_days=h.holding_trading_days.median(),max_days=h.holding_trading_days.max()))
        signals=pd.read_parquet(out/f"{case['frequency']}_{case['policy']}_signals.parquet")
        ranks=signals.rank(axis=1,ascending=False,method='first')
        for trade in t[t.side.eq('buy')].itertuples():
            buys.append(dict(name=case['name'],date=trade.date,signal_date=trade.signal_date,ts_code=trade.ts_code,
                candidate_rank=ranks.loc[trade.signal_date,trade.ts_code],gross=trade.gross,shares=trade.shares,price=trade.price))
    b=pd.DataFrame(buys);b.to_csv(out/'actual_buy_ranks.csv',index=False)
    b.groupby('name').agg(mean_candidate_rank=('candidate_rank','mean'),max_candidate_rank=('candidate_rank','max'),
        n_buys=('ts_code','size'),max_buy_price=('price','max'),median_buy_price=('price','median')).to_csv(out/'actual_buy_rank_summary.csv')
    pd.DataFrame(holding_rows).to_csv(out/'holding_duration_summary.csv',index=False)
    for capital in [20000,50000,100000]:
        for frequency in ['monthly','weekly']:
            folders=[out/f'base_{capital}_{frequency}_{policy}' for policy in ['full_point','full_confidence']]
            if not all((f/'holdings.csv').exists() for f in folders):continue
            hs=[pd.read_csv(f/'holdings.csv',parse_dates=['date']) for f in folders]
            series=[]
            for h in hs:
                series.append(h.groupby('date').apply(lambda g:tuple(sorted(zip(g.ts_code,g.shares))),include_groups=False))
            aligned=pd.concat(series,axis=1,keys=['point','confidence']);different=aligned.point!=aligned.confidence
            first=aligned.index[different][0] if different.any() else pd.NaT
            common=[len(set(hs[0].loc[hs[0].date.eq(d),'ts_code'])&set(hs[1].loc[hs[1].date.eq(d),'ts_code'])) for d in aligned.index]
            row=dict(capital=capital,frequency=frequency,first_holdings_difference=first,
                same_quantity_day_fraction=(~different).mean(),mean_common_names=np.mean(common))
            differences.append(row)
            if pd.notna(first):
                events=[pd.read_csv(f/'events.csv',parse_dates=['date']) for f in folders]
                for label,e in zip(['point','confidence'],events):
                    e[e.date.eq(first)].to_csv(out/f'first_difference_{capital}_{frequency}_{label}.csv',index=False)
            pnl=[pd.read_csv(f/'asset_pnl.csv',index_col=0).pnl_yuan for f in folders]
            pd.concat(pnl,axis=1,keys=['point','confidence']).fillna(0).assign(
                difference=lambda d:d.confidence-d.point).sort_values('difference',ascending=False).to_csv(out/f'pnl_difference_{capital}_{frequency}.csv')
    pd.DataFrame(differences).to_csv(out/'point_confidence_path_difference.csv',index=False)
    (out/'path_diagnostics_addendum.json').write_text(json.dumps(dict(posthoc=True,new_strategy=False,
        notes=['different capital changes integer portfolio; not pure exposure comparison',
        'constant return control only ranks candidate pool; solver does not optimize turnover rank',
        'completed holding durations exclude censored positions; show all observed separately',
        'per-stock profit difference is attribution, not a counterfactual strategy']),indent=2))


if __name__=='__main__':main()
