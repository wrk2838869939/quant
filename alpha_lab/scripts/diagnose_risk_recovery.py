"""日历分布、2×2交互和恢复后再次减仓路径，避免只看最好起点。"""
from pathlib import Path
import sys
import pandas as pd
from research_risk_recovery import ROOT, verify, digest, save

run = Path(sys.argv[1]).resolve()
cfg, manifest = verify(run)
out = run / 'diagnostics'
out.mkdir()
s = pd.read_csv(run / 'analysis/summary.csv')
p = pd.read_csv(run / 'analysis/paired.csv')
rows, interactions = [], []
for (window, policy), group in s[s.stress.eq('main')].groupby(['window','policy']):
    pairs = p[p.window.eq(window) & p.policy.eq(policy) & p.stress.eq('main')]
    rows.append(dict(window=window, policy=policy, n=len(group), ann=group.ann.median(),
        mdd=group.mdd.median(), sharpe=group.sharpe.median(), trades=group.n_trades.median(),
        trade_days=group.trade_days.median(), cash_fraction=group.mean_cash_fraction.median(),
        ann_range=group.ann.max()-group.ann.min(), return_improved=int(pairs.total_delta.gt(0).sum()),
        drawdown_improved=int(pairs.mdd_delta.gt(0).sum()), both_improved=int(pairs.both_improved.sum()),
        worst_return_delta=pairs.total_delta.min(), worst_mdd_delta=pairs.mdd_delta.min()))
for (window, schedule), group in s[s.stress.eq('main')].groupby(['window','schedule']):
    z = group.set_index('policy')
    interactions.append(dict(window=window, schedule=schedule,
        restore_total=z.at['restore','total']-z.at['baseline','total'],
        restore_with_grace_total=z.at['combined','total']-z.at['grace','total'],
        grace_total=z.at['grace','total']-z.at['baseline','total'],
        interaction_total=z.at['combined','total']-z.at['grace','total']-z.at['restore','total']+z.at['baseline','total']))
pd.DataFrame(rows).to_csv(out / 'calendar_robustness.csv', index=False)
pd.DataFrame(interactions).to_csv(out / 'interactions.csv', index=False)
restores = pd.read_csv(run / 'analysis/actual_restores.csv', parse_dates=['date'])
prices = pd.read_parquet(ROOT / cfg['signal_parent'] / 'inputs/daily_close.parquet')
aftermath = []
for case, group in restores.groupby('name'):
    folder = run / 'execution' / case
    trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date'])
    events = pd.read_csv(folder / 'events.csv')
    reasons = events[events.kind.eq('order_planned')][['order_id','reason']]
    sales = trades[trades.side.eq('sell')].merge(reasons, on='order_id', validate='one_to_one')
    for row in group.itertuples():
        future = sales[sales.ts_code.eq(row.ts_code) & sales.date.gt(row.date)].sort_values('date')
        item = dict(name=case, date=row.date, ts_code=row.ts_code, shares=row.shares)
        if len(future):
            sale = future.iloc[0]
            distance = prices.index.get_loc(sale.date)-prices.index.get_loc(row.date)
            item.update(next_sale_date=sale.date, next_sale_reason=sale.reason, days_to_next_sale=distance,
                risk_resale_within60=distance <= 60 and sale.reason == 'single_stock_cap',
                next_sale_price_change=sale.price/row.price-1)
        else:
            item.update(next_sale_date=pd.NaT, next_sale_reason='no_observed_sale', risk_resale_within60=False)
        aftermath.append(item)
pd.DataFrame(aftermath).to_csv(out / 'restore_next_sale.csv', index=False)
save(out / 'completion.json', dict(status='complete', source_sha256=digest(__file__),
    artifacts={f.name:digest(f) for f in out.iterdir()}))
print(pd.DataFrame(rows).round(4).to_string(index=False))
