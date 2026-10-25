"""逐账户独立对账、同日历配对及风险减仓后的现金过程诊断。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from research_risk_budget import ROOT, verify, digest, save

run = Path(sys.argv[1]).resolve()
cfg, manifest = verify(run)
out = run / 'analysis'
out.mkdir()
summary, yearly, monthly, assets, audits, reductions, stranded = [], [], [], [], [], [], []
discounts = pd.read_parquet(run / 'risk/discounts.parquet')
annual = pd.read_csv(run / 'risk/annual_reference.csv').set_index('year')
feature_known = pd.read_parquet(run / 'risk/predictions.parquet').set_index(['date', 'ts_code']).features_complete
total_orders = total_fills = 0
for case in manifest['cases']:
    folder = run / 'execution' / case['name']
    receipt = json.loads((folder / 'completion.json').read_text())
    for name, expected in receipt['hashes'].items():
        assert digest(folder / name) == expected, (case['name'], name)
    s = json.loads((folder / 'summary.json').read_text())
    d = pd.read_csv(folder / 'daily.csv', index_col='date', parse_dates=True)
    t = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'decision_date'])
    h = pd.read_csv(folder / 'holdings.csv', parse_dates=['date'])
    e = pd.read_csv(folder / 'events.csv', parse_dates=['date'])
    capital = case['capital']
    ret = d.equity.pct_change()
    ret.iloc[0] = d.equity.iloc[0] / capital - 1
    metrics = dict(total=d.equity.iloc[-1]/capital-1,
        ann=(d.equity.iloc[-1]/capital)**(244/len(d))-1,
        mdd=(d.equity/d.equity.cummax().clip(lower=capital)-1).min(),
        sharpe=ret.mean()/ret.std(ddof=1)*np.sqrt(244))
    for key, value in metrics.items():
        assert abs(s[key]-value) < 1e-10, (case['name'], key)
    assert d.cash.ge(-1e-7).all() and d.n_hold.le(5).all()
    assert (t.date > t.decision_date).all()
    buys = t[t.side.eq('buy')]
    assert buys.shares.mod(100).eq(0).all() and (buys.price <= buys.limit_price+1e-9).all()
    planned = e[e.kind.eq('order_planned')]
    for date, group in planned[planned.side.eq('buy')].groupby('decision_date'):
        assert group.reserved_cash.sum() <= d.at[pd.Timestamp(date), 'cash'] + 1e-7
    total_orders += len(planned)
    total_fills += len(t)
    # 逐股市值变动、成交现金流和分红权利，独立还原每日资产变动。
    values = h.pivot(index='date', columns='ts_code', values='value').reindex(d.index).fillna(0.)
    mark = values.diff()
    mark.iloc[0] = values.iloc[0]
    flow = t.assign(flow=np.where(t.side.eq('sell'), t.gross, -t.gross)-t.fee).pivot_table(
        index='date', columns='ts_code', values='flow', aggfunc='sum')
    ap = mark.add(flow.reindex(d.index).fillna(0.), fill_value=0.)
    div = e[e.kind.eq('dividend_receivable')]
    if len(div):
        ap = ap.add(div.pivot_table(index='date', columns='ts_code', values='amount', aggfunc='sum').reindex(d.index).fillna(0.), fill_value=0.)
    assert not e.kind.eq('delist_writeoff').any(), '退市回收需要额外逐股对账'
    changes = d.equity.diff()
    changes.iloc[0] = d.equity.iloc[0]-capital
    error = float((ap.sum(axis=1)-changes).abs().max())
    assert error < 1e-6
    am = ap.resample('ME').sum().stack().rename('pnl_yuan').reset_index()
    am['name'] = case['name']
    assets.append(am)
    first = buys.date.min()
    s['active_cash_days'] = int(d.loc[first:, 'n_hold'].eq(0).sum())
    s['trade_days'] = int(t.date.nunique())
    s['mean_cash_fraction'] = float((d.cash/d.equity).mean())
    s['end_cash_fraction'] = float(d.cash.iloc[-1]/d.equity.iloc[-1])
    s['median_holding_days'] = float(pd.read_csv(folder / 'holding_periods.csv').holding_trading_days.median())
    risk = e[e.kind.isin(['risk_budget_review', 'risk_budget_entry'])]
    unknown_count = 0
    for row in risk.itertuples():
        date = pd.Timestamp(row.signal_date)
        expected = annual.at[date.year, 'uniform_discount'] if case['policy'] == 'uniform' else discounts.at[date, row.ts_code]
        assert abs(row.multiplier-expected) < 1e-12
        if not feature_known.loc[(date, row.ts_code)]:
            unknown_count += 1
            assert abs(row.multiplier-annual.at[date.year, 'uniform_discount']) < 1e-12
        if row.kind == 'risk_budget_review':
            assert abs(row.risk_cap-d.at[row.date, 'equity']*.25*expected) < 1e-7
    s['unknown_risk_decisions'] = unknown_count
    cuts = e[e.kind.eq('hold_risk_cap_reduction')]
    s['cap_reduction_requests'] = len(cuts)
    s['cap_full_exit_requests'] = int(cuts.target_shares.eq(0).sum())
    for row in cuts.itertuples():
        reductions.append(dict(name=case['name'], date=row.date, ts_code=row.ts_code,
            shares_before=row.old_shares, target_shares=row.target_shares, requested_sell=row.sell_shares,
            original_stock_value=values.at[row.date, row.ts_code], whole_exit=row.target_shares == 0))
    if case['policy'] in ['uniform', 'full_risk']:
        rv = risk[risk.kind.eq('risk_budget_review')].merge(h[['date', 'ts_code', 'value', 'price']], on=['date', 'ts_code'], validate='one_to_one')
        rv['unfilled_cap_space'] = (rv.risk_cap-rv.value).clip(lower=0)
        rv['whole_lot_space'] = np.floor(rv.unfilled_cap_space/(rv.price*1.02)/100)*100
        for date, group in rv.groupby('date'):
            possible = group[group.whole_lot_space >= 100]
            lot_gross = possible.price*1.02*100
            lot_cost = lot_gross + np.maximum(5.*case['cost_multiplier'], lot_gross*.001*case['cost_multiplier'])
            if d.at[date, 'n_hold'] == 5 and d.at[date, 'pending_sell'] == 0 and len(possible) and d.at[date, 'cash'] > lot_cost.min():
                stranded.append(dict(name=case['name'], date=date, cash=d.at[date, 'cash'], equity=d.at[date, 'equity'],
                    unfilled_cap_space=possible.unfilled_cap_space.sum(), stocks_with_lot_room=len(possible)))
    summary.append(s)
    yearly.append(pd.read_csv(folder / 'yearly.csv'))
    month = d.equity.resample('ME').last().pct_change()
    month.iloc[0] = d.equity.resample('ME').last().iloc[0]/capital-1
    monthly.extend(dict(name=case['name'], month=str(date.to_period('M')), ret=value) for date, value in month.items())
    audits.append(dict(name=case['name'], orders=len(planned), fills=len(t), asset_pnl_max_error=error,
        risk_decisions_verified=len(risk), cash_max_error=s['cash_max_error']))
s = pd.DataFrame(summary)
s.to_csv(out / 'summary.csv', index=False)
pd.concat(yearly).to_csv(out / 'yearly.csv', index=False)
pd.DataFrame(monthly).to_csv(out / 'monthly.csv', index=False)
pd.concat(assets).to_csv(out / 'monthly_asset_pnl.csv', index=False)
pd.DataFrame(reductions).to_csv(out / 'cap_reductions.csv', index=False)
pd.DataFrame(stranded).to_csv(out / 'cash_with_existing_lot_room.csv', index=False)
paired = []
for row in s.itertuples():
    base = s[s.window.eq(row.window) & s.schedule.eq(row.schedule) & s.stress.eq(row.stress) & s.policy.eq('baseline')].iloc[0]
    item = dict(name=row.name, window=row.window, schedule=row.schedule, stress=row.stress, policy=row.policy)
    for key in ['total', 'ann', 'mdd', 'sharpe', 'n_trades', 'mean_cash_fraction', 'total_fees']:
        item[key+'_delta'] = getattr(row, key)-base[key]
    item['return_and_drawdown_better'] = item['total_delta'] > 0 and item['mdd_delta'] >= 0
    paired.append(item)
pd.DataFrame(paired).to_csv(out / 'paired.csv', index=False)
same = []
for (window, schedule), group in s[s.stress.eq('main')].groupby(['window', 'schedule']):
    z = group.set_index('policy')
    for policy in ['entry_risk', 'full_risk']:
        same.append(dict(window=window, schedule=schedule, policy=policy,
            total_delta=z.at[policy, 'total']-z.at['uniform', 'total'],
            mdd_delta=z.at[policy, 'mdd']-z.at['uniform', 'mdd'],
            cash_delta=z.at[policy, 'mean_cash_fraction']-z.at['uniform', 'mean_cash_fraction']))
pd.DataFrame(same).to_csv(out / 'vs_uniform.csv', index=False)
save(out / 'verification.json', dict(cases=len(s), orders=total_orders, fills=total_fills, accounts=audits))
save(out / 'completion.json', dict(status='complete', source_sha256=digest(__file__),
    artifacts={p.name: digest(p) for p in out.iterdir()}))
print(s[s.stress.eq('main')].groupby(['window', 'policy'])[['ann', 'mdd', 'sharpe', 'n_trades', 'mean_cash_fraction']].median().round(4).to_string())
