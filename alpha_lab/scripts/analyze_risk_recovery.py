"""账户对账、恢复真实成交记忆、整手缓冲条件和配对过程分析。"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from research_risk_recovery import ROOT, verify, digest, save

run = Path(sys.argv[1]).resolve()
cfg, manifest = verify(run)
out = run / 'analysis'
out.mkdir()
discounts = pd.read_parquet(ROOT / cfg['parent'] / 'risk/discounts.parquet')
prices = pd.read_parquet(ROOT / cfg['signal_parent'] / 'inputs/daily_close.parquet')
summaries, years, months, asset_months, audits, restored, graces, june = [], [], [], [], [], [], [], []
total_orders = total_fills = 0
for case in manifest['cases']:
    folder = run / 'execution' / case['name']
    receipt = json.loads((folder / 'completion.json').read_text())
    for name, expected in receipt['hashes'].items():
        assert digest(folder / name) == expected
    d = pd.read_csv(folder / 'daily.csv', index_col='date', parse_dates=True)
    t = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'decision_date'])
    h = pd.read_csv(folder / 'holdings.csv', parse_dates=['date'])
    e = pd.read_csv(folder / 'events.csv', parse_dates=['date'])
    s = json.loads((folder / 'summary.json').read_text())
    capital = case['capital']
    ret = d.equity.pct_change()
    ret.iloc[0] = d.equity.iloc[0]/capital-1
    computed = dict(total=d.equity.iloc[-1]/capital-1,
        ann=(d.equity.iloc[-1]/capital)**(244/len(d))-1,
        mdd=(d.equity/d.equity.cummax().clip(lower=capital)-1).min(),
        sharpe=ret.mean()/ret.std(ddof=1)*np.sqrt(244))
    for key, value in computed.items():
        assert abs(s[key]-value) < 1e-10
    assert d.cash.ge(-1e-7).all() and d.n_hold.le(5).all()
    assert (t.date > t.decision_date).all()
    assert t[t.side.eq('buy')].shares.mod(100).eq(0).all()
    planned = e[e.kind.eq('order_planned')].copy()
    ordermap = planned.set_index('order_id')
    for date, group in planned[planned.side.eq('buy')].groupby('decision_date'):
        assert group.reserved_cash.sum() <= d.at[pd.Timestamp(date), 'cash']+1e-7
    total_orders += len(planned)
    total_fills += len(t)
    values = h.pivot(index='date', columns='ts_code', values='value').reindex(d.index).fillna(0.)
    change = values.diff()
    change.iloc[0] = values.iloc[0]
    flows = t.assign(flow=np.where(t.side.eq('sell'), t.gross, -t.gross)-t.fee).pivot_table(index='date', columns='ts_code', values='flow', aggfunc='sum')
    ap = change.add(flows.reindex(d.index).fillna(0.), fill_value=0.)
    div = e[e.kind.eq('dividend_receivable')]
    if len(div):
        ap = ap.add(div.pivot_table(index='date', columns='ts_code', values='amount', aggfunc='sum').reindex(d.index).fillna(0.), fill_value=0.)
    assert not e.kind.eq('delist_writeoff').any(), '需要单列退市现金回收核验'
    expected = d.equity.diff()
    expected.iloc[0] = d.equity.iloc[0]-capital
    error = float((ap.sum(axis=1)-expected).abs().max())
    assert error < 1e-6
    am = ap.resample('ME').sum().stack().rename('pnl_yuan').reset_index()
    am['name'] = case['name']
    asset_months.append(am)
    signal_folder = ROOT / cfg['signal_parent'] / 'schedules' / case['schedule']
    sig = pd.read_parquet(signal_folder / 'signals.parquet')
    hold = pd.read_parquet(signal_folder / 'hold.parquet')
    recovery_plans = e[e.kind.eq('risk_restore_planned')].copy()
    by_order = recovery_plans.set_index('order_id') if len(recovery_plans) else pd.DataFrame()
    memory = {}
    # 先处理该日真实成交，再核验收盘恢复规划，不能拿未成交卖出创建资格。
    for date in sorted(set(t.date) | set(recovery_plans.date)):
        day_hold = h[h.date.eq(date)].set_index('ts_code')
        for trade in t[t.date.eq(date)].itertuples():
            order = ordermap.loc[trade.order_id]
            code = trade.ts_code
            if trade.side == 'sell':
                if order.reason == 'single_stock_cap':
                    memory[code] = discounts.at[pd.Timestamp(order.signal_date), code]
                if code not in day_hold.index:
                    memory.pop(code, None)
            elif order.reason == 'risk_restore':
                detail = by_order.loc[trade.order_id]
                memory[code] = detail.multiplier
                loc = prices.index.get_loc(date)
                row = dict(name=case['name'], date=date, decision_date=trade.decision_date,
                    ts_code=code, shares=trade.shares, price=trade.price, fee=trade.fee,
                    multiplier=detail.multiplier, anchor=detail.anchor_multiplier)
                for horizon in [20, 60]:
                    path = prices.iloc[loc:loc+horizon+1][code]
                    row[f'postfill_close_return_{horizon}'] = path.iloc[-1]/path.iloc[0]-1 if len(path) == horizon+1 and path.notna().all() else np.nan
                restored.append(row)
                assert trade.shares <= day_hold.at[code, 'shares']-day_hold.at[code, 'available'], '新增股份未保持T+1锁定'
            else:
                memory.pop(code, None)
        for row in recovery_plans[recovery_plans.date.eq(date)].itertuples():
            assert date in sig.index and pd.notna(sig.at[date, row.ts_code]) and hold.at[date, row.ts_code] == 1
            assert row.ts_code in memory and abs(memory[row.ts_code]-row.anchor_multiplier) < 1e-12
            assert row.multiplier > row.anchor_multiplier
            assert abs(row.multiplier-discounts.at[date, row.ts_code]) < 1e-12
            assert abs(row.risk_cap-d.at[date, 'equity']*.25*row.multiplier) < 1e-7
            assert row.frozen_position_notional <= row.risk_cap+1e-7
            assert row.old_shares == day_hold.at[row.ts_code, 'shares']
            assert row.buy_shares % 100 == 0 and row.cash_remaining >= -1e-7
    grace = e[e.kind.eq('risk_cap_grace')]
    for row in grace.itertuples():
        assert 0 < row.required_shares < 100 and row.actual_value <= row.base_cap+1e-7
        assert abs(row.base_cap-d.at[row.date, 'equity']*.25) < 1e-7
        assert abs(row.risk_cap-row.base_cap*discounts.at[row.date, row.ts_code]) < 1e-7
        assert hold.at[row.date, row.ts_code] != 0
        graces.append(dict(name=case['name'], date=row.date, ts_code=row.ts_code,
            risk_excess=row.actual_value-row.risk_cap, suppressed_sell_shares=row.suppressed_sell_shares))
    s['restore_plans'] = len(recovery_plans)
    s['restore_fills'] = int(t.order_id.isin(by_order.index).sum()) if len(by_order) else 0
    s['grace_events'] = len(grace)
    s['trade_days'] = t.date.nunique()
    s['mean_cash_fraction'] = (d.cash/d.equity).mean()
    s['end_cash_fraction'] = d.cash.iloc[-1]/d.equity.iloc[-1]
    first_buy = t[t.side.eq('buy')].date.min()
    s['active_cash_days'] = int(d.loc[first_buy:, 'n_hold'].eq(0).sum())
    s['median_holding_days'] = pd.read_csv(folder / 'holding_periods.csv').holding_trading_days.median()
    summaries.append(s)
    years.append(pd.read_csv(folder / 'yearly.csv'))
    m = d.equity.resample('ME').last().pct_change()
    m.iloc[0] = d.equity.resample('ME').last().iloc[0]/capital-1
    months.extend(dict(name=case['name'], month=str(date.to_period('M')), ret=value) for date, value in m.items())
    start = d.loc[:'2026-05-31'].iloc[-1]
    end = d.loc[:'2026-06-30'].iloc[-1]
    june.append(dict(name=case['name'], june_return=end.equity/start.equity-1,
        may_end_cash_fraction=start.cash/start.equity, june_pnl=end.equity-start.equity))
    audits.append(dict(name=case['name'], orders=len(planned), fills=len(t), asset_pnl_error=error,
        recovery_plans_checked=len(recovery_plans), grace_checked=len(grace)))
s = pd.DataFrame(summaries)
s.to_csv(out / 'summary.csv', index=False)
pd.concat(years).to_csv(out / 'yearly.csv', index=False)
pd.DataFrame(months).to_csv(out / 'monthly.csv', index=False)
pd.concat(asset_months).to_csv(out / 'monthly_asset_pnl.csv', index=False)
pd.DataFrame(restored).to_csv(out / 'actual_restores.csv', index=False)
pd.DataFrame(graces).to_csv(out / 'grace_events.csv', index=False)
pd.DataFrame(june).to_csv(out / 'june.csv', index=False)
pairs = []
for row in s.itertuples():
    baseline = s[s.window.eq(row.window) & s.schedule.eq(row.schedule) & s.stress.eq(row.stress) & s.policy.eq('baseline')].iloc[0]
    item = dict(name=row.name, window=row.window, schedule=row.schedule, stress=row.stress, policy=row.policy)
    for key in ['total','ann','mdd','sharpe','n_trades','trade_days','mean_cash_fraction']:
        item[key+'_delta'] = getattr(row, key)-baseline[key]
    item['both_improved'] = item['total_delta'] > 0 and item['mdd_delta'] >= 0
    pairs.append(item)
pd.DataFrame(pairs).to_csv(out / 'paired.csv', index=False)
save(out / 'verification.json', dict(cases=len(s), orders=total_orders, fills=total_fills, checks=audits))
save(out / 'completion.json', dict(status='complete', source_sha256=digest(__file__),
    artifacts={p.name: digest(p) for p in out.iterdir()}))
print(s[s.stress.eq('main')].groupby(['window','policy'])[['ann','mdd','sharpe','n_trades','trade_days','mean_cash_fraction']].median().round(4).to_string())
