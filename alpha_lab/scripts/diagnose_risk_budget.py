"""六月持仓权重与现金路径，及满仓名额下的整手补仓空间。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from research_risk_budget import ROOT, verify, save, digest

run = Path(sys.argv[1]).resolve()
cfg, manifest = verify(run)
out = run / 'diagnostics'
out.mkdir()
summaries, holdings, paths, events = [], [], [], []
for case in manifest['cases']:
    folder = run / 'execution' / case['name']
    d = pd.read_csv(folder / 'daily.csv', index_col='date', parse_dates=True)
    start = d.loc[:'2026-05-31'].iloc[-1]
    june = d.loc['2026-06-01':'2026-06-30']
    h = pd.read_csv(folder / 'holdings.csv', parse_dates=['date'])
    e = pd.read_csv(folder / 'events.csv', parse_dates=['date'])
    month_start_hold = h[h.date.eq(start.name)].copy()
    month_start_hold['weight'] = month_start_hold.value/start.equity
    month_start_hold['name'] = case['name']
    holdings.append(month_start_hold)
    summaries.append(dict(name=case['name'], june_return=june.equity.iloc[-1]/start.equity-1,
        may_end_cash_fraction=start.cash/start.equity, june_end_cash_fraction=june.cash.iloc[-1]/june.equity.iloc[-1],
        june_mean_invested=(june.stock_value/june.equity).mean(), june_pnl=june.equity.iloc[-1]-start.equity))
    if case['stress'] == 'main' and case['schedule'] == 'calendar_01':
        path = d.loc['2026-04-01':'2026-07-31'].copy()
        path['name'] = case['name']
        paths.append(path.reset_index())
        selected = e[e.date.between('2026-04-01', '2026-06-30') & e.kind.isin([
            'risk_budget_review', 'hold_risk_cap_reduction', 'hold_exit_requested', 'order_planned'])].copy()
        selected['name'] = case['name']
        events.append(selected)
pd.DataFrame(summaries).to_csv(out / 'june_summary.csv', index=False)
pd.concat(holdings).to_csv(out / 'may_end_holdings.csv', index=False)
pd.concat(paths).to_csv(out / 'monthly_account_spring_path.csv', index=False)
pd.concat(events).to_csv(out / 'spring_risk_events.csv', index=False)
summary = pd.read_csv(run / 'analysis/summary.csv')
pairs = pd.read_csv(run / 'analysis/paired.csv')
rows = []
for (window, policy), group in summary[summary.stress.eq('main')].groupby(['window', 'policy']):
    paired = pairs[pairs.window.eq(window) & pairs.policy.eq(policy) & pairs.stress.eq('main')]
    rows.append(dict(window=window, policy=policy, cases=len(group),
        ann_median=group.ann.median(), ann_min=group.ann.min(), ann_max=group.ann.max(),
        mdd_median=group.mdd.median(), worst_mdd=group.mdd.min(), sharpe_median=group.sharpe.median(),
        trades_median=group.n_trades.median(), trade_days_median=group.trade_days.median(),
        cash_fraction_median=group.mean_cash_fraction.median(),
        drawdown_improved=int(paired.mdd_delta.gt(0).sum()), return_improved=int(paired.total_delta.gt(0).sum()),
        both_improved=int(paired.return_and_drawdown_better.sum())))
pd.DataFrame(rows).to_csv(out / 'calendar_robustness.csv', index=False)
save(out / 'completion.json', dict(status='complete', source_sha256=digest(__file__),
    artifacts={p.name: digest(p) for p in out.iterdir()}))
print(pd.DataFrame(rows).round(4).to_string(index=False))
