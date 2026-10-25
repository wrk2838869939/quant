"""只读实盘准入审查：校验封存、重算指标、重放主账户及检查运行接口。"""
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import ast
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from research_risk_budget import ROOT, digest, save, verify, simulate, reconcile


out = ROOT / 'outputs/live_readiness' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
out.mkdir(parents=True)
print(out, flush=True)
parent = ROOT / 'outputs/risk_budget/20260913T160301_084080Z'
recovery = ROOT / 'outputs/risk_recovery/20260914T035750_545354Z'
rolling = ROOT / 'outputs/rolling_review/20260913T141120_325588Z'
counts = {}
for folder in [parent, recovery, rolling]:
    receipt = json.loads((folder / 'completion.json').read_text())
    for name, expected in receipt['artifact_sha256'].items():
        assert digest(folder / name) == expected, (folder, name)
    counts[str(folder.relative_to(ROOT))] = len(receipt['artifact_sha256'])
cfg, manifest = verify(parent)
basics = pd.read_parquet(rolling / 'inputs/basics.parquet')
banks = set(basics[basics.industry.eq('银行')].ts_code)
market = pd.read_parquet(rolling / 'inputs/market.parquet').set_index(['date', 'ts_code'])
rows, orders, metrics, train_checks = [], [], [], []
for year in range(2022, 2027):
    train = pd.read_parquet(parent / f'risk/train_{year}.parquet')
    cutoff = pd.Timestamp(f'{year}-01-01')
    assert train.label_end_60.lt(cutoff).all() and train.date.lt(cutoff).all()
    train_checks.append(dict(year=year, n=len(train), max_label_end=str(train.label_end_60.max())))
for case in manifest['cases']:
    if case['stress'] != 'main' or case['policy'] != 'full_risk':
        continue
    folder = parent / 'execution' / case['name']
    d = pd.read_csv(folder / 'daily.csv', index_col='date', parse_dates=True)
    t = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'decision_date'])
    h = pd.read_csv(folder / 'holdings.csv', parse_dates=['date'])
    e = pd.read_csv(folder / 'events.csv', parse_dates=['date'])
    result = json.loads((folder / 'summary.json').read_text())
    ret = d.equity.pct_change()
    ret.iloc[0] = d.equity.iloc[0] / case['capital'] - 1
    fresh = dict(total=d.equity.iloc[-1] / case['capital'] - 1,
        ann=(d.equity.iloc[-1] / case['capital']) ** (244 / len(d)) - 1,
        mdd=(d.equity / d.equity.cummax().clip(lower=case['capital']) - 1).min(),
        sharpe=ret.mean() / ret.std(ddof=1) * np.sqrt(244))
    for key, value in fresh.items():
        assert abs(result[key] - value) < 1e-10
    assert d.cash.ge(-1e-7).all() and d.n_hold.le(5).all()
    assert t[t.side.eq('buy')].shares.mod(100).eq(0).all()
    assert t.date.gt(t.decision_date).all()
    for day, group in e[e.kind.eq('order_planned') & e.side.eq('buy')].groupby('decision_date'):
        assert group.reserved_cash.sum() <= d.loc[pd.Timestamp(day), 'cash'] + 1e-7
    bank_weight = h[h.ts_code.isin(banks)].groupby('date').value.sum().reindex(d.index).fillna(0) / d.equity
    values = h.pivot(index='date', columns='ts_code', values='value').reindex(d.index).fillna(0)
    pnl = values.diff(); pnl.iloc[0] = values.iloc[0]
    flow = t.assign(flow=np.where(t.side.eq('sell'), t.gross, -t.gross) - t.fee)
    flow = flow.pivot_table(index='date', columns='ts_code', values='flow', aggfunc='sum')
    pnl = pnl.add(flow.reindex(d.index).fillna(0), fill_value=0)
    div = e[e.kind.eq('dividend_receivable')]
    if len(div):
        pnl = pnl.add(div.pivot_table(index='date', columns='ts_code', values='amount', aggfunc='sum').reindex(d.index).fillna(0), fill_value=0)
    expected = d.equity.diff(); expected.iloc[0] = d.equity.iloc[0] - case['capital']
    assert (pnl.sum(axis=1) - expected).abs().max() < 1e-6
    by_stock = pnl.sum()
    rows.extend(dict(name=case['name'], ts_code=code, pnl_yuan=value,
        bank_in_frozen_basics=code in banks) for code, value in by_stock.items())
    for trade in t.itertuples():
        volume = market.loc[(trade.date, trade.ts_code), 'vol_share']
        orders.append(dict(name=case['name'], date=trade.date, ts_code=trade.ts_code,
            side=trade.side, shares=trade.shares, execution_kind=trade.execution_kind,
            actual_day_volume_fraction=trade.shares / volume if volume > 0 else np.nan))
    metrics.append(dict(name=case['name'], window=case['window'], schedule=case['schedule'], **fresh,
        trade_count=len(t), trade_days=t.date.nunique(),
        bank_mean_weight=bank_weight.mean(), bank_max_weight=bank_weight.max(),
        bank_pnl_fraction=by_stock.reindex(list(banks)).sum() / (d.equity.iloc[-1] - case['capital']),
        cash_days_after_first_buy=int(d.loc[t[t.side.eq('buy')].date.min():].n_hold.eq(0).sum()),
        data_end=str(d.index[-1]), tail_plans=int((e.kind.eq('order_planned') & e.date.eq(d.index[-1])).sum())))
pd.DataFrame(rows).to_csv(out / 'stock_pnl.csv', index=False)
pd.DataFrame(orders).to_csv(out / 'executions.csv', index=False)
pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
tree = ast.parse((ROOT / 'src/risk_budget_portfolio.py').read_text())
entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_backtest')
save(out / 'interface.json', dict(parameters=[arg.arg for arg in entry.args.args],
    has_initial_account_parameter=any('initial' in arg.arg for arg in entry.args.args),
    source_sha256=digest(ROOT / 'src/risk_budget_portfolio.py')))
checks = []
for window in ['cold', 'continued']:
    case = next(c for c in manifest['cases'] if c['window'] == window and
                c['schedule'] == 'calendar_01' and c['policy'] == 'full_risk' and c['stress'] == 'main')
    result = simulate(parent, case)
    reconcile(result, case['capital'])
    for name, frame in [('daily', result.daily), ('trades', result.trades),
                        ('holdings', result.holdings), ('events', result.events)]:
        actual = pd.read_csv(StringIO(frame.to_csv(index=name == 'daily')))
        pd.testing.assert_frame_equal(actual, pd.read_csv(parent / 'execution' / case['name'] / f'{name}.csv'), check_exact=True)
    checks.append(dict(name=case['name'], exact=True, ledgers=4))
    print('REPLAY EXACT', case['name'], flush=True)
for runner, logs in [('run_risk_budget_tests.py', ['risk_budget_unit.log', 'risk_budget_adversarial.log']),
                     ('run_risk_recovery_tests.py', ['recovery_unit.log', 'recovery_risk.log', 'recovery_causal.log'])]:
    subprocess.run([sys.executable, str(ROOT / 'scripts' / runner)], check=True)
    for name in logs:
        shutil.copy2(Path('/tmp') / name, out / name)
shutil.copy2(__file__, out / 'audit_source.py')
save(out / 'verification.json', dict(status='passed', prior_artifacts_verified=counts,
    metrics_recomputed=len(metrics), real_account_replays=checks,
    risk_training_checks=train_checks, original_engine_tests=31, recovery_engine_tests=37,
    tests_are_overlapping_suites=True, no_live_orders=True, source_sha256=digest(__file__),
    root_agents_sha256=digest(ROOT.parent / 'AGENTS.md')))
print('AUDIT COMPLETE', out, flush=True)
