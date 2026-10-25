"""三项风险政策的真实历史截断重放。"""
from pathlib import Path
from io import StringIO
import sys
import numpy as np
import pandas as pd
from research_risk_budget import ROOT, verify, simulate, save, digest, reconcile

run = Path(sys.argv[1]).resolve()
cfg, manifest = verify(run)
stop = pd.Timestamp('2025-06-30')
checks = []
states = pd.read_parquet(run / 'risk/lifecycle_states.parquet')
states = states[states.nonoverlap_primary]
price = pd.read_parquet(ROOT / cfg['history_parent'] / 'data/close.parquet')
for code, group in states.groupby('ts_code'):
    assert (np.diff(price.index.get_indexer(group.sort_values('date').date)) > cfg['nonoverlap_days']).all()
for row in states.itertuples():
    start = price.index.get_loc(row.date) + 1
    end = start + 60
    if end >= len(price):
        assert row.right_censored_60 and np.isnan(row.mae_60)
        continue
    values = price.iloc[start:end+1][row.ts_code].to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        assert np.isnan(row.mae_60)
        continue
    assert abs((values/values[0]-1).min()-row.mae_60) < 1e-12
    assert row.label_end_60 == price.index[end]
for policy in ['uniform', 'entry_risk', 'full_risk']:
    case = next(c for c in manifest['cases'] if c['window'] == 'continued' and c['schedule'] == 'rolling_00' and c['policy'] == policy)
    result = simulate(run, case, stop)
    reconcile(result, case['capital'])
    folder = run / 'execution' / case['name']
    for name, frame in [('daily', result.daily), ('trades', result.trades), ('holdings', result.holdings)]:
        actual = pd.read_csv(StringIO(frame.to_csv(index=name == 'daily')))
        old = pd.read_csv(folder / f'{name}.csv')
        old = old[pd.to_datetime(old.date).le(stop)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, old, check_exact=True)
    old = pd.read_csv(folder / 'events.csv')
    old = old[old.kind.eq('order_planned') & pd.to_datetime(old.date).lt(stop)]
    current = result.events[result.events.kind.eq('order_planned') & result.events.date.lt(stop)]
    fields = ['date', 'ts_code', 'order_id', 'side', 'shares', 'limit_price', 'reserved_cash', 'reason']
    actual = pd.read_csv(StringIO(current[fields].to_csv(index=False)))
    pd.testing.assert_frame_equal(actual, old[fields].reset_index(drop=True), check_exact=True)
    checks.append(dict(name=case['name'], cutoff=stop, exact=True))
    print(case['name'], 'prefix exact', flush=True)
save(run / 'account_prefix_verification.json', dict(status='passed', checks=checks,
    labels_independently_verified=len(states), source_sha256=digest(__file__)))
