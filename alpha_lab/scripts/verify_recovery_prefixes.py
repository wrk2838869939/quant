"""恢复与缓冲账户的真实历史前缀，不让后来现金和风险改写旧订单。"""
from pathlib import Path
from io import StringIO
import sys
import pandas as pd
from research_risk_recovery import verify, simulate, reconcile, save, digest

run = Path(sys.argv[1]).resolve()
cfg, manifest = verify(run)
stop = pd.Timestamp('2025-06-30')
checks = []
for policy in ['restore', 'grace', 'combined']:
    case = next(c for c in manifest['cases'] if c['window'] == 'continued' and c['schedule'] == 'rolling_00' and c['policy'] == policy)
    result = simulate(run, case, stop)
    reconcile(result, case['capital'])
    folder = run / 'execution' / case['name']
    for name, frame in [('daily',result.daily), ('trades',result.trades), ('holdings',result.holdings)]:
        actual = pd.read_csv(StringIO(frame.to_csv(index=name == 'daily')))
        old = pd.read_csv(folder / f'{name}.csv')
        pd.testing.assert_frame_equal(actual, old[pd.to_datetime(old.date).le(stop)].reset_index(drop=True), check_exact=True)
    old = pd.read_csv(folder / 'events.csv')
    old = old[old.kind.eq('order_planned') & pd.to_datetime(old.date).lt(stop)]
    current = result.events[result.events.kind.eq('order_planned') & result.events.date.lt(stop)]
    fields = ['date','ts_code','order_id','side','shares','limit_price','reserved_cash','reason']
    actual = pd.read_csv(StringIO(current[fields].to_csv(index=False)))
    pd.testing.assert_frame_equal(actual, old[fields].reset_index(drop=True), check_exact=True)
    checks.append(dict(name=case['name'], cutoff=stop, exact=True))
    print(case['name'], 'prefix exact', flush=True)
save(run / 'prefix_verification.json', dict(status='passed', checks=checks, source_sha256=digest(__file__)))
