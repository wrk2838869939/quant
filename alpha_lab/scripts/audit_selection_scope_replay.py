"""核对106个作废账户均已重新执行，并记录修正对账本的实际影响。"""
from pathlib import Path
import argparse
import json

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    correction = json.loads((run / 'candidate_scope_correction.json').read_text())
    records = []
    for name in correction['invalidated_accounts']:
        before = run / 'attempts/implicit_candidate_expansion' / name
        after = run / 'accounts' / name
        assert (after / 'summary.json').exists() and not (after / 'failure.json').exists()
        record = dict(account=name)
        for key in ['daily', 'trades', 'holdings', 'events']:
            a = pd.read_parquet(before / f'{key}.parquet'); b = pd.read_parquet(after / f'{key}.parquet')
            record[key + '_exact'] = a.equals(b)
            if key == 'daily': record['max_equity_change'] = float((a.equity-b.equity).abs().max())
        records.append(record)
    frame = pd.DataFrame(records); frame.to_csv(run / 'scope_replay_comparison.csv', index=False)
    (run / 'scope_replay_verification.json').write_text(json.dumps(dict(replayed=len(frame),
        identical_equity=int(frame.max_equity_change.eq(0).sum()), max_equity_change=float(frame.max_equity_change.max()),
        identical_trades=int(frame.trades_exact.sum()), identical_holdings=int(frame.holdings_exact.sum()),
        changed_events=int((~frame.events_exact).sum())), ensure_ascii=False, indent=2))
    print((run / 'scope_replay_verification.json').read_text())


if __name__ == '__main__': main()
