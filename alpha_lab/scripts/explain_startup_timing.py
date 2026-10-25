"""保存最敏感窗口的排名、席位阻断与逐股贡献证据。"""
from pathlib import Path
import argparse
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    out = run / 'analysis'
    manifest = json.loads((run / 'manifest.json').read_text())
    parent = ROOT / manifest['config']['parent']
    detail = json.loads((out / 'verification.json').read_text())['largest_fresh_spread']
    assert detail['cohort'] == '2022-04', '诊断案例变化，应重新检查机制，不套用结论'
    rows, events, stages = [], [], []
    for name in [detail['low'], detail['high']]:
        case = next(c for c in manifest['cases'] if c['name'] == name)
        signals = pd.read_parquet(parent / f"schedules/calendar_{case['offset']:02d}/signals.parquet")
        scores = signals.loc[case['boot_date']].dropna().sort_values(ascending=False, kind='stable').head(10)
        rows.extend(dict(account=name, date=case['boot_date'], rank=i, ts_code=code, score=value)
                    for i, (code, value) in enumerate(scores.items(), 1))
        folder = run / 'accounts' / name
        ev = pd.read_parquet(folder / 'events.parquet')
        mask = ((ev.date.eq(pd.Timestamp('2022-05-05')) & ev.kind.isin(['hold_retained', 'hold_exit_requested', 'hold_risk_cap_reduction']))
                | (ev.date.eq(pd.Timestamp('2023-01-04')) & ev.kind.eq('selection_decision')))
        ev = ev.loc[mask].copy(); ev.insert(0, 'account', name); events.append(ev)
        daily = pd.read_parquet(folder / 'daily.parquet')
        for session in [20, 60, 126, 244]:
            row = daily.iloc[session - 1]
            stages.append(dict(account=name, session=session, date=str(row.name.date()),
                               equity=row.equity, cash=row.cash, total=row.equity / manifest['config']['capital'] - 1))
    pd.DataFrame(rows).to_csv(out / 'example_initial_ranks.csv', index=False)
    pd.concat(events, ignore_index=True).to_csv(out / 'example_review_and_slot_events.csv', index=False)
    pd.DataFrame(stages).to_csv(out / 'example_path_checkpoints.csv', index=False)
    diagnostics = pd.read_csv(out / 'initial_diagnostics.csv')
    summary = pd.read_csv(run / 'summary.csv')
    facts = dict(initial_pairs=len(diagnostics), identical_initial_stock_sets=int(diagnostics.initial_jaccard.eq(1).sum()),
                 first_regular_no_exit=int(diagnostics.first_regular_hold_exits.eq(0).sum()),
                 first_regular_any_cap_reduction=int(diagnostics.first_regular_cap_reductions.gt(0).sum()),
                 max_cash_reconciliation_error=float(summary.cash_max_error.max()),
                 max_share_reconciliation_error=float(summary.shares_max_error.max()))
    (out / 'process_facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
