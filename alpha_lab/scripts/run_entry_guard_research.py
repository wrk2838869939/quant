"""运行事前登记的月频新增仓位2×2消融，复用冻结整手账本与预测。"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_recent_low_turnover as driver
import portfolio_entry_guard
import guarded_selection
from entry_guard import EntryGuard
from run_patient_value_execution import digest


@contextmanager
def reversed_guard_input(enabled, evidence):
    """显式反转新研究实际调用的selector入口，保存接收端入参证据。"""
    original = guarded_selection.plan_selection_first
    if enabled:
        def reversed_plan(candidates, **kwargs):
            actual = candidates.iloc[::-1].copy()
            evidence.append(dict(before=candidates.ts_code.tolist(), actual=actual.ts_code.tolist(),
                                 decision_date=str(kwargs['decision_date']),
                                 function_module='guarded_selection'))
            return original(actual, **kwargs)
        guarded_selection.plan_selection_first = reversed_plan
    try:
        yield
    finally:
        guarded_selection.plan_selection_first = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--case-name', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, expected in manifest['source_hashes'].items():
        assert digest(ROOT / name) == expected, f'源码漂移 {name}'
    for name, expected in manifest['input_hashes'].items():
        assert digest(run / name) == expected, f'输入漂移 {name}'
    case = next(c for c in manifest['cases'] if c['name'] == args.case_name)
    returns = pd.read_parquet(run / 'correlation_returns.parquet')
    guard = EntryGuard(returns, case['variant'], manifest['parameters'])
    guarded_selection.ACTIVE_GUARD = guard
    driver.run_backtest = portfolio_entry_guard.run_backtest
    try:
        evidence = []
        with reversed_guard_input(case['reverse_input'], evidence):
            driver.run_case(run, manifest, case)
        folder = run / 'execution' / case['name']
        if case['reverse_input']:
            (folder / 'reverse_input_evidence.json').write_text(json.dumps(
                dict(records=evidence, input_reversal_performed=True), ensure_ascii=False, indent=2))
        for name, records in [('guard_reviews', guard.review_audit), ('guard_pairs', guard.pair_audit),
                              ('guard_candidates', guard.candidate_audit)]:
            pd.DataFrame(records).to_csv(folder / f'{name}.csv', index=False)
    finally:
        guarded_selection.ACTIVE_GUARD = None


if __name__ == '__main__':
    main()
