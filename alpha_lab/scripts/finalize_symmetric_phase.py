#!/usr/bin/env python3
"""校验完整产物后记录全部案例，收束P0/P1研究，不变更交易策略。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from ledger import ExperimentLedger
from run_symmetric_phase_execution import verify_frozen_inputs


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    read = lambda name: json.loads((run / name).read_text())
    manifest = read('manifest.json')
    execution = read('execution/manifest.json')
    verification = read('verification/report.json')
    holding = read('holding_evidence/manifest.json')
    bridge = read('entry_bridge/manifest.json')
    assert execution['status'] == 'complete' and execution['completed_cases'] == 21
    assert execution['independent_verification_pass'] and execution['order_invariance_pass']
    assert verification['status'] == 'passed' and verification['models_verified'] == 16
    assert holding['status'] == 'complete' and holding['original_calendar_exact']
    assert bridge['status'] == 'complete' and bridge['actual_cases'] == 21
    assert bridge['posthoc_not_selection'] and not bridge['test_market_data_used']
    assert read('verification/entry_bridge_review.json')['status'] == 'passed'
    assert not manifest['test_market_data_used'] and not execution['test_market_data_used']
    verify_frozen_inputs(run, execution)
    for name, expected in manifest['source_hashes'].items():
        assert digest(ROOT / name) == expected, name
    for child, folder in [(holding, run / 'holding_evidence'), (bridge, run / 'entry_bridge')]:
        for name, expected in child['output_hashes'].items():
            path = Path(name)
            assert digest(path if path.is_absolute() else folder / path) == expected, name
    tests = (run / 'execution/full_test_suite.log').read_text()
    assert 'Ran 199 tests' in tests and tests.rstrip().endswith('OK')
    parity = pd.read_csv(run / 'execution/holding_variant_csv_parity.csv')
    assert len(parity) == 15 and parity.identical.all()
    summary = pd.read_csv(run / 'execution/execution_summary.csv')
    assert len(summary) == 21 and summary.name.nunique() == 21
    report = ROOT / 'docs/SYMMETRIC_PHASE_VALIDATION_20260913.md'
    assert report.exists()
    ledger = ExperimentLedger()
    identifiers = set(ledger.df.get('experiment_id', pd.Series(dtype=str)).dropna())
    events = []
    for case in json.loads(summary.to_json(orient='records')):
        identifier = f'{run.name}:{case["name"]}'
        events.append(identifier)
        if identifier in identifiers:
            continue
        ledger.log('symmetric_phase_' + case['policy'], 'ret_60_gross_and_patient_execution',
            '2022-2023_reused_development', float('nan'), float('nan'), float('nan'), float('nan'),
            params=dict(output=str(run), execution=case, admission='NOT_ADMITTED'),
            verdict='NOT_ADMITTED',
            note='P0/P1完整21案例；倒序是实现验证；策略、资金与窗口依赖，不代表独立试验。',
            experiment_id=identifier, parent_experiment='targeted_research_20260913_P0_P1',
            source_hashes=manifest['source_hashes'], input_hashes=manifest['input_hashes'],
            selection_note='全部登记结果保留；无TEST、无新OOS；桥接分析为posthoc_not_selection；DSR未评估。')
    supplemental = [report, ROOT / 'README.md', Path(__file__).resolve(),
        ROOT / 'scripts/summarize_symmetric_phase.py', ROOT / 'scripts/diagnose_symmetric_entry_bridge.py']
    outputs = [run / name for name in ['capital_comparison.csv', 'process_comparison.csv',
        'actual_phase_pnl.csv', 'monthly_returns.csv', 'training_transfer_comparison.csv',
        'execution_paths.png', 'execution/manifest.json', 'execution/full_test_suite.log',
        'verification/report.json', 'holding_evidence/manifest.json', 'entry_bridge/manifest.json',
        'verification/verify_entry_bridge_review.py']]
    outputs.extend((run / 'verification').glob('entry_bridge_review*'))
    manifest.update(status='complete', admission='NOT_ADMITTED', completed_execution_cases=21,
        complete_scope='P0/P1; P2/P3 not implemented', tests_passed=199,
        independent_model_verification=True, independent_execution_verification=True,
        reverse_csv_exact_comparisons=60, holding_variant_csv_exact_comparisons=15,
        posthoc_explanatory_bridge='entry_bridge/manifest.json', ledger_experiment_ids=events,
        report=str(report.relative_to(ROOT)),
        supplemental_hashes={str(p.relative_to(ROOT)): digest(p) for p in supplemental},
        completion_output_hashes={str(p.relative_to(run)): digest(p) for p in outputs})
    (run / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(status='complete', cases=21, tests=199, admission='NOT_ADMITTED')))


if __name__ == '__main__':
    main()
