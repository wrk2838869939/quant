"""独立审查汇总：校验封存完整性和机器证据，不代替原始审查。"""
import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'alpha_lab/outputs/conditional_hold/20260913T123032_500204Z'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


checks = []


def verify_hashes(group, mapping, base):
    failures = []
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = base / path
        if not path.is_file() or sha(path) != expected:
            failures.append(str(path))
    checks.append(dict(name=group, count=len(mapping), failures=failures,
                       passed=not failures))


verify_hashes('sealed_run_artifacts', read(RUN / 'completion.json')['artifact_sha256'], RUN)
scope = read(HERE / 'scope_and_integrity.json')
verify_hashes('opening_scope', {x['path']: x['expected'] for x in scope['checks']}, ROOT)
execution = read(HERE / 'execution/audit_bundle_manifest.json')
for group in ('source_inputs', 'artifacts'):
    verify_hashes('execution_' + group, execution[group], ROOT)
verify_hashes('signals_provenance', read(HERE / 'signals/provenance.json'), ROOT)
data = read(HERE / 'data/data_audit.json')
verify_hashes('data_sources', data['sources'], ROOT)
public = read(HERE / 'data/public_source_checks.json')
verify_hashes('public_source_files', {x['file']: x['sha256'] for x in public['sources']}, HERE / 'data')

coverage = pd.read_csv(HERE / 'execution/all_30_case_execution_coverage.csv')
actual = []
for case in coverage['case']:
    trades = pd.read_csv(RUN / 'execution' / case / 'trades.csv')
    evidence = coverage.loc[coverage['case'].eq(case)].iloc[0]
    before = (pd.to_datetime(trades.signal_date) < pd.to_datetime(trades.date)).sum()
    actual.append(len(trades) == evidence.trades and before == evidence.strict_signal_date_before_fill)
checks.append(dict(name='all_case_trade_counts_and_signal_dates', count=len(actual), passed=all(actual)))

signal = read(HERE / 'signals/results.json')
assert signal['label_reconstruction']['rows'] == 51883
assert signal['label_reconstruction']['bad_rows'] == 0
assert signal['hybrid_prefix']['review_cutoffs'] == 57
assert signal['hybrid_prefix']['all_prefix_exact']
assert all(x['full_A_features_exact'] and x['predictions_exact'] for x in signal['all_A_prefix'])
assert all(x['historical_hybrid_exact'] for x in read(HERE / 'signals/future_poison.json'))
data_failures = [x['name'] for x in data['checks'] if not x['passed']]
assert data_failures == ['panel_stock_identity_covered']
result = dict(
    integrity_and_evidence_checks=checks,
    verification_passed=all(x['passed'] for x in checks),
    original_artifacts_preserved_count=730,
    trade_count=int(coverage.trades.sum()),
    full_day_volume_gate_counterfactual_scope='cold_hybrid and continued_hybrid only',
    data_check_count=len(data['checks']),
    data_failed_checks=data_failures,
    audit_verdict='FINDINGS_CONFIRMED_NOT_ADMITTED',
    limitation='Evidence consistency verification is not a new independent market-data or execution certification.',
)
(HERE / 'root_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result['verification_passed']
