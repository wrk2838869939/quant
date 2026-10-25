#!/usr/bin/env python3
"""核对历史代码补正证据，追加台账并链接原报告；保留原模型与原清单快照。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ledger import ExperimentLedger


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    read = lambda path: json.loads(path.read_text())
    alias = read(run / 'alias_manifest.json')
    parent = Path(alias['parent'])
    assert alias['status'] == 'complete' and not alias['gate_passed']
    assert alias['original_cache_hashes_unchanged']
    for name, expected in alias['original_frozen_hashes'].items():
        assert sha(parent / name) == expected, name
    for name, expected in alias['source_hashes'].items():
        assert sha(name) == expected, name
    model = read(run / 'verification/policy_model_review.json')
    compare = read(run / 'verification/alias_correction_comparison.json')
    scalar = read(run / 'verification/alias_label_scalar_verification.json')
    assert model['status'] == 'passed' and model['models_verified'] == 8
    assert compare['status'] == 'complete' and not compare['new_overall_pass']
    assert scalar['status'] == 'complete_pass'
    assert scalar['scalar_paths'] == 144 and scalar['scalar_field_checks'] == 4608
    for manifest, key in [(model, 'source_hashes'), (compare, 'source_hashes'), (scalar, 'sources')]:
        for name, expected in manifest[key].items():
            assert sha(name) == expected, name
    tests = (run / 'full_test_suite.log').read_text()
    match = re.search(r'Ran (\d+) tests', tests)
    assert match and tests.rstrip().endswith('OK')
    count = int(match.group(1))
    assert count >= 244
    metadata = read(run / 'policy_model_metadata.json')
    ledger = ExperimentLedger()
    existing = set(ledger.df.get('experiment_id', pd.Series(dtype=str)).dropna())
    events = []
    for stage, info in metadata.items():
        for target, spec in info['models'].items():
            events.append((f'target_model:{stage}:{target}', 'NOT_ADMITTED',
                           dict(stage=stage, target=target, model=spec,
                                common_training_rows=info['common_training_rows'])))
    events.append(('repair:historical_code_alias', 'IMPLEMENTATION_VERIFIED', alias['findings']))
    identifiers = []
    for name, verdict, params in events:
        identifier = f'{run.name}:alias:{name}'
        identifiers.append(identifier)
        if identifier in existing:
            continue
        ledger.log('policy_cash_alias', 'data_corrected_policy_120_proxy',
                   'reused_development_through_2023', *([float('nan')] * 4),
                   params=dict(params, output=str(run)), verdict=verdict,
                   experiment_id=identifier, parent_experiment=parent.name,
                   source_hashes=alias['source_hashes'],
                   input_hashes={'policy_labels.parquet': sha(run / 'policy_labels.parquet')},
                   note='数据连接补正后重拟合，非独立策略试验；原结果保留，停止门仍未通过。',
                   selection_note='无TEST；无新增现金回放；事件数不代表独立试验数。')
    report = ROOT / 'docs/POLICY_CASH_VALIDATION_20260913.md'
    files = [p for p in run.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.name != 'completion_manifest.json']
    completion = dict(status='complete', admission='NOT_ADMITTED', parent=str(parent),
                      original_model_fits=8, corrected_model_refits=8,
                      original_cash_cases=16, adaptive_cash_cases=5,
                      tests_passed=count, scalar_label_review_passed=True,
                      model_review_passed=True, gate_passed=False,
                      conditional_model_replays='not_triggered_by_registered_gate',
                      test_market_data_used=False, fresh_holdout=False,
                      ledger_experiment_ids=identifiers,
                      ledger_semantics='9 dependent refit/repair events; original 32 retained',
                      output_hashes={str(p.relative_to(run)): sha(p) for p in files},
                      report_hashes={str(p.relative_to(ROOT)): sha(p)
                                     for p in [report, ROOT / 'README.md', Path(__file__).resolve()]})
    target = run / 'completion_manifest.json'
    target.write_text(json.dumps(completion, ensure_ascii=False, indent=2))
    original = read(parent / 'manifest.json')
    assert original['status'] == 'complete' and original['model_count'] == 8
    original['alias_followup'] = dict(path=str(run), completion_sha256=sha(target),
                                     tests_passed=count, additional_model_refits=8,
                                     original_manifest_snapshot=str(run / 'prior_report_snapshot/original_root_manifest.json'))
    for path in [report, ROOT / 'README.md']:
        original['supplemental_hashes'][str(path.relative_to(ROOT))] = sha(path)
    (parent / 'manifest.json').write_text(json.dumps(original, ensure_ascii=False, indent=2))
    print(json.dumps(dict(status='complete', tests=count, total_model_fits=16,
                          total_cash_cases=21, additional_ledger_events=len(identifiers))))


if __name__ == '__main__':
    main()
