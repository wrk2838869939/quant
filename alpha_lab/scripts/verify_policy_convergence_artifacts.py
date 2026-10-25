#!/usr/bin/env python3
"""续训结束后核对全部归档指纹、评价数量和独立账本审计记录。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

PROJECT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = read_json(run / 'manifest.json')
    assert manifest['status'] == 'complete'
    hashes = read_json(run / 'artifact_hashes.json')
    for i, (relative, expected) in enumerate(hashes.items(), 1):
        assert digest(run / relative) == expected, relative
        if i % 50000 == 0:
            print(f'ARTIFACT HASHES {i}/{len(hashes)}', flush=True)
    for relative, expected in manifest['source_hashes'].items():
        assert digest(PROJECT / relative) == expected, relative
    replay_count = 0
    if 'execution_amendment_sha256' in manifest:
        amendment_path = run / 'execution_amendment.json'
        assert digest(amendment_path) == manifest['execution_amendment_sha256']
        amendment = read_json(amendment_path)
        for relative, expected in amendment['source_hashes'].items():
            assert digest(PROJECT / relative) == expected, relative
        proof_path = Path(amendment['verification'])
        assert digest(proof_path) == amendment['verification_sha256']
        proof = read_json(proof_path)
        assert proof['status'] == 'passed' and sum(c['candidates'] for c in proof['checks']) == 24
        assert all(c['six_tables_each_exact'] and c['center_champion_rng_exact'] for c in proof['checks'])
        interruption = read_json(run / 'intentional_interruption.json')
        for item in interruption['completed_uncommitted_candidates']:
            assert digest(run / item['folder'] / 'trial.json') == item['trial_sha256'], item['folder']
            replay_count += 1
    parent = PROJECT / manifest['parent_run']
    assert digest(parent / 'manifest.json') == manifest['parent_manifest_sha256']
    decisions = read_json(run / 'extension_decisions.json')
    assert len(decisions) == 12
    for item in decisions:
        for mode in ['baseline', 'random']:
            folder = run / 'search' / str(item['year']) / item['host'] / mode
            state = read_json(folder / f'state_g{item["target"]:03d}.json')
            assert state['generation'] == item['target']
            assert len(state['archive']) == item['target'] + 1
            assert state['champion'] == state['archive'][-1]
            assert len(list((folder / 'train').glob('*/trial.json'))) == 12 * (item['target'] - 8)
    assert manifest['total_training_evaluations'] == sum(2 * (1 + 12 * d['target']) for d in decisions)
    assert manifest['new_training_evaluations'] == manifest['total_training_evaluations'] - 2328
    assert digest(run / 'all_locked_choices.json') == manifest['locked_choices_sha256']
    prefixes = read_json(run / 'prefix_checks.json')
    assert len(prefixes) == manifest['prefix_checks'] == 96
    assert all(row['passed'] for row in prefixes)
    parity = read_json(run / 'cached_engine_parity.json')
    assert len(parity) == 24 and all(row['exact_four_tables'] for row in parity)
    extremes = read_json(run / 'diagnostics' / 'high_return_engine_parity_final.json')
    assert len(extremes) == 3 and all(row['four_tables_original_cached_archive_exact'] for row in extremes)
    error_fields = ['cash_max_error', 'shares_max_error', 'equity_identity_max_error',
                    'trade_cash_max_error', 'fees_max_error']
    max_errors = dict.fromkeys(error_fields, 0.)
    audit_count = trade_count = 0
    for path in run.rglob('audit.json'):
        audit = read_json(path)
        assert all(audit[k] for k in ['t_plus_one_pass', 'official_limits_pass', 'no_negative_cash']), str(path)
        for field in error_fields:
            assert 0 <= audit[field] < 1e-6, (str(path), field, audit[field])
            max_errors[field] = max(max_errors[field], audit[field])
        audit_count += 1
        trade_count += audit['trades_checked']
    assert audit_count == (manifest['new_training_evaluations']
                           + manifest['validation_unique_evaluations'] + manifest['outer_accounts'])
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  core_artifact_hashes_checked=len(hashes), source_hashes_checked=len(manifest['source_hashes']),
                  archived_audits_checked=audit_count, trade_records_checked=trade_count,
                  maximum_reconciliation_errors=max_errors,
                  parent_manifest_unchanged=True, prefix_checks=len(prefixes),
                  completed_interrupted_candidates_replayed_exact=replay_count,
                  cached_engine_original_champions=24, final_extreme_original_engine_replays=3,
                  verifier_sha256=digest(Path(__file__)))
    output = run / 'diagnostics' / 'completion_verification.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
