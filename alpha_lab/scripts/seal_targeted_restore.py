"""核验并封存定向恢复研究；完整性回执不代表策略获准实盘。"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    if (run / 'completion.json').exists():
        raise FileExistsError('拒绝覆盖既有封存')
    protocol = read(run / 'protocol.json')
    for name, expected in protocol['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in protocol['parent_seals'].items():
        parent = Path(name)
        assert digest(parent / 'completion.json') == expected, name
        for artifact, value in read(parent / 'completion.json')['artifact_sha256'].items():
            assert digest(parent / artifact) == value, artifact
    assert digest(ROOT.parent / 'AGENTS.md') == protocol['root_agents_sha256']
    execution = read(run / 'execution_complete.json')
    verification = read(run / 'verification.json')
    independent = read(run / 'independent_audits/restore_performance_independent.json')
    assert execution == dict(accounts=64, parent_parities=32, trades=10127, all_reconciled=True)
    assert verification['accounts'] == 64 and verification['parent_parities'] == 32
    assert verification['cash_and_share_reconciliation'] and verification['root_agents_unchanged']
    assert len(verification['prefixes']) == 3
    assert all(p['all_four_ledgers_exact'] for p in verification['prefixes'])
    assert verification['independently_rebuilt_quota_plans'] == 924
    assert verification['independently_rebuilt_restore_fills'] == 923
    assert independent['summary_rows_verified'] == 64 and independent['rank_accounts_verified'] == 16
    assert sum(row['ann_worse'] for row in independent['matched_results']) == 32
    logs = {
        'logs/targeted_restore_tests.log': 6,
        'independent_audits/audit_targeted_restore_final.log': 15,
        'independent_audits/audit_targeted_restore_existing_final.log': 25,
        'independent_audits/audit_targeted_restore_live_retest.log': 9,
    }
    for path, count in logs.items():
        log = (run / path).read_text()
        assert f'Ran {count} tests' in log and log.rstrip().endswith('OK'), path
    extra = [
        'scripts/analyze_targeted_restore.py', 'scripts/verify_targeted_restore.py',
        'scripts/plot_targeted_restore.py', 'scripts/seal_targeted_restore.py',
        'tests/test_targeted_restore.py', 'README.md',
        'docs/TARGETED_RESTORE_VALIDATION_20260914.md',
        'docs/INDEPENDENT_TARGETED_RESTORE_CAUSAL_20260914.md',
        'docs/INDEPENDENT_TARGETED_RESTORE_LIVE_20260914.md',
        'docs/INDEPENDENT_TARGETED_RESTORE_READINESS_20260914.md',
    ]
    for name in sorted(set(protocol['source_sha256']) | set(extra)):
        target = run / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    decision = dict(structural_fix_verified=True, economic_upgrade_passed=False,
                    manual_adapter_integrated=False, independent_oos=False,
                    live_approved=False, orders_or_emails_sent=False,
                    reason='32/32恢复账户年化低于同日期同费用rank；人工状态事务尚未接入')
    (run / 'deployment_decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2))
    artifacts = {str(p.relative_to(run)): digest(p) for p in sorted(run.rglob('*')) if p.is_file()}
    receipt = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                   execution=execution, verification=verification, test_logs=logs,
                   independent_metrics_verified=64, deployment_decision=decision,
                   artifact_sha256=artifacts)
    (run / 'completion.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    for name, value in read(run / 'completion.json')['artifact_sha256'].items():
        assert digest(run / name) == value, name
    print(json.dumps(dict(run=str(run), artifacts_verified=len(artifacts), live_approved=False)))


if __name__ == '__main__':
    main()
