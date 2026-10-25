"""核验76账户及恢复/缓冲条件，封存本轮完整结果。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import sys
from research_risk_recovery import ROOT, verify, digest, save

run = Path(sys.argv[1]).resolve()
assert not (run / 'completion.json').exists()
cfg, manifest = verify(run)
assert len(manifest['cases']) == 76
assert digest(ROOT.parent / 'AGENTS.md') == manifest['root_agents_sha256']
for name, n in [('recovery_unit.log',6),('recovery_risk.log',6),('recovery_causal.log',25)]:
    text = (run / name).read_text()
    assert f'Ran {n} tests' in text and text.rstrip().endswith('OK')
for case in manifest['cases']:
    folder = run / 'execution' / case['name']
    receipt = json.loads((folder / 'completion.json').read_text())
    assert receipt['status'] == 'complete'
    for name, expected in receipt['hashes'].items():
        assert digest(folder / name) == expected
for folder, source in [('analysis','analyze_risk_recovery.py'), ('diagnostics','diagnose_risk_recovery.py'), ('figures','plot_risk_recovery.py')]:
    receipt = json.loads((run / folder / 'completion.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['source_sha256'] == digest(ROOT / 'scripts' / source)
    for name, expected in receipt['artifacts'].items():
        assert digest(run / folder / name) == expected
prefix = json.loads((run / 'prefix_verification.json').read_text())
assert prefix['status'] == 'passed' and len(prefix['checks']) == 3
assert all(row['exact'] for row in prefix['checks'])
assert prefix['source_sha256'] == digest(ROOT / 'scripts/verify_recovery_prefixes.py')
counts = {}
for path in [cfg['parent'], cfg['signal_parent'], 'outputs/causal_execution/20260913T130810_374654Z']:
    folder = ROOT / path
    receipt = json.loads((folder / 'completion.json').read_text())
    for name, expected in receipt['artifact_sha256'].items():
        assert digest(folder / name) == expected, (path,name)
    counts[path] = len(receipt['artifact_sha256'])
report = ROOT / 'docs/RISK_RECOVERY_RESEARCH_20260914.md'
assert digest(report) == digest(run / 'REPORT.md')
for source in ['analyze_risk_recovery.py','diagnose_risk_recovery.py','verify_recovery_prefixes.py',
               'plot_risk_recovery.py','report_risk_recovery.py','seal_risk_recovery.py']:
    target = run / 'sources/scripts' / source
    assert not target.exists()
    shutil.copy2(ROOT / 'scripts' / source, target)
audit = json.loads((run / 'analysis/verification.json').read_text())
assert audit['cases'] == 76
save(run / 'completion.json', dict(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(),
    cases=76, tests_passed=37, baseline_accounts_exact=20, main_baselines_exact=16,
    actual_account_prefixes=3, orders_verified=audit['orders'], fills_verified=audit['fills'],
    restore_plans_verified=sum(row['recovery_plans_checked'] for row in audit['checks']),
    grace_events_verified=sum(row['grace_checked'] for row in audit['checks']),
    prior_artifacts_unchanged=counts, root_agents_unchanged=True,
    new_strategy_admitted=False, live_actions=False, fresh_holdout=False,
    document_sha256=digest(report),
    artifact_sha256={str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*')) if p.is_file()}))
receipt = json.loads((run / 'completion.json').read_text())
for name, expected in receipt['artifact_sha256'].items():
    assert digest(run / name) == expected
print('SEALED',len(receipt['artifact_sha256']),'artifacts;',audit['orders'],'orders;',audit['fills'],'fills;',sum(counts.values()),'prior artifacts unchanged')
