"""核验退出事件研究的证据链和旧产物，封存未通过准入的结果。"""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys

import pandas as pd

from research_exit_evidence import ROOT, digest, save, verify

run = Path(sys.argv[1]).resolve()
assert not (run / 'completion.json').exists()
cfg = verify(run)

for folder, receipt_name in [('data', 'data_completion.json'), ('models', 'models/completion.json')]:
    receipt = json.loads((run / receipt_name).read_text())
    assert receipt['status'] == 'complete'
    for name, expected in receipt['hashes'].items():
        assert digest(run / folder / name) == expected, (folder, name)

for folder, source in [('diagnostics', 'diagnose_exit_evidence.py'), ('figures', 'plot_exit_evidence.py')]:
    receipt = json.loads((run / folder / 'completion.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['source_sha256'] == digest(ROOT / 'scripts' / source)
    for name, expected in receipt['artifacts'].items():
        assert digest(run / folder / name) == expected, (folder, name)

independent = json.loads((run / 'independent_verification.json').read_text())
assert independent['status'] == 'passed'
assert independent['source_sha256'] == digest(ROOT / 'scripts/verify_exit_evidence.py')
assert independent['labels_recomputed'] and independent['nonoverlap_checked']
assert len(independent['models']) == 12
assert independent['recovery_events_verified'] == 86
assert all(item['manual_ridge_max_error'] < 1e-10 and item['train_labels_mature']
           for item in independent['models'])
tests = (run / 'tests.log').read_text()
assert 'Ran 6 tests' in tests and tests.rstrip().endswith('OK')
audit = pd.read_csv(run / 'models/training_audit.csv')
assert len(audit) == 12 and audit.future_label_perturbation_exact.all()
assert audit[audit.cohort.eq('exit')].model_fitted.all()
assert not audit[audit.cohort.eq('recovery')].model_fitted.any()
coverage = pd.read_csv(run / 'data/coverage.csv').set_index('cohort')
assert coverage.at['exit', 'primary_events'] == 78
assert coverage.at['recovery', 'primary_events'] == 34
prefix = json.loads((run / 'data_completion.json').read_text())['prefix_checks']
assert sum(item['features'] for item in prefix) == 18
gates = json.loads((run / 'models/gate.json').read_text())
assert set(gates) == {'exit', 'recovery'}
for gate in gates.values():
    assert gate['passed'] is False
    for horizon in gate['horizons']:
        assert all(type(value) is bool for value in horizon['checks'].values())

old_counts = {}
for path in [cfg['parent'], cfg['mechanism_parent'],
             'outputs/causal_execution/20260913T130810_374654Z',
             'outputs/profit_defense/20260913T135233_215310Z']:
    folder = ROOT / path
    receipt = json.loads((folder / 'completion.json').read_text())
    for name, expected in receipt['artifact_sha256'].items():
        assert digest(folder / name) == expected, (path, name)
    old_counts[path] = len(receipt['artifact_sha256'])

report = ROOT / 'docs/EXIT_EVIDENCE_RESEARCH_20260913.md'
assert digest(report) == digest(run / 'REPORT.md')
for source in ['diagnose_exit_evidence.py', 'verify_exit_evidence.py',
               'plot_exit_evidence.py', 'report_exit_evidence.py', 'seal_exit_evidence.py']:
    target = run / 'sources/scripts' / source
    assert not target.exists()
    shutil.copy2(ROOT / 'scripts' / source, target)

save(run / 'completion.json', dict(
    status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(),
    exit_primary_events=78, recovery_primary_events=34,
    exit_models=6, recovery_mean_fallbacks=6, tests_passed=6,
    real_feature_prefix_checks=18, future_label_perturbation_checks=12,
    independent_prediction_reconstructions=12, recovery_events_verified=86,
    portfolio_runs=0, new_strategy_admitted=False, live_actions=False,
    fresh_holdout=False, gates=gates, prior_artifacts_unchanged=old_counts,
    document_sha256=digest(report),
    artifact_sha256={str(p.relative_to(run)): digest(p)
                     for p in sorted(run.rglob('*')) if p.is_file()},
))
receipt = json.loads((run / 'completion.json').read_text())
for name, expected in receipt['artifact_sha256'].items():
    assert digest(run / name) == expected, name
print('SEALED', len(receipt['artifact_sha256']), 'artifacts;',
      sum(old_counts.values()), 'prior artifacts unchanged; no strategy admitted')
