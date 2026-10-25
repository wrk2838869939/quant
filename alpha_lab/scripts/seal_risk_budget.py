"""核验76账户、风险模型、前缀回放与原封存，完成研究归档。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import sys
import platform
import numpy as np
import pandas as pd
import sklearn
from research_risk_budget import ROOT, verify, digest, save

run = Path(sys.argv[1]).resolve()
assert not (run / 'completion.json').exists()
cfg, manifest = verify(run)
assert len(manifest['cases']) == 76
assert digest(ROOT.parent / 'AGENTS.md') == manifest['root_agents_sha256']
for name, count in [('risk_budget_unit.log', 6), ('risk_budget_adversarial.log', 25)]:
    log = (run / name).read_text()
    assert f'Ran {count} tests' in log and log.rstrip().endswith('OK')
for case in manifest['cases']:
    folder = run / 'execution' / case['name']
    receipt = json.loads((folder / 'completion.json').read_text())
    assert receipt['status'] == 'complete'
    for name, expected in receipt['hashes'].items():
        assert digest(folder / name) == expected, (case['name'], name)
for folder, source in [('risk', None), ('analysis', 'analyze_risk_budget.py'),
                        ('diagnostics', 'diagnose_risk_budget.py'), ('figures', 'plot_risk_budget.py')]:
    receipt = json.loads((run / folder / 'completion.json').read_text())
    assert receipt['status'] == 'complete'
    if source:
        assert receipt['source_sha256'] == digest(ROOT / 'scripts' / source)
    for name, expected in receipt['artifacts'].items():
        assert digest(run / folder / name) == expected, (folder, name)
models = json.loads((run / 'risk/verification.json').read_text())
assert len(models['annual_checks']) == 5
assert all(row['future_perturbation_exact'] and row['manual_max_error'] < 1e-10 for row in models['annual_checks'])
prefix = json.loads((run / 'account_prefix_verification.json').read_text())
assert prefix['status'] == 'passed' and len(prefix['checks']) == 3
assert prefix['labels_independently_verified'] == 2470
assert prefix['source_sha256'] == digest(ROOT / 'scripts/verify_risk_budget_prefixes.py')
assert all(row['exact'] for row in prefix['checks'])
accounts = json.loads((run / 'analysis/verification.json').read_text())
assert accounts['cases'] == 76
old_counts = {}
for path in [cfg['parent'], cfg['history_parent'], 'outputs/causal_execution/20260913T130810_374654Z']:
    folder = ROOT / path
    old = json.loads((folder / 'completion.json').read_text())
    for name, expected in old['artifact_sha256'].items():
        assert digest(folder / name) == expected, (path, name)
    old_counts[path] = len(old['artifact_sha256'])
for source in ['analyze_risk_budget.py', 'verify_risk_budget_prefixes.py', 'diagnose_risk_budget.py',
               'plot_risk_budget.py', 'report_risk_budget.py', 'seal_risk_budget.py']:
    target = run / 'sources/scripts' / source
    assert not target.exists()
    shutil.copy2(ROOT / 'scripts' / source, target)
report = ROOT / 'docs/RISK_BUDGET_RESEARCH_20260914.md'
assert digest(report) == digest(run / 'REPORT.md')
save(run / 'environment.json', dict(python=platform.python_version(), pandas=pd.__version__,
    numpy=np.__version__, sklearn=sklearn.__version__))
save(run / 'completion.json', dict(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(),
    cases=76, tests_passed=31, baseline_accounts_exact=16, lifecycle_states=7714, primary_states=2470,
    annual_models=5, annual_future_perturbation_checks=5, independent_label_checks=2470,
    account_prefix_replays=3, orders_verified=accounts['orders'], fills_verified=accounts['fills'],
    prior_artifacts_unchanged=old_counts, root_agents_unchanged=True, new_strategy_admitted=False,
    live_actions=False, fresh_holdout=False, document_sha256=digest(report),
    artifact_sha256={str(p.relative_to(run)): digest(p) for p in sorted(run.rglob('*')) if p.is_file()}))
receipt = json.loads((run / 'completion.json').read_text())
for name, expected in receipt['artifact_sha256'].items():
    assert digest(run / name) == expected, name
print('SEALED', len(receipt['artifact_sha256']), 'artifacts;', accounts['orders'], 'orders;',
      accounts['fills'], 'fills;', sum(old_counts.values()), 'prior artifacts unchanged')
