"""核验历史事件、预测及事后对照证据，生成不可覆盖的完成回执。"""
from pathlib import Path
from datetime import datetime, timezone
import sys
import json
import shutil
import platform
import numpy as np
import pandas as pd
import sklearn
from research_exit_history import ROOT, verify, save, digest

run = Path(sys.argv[1]).resolve()
assert not (run / 'completion.json').exists()
cfg = verify(run)
manifest = json.loads((run / 'manifest.json').read_text())
assert digest(ROOT.parent / 'AGENTS.md') == manifest['root_agents_sha256']
for name, expected in manifest['input_hashes'].items():
    assert digest(name) == expected, name
for folder, receipt_path in [('data', 'data_completion.json'), ('models', 'models/completion.json')]:
    receipt = json.loads((run / receipt_path).read_text())
    assert receipt['status'] == 'complete'
    for name, expected in receipt['hashes'].items():
        assert digest(run / folder / name) == expected, name
for folder, source in [('analysis', 'analyze_exit_history.py'), ('controls', 'control_exit_history.py'), ('figures', 'plot_exit_history.py')]:
    receipt = json.loads((run / folder / 'completion.json').read_text())
    assert receipt['status'] == 'complete' and receipt['source_sha256'] == digest(ROOT / 'scripts' / source)
    for name, expected in receipt['artifacts'].items():
        assert digest(run / folder / name) == expected, name
independent = json.loads((run / 'independent_verification.json').read_text())
assert independent['status'] == 'passed'
assert independent['source_sha256'] == digest(ROOT / 'scripts/verify_exit_history.py')
assert independent['exits_verified'] == 1062 and independent['entries_verified'] == 1106
assert len(independent['models']) == 36
assert all(row['manual_prediction_max_error'] < 1e-10 for row in independent['models'])
controls = json.loads((run / 'controls/completion.json').read_text())
assert controls['posthoc'] is True and len(controls['manual_checks']) == 18
assert all(row['manual_error'] < 1e-10 for row in controls['manual_checks'])
audits = pd.read_csv(run / 'models/training_audit.csv')
assert len(audits) == 36 and audits.future_perturbation_exact.all() and audits.model_fitted.all()
data = json.loads((run / 'data_completion.json').read_text())
assert data['primary_events'] == 1036 and data['old_monthly_signals_exact']
assert len(data['prefix_checks']) == 2
assert all(row['events_exact'] and row['hold_exact'] and row['features_exact'] == 9 for row in data['prefix_checks'])
previous = {}
for path in [cfg['parent'], cfg['rolling_parent'], cfg['execution_parent']]:
    folder = ROOT / path
    receipt = json.loads((folder / 'completion.json').read_text())
    for name, expected in receipt['artifact_sha256'].items():
        assert digest(folder / name) == expected, (path, name)
    previous[path] = len(receipt['artifact_sha256'])
report = ROOT / 'docs/EXIT_HISTORY_RESEARCH_20260913.md'
assert digest(report) == digest(run / 'REPORT.md')
for source in ['analyze_exit_history.py', 'verify_exit_history.py', 'control_exit_history.py',
               'plot_exit_history.py', 'report_exit_history.py', 'seal_exit_history.py']:
    target = run / 'sources/scripts' / source
    assert not target.exists()
    shutil.copy2(ROOT / 'scripts' / source, target)
save(run / 'environment.json', dict(python=platform.python_version(), numpy=np.__version__,
    pandas=pd.__version__, sklearn=sklearn.__version__))
save(run / 'completion.json', dict(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(),
    shadow_entries=1106, shadow_exits=1062, primary_shadow_events=1036, prior_artifacts_unchanged=previous,
    real_feature_prefix_checks=18, lifecycle_prefix_checks=2, annual_models=36,
    future_label_perturbation_checks=36, independent_prediction_reconstructions=54,
    additional_controls_posthoc=True, portfolio_runs=0, new_strategy_admitted=False,
    fresh_holdout=False, live_actions=False, root_agents_unchanged=True, document_sha256=digest(report),
    artifact_sha256={str(p.relative_to(run)): digest(p) for p in sorted(run.rglob('*')) if p.is_file()}))
receipt = json.loads((run / 'completion.json').read_text())
for name, expected in receipt['artifact_sha256'].items():
    assert digest(run / name) == expected, name
print('SEALED', len(receipt['artifact_sha256']), 'artifacts;', sum(previous.values()), 'prior artifacts unchanged')
