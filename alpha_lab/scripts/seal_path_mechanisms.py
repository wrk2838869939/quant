"""核验完整80账户及旧封存哈希，生成唯一完成回执。"""
import sys,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from research_path_mechanisms import ROOT,verify,save,digest
run=Path(sys.argv[1]).resolve();assert not (run/'completion.json').exists();cfg,m=verify(run)
for case in m['cases']:
    folder=run/'execution'/case['name'];receipt=json.loads((folder/'completion.json').read_text())
    for name,h in receipt['hashes'].items():assert digest(folder/name)==h,(case['name'],name)
for name,source in [('analysis','analyze_path_mechanisms.py'),('diagnostics','diagnose_path_mechanisms.py'),('figures','plot_path_mechanisms.py')]:
    folder=run/name;receipt=json.loads((folder/'completion.json').read_text());assert digest(ROOT/'scripts'/source)==receipt['source_sha256']
    for name,h in receipt['artifacts'].items():assert digest(folder/name)==h,name
for name,source in [('account_prefix_verification.json','verify_path_prefixes.py'),('mechanism_event_verification.json','verify_path_mechanism_events.py')]:
    receipt=json.loads((run/name).read_text());assert receipt['status']=='passed' and receipt['source_sha256']==digest(ROOT/'scripts'/source)
assert (run/'unit_tests.log').read_text().rstrip().endswith('OK') and 'Ran 7 tests' in (run/'unit_tests.log').read_text()
assert (run/'adversarial_tests.log').read_text().rstrip().endswith('OK') and 'Ran 25 tests' in (run/'adversarial_tests.log').read_text()
assert '7 mechanism tests and 25 causal adversarial tests passed.' in (run/'test_runner_verification.log').read_text()
old_counts={}
for path in [cfg['parent'],'outputs/causal_execution/20260913T130810_374654Z','outputs/profit_defense/20260913T135233_215310Z']:
    folder=ROOT/path;receipt=json.loads((folder/'completion.json').read_text())
    for name,h in receipt['artifact_sha256'].items():assert digest(folder/name)==h,(path,name)
    old_counts[path]=len(receipt['artifact_sha256'])
for source in ['analyze_path_mechanisms.py','diagnose_path_mechanisms.py','verify_path_prefixes.py','verify_path_mechanism_events.py','plot_path_mechanisms.py','report_path_mechanisms.py','run_path_mechanism_tests.py','seal_path_mechanisms.py']:
    target=run/'sources/scripts'/source;assert not target.exists();shutil.copy2(ROOT/'scripts'/source,target)
report=ROOT/'docs/PATH_MECHANISM_RESEARCH_20260913.md';assert digest(report)==digest(run/'REPORT.md')
v=json.loads((run/'analysis/verification.json').read_text());assert v['cases']==80
save(run/'completion.json',dict(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),cases=80,orders_verified=v['orders'],fills_verified=v['fills'],tests_passed=32,baseline_accounts_exact=16,real_duration_prefix_checks=16,real_account_prefix_replays=4,prior_artifacts_unchanged=old_counts,fresh_holdout=False,new_strategy_admitted=False,live_actions=False,document_sha256=digest(report),artifact_sha256={str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*')) if p.is_file()}))
receipt=json.loads((run/'completion.json').read_text())
for name,h in receipt['artifact_sha256'].items():assert digest(run/name)==h,name
print('SEALED',len(receipt['artifact_sha256']),'artifacts; 80 accounts;',v['orders'],'orders;',v['fills'],'fills;',sum(old_counts.values()),'prior artifacts unchanged')
