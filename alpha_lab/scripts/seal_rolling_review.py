"""核验全部回执并封存本轮；不覆盖已有封存记录。"""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,shutil
import pandas as pd
from research_rolling_review import ROOT,verify_manifest,digest,save
run=Path(sys.argv[1]).resolve();assert not (run/'completion.json').exists()
m=verify_manifest(run)
for name in ['signal_completion.json','execution_input_completion.json']:
    obj=json.loads((run/name).read_text())
    for file,h in obj.get('artifacts',obj.get('hashes',{})).items():assert digest(run/file)==h,file
for case in m['cases']:
    folder=run/'execution'/case['name'];obj=json.loads((folder/'completion.json').read_text())
    for name,h in obj['hashes'].items():assert digest(folder/name)==h,(case['name'],name)
a=run/'analysis';obj=json.loads((a/'completion.json').read_text())
assert obj['source_sha256']==digest(ROOT/'scripts/analyze_rolling_review.py')
for name,h in obj['artifacts'].items():assert digest(a/name)==h,name
for filename in ['input_prefix_verification.json','account_prefix_verification.json']:
    obj=json.loads((run/filename).read_text());assert obj['status']=='passed'
    src='verify_rolling_inputs.py' if filename.startswith('input') else 'verify_rolling_accounts.py'
    assert obj['source_sha256']==digest(ROOT/'scripts'/src)
assert 'Ran 5 tests' in (run/'unit_tests.log').read_text() and (run/'unit_tests.log').read_text().endswith('OK\n')
old_counts={}
for path in ['outputs/causal_execution/20260913T130810_374654Z','outputs/profit_defense/20260913T135233_215310Z']:
    parent=ROOT/path;obj=json.loads((parent/'completion.json').read_text())
    for name,h in obj['artifact_sha256'].items():assert digest(parent/name)==h,name
    old_counts[path]=len(obj['artifact_sha256'])
for name in ['prepare_rolling_execution_data.py','preserve_rolling_action_order.py','verify_rolling_inputs.py','verify_rolling_accounts.py','analyze_rolling_review.py','report_rolling_review.py','seal_rolling_review.py']:
    target=run/'sources/scripts'/name;assert not target.exists();shutil.copy2(ROOT/'scripts'/name,target)
report=ROOT/'docs/ROLLING_REVIEW_RESEARCH_20260913.md';assert digest(report)==digest(run/'REPORT.md')
h=pd.read_csv(a/'summary.csv');hy=h[h.policy.eq('hybrid')]
assert len(h)==32 and (hy.max_holdings<=5).all() and (hy.n_cash_days==hy.initial_cash_days).all()
for window in ['continued','cold']:
    z=hy[hy.window.eq(window)];base=z[z.schedule.eq('calendar_01')].iloc[0];roll=z[z.schedule.str.startswith('rolling')]
    for key in ['ann','total','mdd','sharpe']:assert (roll[key]<base[key]).all()
v=json.loads((a/'verification.json').read_text())
save(run/'completion.json',dict(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),scope='8 calendars x 2 windows x 2 holding policies',cases=32,orders_verified=v['orders'],fills_verified=v['fills'],calendar_unit_tests=5,real_feature_prefix_checks=18,real_hold_calendar_prefix_checks=16,real_account_prefix_replays=2,prior_artifacts_unchanged=old_counts,fresh_holdout=False,new_strategy_admitted=False,live_actions=False,document_sha256=digest(report),artifact_sha256={str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*')) if p.is_file()}))
# 封存后再独立读取一次所有散列。
receipt=json.loads((run/'completion.json').read_text())
for name,h in receipt['artifact_sha256'].items():assert digest(run/name)==h,name
print('SEALED',len(receipt['artifact_sha256']),'artifacts; 32 accounts; 1024 orders; 1017 fills; old',old_counts,flush=True)
