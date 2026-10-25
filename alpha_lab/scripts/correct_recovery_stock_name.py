"""保留首次封存，依据冻结证券主表更正报告中的股票简称。"""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import pandas as pd

from research_risk_recovery import ROOT, digest, save

run = Path(sys.argv[1]).resolve()
receipt = json.loads((run / 'completion.json').read_text())
for name, expected in receipt['artifact_sha256'].items():
    assert digest(run / name) == expected
basics = ROOT / 'outputs/rolling_review/20260913T141120_325588Z/inputs/basics.parquet'
data = pd.read_parquet(basics)
assert data.loc[data.ts_code.eq('603868.SH'), 'name'].iloc[0] == '飞科电器'
archive = run / 'report_name_correction'
archive.mkdir(exist_ok=True)
for source, target in [(run / 'completion.json', 'previous_completion.json'),
                       (run / 'REPORT.md', 'previous_REPORT.md'),
                       (run / 'sources/scripts/report_risk_recovery.py', 'previous_report_source.py')]:
    if (archive / target).exists():
        assert digest(source) == digest(archive / target)
    else:
        shutil.copy2(source, archive / target)
source = ROOT / 'scripts/report_risk_recovery.py'
assert source.read_text().count('健之佳') == 2
source.write_text(source.read_text().replace('健之佳', '飞科电器'))
subprocess.run([sys.executable, str(source), str(run)], check=True)
shutil.copy2(source, run / 'sources/scripts/report_risk_recovery.py')
shutil.copy2(__file__, run / 'sources/scripts/correct_recovery_stock_name.py')
save(archive / 'correction.json', dict(
    reason='603868.SH简称误写为健之佳；冻结证券主表名称为飞科电器。仅报告文字更正，代码、成交及绩效均未改变。',
    security_master_sha256=digest(basics), previous_receipt='previous_completion.json',
    source_sha256=digest(__file__)))
receipt['document_sha256'] = digest(ROOT / 'docs/RISK_RECOVERY_RESEARCH_20260914.md')
receipt['report_name_correction'] = 'report_name_correction/correction.json'
receipt['artifact_sha256'] = {str(p.relative_to(run)): digest(p)
    for p in sorted(run.rglob('*')) if p.is_file() and p != run / 'completion.json'}
save(run / 'completion.json', receipt)
for name, expected in receipt['artifact_sha256'].items():
    assert digest(run / name) == expected
assert receipt['document_sha256'] == digest(run / 'REPORT.md')
print('CORRECTED AND VERIFIED', len(receipt['artifact_sha256']), 'artifacts; execution unchanged')
