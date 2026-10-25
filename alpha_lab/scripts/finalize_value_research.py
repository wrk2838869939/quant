#!/usr/bin/env python3
"""汇集完整检验、保留全部反证并登记不准入；不从回测中选择策略。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ledger import ExperimentLedger


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);args=p.parse_args();out=args.run
    checks={name:json.loads((out/path).read_text()) for name,path in {
        'model':'independent_verification.json','execution':'execution/final_checks.json',
        'cash_overlay':'risk_overlay/verification.json','allocation_order':'allocation_identifiability/manifest.json',
        'missing_marks':'execution/missing_mark_and_unknown_hold_audit.json'}.items()}
    assert checks['model']['status']=='passed'
    assert checks['execution']['cases']==21 and checks['execution']['failed_cases']==0
    assert checks['cash_overlay']['case_count']==6 and checks['cash_overlay']['status'].startswith('passed')
    assert checks['allocation_order']['completed_cases']==2
    tests=(out/'tests.log').read_text();count=int(re.search(r'Ran (\d+) tests',tests).group(1))
    assert count>=142 and tests.rstrip().endswith('OK')
    cases=pd.concat([pd.read_csv(out/'execution/execution_summary.csv'),pd.read_csv(out/'risk_overlay/summary.csv'),
        pd.read_csv(out/'allocation_identifiability/comparison.csv')],ignore_index=True)
    assert len(cases)==29 and cases.max_holdings.le(5).all()
    assert cases.cash_max_error.max()<1e-6 and cases.shares_max_error.max()==0
    assert cases.asset_pnl_max_error.max()<1e-6
    cases.to_csv(out/'all_execution_cases.csv',index=False)
    verdict=dict(status='NOT_ADMITTED',production_changed=False,fresh_holdout=False,test_market_data_used=False,
        completed_execution_cases=29,tests_passed=count,
        reasons=['full return magnitude calibration worse than constant',
            'cross-sectional entry confidence does not create active cash',
            'uniform-return top20 allocation fails input-order economic identifiability',
            'market risk overlay loses holding returns in reused development window',
            'suspension gaps can keep long-window holding state unknown for months'],
        engineering_completed=['patient <=5 holdings with integer/cash constraints',
            'confirmed exits persist until filled','dated model fitting and uncertainty diagnostics',
            'strict evidence-based corporate-action source correction','all candidate paths retained and independently reconciled'],
        warning='accounting verification is distinct from strategy admission; no candidate champion selected')
    (out/'admission.json').write_text(json.dumps(verdict,indent=2))
    source_paths=[ROOT/'configs/autonomous_value_research.yaml',ROOT/'configs/corporate_action_components.yaml',
        ROOT/'src/value_forecast.py',ROOT/'src/portfolio.py',ROOT/'src/retail_rebalance.py',ROOT/'src/patient_trend.py',
        ROOT/'src/trend_entry.py',ROOT/'src/risk_states.py']
    scripts=['run_value_iterations','verify_value_iterations','diagnose_value_states','run_patient_value_execution',
        'run_patient_risk_overlay','verify_patient_risk_overlay','check_allocation_identifiability',
        'diagnose_patient_execution','plot_value_research','finalize_value_research','audit_monthly_portfolio']
    source_paths += [ROOT/'scripts'/(name+'.py') for name in scripts]
    source_paths += list((ROOT/'tests').glob('test_*value*.py'))+list((ROOT/'tests').glob('test_patient*.py'))+[ROOT/'tests/test_action_source_corrections.py']
    hashes={}
    for path in dict.fromkeys(source_paths):
        target=out/'sources_final'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        hashes[str(path.relative_to(ROOT))]=sha(path)
    cfg=yaml.safe_load((out/'config.yaml').read_text())
    if not (out/'ledger_complete.json').exists():
        ledger=ExperimentLedger()
        for name,label in [('return_calibration','constant_style_full_and_censoring'),
                           ('uncertainty_gate','calendar_month_blocks_and_process_diagnostics'),
                           ('patient_execution','21_fixed_capital_frequency_execution_cases'),
                           ('market_cash_overlay','6_fixed_market_gate_and_integer_cash_controls'),
                           ('allocation_identifiability','2_input_order_counterexamples')]:
            ledger.log('autonomous_value_'+name,label,'reused development 2014-2023; TEST frozen',
                np.nan,np.nan,np.nan,np.nan,params=cfg,verdict='NOT_ADMITTED',
                experiment_id=str(out.relative_to(ROOT))+'/'+name,source_hashes=hashes,
                note='All alternatives retained; logged events not independent trials; see admission.json',
                selection_note='No champion or production promotion; adaptive additions separately registered')
        (out/'ledger_complete.json').write_text(json.dumps(dict(events=5,independent_trial_count='not_estimated'),indent=2))
    verification=dict(status='complete_with_documented_strategy_rejections',execution_cases=29,tests=count,
        max_cash_error=float(cases.cash_max_error.max()),max_share_error=0,
        max_asset_pnl_error=float(cases.asset_pnl_max_error.max()),max_holdings=int(cases.max_holdings.max()),
        checks=checks,source_hashes=hashes,admission='NOT_ADMITTED')
    (out/'final_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
    manifest=json.loads((out/'manifest.json').read_text());manifest.update(status='complete',
        original_rounds_completed=3,supplementary_market_risk_cases=6,allocation_order_counterexamples=2,
        tests_passed=count,execution_cases=29,admission='NOT_ADMITTED',source_hashes=hashes,
        results='all_execution_cases.csv',report='docs/AUTONOMOUS_VALUE_ITERATIONS_20260913.md')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    artifacts=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name!='artifact_sha256.csv':
            artifacts.append(dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=sha(path)))
    pd.DataFrame(artifacts).to_csv(out/'artifact_sha256.csv',index=False)
    print(json.dumps({k:verification[k] for k in ['status','execution_cases','tests','max_cash_error','max_holdings','admission']}))


if __name__=='__main__':main()
