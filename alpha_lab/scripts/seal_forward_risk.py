"""封存修复后的风险证据与两轮预算账本，保留失败运行。"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def check_seal(path):
    for name,value in read(path/'completion.json')['artifact_sha256'].items():
        assert digest(path/name)==value,name


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run.resolve();assert not (run/'completion.json').exists()
    cfg=read(run/'protocol.json');is_evidence='families' in cfg
    for name,value in cfg['source_sha256'].items():assert digest(ROOT/name)==value,name
    assert digest(ROOT.parent/'AGENTS.md')==cfg['root_agents_sha256']
    shared=['docs/FORWARD_RISK_EVIDENCE_20260914.md','docs/DATA_SCHEMA.md','README.md',
        'configs/monthly_predicted_risk_candidate.yaml','scripts/seal_forward_risk.py']
    logs=[]
    if is_evidence:
        for name,value in cfg['input_sha256'].items():assert digest(Path(name))==value,name
        check_seal(Path(cfg['parent']))
        assert digest(Path(cfg['parent'])/'completion.json')==cfg['parent_completion_sha256']
        formula=read(run/'independent_formula_checks.json')
        assert formula['weekly_labels_reconstructed']==601 and formula['parameter_sets_reconstructed']==7
        assert formula['training_coefficients_unchanged_after_2021_repair'] and formula['early_2022_prediction_restored']
        assert read(run/'data_audit.json')['missing_market_sessions']==0
        assert read(run/'acceptance.json')['eligible']==[]
        checks=read(run/'verification.json')
        assert len(checks['full_market_feature_prefixes'])==2 and checks['root_agents_unchanged']
        extra=['scripts/transfer_forward_risk.py','scripts/verify_forward_risk.py','scripts/plot_forward_risk.py','tests/test_forward_risk_evidence.py']
        logs=['forward_risk_tests.log','forward_risk_execution.log','forward_risk_execution_repaired.log',
              'forward_risk_formula.log','forward_risk_transfer.log','forward_risk_transfer_repaired.log']
        test=Path('/tmp/forward_risk_tests.log').read_text();assert 'Ran 5 tests' in test and test.rstrip().endswith('OK')
        shutil.copytree(ROOT/'outputs/forward_risk/repair_20211213',run/'repair_20211213')
        shutil.copytree(ROOT/'outputs/forward_risk/20260914T103508_365515Z',run/'superseded_initial_evidence')
        failed=ROOT/'outputs/predicted_budget/20260914T103939_688190Z'
        shutil.copytree(failed,run/'aborted_initial_budget')
        old_source=(ROOT/'scripts/research_predicted_budget.py').read_text().replace('20260914T104245_763609Z','20260914T103508_365515Z')
        expected=read(failed/'protocol.json')['source_sha256']['scripts/research_predicted_budget.py']
        assert hashlib.sha256(old_source.encode()).hexdigest()==expected
        (run/'aborted_initial_budget/runner_at_failure.py').write_text(old_source)
        receipt=dict(kind='forward_risk_evidence',checks=checks,formula_checks=formula,cash_timing_eligible=[],
            canonical_panel_modified=False,repair_projection_applied=True,new_state_tests=5)
    else:
        evidence=Path(cfg['evidence']);check_seal(evidence)
        for name,value in cfg['evidence_sha256'].items():assert digest(evidence/name)==value,name
        execution=read(run/'execution_complete.json');checks=read(run/'verification.json')
        metrics=read(run/'metrics_audit/verification.json')
        assert execution['accounts']==48 and execution['all_reconciled']
        assert checks['parent_parities']==16 and len(checks['prefixes'])==3
        assert all(x['all_four_ledgers_exact'] for x in checks['prefixes'])
        assert metrics['summary_rows_verified']==48 and metrics['rank_accounts_verified']==16
        monthly='weekly_comparator' in cfg
        if monthly:
            check_seal(Path(cfg['weekly_comparator']))
            assert digest(Path(cfg['weekly_comparator'])/'protocol.json')==cfg['weekly_protocol_sha256']
        extra=['scripts/analyze_predicted_budget.py','scripts/verify_predicted_budget.py','scripts/verify_monthly_predicted_budget.py',
            'scripts/verify_cash_economics_metrics.py','tests/test_predicted_risk_budget.py','tests/test_targeted_restore.py']
        prefix='monthly_predicted_budget' if monthly else 'predicted_budget'
        logs=[f'{prefix}_{suffix}.log' for suffix in ['execution' if monthly else 'execution_repaired','analysis','verification','metrics']]
        logs+=['predicted_budget_tests.log','predicted_budget_restore_regression.log']
        for name,n in [('predicted_budget_tests.log',3),('predicted_budget_restore_regression.log',6)]:
            content=(Path('/tmp')/name).read_text();assert f'Ran {n} tests' in content and content.rstrip().endswith('OK')
        receipt=dict(kind='monthly_budget' if monthly else 'weekly_budget',execution=execution,checks=checks,
            metric_accounts=48,new_state_tests=3,restore_regression_tests=6,
            evidence_completion_sha256=digest(evidence/'completion.json'))
    for name in sorted(set(cfg['source_sha256'])|set(shared+extra)):
        target=run/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    (run/'logs').mkdir(exist_ok=True)
    for name in logs:shutil.copy2(Path('/tmp')/name,run/'logs'/name)
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),independent_oos=False,
        production_changed=False,orders_or_emails_sent=False,independent_agent_review_this_turn=False,
        candidate='monthly_50_percent_risk_budget_research_only')
    artifact={str(path.relative_to(run)):digest(path) for path in sorted(run.rglob('*')) if path.is_file()}
    receipt['artifact_sha256']=artifact
    (run/'completion.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    check_seal(run)
    print(json.dumps(dict(run=str(run),artifacts_verified=len(artifact),production_changed=False)))


if __name__=='__main__':main()
