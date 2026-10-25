"""封存机制实验和邻域检验；不把最低筛选通过误写为有效防守。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run.resolve()
    assert not (run/'completion.json').exists(),'不覆盖既有封存'
    protocol=read(run/'protocol.json')
    for name,value in protocol['source_sha256'].items():
        assert digest(ROOT/name)==value,name
    assert digest(ROOT.parent/'AGENTS.md')==protocol['root_agents_sha256']
    parent=Path(protocol['parent'])
    for name,value in read(parent/'completion.json')['artifact_sha256'].items():
        assert digest(parent/name)==value,name
    for name,value in protocol.get('dependency_sha256',{}).items():
        assert digest(Path(name))==value,name
    if 'parent_seal_sha256' in protocol:
        assert digest(parent/'completion.json')==protocol['parent_seal_sha256']
    if 'mechanism_parent' in protocol:
        mechanism=Path(protocol['mechanism_parent'])
        for name,value in read(mechanism/'completion.json')['artifact_sha256'].items():
            assert digest(mechanism/name)==value,name
    execution=read(run/'execution_complete.json');verification=read(run/'verification.json')
    metrics=read(run/'metrics_audit/verification.json')
    assert execution['accounts']==80 and execution['all_reconciled']
    assert verification['accounts']==80 and verification['parent_parities']==16
    assert len(verification['prefixes'])==4 and all(x['all_four_ledgers_exact'] for x in verification['prefixes'])
    assert verification['cash_and_share_reconciliation'] and verification['root_agents_unchanged']
    assert metrics['summary_rows_verified']==80 and metrics['rank_accounts_verified']==16
    test=Path('/tmp/cash_economics_tests_v2.log').read_text()
    assert 'Ran 9 tests' in test and test.rstrip().endswith('OK')
    extra=['src/cash_economics_state.py','src/robust_cash_state.py','scripts/diagnose_cash_economics.py',
        'scripts/attribute_restore_roundtrips.py','scripts/research_cash_economics.py','scripts/research_cash_robustness.py',
        'scripts/analyze_cash_economics.py','scripts/verify_cash_economics.py','scripts/verify_cash_robustness.py',
        'scripts/verify_cash_economics_metrics.py','scripts/plot_cash_economics.py','scripts/seal_cash_economics.py',
        'tests/test_cash_economics.py','docs/CASH_ECONOMICS_RESEARCH_20260914.md','README.md']
    for name in sorted(set(extra)|set(protocol['source_sha256'])):
        target=run/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    logs=run/'logs';logs.mkdir(exist_ok=True)
    prefix='cash_robustness' if 'mechanism_parent' in protocol else 'cash_economics'
    for suffix in ['execution','analysis','analysis_final','verification','metrics','metrics_final']:
        name=f'{prefix}_{suffix}.log';shutil.copy2(Path('/tmp')/name,logs/name)
    shutil.copy2('/tmp/cash_economics_tests_v2.log',logs/'cash_economics_tests_v2.log')
    if 'mechanism_parent' not in protocol:
        for name in ['diagnosis_20260914','diagnosis_20260914_v2','roundtrips_20260914','roundtrips_20260914_v2','roundtrips_20260914_v3']:
            shutil.copytree(ROOT/'outputs/cash_economics'/name,run/'initial_diagnostics'/name)
        for name in ['cash_economics_diagnosis.log','cash_economics_roundtrips.log','cash_economics_roundtrips_v2.log',
                     'cash_economics_roundtrips_v3.log','cash_economics_new_roundtrips.log','cash_economics_new_diagnosis.log']:
            shutil.copy2(Path('/tmp')/name,logs/name)
        # 有无全退分支的历史行为一致，不能据此宣告全退有效。
        import pandas as pd
        for cohort in ['continued','cold']:
            for day in [1,8,15,22]:
                for cost in [0,30]:
                    for key in ['daily','trades','holdings','events']:
                        a=run/f'accounts/{cohort}_{day:02d}_staged_cost{cost}_fast_restore_trend_break/{key}.parquet'
                        b=run/f'accounts/{cohort}_{day:02d}_staged_cost{cost}_fast_restore_trend_half/{key}.parquet'
                        pd.testing.assert_frame_equal(pd.read_parquet(a),pd.read_parquet(b),check_exact=True)
        (run/'full_exit_ablation.json').write_text(json.dumps(dict(pairs=16,all_four_ledgers_exact=True)))
    decision=dict(economic_harm_reduced=True,central_screen_passed=True,robustness_passed=False,
        predictive_defense_proven=False,production_changed=False,live_approved=False,
        independent_oos=False,orders_or_emails_sent=False,
        note='中央参数仅保留研究对照；恢复/风险状态工程留待小资金测试，不是本轮阻断。')
    (run/'research_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    artifact={str(path.relative_to(run)):digest(path) for path in sorted(run.rglob('*')) if path.is_file()}
    receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),execution=execution,verification=verification,
        metric_accounts=80,new_state_tests=9,independent_agent_review_this_turn=False,
        research_decision=decision,parent_completion_sha256=digest(parent/'completion.json'),artifact_sha256=artifact)
    if 'mechanism_parent' in protocol:
        receipt['mechanism_completion_sha256']=digest(Path(protocol['mechanism_parent'])/'completion.json')
    (run/'completion.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    for name,value in read(run/'completion.json')['artifact_sha256'].items():
        assert digest(run/name)==value,name
    print(json.dumps(dict(run=str(run),artifacts_verified=len(artifact),live_approved=False)))


if __name__=='__main__':main()
