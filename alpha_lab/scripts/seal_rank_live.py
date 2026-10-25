"""封存排名候选实盘检验的来源、原始账本和未通过事项。"""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_selection_stability import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    assert not (run/'completion.json').exists()
    protocol = json.loads((run/'protocol.json').read_text())
    parent = Path(protocol['parent'])
    assert digest(parent/'completion.json')==protocol['parent_seal_sha256']
    for name, expected in json.loads((parent/'completion.json').read_text())['artifact_sha256'].items():
        assert digest(parent/name)==expected
    for name, expected in protocol['source_sha256'].items(): assert digest(ROOT/name)==expected
    assert digest(ROOT.parent/'AGENTS.md')==protocol['root_agents_sha256']
    done = json.loads((run/'execution_complete.json').read_text())
    verify = json.loads((run/'verification.json').read_text())
    assessment = json.loads((run/'analysis/assessment.json').read_text())
    assert done['accounts']==112 and done['trades']==9811 and done['all_reconciled']
    assert verify['rule_prefixes']==144 and len(verify['account_prefixes'])==3
    assert verify['parent_account_parities']==16
    assert not list((run/'accounts').glob('*/failure.json'))
    for name, count in [('manual_tests.log',57),('signal_tests.log',22)]:
        log = (run/name).read_text()
        assert f'Ran {count} test' in log and log.rstrip().endswith('OK')
    sources = list(protocol['source_sha256']) + ['scripts/analyze_rank_live.py', 'scripts/verify_rank_live.py',
        'scripts/seal_rank_live.py','scripts/build_selection_stability.py','scripts/analyze_selection_stability.py',
        'src/manual_execution.py','src/shadow_account.py','src/shadow_planner.py','src/shadow_inputs.py',
        'src/lifecycle_risk.py','src/portfolio.py','configs/shadow_execution.yaml','configs/manual_execution.yaml',
        'docs/RANK_LIVE_VALIDATION_20260914.md','README.md']
    sources += ['tests/'+name+'.py' for name in ['test_manual_continuation','test_manual_execution',
        'test_shadow_execution','test_shadow_inputs','test_execution_metadata','test_mail_outbox',
        'test_selection_stability','test_rank_entry_alignment','test_selection_scope','test_startup_timing','test_risk_budget']]
    for name in sorted(set(sources)):
        target = run/'sources'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    save(run/'deployment_decision.json',dict(candidate='rank10_confirm2',promoted=False,
        reason='收益一致性与防守筛查未通过，未取得新增样本外或真实成交证据',
        production_changed=False,real_account_initialized=False,email_sent=False,
        priorities=['组合共同风险与低频主动空仓/重入','人工漏单后的有限期重算验证','个人红利税对资金路径的反馈'],
        research_screens=assessment['predeclared_screens']))
    hashes = {str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*')) if p.is_file()}
    save(run/'completion.json',dict(status='complete',accounts=112,trades=9811,tests=79,
        rule_prefixes=144,account_prefixes=3,parent_account_parities=16,
        independent_oos=False,live_approved=False,root_agents_unchanged=True,artifact_sha256=hashes))
    for name, expected in hashes.items(): assert digest(run/name)==expected
    print(json.dumps(dict(artifacts=len(hashes),tests=79,accounts=112,all_hashes_verified=True),ensure_ascii=False))


if __name__ == '__main__': main()
