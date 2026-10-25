"""封存校正后的完整研究、失败原件、验证回执及最终源码。"""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_selection_stability import digest, save


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    assert not (run / 'completion.json').exists()
    manifest = json.loads((run / 'manifest.json').read_text())
    execution = json.loads((run / 'execution_manifest.json').read_text())
    for name, value in manifest['source_sha256'].items(): assert digest(ROOT / name) == value
    for name, value in manifest['input_sha256'].items(): assert digest(name) == value
    assert digest(run / 'PROTOCOL.md') == manifest['protocol_sha256']
    assert digest(ROOT.parent / 'AGENTS.md') == manifest['root_agents_sha256']
    assert digest(ROOT / 'scripts/run_selection_stability.py') == execution['runner_sha256']
    assert digest(run / 'signal_hashes.json') == execution['signal_hashes_sha256']
    assert digest(run / 'execution_input_completion.json') == execution['inputs_receipt_sha256']
    for name, value in json.loads((run / 'signal_hashes.json').read_text()).items(): assert digest(run / name) == value
    data = json.loads((run / 'execution_input_completion.json').read_text())
    for name, value in data['input_sha256'].items(): assert digest(run / name) == value
    done = json.loads((run / 'execution_complete.json').read_text())
    verify = json.loads((run / 'verification.json').read_text())
    analysis = json.loads((run / 'analysis/verification.json').read_text())
    assert done['new_accounts'] == 183 and done['reused_baselines'] == 45
    assert analysis['accounts'] == 228 and analysis['all_asset_pnl_reconciled']
    assert len(verify['prefixes']) == 3 and verify['rank_buffer_first_fill_parities'] == 45
    assert verify['all_candidate_execution_and_risk_coverage'] and verify['corrected_top20_scope_checks'] == 452
    assert not list((run / 'accounts').glob('*/failure.json'))
    assert json.loads((run / 'scope_replay_verification.json').read_text())['replayed'] == 106
    assert json.loads((run / 'replay_preparation_smoke.json').read_text())['all_four_ledgers_exact']
    total_tests = 0
    for name, count in [('stability_tests.log', 8), ('alignment_tests.log', 1), ('scope_tests.log', 2),
                        ('startup_tests.log', 5), ('risk_budget_unit.log', 6), ('risk_budget_adversarial.log', 25)]:
        log = (run / name).read_text(); assert f'Ran {count} test' in log and log.rstrip().endswith('OK')
        total_tests += count
    sources = ['scripts/prepare_selection_stability_data.py', 'scripts/align_rank_entry_permission.py',
        'scripts/run_selection_stability.py', 'scripts/analyze_selection_stability.py', 'scripts/verify_selection_stability.py',
        'scripts/plot_selection_stability.py', 'scripts/diagnose_selection_scores.py', 'scripts/explain_selection_stability.py',
        'scripts/repair_selection_dividend.py', 'scripts/fix_selection_candidate_scope.py', 'scripts/audit_selection_scope_replay.py',
        'scripts/prepare_selection_stability_replay.py', 'scripts/seal_selection_stability.py',
        'scripts/audit_monthly_portfolio.py', 'scripts/run_risk_budget_tests.py', 'src/portfolio.py',
        'tests/test_rank_entry_alignment.py', 'tests/test_selection_scope.py', 'tests/test_startup_timing.py',
        'tests/test_risk_budget.py', 'tests/test_causal_portfolio_adversarial.py',
        'docs/SELECTION_STABILITY_RESEARCH_20260914.md']
    for name in sources:
        target = run / 'sources' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    hashes = {str(p.relative_to(run)): digest(p) for p in sorted(run.rglob('*')) if p.is_file()}
    save(run / 'completion.json', dict(status='complete', new_accounts=183, reused_accounts=45, final_trades=done['trades'],
        tests=total_tests, parent_signal_parities=4, parent_account_parities=3, account_prefixes=3,
        scope_corrected_replays=106, root_agents_unchanged=True, live_orders=False,
        research_candidate='rank_buffer_only_not_live_approved', artifact_sha256=hashes))
    for name, value in hashes.items(): assert digest(run / name) == value
    print(json.dumps(dict(artifacts=len(hashes), tests=total_tests, new_accounts=183, reused_accounts=45,
                         final_trades=done['trades'], all_hashes_verified=True), ensure_ascii=False))


if __name__ == '__main__': main()
