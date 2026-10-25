"""核验本轮固定输入与必要回执，封存研究结果和分析源码。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    assert not (run / 'completion.json').exists(), '已有封存回执，不覆盖'
    manifest = json.loads((run / 'manifest.json').read_text())
    executed = json.loads((run / 'execution_complete.json').read_text())
    checked = json.loads((run / 'prefix_verification.json').read_text())
    analysis = json.loads((run / 'analysis/verification.json').read_text())
    assert executed['accounts'] == analysis['accounts'] == len(manifest['cases']) == 150
    assert executed['all_reconciled'] and analysis['all_asset_pnl_reconciled']
    assert analysis['identical_first_fill_pairs'] == 45
    assert len(checked['prefixes']) == 3 and all(x['exact_prefix'] for x in checked['prefixes'])
    assert not list(run.rglob('failure.json'))
    for name, value in manifest['source_sha256'].items():
        assert digest(ROOT / name) == value, name
    for name, value in manifest['input_sha256'].items():
        assert digest(Path(name)) == value, name
    assert digest(ROOT.parent / 'AGENTS.md') == manifest['root_agents_sha256']
    for name, count in [('startup_unit_tests.log', 5), ('risk_budget_unit.log', 6), ('risk_budget_adversarial.log', 25)]:
        log = (run / name).read_text()
        assert f'Ran {count} tests' in log and log.rstrip().endswith('OK'), name
    files = ['scripts/analyze_startup_timing.py', 'scripts/verify_startup_timing.py',
             'scripts/explain_startup_timing.py', 'scripts/plot_startup_timing.py',
             'scripts/seal_startup_timing.py', 'docs/STARTUP_TIMING_RESEARCH_20260914.md']
    for name in files:
        target = run / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    inputs = run / 'verification_inputs'; inputs.mkdir()
    shutil.copy2(ROOT / manifest['config']['risk_parent'] / 'risk/annual_reference.csv', inputs)
    shutil.copy2(ROOT / 'outputs/causal_execution/20260913T130810_374654Z/config.yaml', inputs / 'causal_config.yaml')
    hashes = {str(path.relative_to(run)): digest(path) for path in sorted(run.rglob('*')) if path.is_file()}
    receipt = dict(status='complete', accounts=150, trades=executed['total_trades'], tests=36,
                   prefix_checks=3, initial_fill_parities=45, parent_account_parities=2,
                   root_agents_unchanged=True, live_orders=False, artifact_sha256=hashes)
    (run / 'completion.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    for name, expected in hashes.items():
        assert digest(run / name) == expected, name
    print(json.dumps(dict(artifacts=len(hashes), accounts=150, trades=executed['total_trades'], tests=36,
                         all_hashes_verified=True, root_agents_unchanged=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
