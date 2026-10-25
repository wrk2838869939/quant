#!/usr/bin/env python3
"""仅恢复报告序列化失败；核对交易函数未变，不重跑已成功的账户。"""
from pathlib import Path
import argparse
import ast
import json
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_left_side_rules as R
from run_classic_strategy_smoke import digest, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    m = json.loads((run / 'manifest.json').read_text())
    assert m['status'] == 'failed' and 'not JSON serializable' in m['error']
    old = run / 'sources/run_left_side_rules.py'
    new = Path(R.__file__)
    key = str(new.relative_to(ROOT.parent))
    assert digest(old) == m['source_hashes'][key]
    # 限定为这一处报告序列化修复；特征、策略、账本和参数不允许变动。
    replacement = ('fixed=json.dumps(dict(zip(others, key)))',
                   'fixed=json.dumps(dict(zip(others, key)), default=float)')
    assert old.read_text().replace(*replacement) == new.read_text()
    assert all(digest(ROOT.parent / p) == h for p, h in m['source_hashes'].items() if p != key)
    supplement = ROOT.parent / m['supplement']
    assert digest(supplement / 'manifest.json') == m['supplement_manifest_sha256']
    s = json.loads((supplement / 'manifest.json').read_text())
    assert all(digest(supplement / p) == h for p, h in s['hashes'].items())
    cfg = yaml.safe_load((run / 'sources/left_side_rules_v1.yaml').read_text())
    R.B.setup(ROOT.parent / cfg['loader_config'], run)
    tasks = json.loads((run / 'tasks.json').read_text())
    rows = []
    for name, params, mode in tasks:
        folder = run / 'accounts' / name
        row = json.loads((folder / 'metrics.json').read_text())
        assert row['id'] == name and row['mode'] == mode and all(row[k] == v for k, v in params.items())
        audit = json.loads((folder / 'audit.json').read_text())
        assert audit['no_negative_cash'] and audit['t_plus_one_pass'] and audit['official_limits_pass']
        for filename in ['daily.parquet', 'trades.parquet', 'events.parquet', 'holdings.parquet', 'signals.parquet', 'position_decisions.parquet', 'actions.parquet']:
            assert (folder / filename).exists()
        rows.append(row)
    assert (run / 'feature_prefix_check.json').exists() and (run / 'accounts/default/prefix_check.json').exists()
    report = R.report(run, cfg, sorted(rows, key=lambda r: r['id']))
    m['report_recovery'] = dict(original_error=m.pop('error'), original_execution_script_sha256=digest(old),
                                change='仅将参数三元组中 numpy 标量转换为 JSON 数值',
                                recovery_source_sha256=digest(Path(__file__)))
    m['source_hashes'][key] = digest(new)
    m['source_hashes'][str(Path(__file__).relative_to(ROOT.parent))] = digest(Path(__file__))
    m.update(status='complete', accounts=len(rows), report=str(report.relative_to(ROOT.parent)), report_sha256=digest(report))
    save_json(run / 'artifact_hashes.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file() and p.name not in ['manifest.json', 'artifact_hashes.json']})
    save_json(run / 'manifest.json', m)
    print('COMPLETE', report, 'accounts', len(rows), flush=True)


if __name__ == '__main__':
    main()
