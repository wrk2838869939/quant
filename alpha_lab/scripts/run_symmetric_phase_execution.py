#!/usr/bin/env python3
"""对称阶段研究真实执行与独立账本核验：冻结输入、主板情景、完整倒序复跑。"""
import argparse
from copy import deepcopy
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_divide_selection_execution import make_matrices, run_case
from run_patient_value_execution import digest, json_write
from verify_divide_selection_execution import verify_case, require


def is_mainboard(code):
    """沪深主板代码空间；账户权限仅作为研究情景。"""
    return bool(re.fullmatch(r'(?:(?:000|001|002|003)[0-9]{3}\.SZ|(?:600|601|603|605)[0-9]{3}\.SH)', str(code)))


def execution_cases(cfg):
    """在执行前登记全部案例；迁移对照仅运行参考资金。"""
    policies = cfg['policies']
    cases = [dict(name=f'base_{capital:g}_{policy}', capital=capital, policy=policy,
                  reverse_input=False, training_universe='mainboard', execution_universe='mainboard')
             for capital in cfg['capitals'] for policy in policies]
    cases += [dict(name=f'reverse_{cfg["reference_capital"]:g}_{policy}',
                   capital=cfg['reference_capital'], policy=policy, reverse_input=True,
                   training_universe='mainboard', execution_universe='mainboard') for policy in policies]
    transfer = cfg.get('execution_transfer_policies', [f'transfer_{policy}' for policy in policies])
    cases += [dict(name=f'transfer_{cfg["reference_capital"]:g}_{policy}',
                   capital=cfg['reference_capital'], policy=policy, reverse_input=False,
                   training_universe='all_a', execution_universe='mainboard') for policy in transfer]
    if cfg.get('holding_variant'):
        variant = cfg['holding_variant']
        cases.append(dict(name=f'variant_{variant["capital"]:g}_{variant["policy"]}_{variant["name"]}',
                          capital=variant['capital'], policy=variant['policy'], reverse_input=False,
                          training_universe='mainboard', execution_universe='mainboard',
                          hold_variant=variant['name']))
    require(len({case['name'] for case in cases}) == len(cases), '案例名称重复')
    return cases


def verify_frozen_inputs(run, manifest):
    """逐进程检查配置、源码、预测、持有日历和执行矩阵未漂移。"""
    out = run / 'execution'
    for name, expected in manifest['source_hashes'].items():
        require(digest(out / 'sources' / name) == expected, f'源码快照漂移: {name}')
        require(digest(ROOT / name) == expected, f'当前源码漂移: {name}')
    for name, expected in manifest['input_hashes'].items():
        require(digest(run / name) == expected, f'冻结输入漂移: {name}')


def compare_all_csv(out, cfg):
    """独立倒序案例的所有CSV严格逐值一致，含年度、阶段及未知持有记录。"""
    rows = []
    for policy in cfg['policies']:
        base = out / f'base_{cfg["reference_capital"]:g}_{policy}'
        reverse = out / f'reverse_{cfg["reference_capital"]:g}_{policy}'
        files = sorted(path.name for path in base.glob('*.csv'))
        require(files == sorted(path.name for path in reverse.glob('*.csv')), '倒序文件集不一致')
        require(len(files) == 15, f'案例CSV产物不是15份: {policy}')
        for name in files:
            a, b = pd.read_csv(base / name), pd.read_csv(reverse / name)
            if name == 'yearly.csv':
                a, b = a.drop(columns='name'), b.drop(columns='name')
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            rows.append(dict(policy=policy, file=name, identical=True))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'candidate_order_invariance.csv', index=False)
    return frame


def independent_verify(run, cfg, manifest):
    """仅从冻结预测及执行CSV重建账户，不访问行情面板。"""
    out = run / 'execution'
    verify_frozen_inputs(run, manifest)
    predictions = pd.read_parquet(run / 'execution_predictions.parquet')
    execution_cfg = yaml.safe_load((out / 'execution_config.yaml').read_text())
    rows = []
    for case in manifest['cases']:
        folder = out / case['name']
        for name in ['trades', 'holdings', 'selection_trace', 'actual_entry_priorities']:
            frame = pd.read_csv(folder / f'{name}.csv')
            require(frame.ts_code.dropna().map(is_mainboard).all(), f'{case["name"]}: 非主板{name}')
        rows.append(verify_case(folder, predictions, cfg, execution_cfg))
    compared = compare_all_csv(out, cfg)
    report = dict(status='complete', market_data_read=False, test_market_data_used=False,
                  cases=rows, reverse_csv_count=len(compared),
                  source_verification='snapshot_and_current_sha256',
                  checks=['independent_per_stock_daily_shares', 'per_trade_and_daily_cash',
                          'receivables_and_equity', 'retained_minimum_and_no_topups',
                          'frozen_priority_separate_from_mu', 'economic_reservation_filled',
                          'known_entry_phase_and_episode_pnl', 'all_15_reverse_csv_exact_equality',
                          'mainboard_only_real_execution', 'unchanged_monthly_hold_allowed'])
    json_write(out / 'independent_execution_verification.json', report)
    return report


def prepare(run, cfg, config_path, cases, variant_hold=None, variant_sources=()):
    """先登记并冻结全部情景，原持有日历按字节复制，缓存仅补缺。"""
    out = run / 'execution'
    out.mkdir(exist_ok=True)
    parent = ROOT / cfg['execution_parent']
    require(pd.Timestamp(cfg['execution'][1]) < pd.Timestamp('2024-01-01'), '禁止TEST行情')
    require(cfg['retail']['selection_first'], '必须先选名单后定股数')
    require(digest(config_path) == digest(run / 'config.yaml'), '传入配置与研究冻结配置不一致')
    sources = [Path(__file__), config_path, ROOT / 'src/portfolio.py',
               ROOT / 'src/retail_rebalance.py', ROOT / 'src/selection_first.py',
               ROOT / 'scripts/audit_monthly_portfolio.py',
               ROOT / 'scripts/run_patient_value_execution.py',
               ROOT / 'scripts/run_divide_selection_execution.py',
               ROOT / 'scripts/verify_divide_selection_execution.py',
               ROOT / 'scripts/prepare_monthly_audit.py',
               ROOT / 'configs/corporate_action_components.yaml']
    sources += list(variant_sources)
    if cfg.get('holding_variant'):
        require(variant_hold is not None and variant_hold.exists(), '第21case持有日历未就绪')
    manifest = dict(status='registered', fresh_holdout=False, test_market_data_used=False,
        cases=cases, source_hashes={str(path.relative_to(ROOT)): digest(path) for path in sources},
        input_hashes={'config.yaml': digest(run / 'config.yaml'),
                      'execution_predictions.parquet': digest(run / 'execution_predictions.parquet')},
        parent_hold_hash=digest(parent / 'monthly_hold_allowed.parquet'),
        account_scope='research_scenario_only_not_user_permission_declaration',
        feature_scope='existing_all_a_pit_rank_unchanged',
        order_test='independent_process_reverse_adapter_rows_only_all_15_csv',
        stage_attribution='known_entry_phase_fixed_per_real_position_episode',
        hold_semantics='original_monthly_hold_allowed_complete_unchanged')
    json_write(out / 'manifest.json', manifest)
    for path in sources:
        target = out / 'sources' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(parent / 'monthly_hold_allowed.parquet', out / 'monthly_hold_allowed.parquet')
    hold = pd.read_parquet(out / 'monthly_hold_allowed.parquet')
    if cfg.get('holding_variant'):
        variant_frame = pd.read_parquet(variant_hold)
        require(variant_frame.index.equals(hold.index) and variant_frame.columns.equals(hold.columns),
                '变体hold必须完整保留原始日历与列集合')
        variant_dest = out / 'keep_failure_memory_hold_allowed.parquet'
        shutil.copyfile(variant_hold, variant_dest)
        manifest['input_hashes'][f'execution/{variant_dest.name}'] = digest(variant_dest)
        manifest['variant_hold_source'] = str(variant_hold)
        manifest['variant_hold_hash'] = digest(variant_hold)
    predictions = pd.read_parquet(run / 'execution_predictions.parquet')
    require(predictions.ts_code.str.fullmatch(cfg['mainboard_pattern']).all(), '预测超出配置主板空间')
    require(predictions.ts_code.map(is_mainboard).all(), '预测包含非主板代码，拒绝执行')
    matrix_cfg = deepcopy(cfg)
    matrix_cfg['policies'] = list(dict.fromkeys(case['policy'] for case in cases))
    matrices = make_matrices(predictions, hold, matrix_cfg)
    p = yaml.safe_load((parent / 'execution_config.yaml').read_text())
    p['valid'] = cfg['execution']
    p['portfolio']['top_n'] = cfg['retail']['max_positions']
    p['data_workers'] = 2
    (out / 'execution_config.yaml').write_text(yaml.safe_dump(p, allow_unicode=True, sort_keys=False))
    codes, coverage = set(), []
    for policy, (signals, forecasts) in matrices.items():
        require(not signals.loc[:, ~signals.columns.map(is_mainboard)].notna().any().any(), '非主板有效信号')
        signals.to_parquet(out / f'{policy}_signals.parquet')
        forecasts.to_parquet(out / f'{policy}_point_forecasts.parquet')
        for date, row in signals.iterrows():
            ranked = row.dropna().sort_values(ascending=False, kind='stable')
            codes.update(ranked.head(cfg['retail']['candidate_pool']).index)
            coverage.append(dict(policy=policy, date=date, admitted=len(ranked),
                                 candidate_pool=min(len(ranked), cfg['retail']['candidate_pool'])))
    manifest['calendar_only_anchor'] = not bool(codes)
    if not codes:
        codes.add('000001.SZ')
    require(all(map(is_mainboard, codes)), '行情代码并集含非主板')
    pd.DataFrame({'ts_code': sorted(codes)}).to_csv(out / 'execution_codes.csv', index=False)
    pd.DataFrame(coverage).to_csv(out / 'entry_coverage.csv', index=False)
    cache = Path((parent / 'execution_cache.txt').read_text().strip())
    missing = [code for code in sorted(codes)
               if any(not (cache / api / f'{code}.parquet').exists() for api in ['dividend', 'stk_limit'])]
    pd.DataFrame({'ts_code': missing}).to_csv(out / 'missing_cache_codes.csv', index=False)
    print(f'{len(cases)}案例已登记；主板候选并集{len(codes)}，缺缓存{len(missing)}', flush=True)
    if missing:
        from prepare_monthly_audit import fetch_execution_data
        cache = fetch_execution_data(missing, tuple(map(pd.Timestamp, cfg['execution'])), p)
    (out / 'execution_cache.txt').write_text(str(cache))
    frozen = ['monthly_hold_allowed.parquet', 'execution_config.yaml', 'execution_codes.csv', 'execution_cache.txt']
    frozen += [f'{policy}_{suffix}.parquet' for policy in matrices for suffix in ['signals', 'point_forecasts']]
    manifest['input_hashes'].update({f'execution/{name}': digest(out / name) for name in frozen})
    require(digest(out / 'monthly_hold_allowed.parquet') == manifest['parent_hold_hash'], '持有日历字节发生变化')
    manifest.update(status='prepared', execution_codes=len(codes), downloaded_missing_codes=len(missing))
    json_write(out / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--case-name')
    parser.add_argument('--variant-hold', type=Path)
    parser.add_argument('--variant-source', type=Path, action='append', default=[])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    config_path = (args.config or run / 'config.yaml').resolve()
    cfg = yaml.safe_load(config_path.read_text())
    cases, out = execution_cases(cfg), run / 'execution'
    manifest_path = out / 'manifest.json'
    if args.case_name or args.verify_only:
        manifest = json.loads(manifest_path.read_text())
        require(cases == manifest['cases'], '案例矩阵漂移')
        verify_frozen_inputs(run, manifest)
        if args.verify_only:
            require(manifest['status'] == 'complete', '案例未完成')
            independent_verify(run, cfg, manifest)
        else:
            case = next(case for case in cases if case['name'] == args.case_name)
            (out / case['name']).mkdir(exist_ok=True)
            if case.get('hold_variant'):
                case_run = out / 'variant_workspace'
                case_out = case_run / 'execution'
                case_out.mkdir(parents=True, exist_ok=True)
                for path in [run / 'execution_predictions.parquet']:
                    target = case_run / path.name
                    if not target.exists():
                        target.symlink_to(path)
                for name in ['execution_config.yaml', 'execution_cache.txt', 'execution_codes.csv',
                             f'{case["policy"]}_signals.parquet', f'{case["policy"]}_point_forecasts.parquet']:
                    target = case_out / name
                    if not target.exists():
                        target.symlink_to(out / name)
                hold_link = case_out / 'monthly_hold_allowed.parquet'
                if not hold_link.exists():
                    hold_link.symlink_to(out / 'keep_failure_memory_hold_allowed.parquet')
                case_link = case_out / case['name']
                if not case_link.exists():
                    case_link.symlink_to(out / case['name'], target_is_directory=True)
                run_case(case_run, cfg, case)
            else:
                run_case(run, cfg, case)
        return
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        require(manifest['status'] in ['prepared', 'running'], '已有执行不可覆盖，使用新run')
        require(cases == manifest['cases'], '案例矩阵漂移')
        verify_frozen_inputs(run, manifest)
    else:
        manifest = prepare(run, cfg, config_path, cases,
                           args.variant_hold or run / 'holding_variant/monthly_hold_allowed.parquet',
                           [path.resolve() for path in args.variant_source])
    if args.prepare_only:
        return
    summaries, failures, yearly = [], [], []
    for i, case in enumerate(cases, 1):
        folder = out / case['name']
        folder.mkdir(exist_ok=True)
        print(f'[{i}/{len(cases)}] {case["name"]}', flush=True)
        with (folder / 'run.log').open('w') as log:
            completed = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                '--run', str(run), '--config', str(config_path), '--case-name', case['name']],
                stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            error = (folder / 'run.log').read_text().splitlines()[-1]
            failures.append(dict(name=case['name'], error=error, returncode=completed.returncode))
            print(error, flush=True)
        else:
            result = json.loads((folder / 'summary.json').read_text())
            summaries.append(result)
            yearly.append(pd.read_csv(folder / 'yearly.csv'))
            print(json.dumps({key: result[key] for key in ['name', 'total', 'mdd', 'n_trades']}, ensure_ascii=False), flush=True)
        pd.DataFrame(summaries).to_csv(out / 'execution_summary.csv', index=False)
        pd.DataFrame(failures, columns=['name', 'error', 'returncode']).to_csv(out / 'execution_failures.csv', index=False)
        manifest.update(status='running', completed_cases=len(summaries), failed_cases=len(failures))
        json_write(manifest_path, manifest)
        if failures:
            manifest.update(status='failed')
            json_write(manifest_path, manifest)
            raise RuntimeError('真实执行异常，停止剩余案例，禁止绕过公司行动检查')
    pd.concat(yearly, ignore_index=True).to_csv(out / 'execution_yearly.csv', index=False)
    try:
        independent_verify(run, cfg, manifest)
    except Exception:
        manifest.update(status='verification_failed')
        json_write(manifest_path, manifest)
        raise
    manifest.update(status='complete', order_invariance_pass=True, independent_verification_pass=True)
    json_write(manifest_path, manifest)


if __name__ == '__main__':
    main()
