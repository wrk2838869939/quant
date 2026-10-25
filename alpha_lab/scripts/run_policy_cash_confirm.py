#!/usr/bin/env python3
"""P3结果后的自适应追加：月度市场警报与可观察个股趋势失败共同触发提前退出。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_patient_value_execution import digest, json_write
from verify_divide_selection_execution import require
import run_policy_cash_execution as cash


def confirmed_alarm_hold(original, risk, dates, codes, trend, observable):
    """仅覆写有覆盖且当次已知的共同证据；原失败退出、未知与无覆盖列保留。"""
    dates, codes = pd.DatetimeIndex(dates), pd.Index(codes)
    require(dates.max() < pd.Timestamp('2024-01-01'), '禁止TEST日期')
    require(not dates.has_duplicates and not codes.has_duplicates, '原始证据日期或代码重复')
    review_indices = dates.get_indexer(original.index)
    require((review_indices >= 0).all(), '原复核日历不在证据内')
    common = original.columns.intersection(codes)
    columns = codes.get_indexer(common)
    raw = pd.DataFrame(trend[np.ix_(review_indices, columns)], index=original.index, columns=common)
    known = pd.DataFrame(observable[np.ix_(review_indices, columns)], index=original.index, columns=common)
    gate = risk.reindex(original.index)
    require(gate.isin([0, 1]).all(), '月度警报缺失')
    trigger = known & ~raw
    trigger.loc[gate.eq(1), :] = False
    changed = trigger & ~original[common].eq(False)
    result = original.astype(float)
    result.loc[:, common] = original[common].astype(float).mask(trigger, 0.)
    records = []
    for date, row in changed.iterrows():
        for code in row.index[row]:
            records.append(dict(date=date, ts_code=code, market_allowed=int(gate.loc[date]),
                observable=bool(known.loc[date, code]), raw_trend=bool(raw.loc[date, code]),
                original_hold=original.loc[date, code], new_hold=False))
    evidence = pd.DataFrame(records, columns=['date', 'ts_code', 'market_allowed', 'observable', 'raw_trend', 'original_hold', 'new_hold'])
    # Parquet使用唯一数值编码；混合bool/float列不能交给Arrow隐式推断。
    return result.astype(float), evidence


def prepare(run):
    out, parent = run / 'cash_confirm', run / 'cash_execution'
    registration = json.loads((out / 'registration.json').read_text())
    require(registration['adaptive_followup_after_P3'], '必须声明看过P3后的追加假设')
    for name, expected in registration['input_hashes'].items():
        require(digest(ROOT / name) == expected, f'登记输入漂移: {name}')
    prior = json.loads((parent / 'manifest.json').read_text())
    require(prior['status'] == 'complete', '原16case未完成')
    sources = dict(prior['source_hashes'])
    for path in [Path(__file__), ROOT / 'tests/test_policy_cash_confirm.py']:
        sources[str(path.relative_to(ROOT))] = digest(path)
    manifest = dict(status='registered', adaptive_followup_after_P3=True, fresh_holdout=False,
        test_market_data_used=False, cases=registration['cases'], source_hashes=sources, input_hashes={},
        proxy_anchor='trigger_signal_date_adjusted_close; subsequent_5_10_20_market_sessions',
        proxy_not_executable_counterfactual=True, original_cash16_unchanged=True)
    json_write(out / 'manifest.json', manifest)
    for name, expected in sources.items():
        require(digest(ROOT / name) == expected, f'源代码漂移: {name}')
        target = out / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    files = ['execution_config.yaml', 'execution_codes.csv', 'execution_cache.txt',
             'low_turnover_signals.parquet', 'low_turnover_point_forecasts.parquet',
             'execution_predictions.parquet', 'risk_plans.csv', 'config.yaml']
    for name in files:
        shutil.copyfile(parent / name, out / name)
    shutil.copyfile(parent / 'monthly_hold_allowed.parquet', out / 'original_hold_allowed.parquet')
    (out / 'policy_market.npz').symlink_to(run / 'policy_market.npz')
    risk = pd.read_csv(out / 'risk_plans.csv', index_col=0, parse_dates=True).extreme_recovery
    original = pd.read_parquet(out / 'original_hold_allowed.parquet')
    with np.load(out / 'policy_market.npz', allow_pickle=False) as data:
        holds, evidence = confirmed_alarm_hold(original, risk, data['dates'], data['codes'], data['trend'], data['observable'])
        selected_dates = pd.DatetimeIndex(data['dates'])[data['review']]
        require(original.index.isin(selected_dates).all(), '追加策略改变原月度reviewclock')
    holds.to_parquet(out / 'monthly_hold_allowed.parquet')
    evidence.to_csv(out / 'hold_changed_cells.csv', index=False)
    files += ['original_hold_allowed.parquet', 'monthly_hold_allowed.parquet', 'hold_changed_cells.csv', 'registration.json', 'policy_market.npz']
    manifest['input_hashes'] = {name: digest(out / name) for name in files}
    manifest.update(status='prepared', changed_hold_cells=len(evidence), changed_stocks=evidence.ts_code.nunique())
    json_write(out / 'manifest.json', manifest)
    workspace = out / 'workspace'
    workspace.mkdir()
    (workspace / 'cash_execution').symlink_to(out, target_is_directory=True)
    # 冻结旧核验器的共同窗口比较只读这些无门槛对照，绝不重新写原案例。
    for capital in [20000, 50000, 100000]:
        (out / f'base_{capital}_none').symlink_to(parent / f'base_{capital}_none', target_is_directory=True)
    (out / 'cost2_50000_none').symlink_to(parent / 'cost2_50000_none', target_is_directory=True)
    return manifest


def verify_triggers_and_paths(run, manifest):
    """独立逐格确认新增退出原因，并单列实际请求后的固定价格代理路径。"""
    out = run / 'cash_confirm'
    original = pd.read_parquet(out / 'original_hold_allowed.parquet')
    current = pd.read_parquet(out / 'monthly_hold_allowed.parquet')
    risk = pd.read_csv(out / 'risk_plans.csv', index_col=0, parse_dates=True).extreme_recovery
    with np.load(out / 'policy_market.npz', allow_pickle=False) as data:
        dates, codes = pd.DatetimeIndex(data['dates']), pd.Index(data['codes'])
        raw, known, close = data['trend'], data['observable'], data['hfq_close']
        di, ci = {d: i for i, d in enumerate(dates)}, {c: i for i, c in enumerate(codes)}
        changed = set()
        for date in original.index:
            for code in original.columns:
                prior = original.loc[date, code]
                value = current.loc[date, code]
                extra = code in ci and risk.loc[date] == 0 and bool(known[di[date], ci[code]]) and not bool(raw[di[date], ci[code]])
                expected = False if extra else prior
                require((pd.isna(value) and pd.isna(expected)) or value == expected, '独立逐格持有矩阵核验失败')
                if extra and not (pd.notna(prior) and not bool(prior)):
                    changed.add((date, code))
        recorded = pd.read_csv(out / 'hold_changed_cells.csv', parse_dates=['date'])
        require(changed == set(zip(recorded.date, recorded.ts_code)), '新增单元证据与独立重建不同')
        requests, proxies, fills = [], [], []
        for case in manifest['cases']:
            folder = out / case['name']
            events = pd.read_csv(folder / 'events.csv', parse_dates=['date', 'signal_date'])
            trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'signal_date'])
            episodes = pd.read_csv(folder / 'entry_phase_episodes.csv', parse_dates=['entry_date', 'close_date'])
            extra_requests = events.loc[events.kind.eq('hold_exit_requested') & events.reason.eq('hold_false')].copy()
            for request in extra_requests.itertuples():
                key = (request.signal_date, request.ts_code)
                if key not in changed:
                    continue
                require(request.signal_date in original.index and request.date > request.signal_date, '提前退出不在原月度时序')
                row, column = di[request.signal_date], ci[request.ts_code]
                require(risk.loc[request.signal_date] == 0 and known[row, column] and not raw[row, column], '提前退出缺少共同可观察证据')
                active = episodes.loc[episodes.ts_code.eq(request.ts_code) & episodes.entry_date.le(request.date)
                                      & (episodes.close_date.isna() | episodes.close_date.ge(request.date))]
                require(len(active) == 1, '提前退出无法定位唯一真实持仓段')
                episode = active.iloc[0]
                end = episode.close_date if pd.notna(episode.close_date) else dates[-1]
                actual = trades.loc[trades.side.eq('sell') & trades.ts_code.eq(request.ts_code)
                    & trades.signal_date.eq(request.signal_date) & trades.date.between(request.date, end)]
                requests.append(dict(case=case['name'], signal_date=request.signal_date, request_date=request.date,
                    ts_code=request.ts_code, old_shares=request.old_shares, original_hold=original.loc[key],
                    market_allowed=0, observable=True, raw_trend=False, episode_id=episode.episode_id,
                    first_actual_fill=actual.date.min(), last_actual_fill=actual.date.max(), actual_sold_shares=actual.shares.sum(),
                    sell_deferred_events=int((events.kind.eq('sell_deferred') & events.ts_code.eq(request.ts_code)
                                              & events.date.between(request.date, end)).sum()),
                    episode_closed=bool(pd.notna(episode.close_date))))
                for trade in actual.to_dict('records'):
                    fills.append(dict(case=case['name'], request_date=request.date, **trade))
                anchor = close[row, column]
                require(np.isfinite(anchor) and anchor > 0, '可观察触发缺少有效价格锚点')
                for horizon in [5, 10, 20]:
                    stop = row + horizon
                    values = close[row+1:stop+1, column]
                    complete = stop < len(dates) and len(values) == horizon and np.isfinite(values).all() and (values > 0).all()
                    ratios = values / anchor - 1 if complete else np.array([])
                    proxies.append(dict(case=case['name'], ts_code=request.ts_code, signal_date=request.signal_date,
                        request_date=request.date, horizon=horizon, proxy_anchor='signal_close',
                        anchor_adjusted_close=anchor, label_end=dates[stop] if stop < len(dates) else pd.NaT,
                        complete=complete, close_return=ratios[-1] if complete else np.nan,
                        min_close_return=ratios.min() if complete else np.nan, max_close_return=ratios.max() if complete else np.nan,
                        executable_counterfactual=False))
    pd.DataFrame(requests).to_csv(out / 'additional_exit_requests.csv', index=False)
    pd.DataFrame(fills).to_csv(out / 'additional_exit_fills.csv', index=False)
    pd.DataFrame(proxies).to_csv(out / 'additional_exit_price_proxies.csv', index=False)
    require(len(requests) > 0, '策略没有实际新增退出，不能宣称已检验持仓确认')
    for name, frame in [('requests', pd.DataFrame(requests)), ('proxies', pd.DataFrame(proxies)), ('fills', pd.DataFrame(fills))]:
        a = frame.loc[frame.case.eq('base_50000_entry_only')].drop(columns='case').reset_index(drop=True)
        b = frame.loc[frame.case.eq('reverse_50000_entry_only')].drop(columns='case').reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    comparisons = []
    for case in manifest['cases']:
        current_summary = json.loads((out / case['name'] / 'summary.json').read_text())
        parent_summary = json.loads((run / 'cash_execution' / case['name'] / 'summary.json').read_text())
        comparisons.append(dict(name=case['name'], capital=case['capital'], cost_multiplier=case['cost_multiplier'],
            total=current_summary['total'], parent_entry_only_total=parent_summary['total'],
            total_difference=current_summary['total']-parent_summary['total'], mdd=current_summary['mdd'],
            parent_entry_only_mdd=parent_summary['mdd'], n_trades=current_summary['n_trades'],
            parent_entry_only_trades=parent_summary['n_trades'], mean_invested=current_summary['mean_invested']))
    pd.DataFrame(comparisons).to_csv(out / 'parent_entry_only_comparison.csv', index=False)
    report = dict(status='complete', adaptive_followup_after_P3=True, test_market_data_used=False,
        original_review_clock_preserved=True, original_failure_exits_preserved=True, changed_hold_cells=len(changed),
        additional_exit_requests=len(requests), actual_exit_fills=len(fills), proxy_rows=len(proxies),
        reverse_additional_evidence_exact=True, unknown_causes_no_new_exit=True,
        proxy_limitation='adjusted_close_only; no_tradeable_counterfactual_or_future_label_tuning')
    json_write(out / 'trigger_verification.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--case-name')
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / 'cash_confirm'
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    cfg['cash_actions'] = ['entry_only']
    cfg['cash_cases'] = 5
    workspace = out / 'workspace'
    if args.case_name:
        manifest = json.loads((out / 'manifest.json').read_text())
        cash.frozen_check(workspace, manifest)
        case = next(case for case in manifest['cases'] if case['name'] == args.case_name)
        cash.run_case(workspace, cfg, case)
        return
    require(not (out / 'manifest.json').exists(), '追加执行已登记，拒绝覆盖')
    manifest = prepare(run)
    results, failures = [], []
    for number, case in enumerate(manifest['cases'], 1):
        folder = out / case['name']
        folder.mkdir()
        print(f'[{number}/5] {case["name"]}', flush=True)
        with (folder / 'run.log').open('w') as log:
            process = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--run', str(run),
                                      '--case-name', case['name']], stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            error = (folder / 'run.log').read_text().splitlines()[-1]
            failures.append(dict(name=case['name'], returncode=process.returncode, error=error))
            print(error, flush=True)
        else:
            summary = json.loads((folder / 'summary.json').read_text())
            results.append(summary)
            print(json.dumps({key: summary[key] for key in ['name', 'total', 'mdd', 'n_trades']}, ensure_ascii=False), flush=True)
        pd.DataFrame(results).to_csv(out / 'execution_summary.csv', index=False)
        pd.DataFrame(failures, columns=['name', 'returncode', 'error']).to_csv(out / 'execution_failures.csv', index=False)
        manifest.update(status='running', completed_cases=len(results), failed_cases=len(failures))
        json_write(out / 'manifest.json', manifest)
    if failures:
        manifest['status'] = 'failed'
        json_write(out / 'manifest.json', manifest)
        raise RuntimeError('追加案例失败，全部失败记录已保留')
    try:
        verification = cash.verify_outputs(workspace, cfg, manifest)
        verification.pop('fourth_action', None)
        verification.update(adaptive_followup_after_P3=True,
            policy_variant='market_warning_and_observed_raw_trend_failure_early_exit_distinct_from_original_two_failure')
        json_write(out / 'independent_verification.json', verification)
        verify_triggers_and_paths(run, manifest)
    except Exception:
        manifest['status'] = 'verification_failed'
        json_write(out / 'manifest.json', manifest)
        raise
    manifest.update(status='complete', independent_verification_pass=True, trigger_verification_pass=True)
    json_write(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
