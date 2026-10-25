#!/usr/bin/env python3
"""P3冻结市场警报三动作：真实整手执行、逐股独立核验及共同事件窗口。"""
import argparse
from copy import deepcopy
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
from portfolio import run_backtest, daily_metrics
from audit_monthly_portfolio import load_inputs, reconcile
from run_patient_value_execution import process_report, digest, json_write
from run_divide_selection_execution import reversed_candidate_input, entry_phase_attribution
from run_symmetric_phase_execution import is_mainboard
from verify_divide_selection_execution import verify_case, require
from check_cash_reserve import add_cash_reserve


def execution_cases(cfg):
    """九个基线、三倒序、三倍费和一个固定现金对照。"""
    def case(name, action, capital, reverse=False, cost=1., reserve=0.):
        return dict(name=name, action=action, capital=capital, reverse_input=reverse,
                    cost_multiplier=cost, reserve_fraction=reserve, policy='low_turnover')
    rows = [case(f'base_{capital:g}_{action}', action, capital)
            for capital in cfg['capitals'] for action in cfg['cash_actions']]
    ref = cfg['reference_capital']
    rows += [case(f'reverse_{ref:g}_{action}', action, ref, reverse=True) for action in cfg['cash_actions']]
    rows += [case(f'cost2_{ref:g}_{action}', action, ref, cost=cfg['cash_stress_cost_multiplier'])
             for action in cfg['cash_actions']]
    rows += [case(f'fixed_half_{ref:g}_none', 'none', ref, reserve=cfg['cash_fixed_reserve_fraction'])]
    require(len(rows) == cfg['cash_cases'], '案例数量偏离预登记')
    return rows


def action_inputs(signals, hold, risk, action, forecasts):
    """只过滤当次月度新候选；收益点估计和原持有许可保持不变。"""
    require(action in ['none', 'entry_only', 'liquidate'], '未登记动作')
    result = signals.copy()
    if action == 'entry_only':
        gate = risk.reindex(signals.index)
        require(gate.notna().all(), '月度警报缺失')
        result.loc[gate.eq(0), :] = np.nan
    return result, hold.copy(), forecasts.copy()


def portfolio_config(base, case):
    """费用比例与最低费同步变更；现金储备在真实分配之前划出。"""
    cfg = deepcopy(base)
    cfg['capital'] = case['capital'] * (1 - case['reserve_fraction'])
    for key in ['buy_cost', 'sell_cost', 'min_fee']:
        cfg[key] *= case['cost_multiplier']
    return cfg


def common_alarms(dates, risk):
    """共同事件以同一收盘警报次日执行日期定义，不依赖某个账户是否清仓。"""
    gate = risk.reindex(dates)
    require(gate.notna().all(), '警报没有覆盖执行日历')
    gate = gate.shift(1).fillna(1).astype(int)
    starts = dates[gate.eq(0) & gate.shift(1).fillna(1).eq(1)]
    rows = []
    for number, date in enumerate(starts, 1):
        i = dates.get_loc(date)
        following = dates[(dates > date) & gate.eq(1)]
        recovery = following[0] if len(following) else pd.NaT
        next_start = starts[number] if number < len(starts) else pd.NaT
        rows.append(dict(alarm_id=number, request_date=date, signal_date=dates[i-1],
                         recovery_execution_date=recovery, next_alarm_date=next_start))
    return pd.DataFrame(rows, columns=['alarm_id', 'request_date', 'signal_date',
                                       'recovery_execution_date', 'next_alarm_date'])


def alarm_reports(result, risk, signals, action, horizons):
    """请求、交易受阻与重入分开记录；共同观察窗不依赖实际重入时点。"""
    daily, trades, events = result.daily, result.trades, result.events
    dates = daily.index
    alarms = common_alarms(dates, risk)
    scheduled = {dates[dates.get_loc(d)+1]: d for d in signals.index
                 if d in dates and dates.get_loc(d)+1 < len(dates)}
    rows, trace, windows = [], [], []
    for alarm in alarms.to_dict('records'):
        start, recovery, next_alarm = alarm['request_date'], alarm['recovery_execution_date'], alarm['next_alarm_date']
        limit = next_alarm if pd.notna(next_alarm) else dates[-1] + pd.Timedelta(days=1)
        end = dates[dates < limit][-1]
        in_period = events.date.between(start, end)
        event_frame = events.loc[in_period].copy()
        relevant = event_frame.kind.isin(['hold_exit_requested', 'sell_deferred', 'risk_transition',
            'risk_exit_complete', 'reentry_wait', 'reentry_attempt', 'buy_skipped', 'selection_decision'])
        for record in event_frame.loc[relevant].to_dict('records'):
            trace.append(dict(alarm_id=alarm['alarm_id'], record_type='engine_event', **record))
        for record in trades.loc[trades.date.between(start, end)].to_dict('records'):
            trace.append(dict(alarm_id=alarm['alarm_id'], record_type='actual_trade', **record))
        blocked_reviews = []
        for execution_date, signal_date in scheduled.items():
            if start <= execution_date <= end and risk.loc[signal_date] == 0:
                blocked = action != 'none'
                if blocked:
                    blocked_reviews.append(execution_date)
                trace.append(dict(alarm_id=alarm['alarm_id'], record_type='scheduled_review',
                    date=execution_date, signal_date=signal_date, kind='market_entry_veto' if blocked else 'market_alarm_observed',
                    ts_code='__portfolio__', reason=action, admitted_before_gate=int(signals.loc[signal_date].notna().sum())))
        requests = event_frame.loc[event_frame.kind.eq('hold_exit_requested')]
        risk_requests = requests.loc[requests.reason.eq('portfolio_risk')]
        sells = trades.loc[trades.side.eq('sell') & trades.date.between(start, end)]
        buys = trades.loc[trades.side.eq('buy') & trades.date.between(start, end)]
        recovered_buys = buys.loc[buys.date.ge(recovery)] if pd.notna(recovery) else buys.iloc[:0]
        completed = event_frame.loc[event_frame.kind.eq('risk_exit_complete')]
        review_after = [d for d in scheduled if pd.notna(recovery) and recovery <= d <= end]
        # 无门槛和仅禁新仓不请求全清，不能把未清仓计成失败。
        rows.append(dict(**alarm, action=action, observation_end=end,
            liquidation_requested=bool(len(risk_requests)), liquidation_request_stocks=risk_requests.ts_code.nunique(),
            first_actual_sell_date=sells.date.min(), last_actual_sell_date=sells.date.max(),
            actual_sell_trades=len(sells), sell_deferred_events=int(event_frame.kind.eq('sell_deferred').sum()),
            liquidation_complete_date=completed.date.min(), blocked_monthly_reviews=len(blocked_reviews),
            next_monthly_review_after_recovery=min(review_after) if review_after else pd.NaT,
            first_actual_buy_date=buys.date.min(), first_actual_buy_after_recovery=recovered_buys.date.min(),
            buy_skipped_events=int(event_frame.kind.eq('buy_skipped').sum()),
            no_actual_buy_after_recovery=bool(pd.notna(recovery) and not len(recovered_buys)),
            pending_sell_days=int(daily.loc[start:end, 'pending_sell'].gt(0).sum()),
            full_cash_days=int(daily.loc[start:end, 'n_hold'].eq(0).sum())))
        request_index = dates.get_loc(start)
        before_equity = daily.equity.iloc[request_index-1] if request_index else result.summary['capital']
        alarm_end_index = dates.get_loc(recovery)-1 if pd.notna(recovery) else len(dates)-1
        windows.append(dict(alarm_id=alarm['alarm_id'], window='alarm_period', start=start,
            end=dates[alarm_end_index], observed=True, complete=bool(pd.notna(recovery)),
            start_equity=before_equity, end_equity=daily.equity.iloc[alarm_end_index],
            return_value=daily.equity.iloc[alarm_end_index]/before_equity-1,
            mean_exposure=(daily.stock_value/daily.equity).iloc[request_index:alarm_end_index+1].mean(),
            overlaps_next_alarm=False))
        for horizon in horizons:
            recovery_index = dates.get_loc(recovery) if pd.notna(recovery) else len(dates)
            stop = recovery_index + horizon - 1
            complete = stop < len(dates)
            prior = daily.equity.iloc[recovery_index-1] if complete else np.nan
            last = daily.equity.iloc[stop] if complete else np.nan
            windows.append(dict(alarm_id=alarm['alarm_id'], window=f'recovery_next_{horizon}d',
                start=recovery, end=dates[stop] if complete else pd.NaT, observed=complete, complete=complete,
                start_equity=prior, end_equity=last, return_value=last/prior-1,
                mean_exposure=(daily.stock_value/daily.equity).iloc[recovery_index:stop+1].mean() if complete else np.nan,
                overlaps_next_alarm=bool(complete and pd.notna(next_alarm) and next_alarm <= dates[stop])))
    return (pd.DataFrame(rows), pd.DataFrame(trace), pd.DataFrame(windows))


def frozen_check(run, manifest):
    out = run / 'cash_execution'
    for name, expected in manifest['source_hashes'].items():
        require(digest(ROOT / name) == expected and digest(out / 'sources' / name) == expected, f'源码漂移: {name}')
    for name, expected in manifest['input_hashes'].items():
        require(digest(out / name) == expected, f'输入漂移: {name}')


def run_case(run, cfg, case):
    out, folder = run / 'cash_execution', run / 'cash_execution' / case['name']
    ecfg = yaml.safe_load((out / 'execution_config.yaml').read_text())
    market, basics, st, limits, actions = load_inputs(out, ecfg)
    signals = pd.read_parquet(out / 'low_turnover_signals.parquet')
    forecasts = pd.read_parquet(out / 'low_turnover_point_forecasts.parquet')
    hold = pd.read_parquet(out / 'monthly_hold_allowed.parquet')
    predictions = pd.read_parquet(out / 'execution_predictions.parquet')
    risk = pd.read_csv(out / 'risk_plans.csv', index_col=0, parse_dates=True)[cfg['cash_risk_column']]
    chosen, holds, points = action_inputs(signals, hold, risk, case['action'], forecasts)
    portfolio = portfolio_config(ecfg['portfolio'], case)
    extra = dict(risk_allowed=risk, risk_signal_max_age=cfg['risk_signal_max_age'], reentry_on_schedule_only=True) if case['action'] == 'liquidate' else {}
    rights = yaml.safe_load((ROOT / 'configs/corporate_action_components.yaml').read_text()).get('rights_issues', [])
    with reversed_candidate_input(case['reverse_input']):
        result = run_backtest(chosen, market, basics, st, limits, actions, *cfg['execution'], portfolio,
            rights, retail_forecasts=points, retail_config=cfg['retail'], hold_allowed=holds, **extra)
    reserve = case['capital'] * case['reserve_fraction']
    if reserve:
        result = add_cash_reserve(result, reserve, portfolio['annual_days'])
        result.trades['active_cash_after'] = result.trades.cash_after
        result.trades['cash_after'] += reserve
    check = reconcile(result, case['capital'])
    process, pnl, periods, exits, metrics = process_report(result, case['capital'])
    episodes, phases, buys, error = entry_phase_attribution(result, predictions, chosen)
    trace = result.events.loc[result.events.kind.eq('selection_decision')].copy()
    if len(trace):
        trace = trace.merge(predictions[['date', 'ts_code', 'phase']].rename(columns={'date': 'signal_date'}),
                            on=['signal_date', 'ts_code'], how='left', validate='many_to_one')
    horizons = yaml.safe_load((ROOT / 'configs/active_cash_audit.yaml').read_text())['event_horizons']
    alarm_rows, alarm_trace, windows = alarm_reports(result, risk, signals, case['action'], horizons)
    frames = dict(daily=result.daily, trades=result.trades, holdings=result.holdings, events=result.events,
        process_daily=process, asset_pnl=pnl, holding_periods=periods, exit_reasons=exits,
        entry_phase_episodes=episodes, entry_phase_attribution=phases, actual_entry_priorities=buys,
        selection_trace=trace, stale_holding_marks=result.holdings.loc[result.holdings.stale_days.gt(0)],
        unknown_hold_reviews=result.events.loc[result.events.kind.eq('hold_retained') & result.events.reason.eq('unknown')],
        alarm_episodes=alarm_rows, alarm_event_trace=alarm_trace, common_alarm_windows=windows)
    for name, frame in frames.items():
        if not len(frame.columns):
            frame = pd.DataFrame(columns=['date', 'ts_code'])
        frame.to_csv(folder / f'{name}.csv', index=name in ['daily', 'process_daily'])
    summary = {**case, **result.summary, **check, **metrics, 'phase_pnl_max_error': error,
               'actual_buys': len(buys), 'reserve_cash': reserve,
               'mean_actual_entry_rank': float(buys.priority_rank.mean()) if len(buys) else np.nan}
    json_write(folder / 'summary.json', summary)
    years, previous = [], case['capital']
    for year, daily in result.daily.groupby(result.daily.index.year):
        values, _ = daily_metrics(daily.equity, previous, portfolio['annual_days'])
        years.append(dict(name=case['name'], year=year, **values))
        previous = daily.equity.iloc[-1]
    pd.DataFrame(years).to_csv(folder / 'yearly.csv', index=False)
    json_write(folder / 'portfolio_config.json', portfolio)
    print(json.dumps({key: summary[key] for key in ['name', 'total', 'mdd', 'n_trades', 'mean_invested']}, ensure_ascii=False), flush=True)


def prepare(run, cfg, cp, cases):
    out = run / 'cash_execution'
    out.mkdir(exist_ok=False)
    require(pd.Timestamp(cfg['execution'][1]) < pd.Timestamp('2024-01-01'), '禁止TEST行情')
    require(digest(cp) == digest(run / 'config.yaml'), '冻结配置不一致')
    parent = ROOT / cfg['parent'] / 'execution'
    sources = [Path(__file__), cp, ROOT / 'src/portfolio.py', ROOT / 'src/selection_first.py', ROOT / 'src/retail_rebalance.py',
        ROOT / 'scripts/audit_monthly_portfolio.py', ROOT / 'scripts/run_patient_value_execution.py',
        ROOT / 'scripts/run_divide_selection_execution.py', ROOT / 'scripts/run_symmetric_phase_execution.py',
        ROOT / 'scripts/verify_divide_selection_execution.py', ROOT / 'scripts/check_cash_reserve.py',
        ROOT / 'scripts/prepare_monthly_audit.py', ROOT / 'configs/corporate_action_components.yaml',
        ROOT / 'configs/active_cash_audit.yaml', ROOT / 'configs/retail_validation.yaml',
        ROOT / 'src/risk_states.py', ROOT / 'scripts/prepare_retail_validation.py']
    manifest = dict(status='registered', cases=cases, fresh_holdout=False, test_market_data_used=False,
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources}, input_hashes={},
        alarm_definition='frozen_extreme_recovery_allowed_zero_including_recovery_wait',
        fourth_action='original_double_failure_confirmation_plus_entry_veto_equals_entry_only_not_an_extra_strategy',
        observation_windows='existing_active_cash_audit_event_horizons',
        fixed_cash_semantics='real_reduced_trading_account_plus_constant_unremunerated_reserve')
    json_write(out / 'manifest.json', manifest)
    for path in sources:
        target = out / 'sources' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    files = ['execution_config.yaml', 'execution_codes.csv', 'execution_cache.txt', 'monthly_hold_allowed.parquet',
             'low_turnover_signals.parquet', 'low_turnover_point_forecasts.parquet']
    for name in files:
        shutil.copyfile(parent / name, out / name)
    shutil.copyfile(ROOT / cfg['parent'] / 'execution_predictions.parquet', out / 'execution_predictions.parquet')
    shutil.copyfile(ROOT / cfg['cash_risk_source'], out / 'risk_plans.csv')
    shutil.copyfile(cp, out / 'config.yaml')
    codes = pd.read_csv(out / 'execution_codes.csv').ts_code.tolist()
    require(all(map(is_mainboard, codes)), '执行代码含非主板')
    risk = pd.read_csv(out / 'risk_plans.csv', index_col=0, parse_dates=True)[cfg['cash_risk_column']]
    require(risk.index.max() < pd.Timestamp('2024-01-01') and risk.isin([0, 1]).all(), '非法风险计划')
    cache = Path((out / 'execution_cache.txt').read_text().strip())
    missing = [code for code in codes if any(not (cache / api / f'{code}.parquet').exists() for api in ['dividend', 'stk_limit'])]
    pd.DataFrame({'ts_code': missing}).to_csv(out / 'missing_cache_codes.csv', index=False)
    print(f'16案例已登记，候选数据{len(codes)}只，缺缓存{len(missing)}只', flush=True)
    if missing:
        from prepare_monthly_audit import fetch_execution_data
        p = yaml.safe_load((out / 'execution_config.yaml').read_text())
        p['data_workers'] = 2
        try:
            cache = fetch_execution_data(missing, tuple(map(pd.Timestamp, cfg['execution'])), p)
        except Exception as exc:
            manifest.update(status='cache_failed', error_type=type(exc).__name__)
            json_write(out / 'manifest.json', manifest)
            if (cache / 'failures.csv').exists():
                shutil.copyfile(cache / 'failures.csv', out / 'cache_failures.csv')
            raise
        (out / 'execution_cache.txt').write_text(str(cache))
    manifest['input_hashes'] = {name: digest(out / name) for name in files + ['execution_predictions.parquet', 'risk_plans.csv', 'config.yaml']}
    manifest.update(status='prepared', missing_cache_codes=len(missing))
    json_write(out / 'manifest.json', manifest)
    return manifest


def verify_outputs(run, cfg, manifest):
    out = run / 'cash_execution'
    frozen_check(run, manifest)
    predictions = pd.read_parquet(out / 'execution_predictions.parquet')
    signals = pd.read_parquet(out / 'low_turnover_signals.parquet')
    hold = pd.read_parquet(out / 'monthly_hold_allowed.parquet')
    risk = pd.read_csv(out / 'risk_plans.csv', index_col=0, parse_dates=True)[cfg['cash_risk_column']]
    verified = []
    for case in manifest['cases']:
        folder = out / case['name']
        ecfg = dict(portfolio=json.loads((folder / 'portfolio_config.json').read_text()))
        verified.append(verify_case(folder, predictions, cfg, ecfg))
        daily = pd.read_csv(folder / 'daily.csv', index_col=0, parse_dates=True)
        trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'signal_date'])
        events = pd.read_csv(folder / 'events.csv', parse_dates=['date', 'signal_date'])
        holdings = pd.read_csv(folder / 'holdings.csv', parse_dates=['date'])
        require(holdings.ts_code.map(is_mainboard).all() and trades.ts_code.map(is_mainboard).all(), '出现非主板持仓或交易')
        buys = trades.loc[trades.side.eq('buy')]
        schedule = {daily.index[daily.index.get_loc(d)+1] for d in signals.index if d in daily.index and daily.index.get_loc(d)+1 < len(daily)}
        require(set(buys.date).issubset(schedule), '出现非月度买入')
        for row in buys.itertuples():
            if case['action'] != 'none':
                require(risk.loc[row.signal_date] == 1, '警报期间新买入')
        if case['action'] == 'liquidate':
            pd.testing.assert_series_equal(daily.risk_target, risk.reindex(daily.index).shift(1).fillna(1).astype(int), check_names=False)
            for day in daily.index[daily.risk_target.eq(0) & daily.n_hold.gt(0)]:
                remaining = set(holdings.loc[holdings.date.eq(day), 'ts_code'])
                deferred = set(events.loc[events.date.eq(day) & events.kind.eq('sell_deferred'), 'ts_code'])
                require(remaining.issubset(deferred), '风险禁持却无卖出受阻证据')
        if case['reserve_fraction']:
            require((daily.cash >= case['capital']*case['reserve_fraction']-1e-6).all(), '动用了固定现金储备')
            np.testing.assert_allclose(trades.cash_after-trades.active_cash_after, case['capital']*case['reserve_fraction'], atol=1e-8)
    comparisons = []
    for action in cfg['cash_actions']:
        base = out / f'base_{cfg["reference_capital"]:g}_{action}'
        reverse = out / f'reverse_{cfg["reference_capital"]:g}_{action}'
        files = sorted(p.name for p in base.glob('*.csv'))
        require(files == sorted(p.name for p in reverse.glob('*.csv')), '倒序产物集不同')
        for name in files:
            a, b = pd.read_csv(base/name), pd.read_csv(reverse/name)
            if name == 'yearly.csv':
                a, b = a.drop(columns='name'), b.drop(columns='name')
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            comparisons.append(dict(action=action, file=name, identical=True))
    pd.DataFrame(comparisons).to_csv(out / 'candidate_order_invariance.csv', index=False)
    # 共同窗口以无门槛同资金同费用分支对照；固定现金单列解释暴露差。
    windows = []
    for case in manifest['cases']:
        frame = pd.read_csv(out / case['name'] / 'common_alarm_windows.csv')
        base = pd.read_csv(out / f'base_{case["capital"]:g}_none' / 'common_alarm_windows.csv')
        if case['cost_multiplier'] != 1:
            base = pd.read_csv(out / f'cost2_{case["capital"]:g}_none' / 'common_alarm_windows.csv')
        paired = frame.merge(base[['alarm_id', 'window', 'return_value', 'mean_exposure']], on=['alarm_id', 'window'],
                             suffixes=('', '_none'), validate='one_to_one')
        paired['case'] = case['name']
        paired['paired_return_difference'] = paired.return_value-paired.return_value_none
        paired['paired_exposure_difference'] = paired.mean_exposure-paired.mean_exposure_none
        windows.append(paired)
    pd.concat(windows, ignore_index=True).to_csv(out / 'common_alarm_path_comparison.csv', index=False)
    report = dict(status='complete', cases=verified, reverse_csv_comparisons=len(comparisons), market_data_read=False,
        test_market_data_used=False, source_and_input_hashes_verified=True, monthly_entry_only=True,
        fourth_action='structurally_identical_to_entry_only_if_confirmation_is_original_hold_false',
        event_limitation='evolving_portfolios_and_exposures; differences_are_descriptive_not_isolated_timing_alpha')
    json_write(out / 'independent_verification.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--case-name')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run, cp = args.run.resolve(), args.config.resolve()
    cfg = yaml.safe_load(cp.read_text())
    cases, out = execution_cases(cfg), run / 'cash_execution'
    mp = out / 'manifest.json'
    if args.case_name or args.verify_only:
        manifest = json.loads(mp.read_text())
        require(cases == manifest['cases'], '案例漂移')
        frozen_check(run, manifest)
        if args.verify_only:
            verify_outputs(run, cfg, manifest)
        else:
            run_case(run, cfg, next(case for case in cases if case['name'] == args.case_name))
        return
    require(not out.exists(), '新执行目录已存在，拒绝覆盖')
    manifest = prepare(run, cfg, cp, cases)
    summaries, failures = [], []
    for i, case in enumerate(cases, 1):
        folder = out / case['name']
        folder.mkdir()
        print(f'[{i}/{len(cases)}] {case["name"]}', flush=True)
        with (folder / 'run.log').open('w') as log:
            completed = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--run', str(run),
                '--config', str(cp), '--case-name', case['name']], stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            error = (folder / 'run.log').read_text().splitlines()[-1]
            failures.append(dict(name=case['name'], error=error, returncode=completed.returncode))
            print(error, flush=True)
        else:
            summary = json.loads((folder / 'summary.json').read_text())
            summaries.append(summary)
            print(json.dumps({key: summary[key] for key in ['name', 'total', 'mdd', 'n_trades']}, ensure_ascii=False), flush=True)
        pd.DataFrame(summaries).to_csv(out / 'execution_summary.csv', index=False)
        pd.DataFrame(failures, columns=['name', 'error', 'returncode']).to_csv(out / 'execution_failures.csv', index=False)
        manifest.update(status='running', completed_cases=len(summaries), failed_cases=len(failures))
        json_write(mp, manifest)
    if failures:
        manifest.update(status='failed')
        json_write(mp, manifest)
        raise RuntimeError('案例失败已完整留档，禁止绕过真实执行异常')
    try:
        verify_outputs(run, cfg, manifest)
    except Exception:
        manifest.update(status='verification_failed')
        json_write(mp, manifest)
        raise
    manifest.update(status='complete', independent_verification_pass=True, order_invariance_pass=True)
    json_write(mp, manifest)


if __name__ == '__main__':
    main()
