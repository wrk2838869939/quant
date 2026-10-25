#!/usr/bin/env python3
"""固定价值模型的周/月耐心持仓开发账本；仅访问2024年前行情，不选择冠军。"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from patient_trend import review_dates, trend_state
from trend_entry import entry_features
from portfolio import run_backtest, daily_metrics, AccountingError
from audit_monthly_portfolio import load_inputs, reconcile


def digest(path):
    """只散列显式登记的研究文件，不扫描配置以外的凭据文件。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, content):
    """登记写入临时文件后原子替换，保留可恢复的阶段状态。"""
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(content, ensure_ascii=False, indent=2, default=str))
    temp.replace(path)


def confirmed_hold_states(trend, observable, confirmations):
    """连续可观察复核失败才确认退出；信息缺失中断失败计数，输出未知。"""
    if (not trend.index.equals(observable.index) or not trend.columns.equals(observable.columns)
            or not isinstance(confirmations, int) or confirmations < 1):
        raise ValueError('持有状态输入或确认次数非法')
    failures = np.zeros(len(trend.columns), dtype=int)
    states = np.full(trend.shape, np.nan)
    for i in range(len(trend)):
        known = observable.iloc[i].to_numpy(bool)
        good = trend.iloc[i].to_numpy(bool)
        failures = np.where(known & ~good, failures + 1, 0)
        states[i, known] = (failures[known] < confirmations).astype(float)
    return pd.DataFrame(states, index=trend.index, columns=trend.columns)


def build_review_inputs(raw, basics, names, cfg):
    """与entry审计相同的PIT资格和日期截面rank；持有可观察性独立于财务/换手缺失。"""
    lo, hi = map(pd.Timestamp, cfg['execution'])
    if hi >= pd.Timestamp('2024-01-01') or raw.date.max() > hi:
        raise ValueError('禁止读取或使用TEST行情')
    if raw.duplicated(['date', 'ts_code']).any():
        raise ValueError('行情代码/日期重复')
    columns = ['hfq_close', 'total_mv_yuan', 'turnover_rate']
    wide = {c: raw.pivot(index='date', columns='ts_code', values=c).sort_index() for c in columns}
    close = wide['hfq_close']
    spec = cfg['trend']
    features = entry_features(close, wide['total_mv_yuan'], wide['turnover_rate'],
                              medium=spec['medium_window'], long=spec['long_window'])
    score, trend = trend_state(close, spec)
    observable = (close.gt(0) & close.shift(spec['medium_window']).gt(0)
        & close.shift(spec['long_window']).gt(0))
    if spec['require_above_long_ma']:
        observable &= close.where(close.gt(0)).rolling(spec['long_window'],
            min_periods=spec['long_window']).mean().notna()
    dates = close.index[(close.index >= lo) & (close.index <= hi)]
    calendars = {f: review_dates(dates, f) for f in cfg['review_frequencies']}
    holds = {f: confirmed_hold_states(trend.loc[ds], observable.loc[ds], cfg['exit_confirmations'])
             for f, ds in calendars.items()}
    basics = basics.set_index('ts_code')
    listing = pd.to_datetime(basics.list_date).reindex(close.columns)
    delisting = pd.to_datetime(basics.delist_date).reindex(close.columns)
    st = names[names.name.str.contains('ST', case=False, na=False)].copy()
    st['start'] = pd.to_datetime(st.start_date)
    st['end'] = pd.to_datetime(st.end_date)
    blocks = []
    for date in sorted(set().union(*map(set, calendars.values()))):
        frame = pd.DataFrame({key: value.loc[date] for key, value in features.items()})
        eligible = ((date-listing).dt.days.ge(cfg['min_listing_days'])
            & (delisting.isna() | delisting.gt(date)) & np.isfinite(frame).all(axis=1))
        active_st = st.loc[st.start.le(date) & (st.end.isna() | st.end.ge(date)), 'ts_code']
        eligible.loc[eligible.index.isin(active_st)] = False
        frame = frame.loc[eligible].copy()
        # 不先筛趋势或价格；与旧审计的全合格截面rank保持一致。
        ranks = frame.rank(pct=True) - .5
        for key in features:
            frame['x_'+key] = ranks[key]
        frame['date'] = date
        frame['trend'] = trend.loc[date].reindex(frame.index)
        frame['momentum'] = score.loc[date].reindex(frame.index)
        for frequency, ds in calendars.items():
            frame[frequency] = date in ds
        blocks.append(frame.reset_index(names='ts_code'))
    table = pd.concat(blocks, ignore_index=True)
    # 两个历史前缀截断检查：未来行情不能改写已知特征或趋势。
    checked = 0
    for stop in dates[[len(dates)//3, 2*len(dates)//3]]:
        short = entry_features(close.loc[:stop], wide['total_mv_yuan'].loc[:stop],
            wide['turnover_rate'].loc[:stop], medium=spec['medium_window'], long=spec['long_window'])
        for key in features:
            pd.testing.assert_frame_equal(short[key], features[key].loc[:stop])
            checked += 1
    return table, holds, dict(prefix_checks=checked, warmup_start=str(raw.date.min()),
        market_end=str(raw.date.max()), dates=len(dates), feature_rows=len(table),
        monthly_reviews=len(calendars.get('monthly', [])), weekly_reviews=len(calendars.get('weekly', [])))


def make_policy_matrices(table, holds, models, cfg):
    """准入门槛仅控制新仓；整手目标函数始终使用收益点预测。"""
    table = table.copy()
    prediction = models['full'].predict(table)
    for col in prediction:
        table[col] = prediction[col]
    constant = models['constant'].predict(table.iloc[:1]).expected_return.iloc[0]
    buffer = cfg['entry']['mean_lower_buffer']
    adverse = cfg['entry']['max_expected_adverse']
    matrices, coverage = {}, []
    for frequency, hold in holds.items():
        sample = table[table[frequency]].copy()
        for policy in cfg['execution_policies']:
            point = sample.expected_return
            if policy == 'low_turnover':
                score = -sample.x_turnover
                gate = sample.trend & (constant > 0)
                mu = pd.Series(constant, index=sample.index)
            elif policy == 'full_point':
                score, mu = point, point
                gate = sample.trend & point.gt(buffer)
            elif policy == 'full_confidence':
                score, mu = point, point
                gate = (sample.trend & sample.mean_lower.gt(buffer)
                    & sample.expected_adverse.le(adverse))
            else:
                raise ValueError(f'未知固定方案: {policy}')
            # 保留所有可预测旧仓的点估计；不因为准入失败把旧仓收益改成未知。
            values = sample[['date', 'ts_code']].copy()
            values['score'] = score.where(gate)
            values['mu'] = mu
            signals = values.pivot(index='date', columns='ts_code', values='score').reindex(
                index=hold.index, columns=hold.columns)
            forecasts = values.pivot(index='date', columns='ts_code', values='mu').reindex(
                index=hold.index, columns=hold.columns)
            matrices[(frequency, policy)] = (signals, forecasts)
            for date, g in sample.assign(gate=gate).groupby('date'):
                coverage.append(dict(frequency=frequency, policy=policy, date=date,
                    eligible=int(len(g)), trend=int(g.trend.sum()), admitted=int(g.gate.sum()),
                    constant_expected_return=float(constant)))
    return matrices, pd.DataFrame(coverage), table


def case_matrix(cfg):
    """完整登记18个基线与3个联合执行压力，不能选择最好参数。"""
    cases = []
    for capital in cfg['capitals']:
        for frequency in cfg['review_frequencies']:
            for policy in cfg['execution_policies']:
                cases.append(dict(name=f'base_{capital:g}_{frequency}_{policy}', capital=capital,
                    frequency=frequency, policy=policy, delay=0, slippage_rate=0.))
    for policy in cfg['execution_policies']:
        cases.append(dict(name=f'stress_{cfg["reference_capital"]:g}_monthly_{policy}',
            capital=cfg['reference_capital'], frequency='monthly', policy=policy,
            delay=cfg['stress']['extra_execution_delay'], slippage_rate=cfg['stress']['slippage_rate']))
    return cases


def process_report(result, capital):
    """独立逐股利润、期间费用和暴露；空仓时仍输出完整且可读的表头。"""
    daily = result.daily.copy()
    trades = result.trades.reindex(columns=['date', 'signal_date', 'ts_code', 'side',
        'shares', 'price', 'gross', 'fee', 'cash_after', 'slippage_cost'])
    events = result.events
    holdings = result.holdings.reindex(columns=['date', 'ts_code', 'shares', 'available',
                                              'price', 'value', 'stale_days'])
    pnl = pd.Series(dtype=float, name='pnl_yuan')
    if len(trades):
        flow = np.where(trades.side.eq('sell'), trades.gross, -trades.gross) - trades.fee
        pnl = pd.Series(flow, index=trades.index).groupby(trades.ts_code).sum()
    if len(events):
        for kind, column in [('dividend_receivable', 'amount'), ('delist_writeoff', 'cash_recovery')]:
            selected = events[events.kind.eq(kind)]
            if len(selected):
                pnl = pnl.add(selected.groupby('ts_code')[column].sum(), fill_value=0.)
    last = holdings[holdings.date.eq(daily.index[-1])].set_index('ts_code').value
    pnl = pnl.add(last, fill_value=0.).sort_values(ascending=False)
    pnl_error = float(abs(pnl.sum() - (daily.equity.iloc[-1] - capital)))
    if pnl_error > 1e-6:
        raise AccountingError(f'逐股净利润对账失败: {pnl_error}')
    for side in ['buy', 'sell']:
        gross = trades[trades.side.eq(side)].groupby('date').gross.sum()
        daily[f'{side}_gross'] = gross.reindex(daily.index, fill_value=0.)
    daily['fees'] = trades.groupby('date').fee.sum().reindex(daily.index, fill_value=0.)
    daily['cumulative_fees'] = daily.fees.cumsum()
    daily['two_sided_turnover'] = (daily.buy_gross+daily.sell_gross)/daily.equity
    daily['cash_fraction'] = daily.cash/daily.equity
    daily['stock_fraction'] = daily.stock_value/daily.equity
    weights = holdings.value / holdings.date.map(daily.equity)
    daily['largest_stock_weight'] = weights.groupby(holdings.date).max().reindex(daily.index, fill_value=0.)
    episodes = []
    day_number = pd.Series(np.arange(len(daily)), index=daily.index)
    for code, group in holdings.groupby('ts_code'):
        group = group.sort_values('date')
        number = group.date.map(day_number)
        for _, period in group.groupby(number.diff().ne(1).cumsum()):
            episodes.append(dict(ts_code=code, first_mark=period.date.min(), last_mark=period.date.max(),
                holding_trading_days=len(period), open_at_end=bool(period.date.max()==daily.index[-1])))
    episodes = pd.DataFrame(episodes, columns=['ts_code', 'first_mark', 'last_mark', 'holding_trading_days', 'open_at_end'])
    reason_rows, reasons = [], {}
    for date in daily.index:
        if len(events):
            for event in events[events.date.eq(date)].to_dict('records'):
                if event['kind'] in ['hold_exit_requested', 'hold_risk_cap_reduction']:
                    reasons[event['ts_code']] = (event['kind'], event.get('reason', ''))
        for trade in trades[trades.date.eq(date) & trades.side.eq('sell')].to_dict('records'):
            kind, reason = reasons.get(trade['ts_code'], ('unspecified', ''))
            reason_rows.append(dict(date=date, ts_code=trade['ts_code'], shares=trade['shares'],
                gross=trade['gross'], fee=trade['fee'], reason_kind=kind, reason=reason))
    exits = pd.DataFrame(reason_rows, columns=['date', 'ts_code', 'shares', 'gross', 'fee', 'reason_kind', 'reason'])
    positive = pnl.clip(lower=0.).sum()
    abs_sum = pnl.abs().sum()
    summary = dict(asset_pnl_max_error=pnl_error, mean_holdings=float(daily.n_hold.mean()),
        max_holdings=int(daily.n_hold.max()), mean_cash_fraction=float(daily.cash_fraction.mean()),
        n_cash_days=int(daily.n_hold.eq(0).sum()), cash_day_fraction=float(daily.n_hold.eq(0).mean()),
        max_marked_stock_weight=float(daily.largest_stock_weight.max()),
        cumulative_two_sided_turnover=float(daily.two_sided_turnover.sum()),
        fee_fraction_of_initial=float(daily.fees.sum()/capital), n_distinct_stocks=len(pnl),
        max_holding_trading_days=int(episodes.holding_trading_days.max()) if len(episodes) else 0,
        best_asset_pnl=float(pnl.max()) if len(pnl) else 0., worst_asset_pnl=float(pnl.min()) if len(pnl) else 0.,
        best_asset_share_of_positive_pnl=float(pnl.max()/positive) if positive > 0 else np.nan,
        largest_abs_asset_share=float(pnl.abs().max()/abs_sum) if abs_sum > 0 else np.nan,
        attributed_return_without_best_asset=float((pnl.sum()-pnl.max())/capital) if len(pnl) else 0.)
    return daily, pnl.rename_axis('ts_code').rename('pnl_yuan').reset_index(), episodes, exits, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=ROOT/'configs/autonomous_value_research.yaml')
    ap.add_argument('--run', type=Path, default=None)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    root = args.run or Path('/tmp/autonomous_value_run_path').read_text().strip()
    root = Path(root)
    out = root/'execution'
    out.mkdir(exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())
    lo, hi = map(pd.Timestamp, cfg['execution'])
    if hi >= pd.Timestamp('2024-01-01'):
        raise ValueError('本实验不允许访问TEST')
    cases = case_matrix(cfg)
    manifest_path = out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('status') == 'complete':
        raise ValueError('已经完成的执行矩阵禁止覆盖')
    parent = ROOT/cfg['execution_parent']
    pc = yaml.safe_load((parent/'execution_config.yaml').read_text())
    pc['valid'] = cfg['execution']
    pc['portfolio'].update(top_n=cfg['retail']['max_positions'],
        min_listing_calendar_days=cfg['min_listing_days'])
    sources = [Path(__file__), args.config, ROOT/'src/portfolio.py', ROOT/'src/retail_rebalance.py',
        ROOT/'src/trend_entry.py', ROOT/'src/patient_trend.py', ROOT/'src/value_forecast.py',
        ROOT/'scripts/audit_monthly_portfolio.py',
        ROOT/'configs/corporate_action_components.yaml', parent/'execution_config.yaml']
    manifest = dict(status='preparing', fresh_holdout=False, test_market_data_used=False,
        period='2022-2023 repeatedly inspected development', cases=cases,
        hold_semantics='two_consecutive_observable_trend_failures; unknown_breaks_confirmation; irreversible_pending_exit',
        entry_semantics='gate_new_positions_only; point_expected_return_in_allocator',
        data_access='date_predicate_below_2024; existing_execution_data_interface_no_credential_output',
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in sources})
    json_write(manifest_path, manifest)
    (out/'execution_config.yaml').write_text(yaml.safe_dump(pc, allow_unicode=True, sort_keys=False))
    source_dir = out/'sources'
    source_dir.mkdir(exist_ok=True)
    for path in sources:
        target = source_dir/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    feature_path = out/'pit_review_features.parquet'
    if feature_path.exists():
        table = pd.read_parquet(feature_path)
        holds = {f: pd.read_parquet(out/f'{f}_hold_allowed.parquet') for f in cfg['review_frequencies']}
        manifest['feature_checks'] = json.loads((out/'feature_checks.json').read_text())
    else:
        raw = pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',
            columns=['date', 'ts_code', 'hfq_close', 'total_mv_yuan', 'turnover_rate'],
            # 相同历史起点也固定滚动浮点累积，避免临界并列值的rank漂移。
            filters=[('date', '>=', pd.Timestamp(cfg['history'][0])), ('date', '<=', hi)])
        basics = pd.read_parquet(ROOT/'data/raw/stock_basic_all.parquet')
        names = pd.read_parquet(ROOT/'data/raw/namechange_all.parquet')
        table, holds, checks = build_review_inputs(raw, basics, names, cfg)
        checks['raw_input_hash'] = hashlib.sha256(pd.util.hash_pandas_object(raw, index=False).values.tobytes()).hexdigest()
        table.to_parquet(feature_path, index=False)
        for frequency, hold in holds.items():
            hold.to_parquet(out/f'{frequency}_hold_allowed.parquet')
        json_write(out/'feature_checks.json', checks)
        manifest['feature_checks'] = checks
        del raw
    manifest['feature_hash'] = digest(feature_path)
    print(f'PIT特征完成 {len(table)}行；月/周持有状态冻结', flush=True)
    json_write(manifest_path, manifest)
    if args.prepare_only:
        manifest['status'] = 'features_prepared'
        json_write(manifest_path, manifest)
        return
    ready = root/'execution_models_ready.json'
    if not ready.exists():
        raise ValueError('等待执行模型完整登记 execution_models_ready.json')
    models = {name: joblib.load(root/'models'/f'execution_{name}.joblib')
              for name in ['constant', 'style', 'full']}
    for name, model in models.items():
        if model.last_label >= lo or pd.Timestamp(model.metadata['asof']) >= lo:
            raise ValueError(f'{name}训练标签越过执行起点')
    manifest['model_hashes'] = {name: digest(root/'models'/f'execution_{name}.joblib') for name in models}
    matrices, coverage, predictions = make_policy_matrices(table, holds, models, cfg)
    coverage.to_csv(out/'entry_coverage.csv', index=False)
    predictions.to_parquet(out/'pit_review_predictions.parquet', index=False)
    codes = set()
    for (frequency, policy), (signals, forecasts) in matrices.items():
        signals.to_parquet(out/f'{frequency}_{policy}_signals.parquet')
        forecasts.to_parquet(out/f'{frequency}_{policy}_point_forecasts.parquet')
        for _, row in signals.iterrows():
            codes.update(row.dropna().sort_values(ascending=False, kind='stable').head(cfg['retail']['candidate_pool']).index)
    # 全现金矩阵仍需一个行情日历载体，绝不把它加入候选或投资组合。
    manifest['calendar_only_anchor'] = not bool(codes)
    if not codes:
        codes.add('000001.SZ')
    pd.DataFrame({'ts_code': sorted(codes)}).to_csv(out/'execution_codes.csv', index=False)
    cache = Path((parent/'execution_cache.txt').read_text().strip())
    missing = [c for c in sorted(codes) if any(not (cache/api/f'{c}.parquet').exists()
                                              for api in ['dividend', 'stk_limit'])]
    pd.DataFrame({'ts_code': missing}).to_csv(out/'missing_cache_codes.csv', index=False)
    print(f'全部历史入场候选并集 {len(codes)}只；待补缓存 {len(missing)}只', flush=True)
    if missing:
        from prepare_monthly_audit import fetch_execution_data
        cache = fetch_execution_data(missing, (lo, hi), pc)
    (out/'execution_cache.txt').write_text(str(cache))
    manifest.update(status='running', execution_codes=len(codes), downloaded_missing_codes=len(missing))
    json_write(manifest_path, manifest)
    try:
        market, basics, st, limits, actions = load_inputs(out, pc)
    except AccountingError as error:
        manifest.update(status='execution_inputs_failed', error_type=type(error).__name__, error=str(error))
        json_write(manifest_path, manifest)
        raise
    if market.date.max() >= pd.Timestamp('2024-01-01'):
        raise ValueError('执行行情越过TEST边界')
    manifest['execution_input_hashes'] = {name: hashlib.sha256(pd.util.hash_pandas_object(frame,
        index=False).values.tobytes()).hexdigest() for name, frame in [('market', market), ('limits', limits), ('actions', actions)]}
    corporate = yaml.safe_load((ROOT/'configs/corporate_action_components.yaml').read_text())
    summaries, years, failures = [], [], []
    for number, case in enumerate(cases, 1):
        name = case['name']
        case_dir = out/name
        case_dir.mkdir(exist_ok=True)
        print(f'[{number}/{len(cases)}] {name}', flush=True)
        p = deepcopy(pc['portfolio'])
        p['capital'] = case['capital']
        signals, forecasts = matrices[(case['frequency'], case['policy'])]
        try:
            result = run_backtest(signals, market, basics, st, limits, actions,
                lo, hi, p, corporate.get('rights_issues', []), retail_forecasts=forecasts,
                retail_config=cfg['retail'], hold_allowed=holds[case['frequency']],
                extra_execution_delay=case['delay'], slippage_rate=case['slippage_rate'])
            check = reconcile(result, case['capital'])
            process, pnl, episodes, exits, metrics = process_report(result, case['capital'])
            if result.daily.n_hold.max() > cfg['retail']['max_positions']:
                raise AccountingError('真实持仓超过5只')
            for suffix, frame in [('daily', result.daily), ('trades', result.trades),
                ('holdings', result.holdings), ('events', result.events), ('process_daily', process),
                ('asset_pnl', pnl), ('holding_episodes', episodes), ('exit_reasons', exits)]:
                if frame.empty and not len(frame.columns):
                    columns = {'trades': ['date', 'ts_code', 'side', 'shares', 'gross', 'fee'],
                        'holdings': ['date', 'ts_code', 'shares', 'value'], 'events': ['date', 'ts_code', 'kind']}
                    frame = frame.reindex(columns=columns.get(suffix, []))
                frame.to_csv(case_dir/f'{suffix}.csv', index=suffix in ['daily', 'process_daily'])
            summary = {**case, **result.summary, **check, **metrics}
            summaries.append(summary)
            json_write(case_dir/'summary.json', summary)
            previous = case['capital']
            for year, daily in result.daily.groupby(result.daily.index.year):
                stats, _ = daily_metrics(daily.equity, previous, p['annual_days'])
                years.append(dict(name=name, year=year, **stats))
                previous = daily.equity.iloc[-1]
            print(json.dumps({key: summary[key] for key in ['name', 'total', 'mdd', 'mean_holdings', 'total_fees']}, ensure_ascii=False), flush=True)
        except Exception as error:
            # 账本错误包含日期/代码，可报告；下载异常由旧接口负责脱敏。
            failure = dict(name=name, error_type=type(error).__name__, error=str(error))
            failures.append(failure)
            json_write(case_dir/'failure.json', failure)
            print(json.dumps(failure, ensure_ascii=False), flush=True)
        pd.DataFrame(summaries).to_csv(out/'execution_summary.csv', index=False)
        pd.DataFrame(years).to_csv(out/'execution_yearly.csv', index=False)
        pd.DataFrame(failures, columns=['name', 'error_type', 'error']).to_csv(out/'execution_failures.csv', index=False)
        manifest.update(completed_cases=len(summaries), failed_cases=len(failures))
        json_write(manifest_path, manifest)
    manifest['status'] = 'failed' if failures else 'complete'
    json_write(manifest_path, manifest)
    if failures:
        raise AccountingError(f'{len(failures)}个固定情景严格失败，未把其收益计作有效结果')


if __name__ == '__main__':
    main()
