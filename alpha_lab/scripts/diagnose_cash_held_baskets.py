#!/usr/bin/env python3
"""警报收盘原持仓的无费用冻结价格代理；不比较不同未来持仓的策略差值。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def frozen_price_path(prices, delist_day, start, stop, weights):
    """冻结初始单位，缺价沿用；摘牌归零且不因随后出现报价复活。

    weights以警报收盘账户equity为分母，非股票部分保持不变，绝不将
    股票权重重新归一化。MAE按账户权重的收盘代理计算，不是盘中MAE。
    """
    prices = np.asarray(prices, float)
    weights = np.asarray(weights, float)
    delist_day = np.asarray(delist_day, float)
    if (prices.ndim != 2 or prices.shape[1] != len(weights) or delist_day.shape != weights.shape
            or not 0 <= start <= stop < len(prices) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() > 1 + 1e-8):
        raise ValueError('篮子形状、日期或冻结权重非法')
    base = np.empty(len(weights))
    baseline_age = np.empty(len(weights), int)
    for j in range(len(weights)):
        available = np.flatnonzero(np.isfinite(prices[:start+1, j]) & (prices[:start+1, j] > 0))
        if not len(available):
            raise ValueError('真实持仓缺少警报当日及更早后复权标记，不能伪装成现金')
        base[j] = prices[available[-1], j]
        baseline_age[j] = start - available[-1]
        if delist_day[j] <= start and weights[j] > 0:
            raise ValueError('已摘牌股票仍有正起始权重，需先核对真实账本')
    relative = np.ones((stop - start + 1, len(weights)))
    missing = np.zeros_like(relative, dtype=bool)
    dead = np.zeros_like(relative, dtype=bool)
    for offset, i in enumerate(range(start + 1, stop + 1), 1):
        quoted = np.isfinite(prices[i]) & (prices[i] > 0)
        missing[offset] = ~quoted
        dead[offset] = delist_day <= i
        relative[offset] = relative[offset - 1]
        active = quoted & ~dead[offset]
        relative[offset, active] = prices[i, active] / base[active]
        relative[offset, dead[offset]] = 0
    contribution = (relative - 1) * weights
    total = contribution.sum(axis=1)
    return dict(base=base, baseline_age=baseline_age, relative=relative,
                missing=missing, dead=dead, contributions=contribution,
                total_return=total, mae=np.minimum.accumulate(np.minimum(total, 0)))


def endpoints(alarm, dates, horizons):
    """既有5/10/20日观察尺度；精确列明信号、恢复执行及首次买入边界。"""
    start = dates.get_loc(alarm.signal_date)
    targets = {}
    for h in horizons:
        end = start + h
        targets[f'alarm_next_{h}d'] = dates[end] if end < len(dates) else pd.NaT
    recovery = alarm.recovery_execution_date
    targets['recovery_signal_close'] = dates[dates.get_loc(recovery) - 1] if pd.notna(recovery) else pd.NaT
    targets['recovery_execution_close'] = recovery
    targets['first_actual_buy_after_recovery_close'] = alarm.first_actual_buy_after_recovery
    for h in horizons:
        end = dates.get_loc(recovery) + h - 1 if pd.notna(recovery) else len(dates)
        targets[f'recovery_next_{h}d'] = dates[end] if end < len(dates) else pd.NaT
    return targets


def run_diagnostic(run):
    """全部登记账本完成后只读三份5万基线，输出9个独立原持仓篮子。"""
    run = Path(run).resolve()
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    execution = run / 'cash_execution'
    manifest_path = execution / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cases = manifest['cases']
    if len(cases) != cfg['cash_cases'] or any(not (execution / c['name'] / 'summary.json').exists() for c in cases):
        raise ValueError('等待全部16个已登记案例完成；不挑选已完成子集')
    horizon_path = ROOT / 'configs/active_cash_audit.yaml'
    horizons = yaml.safe_load(horizon_path.read_text())['event_horizons']
    if list(horizons) != [5, 10, 20]:
        raise ValueError('观察尺度必须沿用既有5/10/20日')
    market_path = run / 'policy_market.npz'
    with np.load(market_path, allow_pickle=False) as archive:
        dates = pd.DatetimeIndex(archive['dates'])
        codes = pd.Index(archive['codes'])
        prices = archive['hfq_close']
        delist_day = archive['delist_day']
    if (dates.has_duplicates or not dates.is_monotonic_increasing
            or dates.max() >= pd.Timestamp('2024-01-01') or dates.min() < pd.Timestamp('2014-01-01')):
        raise ValueError('价格代理日历越过2014—2023边界或不唯一')
    if codes.has_duplicates or prices.shape != (len(dates), len(codes)):
        raise ValueError('价格矩阵与代码不一致')
    source_paths = [Path(__file__), horizon_path, manifest_path, run / 'config.yaml', market_path]
    components, summaries, daily_paths = [], [], []
    baseline_alarm_dates = None
    for action in cfg['cash_actions']:
        name = f'base_{cfg["reference_capital"]:g}_{action}'
        if not any(case['name'] == name for case in cases):
            raise ValueError(f'没有登记基线案例: {name}')
        folder = execution / name
        paths = {key: folder / (key + '.csv') for key in ['daily', 'holdings', 'alarm_episodes', 'trades']}
        source_paths.extend(paths.values())
        daily = pd.read_csv(paths['daily'], parse_dates=['date']).set_index('date')
        holdings = pd.read_csv(paths['holdings'], parse_dates=['date'])
        episodes = pd.read_csv(paths['alarm_episodes'], parse_dates=[
            'signal_date', 'request_date', 'recovery_execution_date',
            'first_actual_buy_after_recovery', 'next_alarm_date'])
        trades = pd.read_csv(paths['trades'], parse_dates=['date'])
        if len(episodes) != 3 or not episodes.signal_date.between('2022-01-01', '2023-12-31').all():
            raise ValueError('必须保留三个既有警报且不得越过执行窗口')
        alarm_dates = list(episodes.signal_date)
        if baseline_alarm_dates is None:
            baseline_alarm_dates = alarm_dates
        if alarm_dates != baseline_alarm_dates:
            raise ValueError('三个策略的共同市场警报不一致')
        for alarm in episodes.itertuples():
            start = dates.get_loc(alarm.signal_date)
            if dates[start + 1] != alarm.request_date:
                raise ValueError('警报必须来自执行请求前一交易日收盘')
            book = holdings[holdings.date.eq(alarm.signal_date)].sort_values('ts_code')
            equity = float(daily.loc[alarm.signal_date, 'equity'])
            if equity <= 0 or book.ts_code.duplicated().any():
                raise ValueError('警报账户净值或股票身份非法')
            np.testing.assert_allclose(book.value.sum(), daily.loc[alarm.signal_date, 'stock_value'], atol=1e-6)
            if len(book) != int(daily.loc[alarm.signal_date, 'n_hold']):
                raise ValueError('原持仓数量与真实每日账本不一致')
            if pd.notna(alarm.first_actual_buy_after_recovery):
                limit = alarm.next_alarm_date if pd.notna(alarm.next_alarm_date) else pd.Timestamp('2024-01-01')
                buys = trades[trades.side.eq('buy') & trades.date.ge(alarm.recovery_execution_date) & trades.date.lt(limit)]
                if buys.date.min() != alarm.first_actual_buy_after_recovery:
                    raise ValueError('首次恢复后真实买入日期与交易流水不一致')
            positions = codes.get_indexer(book.ts_code)
            if (positions < 0).any():
                raise ValueError('真实持仓股票缺少冻结行情代理')
            weights = book.value.to_numpy(float) / equity
            targets = endpoints(alarm, dates, horizons)
            stops = [dates.get_loc(d) for d in targets.values() if pd.notna(d)]
            stop = max([start] + stops)
            path = frozen_price_path(prices[:, positions], delist_day[positions], start, stop, weights)
            common = dict(case=name, action=action, alarm_id=alarm.alarm_id,
                signal_date=alarm.signal_date, request_date=alarm.request_date,
                initial_equity=equity, initial_stock_value=float(book.value.sum()),
                initial_stock_weight=float(weights.sum()), initial_nonstock_weight=float(1 - weights.sum()),
                initial_n_hold=len(book), zero_holdings=len(book) == 0,
                basket_codes='|'.join(book.ts_code), baseline_stale_names=int((path['baseline_age'] > 0).sum()))
            for j, stock in enumerate(book.itertuples()):
                components.append(dict(**common, ts_code=stock.ts_code, shares=stock.shares,
                    original_price=stock.price, original_value=stock.value, frozen_weight=weights[j],
                    baseline_hfq_price=path['base'][j], baseline_hfq_age=int(path['baseline_age'][j]),
                    ledger_stale_days=stock.stale_days))
            for offset, day in enumerate(dates[start:stop+1]):
                daily_paths.append(dict(**common, date=day, return_proxy=path['total_return'][offset],
                    mae_proxy=path['mae'][offset], missing_price_names=int(path['missing'][offset].sum()),
                    delisted_names=int(path['dead'][offset].sum())))
            for window, endpoint in targets.items():
                complete = pd.notna(endpoint)
                offset = dates.get_loc(endpoint) - start if complete else None
                summaries.append(dict(**common, window=window, endpoint=endpoint, complete=complete,
                    return_proxy=path['total_return'][offset] if complete else np.nan,
                    mae_proxy=path['mae'][offset] if complete else np.nan,
                    trading_intervals=offset, missing_price_observations=int(path['missing'][:offset+1].sum()) if complete else None,
                    delisted_names=int(path['dead'][offset].sum()) if complete else None,
                    overlaps_next_alarm=bool(complete and pd.notna(alarm.next_alarm_date) and endpoint >= alarm.next_alarm_date)))
    output = run / 'cash_held_baskets'
    output.mkdir(exist_ok=True)
    pd.DataFrame(components).to_csv(output / 'initial_basket_components.csv', index=False)
    pd.DataFrame(daily_paths).to_csv(output / 'frozen_basket_daily.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'frozen_basket_windows.csv', index=False)
    report = dict(status='complete', cases=3, alarms_per_case=3, windows=len(summaries),
        no_training=True, no_new_strategy=True, no_TEST=True,
        weight_definition='alarm_signal_close_actual_holding_value_divided_by_account_equity_no_renormalization',
        semantics='fee_free_frozen_original_holdings_adjusted_price_proxy_not_actual_trading_profit_or_pure_timing_attribution',
        nonstock='initial_cash_and_receivables_kept_constant_zero_return',
        mark_rules='only_prior_or_current_price_for_baseline; missing_future_close_carry; delist_zero_no_revival',
        mae_definition='minimum_cumulative_account_weighted_close_return_including_zero_start_not_intraday_low',
        endpoints='all_returns_from_alarm_signal_close; recovery_next_h ends hth_close_including_recovery_execution_day',
        limitations=['各策略警报前持仓可能不同，逐篮子保存；不是同一五股跨策略随机对照。',
            '未来真实拆分、股息现金、交易费用和后续买卖账本不加入该价格代理。',
            '零持仓保留，初始非股票资产零收益；不据三个警报挑策略或新阈值。'],
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.run), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
