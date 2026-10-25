#!/usr/bin/env python3
"""扩窗低换手账户独立复核：原账本状态机只替换预登记日期边界。"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
COPY_SOURCE = 'scripts/verify_divide_selection_execution.py'
COPY_SOURCE_SHA256 = '8ba7c0e207e8bb562220f73a5acbffc453672582371a0028b0b5ecc1869452cd'
ORIGINAL_FUNCTION_SHA256 = '757da47480f5f76415690920ca7ac9d69b47a48cd53e73f3ba468900f632183c'
OLD_BOUNDARY = "require(daily.date.max() < pd.Timestamp('2024-01-01'), '禁止TEST产物')"
NEW_BOUNDARY = "require(daily.date.between(*cfg['verification_window']).all(), '账户日期超出预登记窗口')"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, message):
    require(np.isfinite([a, b]).all() and abs(a-b) < 1e-6, f'{message}: {a} != {b}')


def read_csv(path):
    frame = pd.read_csv(path)
    for col in ['date', 'signal_date', 'entry_date', 'close_date']:
        if col in frame:
            frame[col] = pd.to_datetime(frame[col])
    return frame


def verify_case(case_dir, predictions, cfg, execution_config):
    summary = json.loads((case_dir/'summary.json').read_text())
    policy = summary['policy']
    daily = read_csv(case_dir/'daily.csv')
    trades = read_csv(case_dir/'trades.csv')
    events = read_csv(case_dir/'events.csv')
    holdings = read_csv(case_dir/'holdings.csv')
    episodes = read_csv(case_dir/'entry_phase_episodes.csv')
    pred = predictions.set_index(['date', 'ts_code'])
    require(daily.date.is_monotonic_increasing and not daily.date.duplicated().any(), '日历重复或倒序')
    require(daily.date.between(*cfg['verification_window']).all(), '账户日期超出预登记窗口')
    require(daily.n_hold.max() <= 5, '超过五仓')
    cash, claims = float(summary['capital']), 0.
    shares = defaultdict(int)
    active, rebuilt = {}, []
    max_cash_error, max_equity_error = 0., 0.
    known_sells = set()
    nonselected_forecast_overrides = 0
    for day in daily.itertuples():
        day_events = events[events.date.eq(day.date)]
        for event in day_events.to_dict('records'):
            code, kind = event['ts_code'], event['kind']
            if kind in ['hold_exit_requested', 'hold_risk_cap_reduction']:
                known_sells.add(code)
            if kind == 'dividend_receivable':
                claims += event['amount']
                require(code in active, '无持仓段的分红权益')
                active[code]['pnl'] += event['amount']
            elif kind == 'dividend_paid':
                cash += event['amount']; claims -= event['amount']
            elif kind == 'bonus_shares':
                require(code in active, '无持仓段的送转股')
                shares[code] += int(event['shares'])
            elif kind == 'delist_writeoff':
                require(code in active and shares[code] == int(event['shares']), '退市股数错误')
                cash += event['cash_recovery']
                active[code]['pnl'] += event['cash_recovery']
                active[code]['close_date'] = day.date
                del active[code]; shares[code] = 0
            elif kind == 'allocation_decision':
                close(event['target_shares'], event['old_shares'] + event['buy_shares'] - event['sell_shares'], '分配股数')
                require(event['target_shares'] >= event['min_target_shares'], '违反旧仓保留下界')
                require(not (event['old_shares'] > 0 and event['buy_shares'] > 0), '旧仓补仓')
            elif kind == 'selection_decision':
                require(event['signal_date'] < day.date, '使用当日收盘信号成交')
                source = pred.loc[(event['signal_date'], code)]
                close(event['priority'], source[f'{policy}__score'], '优先级偏离冻结分数')
                point = source[f'{policy}__expected_return']
                if event['reason'] == 'selected' or abs(event['expected_return']-point) < 1e-10:
                    close(event['expected_return'], point, '真实μ偏离冻结点预测')
                else:
                    require(event['reason'] == 'buy_not_allowed' and event['expected_return'] in [0., -1.],
                            '非入场预测变更并非ST/未知报价否决')
                    nonselected_forecast_overrides += 1
                if event['reason'] == 'selected':
                    require(event['reserved_shares'] > 0 and event['reserved_net_profit'] > 0, '预约缺少正净利')
        for trade in trades[trades.date.eq(day.date)].to_dict('records'):
            code = trade['ts_code']
            require(trade['shares'] > 0 and trade['shares'] == int(trade['shares']), '成交股数非法')
            close(trade['gross'], trade['shares'] * trade['price'], '成交价款')
            spec = execution_config['portfolio']
            rate = spec['buy_cost'] if trade['side'] == 'buy' else spec['sell_cost']
            close(trade['fee'], max(spec['min_fee'], trade['gross'] * rate), '成交费率')
            if trade['side'] == 'buy':
                require(shares[code] == 0 and code not in active, '排名驱动旧仓补仓或混合入场')
                known_sells.discard(code)
                selections = day_events[day_events.kind.eq('selection_decision') & day_events.ts_code.eq(code)]
                require(len(selections) == 1, '买入缺少唯一预约')
                selected = selections.iloc[0]
                require(selected.reason == 'selected' and trade['shares'] >= selected.reserved_shares, '实际股数不足经济预约')
                phase = pred.loc[(trade['signal_date'], code), 'phase']
                item = dict(ts_code=code, entry_date=day.date, signal_date=trade['signal_date'],
                            phase=phase, pnl=-trade['gross']-trade['fee'], close_date=pd.NaT)
                active[code] = item; rebuilt.append(item)
                shares[code] += int(trade['shares'])
                cash -= trade['gross'] + trade['fee']
            else:
                require(code in active and code in known_sells, '卖出缺少明确退出或必要风控减仓')
                shares[code] -= int(trade['shares'])
                require(shares[code] >= 0, '卖空')
                cash += trade['gross'] - trade['fee']
                active[code]['pnl'] += trade['gross']-trade['fee']
                if not shares[code]:
                    active[code]['close_date'] = day.date
                    del active[code]
            close(cash, trade['cash_after'], '逐笔现金')
        actual = holdings[holdings.date.eq(day.date)]
        actual_shares = {r.ts_code: int(r.shares) for r in actual.itertuples()}
        require(actual_shares == {code: qty for code, qty in shares.items() if qty}, '逐代码每日股份不守恒')
        for row in actual.itertuples():
            close(row.value, row.shares * row.price, '持仓市值')
        close(cash, day.cash, '每日现金')
        close(claims, day.receivables, '分红应收')
        close(actual.value.sum(), day.stock_value, '每日股票资产')
        close(cash + claims + actual.value.sum(), day.equity, '总资产')
        max_cash_error = max(max_cash_error, abs(cash-day.cash))
        max_equity_error = max(max_equity_error, abs(cash+claims+actual.value.sum()-day.equity))
    for row in holdings[holdings.date.eq(daily.date.max())].itertuples():
        active[row.ts_code]['pnl'] += row.value
    require(len(episodes) == len(rebuilt), '入场持仓段数量不同')
    for item, recorded in zip(rebuilt, episodes.to_dict('records')):
        require(item['ts_code'] == recorded['ts_code'] and item['entry_date'] == recorded['entry_date']
                and item['signal_date'] == recorded['signal_date'] and item['phase'] == recorded['entry_phase'], '阶段不是入场时已知阶段')
        require((pd.isna(item['close_date']) and pd.isna(recorded['close_date']))
                or item['close_date'] == recorded['close_date'], '持仓段结束日不同')
        close(item['pnl'], recorded['pnl_yuan'], '逐持仓段净利润')
    close(sum(r['pnl'] for r in rebuilt), daily.equity.iloc[-1]-summary['capital'], '阶段归因总净利润')
    return dict(name=case_dir.name, days=len(daily), trades=len(trades), episodes=len(rebuilt),
                cash_max_error=max_cash_error, equity_max_error=max_equity_error,
                nonselected_forecast_overrides=nonselected_forecast_overrides, verified=True)


def verify_copy_provenance():
    """逐字证明独立状态机只修改一个日期断言，其他要求无变化。"""
    source_path = ROOT / COPY_SOURCE
    source = source_path.read_text()
    require(digest(source_path) == COPY_SOURCE_SHA256, '复制来源代码发生漂移')
    original = next(ast.get_source_segment(source, node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'verify_case')
    current_source = Path(__file__).read_text()
    copied = next(ast.get_source_segment(current_source, node) for node in ast.parse(current_source).body
                  if isinstance(node, ast.FunctionDef) and node.name == 'verify_case')
    require(hashlib.sha256(original.encode()).hexdigest() == ORIGINAL_FUNCTION_SHA256,
            '复制来源函数哈希不同')
    require(original.count(OLD_BOUNDARY) == 1, '原日期边界出现次数错误')
    require(copied == original.replace(OLD_BOUNDARY, NEW_BOUNDARY), '状态机存在日期边界以外修改')
    return dict(source=COPY_SOURCE, source_sha256=COPY_SOURCE_SHA256,
                original_function_sha256=ORIGINAL_FUNCTION_SHA256,
                copied_function_sha256=hashlib.sha256(copied.encode()).hexdigest(),
                sole_change=dict(old=OLD_BOUNDARY, new=NEW_BOUNDARY))


def prepare_execution_evidence(market, limits):
    """只接收冻结的逐股原价及官方涨跌停证据，拒绝重复键。"""
    market = market.copy()
    limits = limits.copy()
    if 'date' not in limits and 'trade_date' in limits:
        limits['date'] = pd.to_datetime(limits['trade_date'].astype(str))
    for frame in [market, limits]:
        frame['date'] = pd.to_datetime(frame['date'])
        require(not frame.duplicated(['date', 'ts_code']).any(), '执行证据日期/代码重复')
    require({'open', 'close', 'vol_share'}.issubset(market.columns), '原价证据字段缺失')
    require({'up_limit', 'down_limit'}.issubset(limits.columns), '官方限价字段缺失')
    return market.set_index(['date', 'ts_code']).sort_index(), limits.set_index(['date', 'ts_code']).sort_index()


def verify_market_fills(case_dir, spec, market, limits, window):
    """逐成交核验原始开盘价、官方限价、成交量及 T+1，逐持仓核验有效收盘估值。"""
    trades = read_csv(case_dir / 'trades.csv')
    holdings = read_csv(case_dir / 'holdings.csv')
    daily = read_csv(case_dir / 'daily.csv')
    expected_days = pd.DatetimeIndex(sorted(set(market.index.get_level_values('date'))))
    expected_days = expected_days[(expected_days >= window[0]) & (expected_days <= window[1])]
    require(pd.DatetimeIndex(daily.date).equals(expected_days), '账户日历与冻结行情日历不完整对应')
    require(spec['require_official_limits'] is True and spec['strict_actions'] is True,
            '官方限价或严格公司行动检查被关闭')
    mainboard = r'(?:(?:000|001|002|003)[0-9]{3}\.SZ|(?:600|601|603|605)[0-9]{3}\.SH)'
    for frame in [trades, holdings]:
        require(frame.ts_code.dropna().map(lambda s: bool(re.fullmatch(mainboard, str(s)))).all(), '出现非主板真实账户记录')
    last_buy = {}
    checked = []
    for trade in trades.itertuples():
        key = (trade.date, trade.ts_code)
        require(key in market.index and key in limits.index, '真实成交缺原价或官方限价证据')
        quote, bound = market.loc[key], limits.loc[key]
        values = [quote['open'], quote['vol_share'], bound['up_limit'], bound['down_limit']]
        require(np.isfinite(values).all() and min(values) > 0, '成交时价格、成交量或官方限价无效')
        close(trade.price, quote['open'], '成交不是未复权原始开盘价')
        half_tick = spec['price_tick'] / 2
        require(bound['down_limit'] - half_tick <= trade.price <= bound['up_limit'] + half_tick,
                '成交价在官方涨跌停范围之外')
        require(trade.side in ['buy', 'sell'], '成交方向非法')
        if trade.side == 'buy':
            require(trade.price < bound['up_limit'] - half_tick, '涨停开盘买入')
            require(trade.signal_date in expected_days, '买入信号不在执行行情日历')
            signal_index = expected_days.get_loc(trade.signal_date)
            require(signal_index + 1 < len(expected_days)
                    and expected_days[signal_index + 1] == trade.date, '买入未使用信号次交易日')
            prior = market.loc[(trade.signal_date, trade.ts_code)]
            require(np.isfinite(prior.vol_share) and prior.vol_share > 0, '信号日成交量无效')
            require(trade.shares <= prior.vol_share * spec['max_volume_participation'] + 1e-8,
                    '买入突破冻结成交量上限')
            rule = spec['boards']['main']
            require(trade.shares >= rule['min_buy']
                    and (trade.shares - rule['min_buy']) % rule['buy_step'] == 0, '买入不符合主板整手规则')
            last_buy[trade.ts_code] = trade.date
        else:
            require(trade.price > bound['down_limit'] + half_tick, '跌停开盘卖出')
            require(trade.ts_code in last_buy and last_buy[trade.ts_code] < trade.date, '违反买入 T+1')
        checked.append(dict(date=trade.date.isoformat(), ts_code=trade.ts_code,
                            side=trade.side, raw_open=float(quote['open']),
                            up_limit=float(bound['up_limit']), down_limit=float(bound['down_limit']), verified=True))
    valid_marks, stale_marks = 0, 0
    for hold in holdings.itertuples():
        key = (hold.date, hold.ts_code)
        raw_close = market.loc[key, 'close'] if key in market.index else np.nan
        if np.isfinite(raw_close) and raw_close > 0:
            close(hold.price, raw_close, '有效报价日持仓不是未复权收盘估值')
            require(hold.stale_days == 0, '有效收盘日仍标记陈旧估值')
            valid_marks += 1
        else:
            require(hold.stale_days > 0, '无效收盘日未标记陈旧估值')
            stale_marks += 1
    return dict(official_limit_and_raw_open_checks=len(checked), valid_close_marks=valid_marks,
                stale_marks=stale_marks, fills=checked)


def performance_report(case_dir, capital, recent_start):
    """独立用总资产计算整段及近期窗口收益，续跑近期以窗前净值作分母。"""
    daily = read_csv(case_dir / 'daily.csv')
    summary = json.loads((case_dir / 'summary.json').read_text())
    values = daily.equity.to_numpy(float)
    whole_return = values[-1] / capital - 1
    mdd = np.min(values / np.maximum.accumulate(np.r_[capital, values])[1:] - 1)
    close(whole_return, summary['total'], '总收益不等于总资产变化')
    close(mdd, summary['mdd'], '回撤未包含初始资金峰值')
    recent = daily[daily.date.ge(recent_start)]
    require(not recent.empty, '近期报告窗口为空')
    before = daily[daily.date.lt(recent_start)]
    opening_equity = float(before.equity.iloc[-1]) if len(before) else float(capital)
    recent_values = recent.equity.to_numpy(float)
    return dict(initial_capital=capital, final_cash=float(daily.cash.iloc[-1]),
                final_equity=float(values[-1]), total_return=float(whole_return),
                mdd=float(mdd), recent_start_equity=opening_equity,
                recent_total_return=float(recent_values[-1] / opening_equity - 1),
                recent_mdd=float(np.min(recent_values / np.maximum.accumulate(np.r_[opening_equity, recent_values])[1:] - 1)))


def compare_reverse(out, cases):
    rows = []
    for reverse in [c for c in cases if c['reverse_input']]:
        candidates = [c for c in cases if not c['reverse_input'] and all(
            c[key] == reverse[key] for key in ['capital', 'start', 'cost_multiplier', 'policy'])]
        require(len(candidates) == 1, '倒序情景缺少唯一基准')
        base = candidates[0]
        left, right = out / base['name'], out / reverse['name']
        files = sorted(p.name for p in left.glob('*.csv'))
        require(files and files == sorted(p.name for p in right.glob('*.csv')), '倒序 CSV 文件集不同')
        for name in files:
            a, b = pd.read_csv(left / name), pd.read_csv(right / name)
            if name == 'yearly.csv' and 'name' in a:
                a, b = a.drop(columns='name'), b.drop(columns='name')
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            rows.append(dict(base=base['name'], reverse=reverse['name'], file=name, identical=True))
    return rows


def compare_prefix_frame(reference, continued, name):
    """只容许事件新增列在旧窗口全部为空；经济数值保持严格相等。"""
    reference, continued = reference.copy(), continued.copy()
    missing = set(reference) - set(continued)
    require(not missing, f'旧前缀缺少原列: {name} {sorted(missing)}')
    added = sorted(set(continued) - set(reference))
    require(not added or name in ['events.csv', 'selection_trace.csv'],
            f'非事件文件增加未核验前缀列: {name} {added}')
    for column in added:
        require(continued[column].isna().all(), f'事件新增列在旧前缀含有效值: {name} {column}')
        reference[column] = np.nan
    continued = continued.reindex(columns=reference.columns)
    dtype_changes = {column: [str(reference[column].dtype), str(continued[column].dtype)]
                     for column in reference if reference[column].dtype != continued[column].dtype}
    pd.testing.assert_frame_equal(reference.reset_index(drop=True), continued.reset_index(drop=True),
                                  check_exact=True, check_dtype=False)
    return dict(file=name, rows=len(reference), identical_economic_values=True,
                added_all_null_columns=added, dtype_changes=dtype_changes)


def verify_continued_prefix(out, cases, recent_start, reference):
    """对照冻结 P3 无警报动作账户，证明 2022–2023 交易前缀保持一致。"""
    continued = [case for case in cases if pd.Timestamp(case['start']) < recent_start]
    require(len(continued) == 1, '续跑前缀缺少唯一案例')
    case = continued[0]
    original_summary = json.loads((reference / 'summary.json').read_text())
    close(case['capital'], original_summary['capital'], '旧前缀参考账户本金不同')
    require(case['cost_multiplier'] == 1 and not case['reverse_input'], '旧前缀参考情景不一致')
    original_daily = read_csv(reference / 'daily.csv')
    require(original_daily.date.max() < recent_start
            and original_daily.date.min() >= pd.Timestamp(case['start']), '旧参考账户不在完整旧窗口')
    files = ['daily.csv', 'trades.csv', 'holdings.csv', 'events.csv',
             'actual_entry_priorities.csv', 'selection_trace.csv', 'process_daily.csv', 'yearly.csv']
    results = []
    for name in files:
        left, right = pd.read_csv(reference / name), pd.read_csv(out / case['name'] / name)
        if name == 'yearly.csv':
            left, right = left.drop(columns='name'), right.drop(columns='name')
            right = right[right.year < recent_start.year].copy()
            # 新年度账本额外保存首尾净值时，从原每日账本重建，不能直接丢弃经济字段。
            for column in ['start_equity', 'end_equity']:
                if column in right and column not in left:
                    previous = float(original_summary['capital'])
                    values = {}
                    for year, frame in original_daily.groupby(original_daily.date.dt.year):
                        values[year] = previous if column == 'start_equity' else float(frame.equity.iloc[-1])
                        previous = float(frame.equity.iloc[-1])
                    left[column] = left.year.map(values)
        else:
            require('date' in left and 'date' in right, f'前缀文件没有日期: {name}')
            right = right[pd.to_datetime(right.date).lt(recent_start)].copy()
        result = compare_prefix_frame(left, right, name)
        result.update(reference_sha256=digest(reference / name),
                      continued_sha256=digest(out / case['name'] / name))
        results.append(result)
    return dict(reference_case=str(reference), continued_case=case['name'],
                before=str(recent_start.date()), comparisons=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.run.resolve() / 'execution'
    manifest = json.loads((run / 'manifest.json').read_text())
    provenance = verify_copy_provenance()
    require(manifest.get('no_result_driven_changes') is True, '未登记固定参数纪律')
    require(manifest.get('refit') is False, '未登记模型不重拟合')
    lo, hi = pd.Timestamp(manifest['start']), pd.Timestamp(manifest['end'])
    require(lo <= hi, '预登记日期顺序错误')
    cases = manifest['cases']
    require(cases and len({c['name'] for c in cases}) == len(cases), '预登记案例为空或重复')
    observed = {p.parent.name for p in out.glob('*/summary.json')}
    require(observed == {c['name'] for c in cases}, '执行案例集合与预登记不同')
    data_manifest = json.loads((out / 'data_manifest.json').read_text())
    required_evidence = {'market.parquet', 'limits.parquet', 'normalized_actions.parquet',
                         'monthly_hold_allowed.parquet', 'low_turnover_signals.parquet',
                         'low_turnover_point_forecasts.parquet', 'execution_config.yaml'}
    require(required_evidence.issubset(data_manifest['hashes']), '预执行证据哈希不完整')
    for name, expected in data_manifest['hashes'].items():
        require(digest(out / name) == expected, '预执行证据发生漂移: ' + name)
    require(digest(run / 'execution_predictions.parquet') == data_manifest['prediction_hash'],
            '预执行预测哈希发生漂移')
    frozen_config = yaml.safe_load((out / 'execution_config.yaml').read_text())
    checked_sources = {}
    for name, expected in manifest['source_hashes'].items():
        require(digest(ROOT / name) == expected, f'执行源码漂移: {name}')
        require(digest(run / 'sources' / name) == expected, f'冻结源码快照漂移: {name}')
        checked_sources[name] = expected
    predictions = pd.read_parquet(run / 'execution_predictions.parquet')
    predictions['date'] = pd.to_datetime(predictions.date)
    earliest = min(pd.Timestamp(c['start']) for c in cases)
    require(predictions.date.between(earliest, hi).all(), '预测超出预登记完整窗口')
    require(not predictions.duplicated(['date', 'ts_code']).any(), '冻结预测日期/代码重复')
    market_path, limit_path = out / 'market.parquet', out / 'limits.parquet'
    market, limits = prepare_execution_evidence(pd.read_parquet(market_path), pd.read_parquet(limit_path))
    reports, fills = [], []
    config_specs = {}
    for case in cases:
        folder = out / case['name']
        saved = json.loads((folder / 'portfolio_config.json').read_text())
        spec = saved['portfolio'] if 'portfolio' in saved else saved
        expected_spec = dict(frozen_config['portfolio'])
        expected_spec.update(capital=case['capital'], top_n=5)
        for field in ['buy_cost', 'sell_cost', 'min_fee']:
            expected_spec[field] *= case['cost_multiplier']
        require(spec == expected_spec, '账户参数不等于冻结配置加预登记本金/费用变化')
        config_specs[case['name']] = spec
        summary = json.loads((folder / 'summary.json').read_text())
        close(spec['capital'], case['capital'], '配置本金偏离登记')
        close(summary['capital'], case['capital'], '摘要本金偏离登记')
        require(summary['policy'] == case['policy'], '策略偏离登记')
        window = (pd.Timestamp(case['start']), hi)
        account = verify_case(folder, predictions, {'verification_window': window}, {'portfolio': spec})
        execution = verify_market_fills(folder, spec, market, limits, window)
        fills.extend(dict(case=case['name'], **r) for r in execution.pop('fills'))
        reports.append({**account, **execution, **performance_report(folder, case['capital'], lo)})
    base_candidates = [c for c in cases if c['capital'] == 50000 and c['cost_multiplier'] == 1
                       and c['start'] == manifest['start'] and not c['reverse_input']]
    require(len(base_candidates) == 1, '费用压力缺少唯一五万元冷启动基准')
    base = config_specs[base_candidates[0]['name']]
    for case in cases:
        for field in ['buy_cost', 'sell_cost', 'min_fee']:
            close(config_specs[case['name']][field], base[field] * case['cost_multiplier'], '费用倍数不符: ' + field)
    reverse = compare_reverse(out, cases)
    reference = Path(manifest.get('prefix_reference_case', ROOT /
        'outputs/policy_cash/20260913T063048_495344Z/cash_execution/base_50000_none'))
    prefix = verify_continued_prefix(out, cases, lo, reference)
    artifacts = [run / 'manifest.json', run / 'execution_predictions.parquet', market_path, limit_path,
                 out / 'data_manifest.json', out / 'execution_config.yaml']
    artifacts += sorted(out.glob('*/portfolio_config.json'))
    artifacts += sorted(out.glob('*hold*.parquet'))
    if (out / 'normalized_actions.parquet').exists():
        artifacts.append(out / 'normalized_actions.parquet')
    report = dict(status='complete', copied_state_machine=provenance, run_window=[str(lo.date()), str(hi.date())],
                  cases=reports, reverse_comparisons=reverse, continued_prefix=prefix,
                  source_hashes=checked_sources,
                  input_hashes={str(p.relative_to(run)): digest(p) for p in artifacts},
                  verifier_sha256=digest(Path(__file__)), market_data_read=True,
                  scope='仅已冻结执行证据；未下载、未修改策略、未重新拟合')
    (out / 'independent_recent_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    pd.DataFrame(fills).to_csv(out / 'independent_fill_checks.csv', index=False)
    print(json.dumps(dict(status='complete', cases=len(reports), fill_checks=len(fills),
                          reverse_csv_checks=len(reverse)), ensure_ascii=False))


if __name__ == '__main__':
    main()
