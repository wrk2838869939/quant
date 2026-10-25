#!/usr/bin/env python3
"""只读冻结产物，独立逐股重建20组执行现金、权益与入场阶段；不读取行情。"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


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
    require(daily.date.max() < pd.Timestamp('2024-01-01'), '禁止TEST产物')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=None)
    args = parser.parse_args()
    run = args.run or Path(Path('/tmp/divide_selection_run_path').read_text().strip())
    out = run/'execution'
    manifest = json.loads((out/'manifest.json').read_text())
    require(manifest['status'] == 'complete', '执行矩阵未完整结束')
    cfg = yaml.safe_load((run/'config.yaml').read_text())
    ecfg = yaml.safe_load((out/'execution_config.yaml').read_text())
    sources = []
    for name, expected in manifest['source_hashes'].items():
        snapshot = digest(out/'sources'/name)
        current = digest(ROOT/name)
        require(snapshot == expected, f'源快照哈希不匹配: {name}')
        require(current == expected, f'执行后源码发生漂移: {name}')
        sources.append(dict(path=name, snapshot_matches=True, current_matches=True))
    require(digest(run/'execution_predictions.parquet') == manifest['prediction_hash'], '预测文件漂移')
    require(digest(out/'monthly_hold_allowed.parquet') == manifest['hold_hash'], '持有日历漂移')
    predictions = pd.read_parquet(run/'execution_predictions.parquet')
    require(predictions.date.max() < pd.Timestamp('2024-01-01'), '预测超出TEST边界')
    rows = [verify_case(out/case['name'], predictions, cfg, ecfg) for case in manifest['cases']]
    compared = []
    for policy in cfg['policies']:
        base = out/f'base_{cfg["reference_capital"]:g}_{policy}'
        reverse = out/f'reverse_{cfg["reference_capital"]:g}_{policy}'
        files = sorted(p.name for p in base.glob('*.csv'))
        for name in files:
            a, b = pd.read_csv(base/name), pd.read_csv(reverse/name)
            if name == 'yearly.csv':
                a, b = a.drop(columns='name'), b.drop(columns='name')
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        compared.append(dict(policy=policy, exact_equal_csv_files=len(files)))
    report = dict(status='complete', market_data_read=False, test_market_data_used=False,
                  cases=rows, source_verification=sources, reverse_comparisons=compared,
                  checks=['independent_per_stock_daily_shares', 'per_trade_and_daily_cash',
                          'receivables_and_equity', 'retained_minimum_and_no_topups',
                          'frozen_priority_separate_from_mu', 'economic_reservation_filled',
                          'known_entry_phase_and_episode_pnl', 'all_reverse_csv_exact_equality'])
    (out/'independent_execution_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(dict(status='complete', cases=len(rows), reversed_policies=len(compared)), ensure_ascii=False))


if __name__ == '__main__':
    main()
