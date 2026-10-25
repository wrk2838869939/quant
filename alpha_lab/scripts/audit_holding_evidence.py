#!/usr/bin/env python3
"""P0：复现冻结持有日历、C1证据记忆对照及15基线三时钟诊断。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from holding_evidence import evidence_hold_states
from patient_trend import review_dates, trend_state


def digest(path):
    """仅对登记文件计算哈希。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_function(path):
    """编译冻结的原始纯状态函数，避免导入数据下载依赖。"""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'confirmed_hold_states')
    scope = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['confirmed_hold_states']


def read_csv(path, date_columns=()):
    """读取已有开发CSV；日期逐列显式转换并验证冻结边界。"""
    frame = pd.read_csv(path)
    for col in date_columns:
        frame[col] = pd.to_datetime(frame[col])
        if frame[col].dropna().ge('2024-01-01').any():
            raise ValueError(f'CSV越过开发边界: {path}:{col}')
    return frame


def diagnose_cases(source, hold, out, inputs):
    """请求日志、实际清仓与行政删失分别记录；部分卖出不结束持仓段。"""
    summary, rows = [], []
    cases = read_csv(source/'execution/execution_summary.csv')
    inputs.append(source/'execution/execution_summary.csv')
    cases = cases[cases.name.str.startswith('base_')]
    if len(cases) != 15:
        raise ValueError('必须审计完整15个基线')
    for case in cases.itertuples():
        folder = source/'execution'/case.name
        files = {name: folder/f'{name}.csv' for name in
                 ['entry_phase_episodes', 'holding_periods', 'events', 'trades', 'daily']}
        inputs.extend(files.values())
        episodes = read_csv(files['entry_phase_episodes'], ['entry_date', 'signal_date', 'close_date'])
        periods = read_csv(files['holding_periods'], ['first_mark', 'last_mark'])
        events = read_csv(files['events'], ['date', 'signal_date'])
        trades = read_csv(files['trades'], ['date', 'signal_date'])
        daily = read_csv(files['daily'], ['date'])
        calendar = pd.DatetimeIndex(daily.date)
        end = calendar[-1]
        if len(episodes) != len(periods):
            raise ValueError(f'{case.name}: 实际入场段与持仓标记段数量不同')
        local = []
        for ep in episodes.itertuples():
            closed = pd.notna(ep.close_date)
            stop = ep.close_date if closed else end
            marks = periods[periods.ts_code.eq(ep.ts_code) & periods.first_mark.eq(ep.entry_date)]
            if len(marks) != 1 or bool(marks.iloc[0].open_at_end) == closed:
                raise ValueError('持仓段闭合/删失标志不一致')
            mark = marks.iloc[0]
            ev = events[events.ts_code.eq(ep.ts_code) & events.date.between(ep.entry_date, stop)]
            req = ev[ev.kind.eq('hold_exit_requested')].sort_values('date')
            request = req.iloc[0] if len(req) else None
            actual = trades[trades.ts_code.eq(ep.ts_code) & trades.date.between(ep.entry_date, stop)]
            buys = actual[actual.side.eq('buy')]
            sells = actual[actual.side.eq('sell')]
            if len(buys) != 1 or buys.iloc[0].date != ep.entry_date:
                raise ValueError('入场日缺少唯一真实买单')
            liquidation_kind = ('sell' if closed and len(sells) and sells.date.max() == ep.close_date
                else 'delist_writeoff' if closed and ev.kind.eq('delist_writeoff').any()
                else 'unresolved' if closed else 'administrative_censor')
            if closed and liquidation_kind == 'unresolved':
                raise ValueError('已闭合段缺少真实终止记录')
            request_date = request.date if request is not None else pd.NaT
            request_signal = request.signal_date if request is not None else pd.NaT
            row = dict(case=case.name, capital=case.capital, policy=case.policy,
                episode_id=ep.episode_id, ts_code=ep.ts_code,
                entry_signal=ep.signal_date, entry_fill=ep.entry_date,
                first_exit_request_decision=request_signal, first_exit_request_logged=request_date,
                exit_fill=ep.close_date, censor_date=pd.NaT if closed else end,
                closed=closed, censored=not closed, liquidation_kind=liquidation_kind,
                first_request_reason=request.reason if request is not None else '',
                request_observed=request is not None,
                complete_request_fill_clock=bool(request is not None and closed),
                signal_to_request_review_intervals=int(((hold.index > ep.signal_date)
                    & (hold.index <= request_signal)).sum()) if request is not None else np.nan,
                fill_to_end_trading_intervals=int(calendar.get_loc(stop)-calendar.get_loc(ep.entry_date)),
                marked_holding_trading_days=int(mark.holding_trading_days),
                request_to_fill_trading_intervals=int(calendar.get_loc(ep.close_date)
                    - calendar.get_loc(request_date)) if request is not None and closed else np.nan,
                pnl_observation_end=end, pnl_observation_includes_post_exit_cash=closed,
                episode_pnl_to_cutoff_yuan=ep.pnl_yuan,
                unknown_hold_reviews=int((ev.kind.eq('hold_retained') & ev.reason.eq('unknown')).sum()),
                sell_deferred_events=int(ev.kind.eq('sell_deferred').sum()),
                closed_mark_duration_is_natural_trend_lifetime=False)
            local.append(row)
        frame = pd.DataFrame(local)
        rows.extend(local)
        summary.append(dict(case=case.name, capital=case.capital, policy=case.policy,
            episodes=len(frame), closed=int(frame.closed.sum()), censored=int(frame.censored.sum()),
            entry_signal_and_fill_observed=len(frame), request_observed=int(frame.request_observed.sum()),
            complete_request_fill_clocks=int(frame.complete_request_fill_clock.sum()),
            terminal_accounting_observed=len(frame),
            unknown_hold_reviews=int(frame.unknown_hold_reviews.sum()),
            sell_deferred_events=int(frame.sell_deferred_events.sum()),
            closed_only_mark_days_median=frame.loc[frame.closed, 'marked_holding_trading_days'].median(),
            all_observed_mark_days_median=frame.marked_holding_trading_days.median(),
            complete_request_fill_wait_max=frame.request_to_fill_trading_intervals.max(),
            pnl_reconciliation_error=float(abs(frame.episode_pnl_to_cutoff_yuan.sum()
                -(daily.equity.iloc[-1]-case.capital)))))
    table = pd.DataFrame(summary)
    if table.pnl_reconciliation_error.max() > 1e-6:
        raise ValueError('截止期持仓段利润无法还原账户')
    table.to_csv(out/'baseline_15_clock_summary.csv', index=False)
    pd.DataFrame(rows).to_csv(out/'baseline_holding_clocks.csv', index=False)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    out = args.run/'holding_evidence'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('拒绝覆盖已经登记的P0研究产物')
    source = args.source_run
    source_cfg = yaml.safe_load((source/'config.yaml').read_text())
    parent = ROOT/source_cfg['execution_parent']
    config_path = parent/'sources/configs/autonomous_value_research.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    original_path = parent/'sources/scripts/run_patient_value_execution.py'
    frozen_path = source/'execution/monthly_hold_allowed.parquet'
    frozen = pd.read_parquet(frozen_path)
    if cfg['exit_confirmations'] != 2 or frozen.index.max() >= pd.Timestamp('2024-01-01'):
        raise ValueError('冻结规则或开发边界不符')
    inputs = [source/'config.yaml', config_path, original_path, frozen_path,
              ROOT/'src/patient_trend.py', ROOT/'src/portfolio.py',
              ROOT/'src/holding_evidence.py', Path(__file__), ROOT/'tests/test_holding_evidence.py']
    filters = [('date', '>=', pd.Timestamp('2014-01-01')),
               ('date', '<=', pd.Timestamp('2023-12-31'))]
    columns = ['date', 'ts_code', 'hfq_close']
    raw = pd.read_parquet(ROOT/'data/panel/daily_panel.parquet', columns=columns, filters=filters)
    if raw.date.min() < filters[0][2] or raw.date.max() > filters[1][2]:
        raise ValueError('行情过滤失败')
    raw_hash = hashlib.sha256(pd.util.hash_pandas_object(raw, index=False).values.tobytes()).hexdigest()
    raw_rows, raw_min, raw_max = len(raw), raw.date.min(), raw.date.max()
    close = raw.pivot(index='date', columns='ts_code', values='hfq_close').sort_index()
    del raw
    spec = cfg['trend']
    _, trend = trend_state(close, spec)
    observable = (close.gt(0) & close.shift(spec['medium_window']).gt(0)
                  & close.shift(spec['long_window']).gt(0))
    if spec['require_above_long_ma']:
        observable &= close.where(close.gt(0)).rolling(spec['long_window'],
            min_periods=spec['long_window']).mean().notna()
    dates = close.index[(close.index >= pd.Timestamp(cfg['execution'][0]))
                        & (close.index <= pd.Timestamp(cfg['execution'][1]))]
    reviews = review_dates(dates, 'monthly')
    if not reviews.equals(frozen.index) or not close.columns.equals(frozen.columns):
        raise ValueError('原始全列或月度日历不一致，禁止生成变体')
    trend, observable = trend.loc[reviews], observable.loc[reviews]
    original = original_function(original_path)(trend, observable, cfg['exit_confirmations'])
    pd.testing.assert_frame_equal(original, frozen, check_exact=True)
    original.to_parquet(out/'reconstructed_original_monthly_hold_allowed.parquet')
    variant, before, after = evidence_hold_states(trend, observable, 2, True)
    variant.to_parquet(out/'keep_failure_memory_monthly_hold_allowed.parquet')
    original_counts = np.zeros(original.shape, dtype=int)
    counts = np.zeros(len(original.columns), dtype=int)
    for i in range(len(original)):
        counts = np.where(observable.iloc[i].to_numpy() & ~trend.iloc[i].to_numpy(), counts+1, 0)
        original_counts[i] = counts
    changed = ~((variant == original) | (variant.isna() & original.isna()))
    records = []
    for i, date in enumerate(reviews):
        for j, code in enumerate(original.columns):
            records.append(dict(date=date, ts_code=code, observable=bool(observable.iat[i,j]),
                trend=bool(trend.iat[i,j]) if observable.iat[i,j] else None,
                original_failure_count_after=int(original_counts[i,j]),
                c1_failure_count_before=int(before.iat[i,j]), c1_failure_count_after=int(after.iat[i,j]),
                original_hold=original.iat[i,j], c1_hold=variant.iat[i,j], changed=bool(changed.iat[i,j])))
    audit = pd.DataFrame(records)
    audit.to_csv(out/'monthly_evidence_transitions.csv', index=False)
    audit[audit.changed].to_csv(out/'changed_dates_stocks.csv', index=False)
    no_unknown = observable.all(axis=0)
    pd.testing.assert_frame_equal(original.loc[:, no_unknown], variant.loc[:, no_unknown])
    for stop in range(1, len(reviews)+1):
        pd.testing.assert_frame_equal(evidence_hold_states(trend.iloc[:stop], observable.iloc[:stop], 2),
                                      variant.iloc[:stop])
    summary = diagnose_cases(source, frozen, out, inputs)
    facts = dict(status='complete', purpose='research_only_C1', confirmations=2,
        unknown_timeout_exit=False, rolling_window_or_clock_changed=False,
        pending_exit_semantics_changed=False, original_calendar_exact=True,
        frozen_shape=list(frozen.shape), changed_cells=int(changed.sum().sum()),
        changed_stocks=int(changed.any().sum()), no_unknown_stock_parity=int(no_unknown.sum()),
        prefix_checks=len(reviews), market_path='data/panel/daily_panel.parquet',
        market_columns=columns, market_filters=[[a,b,str(c)] for a,b,c in filters],
        raw_filtered_rows=raw_rows, raw_filtered_min=str(raw_min), raw_filtered_max=str(raw_max),
        raw_filtered_close_sha256=raw_hash, test_market_data_used=False,
        fresh_holdout=False, trade_replay_performed_here=False,
        baseline_cases=len(summary), baseline_episodes=int(summary.episodes.sum()),
        baseline_closed=int(summary.closed.sum()), baseline_censored=int(summary.censored.sum()),
        baseline_unknown_hold_reviews=int(summary.unknown_hold_reviews.sum()),
        source_hashes={str(p): digest(p) for p in inputs})
    report = ('# P0持有证据与三时钟审计\n\n'
        f'原持有日历逐格完全复现：{frozen.shape}；C1改变{facts["changed_cells"]}个日期股票格，'
        f'涉及{facts["changed_stocks"]}只股票。未知保持失败证据，True清零，False累计至原阈值2。'
        '未修改完整120日窗口、复核时钟、未知超时或原执行引擎不可撤销退出。\n\n'
        f'全15基线共{facts["baseline_episodes"]}个持仓段，已闭合{facts["baseline_closed"]}，'
        f'行政右删失{facts["baseline_censored"]}；未知持有复核{facts["baseline_unknown_hold_reviews"]}次。'
        '各方案计数与时钟覆盖见baseline_15_clock_summary.csv。\n\n'
        '信号时钟以入场signal_date到首次退出请求signal_date的月度间隔计；'
        '退出请求实际日志日期另列，不能把下一交易日事件日志倒称当时已执行。'
        '资金时钟用真实买入到最终清仓或开发截止，部分风险减仓不结束持仓段。'
        '损益统一观察至开发截止，已退出后保留现金；期末旧仓含未实现估值。'
        'closed_only_mark_days_median仅描述已闭合持仓段标记天数，不是自然趋势寿命、最佳持有期或无偏总体估计。\n\n'
        '原始趋势由带日期谓词的既有2014—2023三列行情重算。资格后的PIT特征不能完整恢复原trend，'
        '故未用其缺失位置猜测可观察性。没有读取2024+行情行、网络数据接口或凭据。'
        '本脚本不回放交易、不评估C1收益；状态变化不代表既有持仓会受影响。\n')
    (out/'REPORT.md').write_text(report)
    facts['output_hashes'] = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in facts.items() if k not in ['source_hashes','output_hashes']},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
