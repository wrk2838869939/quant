"""用户授权的近期冻结低换手回放；不修改开发期隔离脚本或策略规则。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from portfolio import run_backtest, normalize_actions, daily_metrics
from audit_monthly_portfolio import reconcile
from prepare_monthly_audit import fetch_execution_data
from run_patient_value_execution import process_report, json_write, digest
from run_divide_selection_execution import entry_phase_attribution, reversed_candidate_input


def prepare(run, manifest):
    out = run / 'execution'
    signals = pd.read_parquet(out / 'low_turnover_signals.parquet')
    # 只按冻结分数选择执行证据并集，不以随后是否盈利选择股票。
    codes = sorted(set().union(*(set(row.dropna().sort_values(ascending=False, kind='stable').head(20).index)
                                 for _, row in signals.iterrows())))
    pd.DataFrame({'ts_code': codes}).to_csv(out / 'execution_codes.csv', index=False)
    config = yaml.safe_load((ROOT / 'outputs/divide_selection/20260913T050617_571652Z/execution/execution_config.yaml').read_text())
    config['valid'] = [manifest['continuation_start'], manifest['end']]
    config['data_workers'] = 2
    (out / 'execution_config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    lo, hi = map(pd.Timestamp, config['valid'])
    cache = fetch_execution_data(codes, (lo, hi), config)
    (out / 'execution_cache.txt').write_text(str(cache))
    market = pd.read_parquet(ROOT / 'data/panel/daily_panel.parquet',
        columns=['ts_code', 'date', 'open', 'close', 'pre_close', 'adj_factor', 'vol_share'],
        filters=[('date', '>=', lo), ('date', '<=', hi), ('ts_code', 'in', codes)])
    basics = pd.read_parquet(ROOT / 'data/raw/stock_basic_all.parquet')
    for col in ['list_date', 'delist_date']:
        basics[col] = pd.to_datetime(basics[col], errors='coerce')
    names = pd.read_parquet(ROOT / 'data/raw/namechange_all.parquet')
    names['start'] = pd.to_datetime(names.start_date, errors='coerce')
    names['end'] = pd.to_datetime(names.end_date, errors='coerce')
    st = names[names.name.str.contains('ST', case=False, na=False)][['ts_code', 'start', 'end']]
    actions = pd.concat([pd.read_parquet(cache / 'dividend' / f'{code}.parquet') for code in codes], ignore_index=True)
    limits = pd.concat([pd.read_parquet(cache / 'stk_limit' / f'{code}.parquet') for code in codes], ignore_index=True)
    limits['date'] = pd.to_datetime(limits.trade_date, errors='raise')
    for col in ['up_limit', 'down_limit']:
        limits[col] = pd.to_numeric(limits[col], errors='coerce')
    assert not limits.duplicated(['date', 'ts_code']).any()
    assert not market.duplicated(['date', 'ts_code']).any()
    coverage = market.merge(limits[['date', 'ts_code', 'up_limit', 'down_limit']],
                            on=['date', 'ts_code'], how='left', validate='one_to_one')
    missing = coverage[coverage[['up_limit', 'down_limit']].isna().any(axis=1)]
    missing.to_csv(out / 'missing_limit_coverage.csv', index=False)
    assert len(missing) == 0, f'行情日官方限价缺失 {len(missing)}，停止执行'
    components = yaml.safe_load((run / 'sources/configs/corporate_action_components.yaml').read_text())
    normalized = normalize_actions(actions, components['components'],
        components.get('reference_adjustments', []), components.get('source_corrections', []))
    json_write(out / 'corporate_action_source_corrections.json',
               {'applied': normalized.attrs.get('source_corrections', []), 'raw_cache_modified': False})
    for name, frame in [('market', market), ('basics', basics), ('st', st), ('limits', limits),
                        ('raw_actions', actions), ('normalized_actions', normalized)]:
        frame.to_parquet(out / f'{name}.parquet', index=False)
    json_write(out / 'data_manifest.json', dict(codes=len(codes), market_rows=len(market),
        market_start=str(market.date.min()), market_end=str(market.date.max()),
        limits_rows=len(limits), missing_limits=len(missing),
        prediction_hash=digest(run / 'execution_predictions.parquet'),
        hashes={p.name: digest(p) for p in [*out.glob('*.parquet'), out / 'execution_config.yaml']}))


def run_case(run, manifest, case):
    out = run / 'execution'
    folder = out / case['name']
    folder.mkdir(exist_ok=True)
    cfg = yaml.safe_load((run / 'sources/configs/symmetric_phase_research.yaml').read_text())
    execution = yaml.safe_load((out / 'execution_config.yaml').read_text())
    p = deepcopy(execution['portfolio'])
    p.update(capital=case['capital'], top_n=5)
    for field in ['buy_cost', 'sell_cost', 'min_fee']:
        p[field] *= case['cost_multiplier']
    json_write(folder / 'portfolio_config.json', p)
    matrices = [pd.read_parquet(out / name) for name in
                ['low_turnover_signals.parquet', 'low_turnover_point_forecasts.parquet', 'monthly_hold_allowed.parquet']]
    signals, forecasts, holds = matrices
    frames = [pd.read_parquet(out / f'{name}.parquet') for name in
              ['market', 'basics', 'st', 'limits', 'normalized_actions']]
    predictions = pd.read_parquet(run / 'execution_predictions.parquet')
    actions_cfg = yaml.safe_load((run / 'sources/configs/corporate_action_components.yaml').read_text())
    with reversed_candidate_input(case['reverse_input']):
        result = run_backtest(signals, *frames, case['start'], manifest['end'], p,
            actions_cfg.get('rights_issues', []), retail_forecasts=forecasts,
            retail_config=cfg['retail'], hold_allowed=holds)
    checks = reconcile(result, case['capital'])
    process, pnl, periods, exits, statistics = process_report(result, case['capital'])
    episodes, phases, buys, phase_error = entry_phase_attribution(result, predictions, signals)
    trace = result.events[result.events.kind.eq('selection_decision')].copy()
    if len(trace):
        trace = trace.merge(predictions[['date', 'ts_code', 'phase']].rename(columns={'date': 'signal_date'}),
                            on=['signal_date', 'ts_code'], how='left', validate='many_to_one')
    unknown = result.events[result.events.kind.eq('hold_retained') & result.events.reason.eq('unknown')]
    for name, frame in [('daily', result.daily), ('trades', result.trades), ('holdings', result.holdings),
        ('events', result.events), ('process_daily', process), ('asset_pnl', pnl), ('holding_periods', periods),
        ('exit_reasons', exits), ('entry_phase_episodes', episodes), ('entry_phase_attribution', phases),
        ('actual_entry_priorities', buys), ('selection_trace', trace),
        ('stale_holding_marks', result.holdings[result.holdings.stale_days.gt(0)]), ('unknown_hold_reviews', unknown)]:
        frame.to_csv(folder / f'{name}.csv', index=name in ['daily', 'process_daily'])
    report = {**case, **result.summary, **checks, **statistics,
              'phase_pnl_max_error': phase_error, 'actual_buys': len(buys)}
    json_write(folder / 'summary.json', report)
    previous, years = case['capital'], []
    for year, daily in result.daily.groupby(result.daily.index.year):
        values, _ = daily_metrics(daily.equity, previous, p['annual_days'])
        years.append(dict(name=case['name'], year=year, start_equity=previous,
                          end_equity=daily.equity.iloc[-1], **values))
        previous = daily.equity.iloc[-1]
    pd.DataFrame(years).to_csv(folder / 'yearly.csv', index=False)
    recent = result.daily.loc['2024-01-01':]
    initial = result.daily.loc[:'2023-12-31'].equity.iloc[-1] if case['start'] < '2024-01-01' else case['capital']
    recent_metrics, _ = daily_metrics(recent.equity, initial, p['annual_days'])
    recent_metrics.update(name=case['name'], initial_equity=initial, end_equity=recent.equity.iloc[-1],
        mean_exposure=float((recent.stock_value/recent.equity).mean()),
        recent_trades=int(result.trades.date.ge(pd.Timestamp('2024-01-01')).sum()))
    json_write(folder / 'recent_summary.json', recent_metrics)
    print(json.dumps(recent_metrics, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--case-name')
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, expected in manifest['source_hashes'].items():
        assert digest(ROOT / name) == expected, f'冻结源码变化 {name}'
    try:
        if args.prepare_only:
            prepare(run, manifest)
        elif args.case_name:
            run_case(run, manifest, next(c for c in manifest['cases'] if c['name'] == args.case_name))
        else:
            raise ValueError('必须明确准备或案例')
    except Exception as exc:
        # 仅记录本地错误；远端异常已由数据准备器屏蔽。
        json_write(run / f'failure_{args.case_name or "prepare"}.json',
                   dict(error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
