#!/usr/bin/env python3
"""仅重排MILP输入行的两组预登记检验，不修改经济输入或筛选规则。"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import retail_rebalance
from portfolio import run_backtest
from run_patient_value_execution import process_report, load_inputs, reconcile, digest, json_write


def main():
    run = Path('/tmp/autonomous_value_run_path').read_text().strip()
    run = Path(run)
    parent = run / 'execution'
    out = run / 'allocation_identifiability'
    out.mkdir(exist_ok=False)
    names = ['code_ascending', 'code_descending']
    manifest = dict(status='registered', registered_at=datetime.now(timezone.utc).isoformat(),
        cases=names, capital=50000, frequency='monthly', policy='low_turnover',
        perturbation='stable ts_code ascending/descending permutation immediately before MILP',
        unchanged=['economic inputs', 'candidate admission and top20 ranking', 'forecasts',
            'fees', 'cash', 'holding rules', 'risk caps', 'solver tolerance'],
        selection='two cases only; no seeds or tolerance scan; do not choose a winner',
        fresh_holdout=False, test_market_data_used=False,
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT/'src/portfolio.py', ROOT/'src/retail_rebalance.py',
            ROOT/'scripts/run_patient_value_execution.py',
            ROOT/'scripts/audit_monthly_portfolio.py', ROOT/'configs/corporate_action_components.yaml']})
    json_write(out/'manifest.json', manifest)
    for name in ['execution_codes.csv', 'execution_cache.txt', 'execution_config.yaml']:
        shutil.copy2(parent/name, out/name)
    inputs = ['monthly_low_turnover_signals.parquet',
              'monthly_low_turnover_point_forecasts.parquet', 'monthly_hold_allowed.parquet']
    manifest['input_hashes'] = {name: digest(parent/name) for name in inputs}
    cfg = yaml.safe_load((run/'config.yaml').read_text())
    pc = yaml.safe_load((out/'execution_config.yaml').read_text())
    lo, hi = map(pd.Timestamp, pc['valid'])
    assert hi < pd.Timestamp('2024-01-01')
    market, basics, st, limits, actions = load_inputs(out, pc)
    assert market.date.max() < pd.Timestamp('2024-01-01')
    signals, forecasts, hold = [pd.read_parquet(parent/name) for name in inputs]
    corporate = yaml.safe_load((ROOT/'configs/corporate_action_components.yaml').read_text())
    original = retail_rebalance.plan_rebalance
    baseline = parent/'base_50000_monthly_low_turnover'
    base_daily = pd.read_csv(baseline/'daily.csv', index_col=0, parse_dates=True)
    base_hold = pd.read_csv(baseline/'holdings.csv', parse_dates=['date'])
    base_events = pd.read_csv(baseline/'events.csv', parse_dates=['date'])
    base_target = base_events[base_events.kind.eq('allocation_decision')]
    summaries = []
    manifest['status'] = 'running'
    json_write(out/'manifest.json', manifest)
    for name in names:
        case_dir = out/name
        case_dir.mkdir()
        calls = []
        def ordered(candidates, **kwargs):
            permuted = candidates.sort_values('ts_code', ascending=name=='code_ascending', kind='stable').reset_index(drop=True)
            pd.testing.assert_frame_equal(candidates.sort_values('ts_code').reset_index(drop=True),
                                          permuted.sort_values('ts_code').reset_index(drop=True))
            result = original(permuted, **kwargs)
            call = len(calls)
            candidates.to_csv(case_dir/f'call_{call:02d}_original_candidates.csv', index=False)
            permuted.to_csv(case_dir/f'call_{call:02d}_permuted_candidates.csv', index=False)
            result['plan'].to_csv(case_dir/f'call_{call:02d}_plan.csv', index=False)
            json_write(case_dir/f'call_{call:02d}_arguments.json', kwargs)
            calls.append(dict(call=call, candidate_count=len(candidates),
                permutation_only=True, **{key:result[key] for key in
                    ['solver_gap', 'solver_absolute_gap', 'expected_net_profit', 'cash_after']}))
            return result
        retail_rebalance.plan_rebalance = ordered
        try:
            p = deepcopy(pc['portfolio'])
            p['capital'] = 50000
            result = run_backtest(signals, market, basics, st, limits, actions, lo, hi,
                p, corporate.get('rights_issues', []), retail_forecasts=forecasts,
                retail_config=cfg['retail'], hold_allowed=hold,
                extra_execution_delay=0, slippage_rate=0.)
        finally:
            retail_rebalance.plan_rebalance = original
        check = reconcile(result, 50000)
        process, pnl, episodes, exits, metrics = process_report(result, 50000)
        assert result.daily.n_hold.max() <= 5
        for suffix, frame in [('daily', result.daily), ('trades', result.trades),
            ('holdings', result.holdings), ('events', result.events),
            ('process_daily', process), ('asset_pnl', pnl),
            ('holding_episodes', episodes), ('exit_reasons', exits)]:
            frame.to_csv(case_dir/f'{suffix}.csv', index=suffix in ['daily', 'process_daily'])
        pd.DataFrame(calls).to_csv(case_dir/'solver_calls.csv', index=False)
        a = base_hold.pivot(index='date', columns='ts_code', values='shares')
        b = result.holdings.pivot(index='date', columns='ts_code', values='shares')
        columns = a.columns.union(b.columns)
        a = a.reindex(index=base_daily.index, columns=columns).fillna(0)
        b = b.reindex(index=base_daily.index, columns=columns).fillna(0)
        divergent = a.ne(b).any(axis=1)
        names_diverge = a.gt(0).ne(b.gt(0)).any(axis=1)
        first = divergent[divergent].index.min() if divergent.any() else None
        targets = result.events[result.events.kind.eq('allocation_decision')]
        details = None
        if first is not None:
            pd.DataFrame({'baseline_shares': a.loc[first], 'permuted_shares': b.loc[first]}).query(
                'baseline_shares != 0 or permuted_shares != 0').to_csv(case_dir/'first_divergence_holdings.csv')
            details = dict(date=first, baseline=base_target[base_target.date.eq(first)].to_dict('records'),
                permuted=targets[targets.date.eq(first)].to_dict('records'))
            json_write(case_dir/'first_divergence.json', details)
        summary = dict(name=name, **result.summary, **check, **metrics,
            first_holdings_divergence=first, differing_holdings_days=int(divergent.sum()),
            differing_name_days=int(names_diverge.sum()),
            max_daily_equity_difference=float((result.daily.equity-base_daily.equity).abs().max()),
            max_solver_gap=float(pd.DataFrame(calls).solver_gap.max()),
            max_solver_absolute_gap=float(pd.DataFrame(calls).solver_absolute_gap.max()),
            all_calls_permutation_only=all(call['permutation_only'] for call in calls))
        summaries.append(summary)
        json_write(case_dir/'summary.json', summary)
        print(json.dumps({key:summary[key] for key in ['name', 'total', 'mdd',
            'first_holdings_divergence', 'differing_name_days', 'max_solver_gap']}, default=str), flush=True)
        manifest['completed_cases'] = len(summaries)
        json_write(out/'manifest.json', manifest)
    pd.DataFrame(summaries).to_csv(out/'comparison.csv', index=False)
    manifest['status'] = 'complete'
    manifest['source_hashes_unchanged'] = all(digest(ROOT/key)==value for key,value in manifest['source_hashes'].items())
    assert manifest['source_hashes_unchanged']
    json_write(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
