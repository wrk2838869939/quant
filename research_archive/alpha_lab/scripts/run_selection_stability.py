# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
"""复现父账户后执行固定持续性/排名缓冲对照，逐账户校验真实资金约束。"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from build_selection_stability import digest, save
from risk_budget_portfolio import run_backtest
from startup_timing import startup_matrices
from audit_monthly_portfolio import reconcile

STATE = {}


def initialize(path):
    run = Path(path); cfg = yaml.safe_load((run / 'config.yaml').read_text())
    inp = run / 'inputs'
    STATE.update(run=run, cfg=cfg,
        frames={k: pd.read_parquet(inp / f'{k}.parquet') for k in ['market', 'basics', 'st', 'limits', 'normalized_actions']},
        schedules={day: {policy: {k: pd.read_parquet(run / f'schedules/calendar_{day:02d}/{policy}/{k}.parquet')
                                 for k in ['signals', 'forecasts', 'hold']} for policy in cfg['policies'] + cfg['neighbors']}
                   for day in cfg['calendar_days']}, risk=pd.read_parquet(run / 'risk.parquet'),
        portfolio=yaml.safe_load((inp / 'execution_config.yaml').read_text())['portfolio'],
        retail=yaml.safe_load((inp / 'symmetric_phase_research.yaml').read_text())['retail'],
        rights=yaml.safe_load((inp / 'corporate_action_components.yaml').read_text()).get('rights_issues', []),
        causal=yaml.safe_load((ROOT / 'outputs/causal_execution/20260913T130810_374654Z/config.yaml').read_text())['causal'])
    STATE['portfolio'].update(capital=cfg['capital'], top_n=cfg['max_positions'])


def simulate(case, stop=None):
    start, end = pd.Timestamp(case['start']), pd.Timestamp(stop or case['end'])
    day, policy = case['offset'], case['policy']
    if case['kind'] == 'startup':
        matrices, scales, lineage = startup_matrices(STATE['schedules'][1][policy], STATE['schedules'][day][policy],
            STATE['risk'], case['boot_date'], start, end, 'fresh_once')
    else:
        matrices = {k: v.loc[v.index <= end] for k, v in STATE['schedules'][day][policy].items()}
        scales = STATE['risk'].reindex(index=matrices['signals'].index, columns=matrices['signals'].columns)
        lineage = dict(following_calendar=f'calendar_{day:02d}', future_source_used=False)
    needed = sorted(set().union(*(set(row.dropna().nlargest(20).index) for _, row in matrices['signals'].iterrows())))
    if not set(needed).issubset(set(STATE['frames']['market'].ts_code)):
        raise ValueError('候选代码完全缺执行行情，禁止当作普通不可成交跳过')
    if scales.reindex(columns=needed).isna().any().any():
        raise ValueError('候选风险覆盖不全，禁止隐式扩池')
    local = [STATE['frames'][k].loc[STATE['frames'][k].ts_code.isin(needed)].copy()
             for k in ['market', 'basics', 'st', 'limits', 'normalized_actions']]
    local[0] = local[0].loc[local[0].date.between(start, end)]
    local[3] = local[3].loc[local[3].date.le(end)]
    local[4] = local[4].loc[local[4].known_date.le(end)]
    result = run_backtest(matrices['signals'], *local, start, end, STATE['portfolio'], STATE['rights'],
        retail_forecasts=matrices['forecasts'], retail_config=STATE['retail'], hold_allowed=matrices['hold'],
        causal_config=STATE['causal'], risk_multipliers=scales, adjust_existing_risk=True)
    return result, lineage


def execute(case):
    folder = STATE['run'] / 'accounts' / case['name']; folder.mkdir(parents=True, exist_ok=True)
    try:
        result, lineage = simulate(case)
        cfg = STATE['cfg']; check = reconcile(result, cfg['capital'])
        assert result.summary['execution_data_sufficient']
        assert result.daily.n_hold.max() <= cfg['max_positions']
        if case['kind'] == 'startup': assert len(result.daily) == 244
        trades = result.trades; ev = result.events
        assert trades.loc[trades.side.eq('buy'), 'shares'].mod(100).eq(0).all()
        assert (trades.decision_date < trades.date).all()
        assert np.max(np.abs(result.daily.equity - result.daily.cash - result.daily.stock_value - result.daily.receivables)) < 1e-6
        orders = ev.loc[ev.kind.eq('order_planned')]
        for _, g in orders.loc[orders.side.eq('buy')].groupby('decision_date'):
            assert g.reserved_cash.sum() <= g.cash_at_decision.iloc[0] + 1e-6
        original = STATE['schedules'][1 if case['kind'] == 'startup' else case['offset']]['baseline']['hold']
        exits = ev.loc[ev.kind.eq('hold_exit_requested') & ev.reason.eq('hold_false')]
        extra = sum(r.date in original.index and original.at[r.date, r.ts_code] == 1 for r in exits.itertuples())
        summary = dict(case, **result.summary, **check, rank_exit_requests=int(extra), status='passed')
        for key in ['daily', 'trades', 'holdings', 'events']:
            getattr(result, key).to_parquet(folder / f'{key}.parquet', index=key == 'daily')
        save(folder / 'lineage.json', lineage); save(folder / 'summary.json', summary)
        return summary
    except Exception as exc:
        save(folder / 'failure.json', dict(case, error_type=type(exc).__name__, message=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    initialize(run); cfg = STATE['cfg']
    for name, expected in json.loads((run / 'signal_hashes.json').read_text()).items(): assert digest(run / name) == expected
    inp_receipt = json.loads((run / 'execution_input_completion.json').read_text())
    for name, expected in inp_receipt['input_sha256'].items(): assert digest(run / name) == expected
    path = run / 'execution_manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        assert digest(Path(__file__)) == manifest['runner_sha256']
    else:
        previous = json.loads((ROOT / cfg['startup_parent'] / 'manifest.json').read_text())
        cases = []; reused = []
        for old in previous['cases']:
            if old['policy'] != 'fresh_once': continue
            reused.append(dict(source=old['name'], cohort=old['cohort'], offset=old['offset']))
            for policy in cfg['policies'][1:]:
                cases.append(dict(name=f"startup_{old['cohort']}_{old['offset']:02d}_{policy}", kind='startup',
                    policy=policy, offset=old['offset'], cohort=old['cohort'], start=old['start'], end=old['end'], boot_date=old['boot_date']))
        for window, start in [('continued', '2022-01-01'), ('cold', '2024-01-01')]:
            for day in cfg['calendar_days']:
                for policy in cfg['policies'] + cfg['neighbors']:
                    cases.append(dict(name=f'{window}_{day:02d}_{policy}', kind='long', policy=policy,
                        offset=day, cohort=window, start=start, end=cfg['end']))
        manifest = dict(cases=cases, reused_startup_baselines=reused, runner_sha256=digest(Path(__file__)),
                        signal_hashes_sha256=digest(run / 'signal_hashes.json'), inputs_receipt_sha256=digest(run / 'execution_input_completion.json'))
        save(path, manifest)
    if not (run / 'parent_parity.json').exists():
        rows = []
        for old_name in ['2022-04_08_fresh_once', '2023-01_15_fresh_once', '2024-07_22_fresh_once']:
            prev = ROOT / cfg['startup_parent'] / 'accounts' / old_name
            old = json.loads((prev / 'summary.json').read_text())
            case = dict(old, kind='startup', policy='baseline')
            result, _ = simulate(case)
            for key in ['daily', 'trades', 'holdings', 'events']:
                pd.testing.assert_frame_equal(getattr(result, key), pd.read_parquet(prev / f'{key}.parquet'), check_exact=True)
            rows.append(dict(account=old_name, all_four_ledgers_exact=True))
            print('parent parity', old_name, flush=True)
        save(run / 'parent_parity.json', rows)
    results = []; pending = []
    for case in manifest['cases']:
        receipt = run / 'accounts' / case['name'] / 'summary.json'
        if receipt.exists(): results.append(json.loads(receipt.read_text()))
        else: pending.append(case)
    with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=get_context('spawn'), initializer=initialize, initargs=(str(run),)) as pool:
        futures = {pool.submit(execute, c): c for c in pending}
        for f in as_completed(futures):
            try: result = f.result()
            except Exception:
                for future in futures: future.cancel()
                raise
            results.append(result)
            print(len(results), '/', len(manifest['cases']), result['name'], round(result['total'] * 100, 2), round(result['mdd'] * 100, 2), flush=True)
    pd.DataFrame(results).sort_values('name').to_csv(run / 'summary.csv', index=False)
    save(run / 'execution_complete.json', dict(new_accounts=len(results), reused_baselines=len(manifest['reused_startup_baselines']),
        trades=sum(r['n_trades'] for r in results), all_reconciled=True, live_orders=False))


if __name__ == '__main__': main()
