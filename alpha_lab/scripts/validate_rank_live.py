"""固定排名邻域及成本压力检验；不修改已封存策略或实盘账户。"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_selection_stability as prior
from selection_stability import buffered_holds
from build_selection_stability import digest, save

ORIGINAL_ENGINE = prior.run_backtest
COST = 0.


def cost_engine(*args, **kwargs):
    return ORIGINAL_ENGINE(*args, **kwargs, slippage_rate=COST)


def initialize(run):
    run = Path(run)
    protocol = json.loads((run / 'protocol.json').read_text())
    prior.initialize(protocol['parent'])
    prior.STATE['run'] = run
    prior.run_backtest = cost_engine
    for day in protocol['days']:
        schedules = prior.STATE['schedules'][day]
        base = schedules['baseline']
        scores = base['signals']
        top = scores.notna() & False
        for date, row in scores.iterrows():
            top.loc[date, row.dropna().nlargest(20).index] = True
        for buffer in protocol['buffers']:
            for confirmations in protocol['confirmations']:
                hold, _, exit_mask = buffered_holds(scores, base['hold'], buffer, confirmations)
                policy = f'rank{buffer}_confirm{confirmations}'
                schedules[policy] = dict(signals=scores.where(top & ~exit_mask),
                                         forecasts=base['forecasts'], hold=hold)
                # 先取原始Top20，再撤回入选许可，绝不递补第21名。
                assert not (schedules[policy]['signals'].notna() & ~top).any().any()
        for key in ['signals', 'hold']:
            pd.testing.assert_frame_equal(schedules['rank10_confirm2'][key], schedules['rank_buffer'][key])


def execute(case):
    global COST
    COST = case['extra_cost_bps'] / 10000
    result = prior.execute(case)
    if COST == 0 and case['policy'] in ['baseline', 'rank10_confirm2']:
        policy = 'baseline' if case['policy'] == 'baseline' else 'rank_buffer'
        protocol = json.loads((prior.STATE['run'] / 'protocol.json').read_text())
        old = Path(protocol['parent']) / 'accounts' / f"{case['cohort']}_{case['offset']:02d}_{policy}"
        new = prior.STATE['run'] / 'accounts' / case['name']
        for key in ['daily', 'trades', 'holdings', 'events']:
            pd.testing.assert_frame_equal(pd.read_parquet(new / f'{key}.parquet'),
                                          pd.read_parquet(old / f'{key}.parquet'), check_exact=True)
        save(new / 'parent_parity.json', dict(all_four_ledgers_exact=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'outputs/selection_stability/20260914T070228_976007Z')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    run = args.run or ROOT / 'outputs/rank_live_validation' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=True)
    path = run / 'protocol.json'
    if not path.exists():
        receipt = json.loads((args.parent / 'completion.json').read_text())
        for name, expected in receipt['artifact_sha256'].items():
            assert digest(args.parent / name) == expected, name
        cases = []
        for cohort, start in [('continued', '2022-01-01'), ('cold', '2024-01-01')]:
            for day in [1, 8, 15, 22]:
                policies = ['baseline'] + [f'rank{b}_confirm{c}' for b in [8, 10, 12] for c in [1, 2, 3]]
                for cost in [0, 10, 30]:
                    for policy in policies if cost == 0 else ['baseline', 'rank10_confirm2']:
                        cases.append(dict(name=f'{cohort}_{day:02d}_{policy}_cost{cost}', kind='long',
                            cohort=cohort, policy=policy, offset=day, start=start, end='2026-09-08', extra_cost_bps=cost))
        sources = ['scripts/validate_rank_live.py', 'scripts/run_selection_stability.py',
            'src/selection_stability.py', 'src/risk_budget_portfolio.py', 'src/selection_first.py',
            'src/retail_rebalance.py', 'scripts/audit_monthly_portfolio.py']
        save(path, dict(parent=str(args.parent.resolve()), parent_seal_sha256=digest(args.parent / 'completion.json'),
            verified_parent_artifacts=len(receipt['artifact_sha256']), days=[1, 8, 15, 22], buffers=[8, 10, 12],
            confirmations=[1, 2, 3], cases=cases, capital=80000, max_positions=5,
            purpose='参数邻域、额外成交成本与原账户逐账本复现；不挑选收益最高参数',
            limitations=['重用已观察样本，并非新增样本外', '成本作为每边额外现金费用，不模拟价格延迟或漏单',
                        '含税分红；未扣个人持有期差别红利税', '日线严格穿价仍不保证集合竞价成交'],
            screens=dict(neighbor_positive_median_ann_fraction_min=.75, candidate_positive_ann_pairs_min=6,
                candidate_mdd_worse_pairs_max=4, cost30_positive_median_ann_delta=True),
            source_sha256={s: digest(ROOT / s) for s in sources}, root_agents_sha256=digest(ROOT.parent / 'AGENTS.md')))
    protocol = json.loads(path.read_text())
    for name, expected in protocol['source_sha256'].items(): assert digest(ROOT / name) == expected
    assert digest(args.parent / 'completion.json') == protocol['parent_seal_sha256']
    print(str(run), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn'), initializer=initialize,
                             initargs=(str(run),)) as pool:
        futures = {pool.submit(execute, case): case for case in protocol['cases']}
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(len(rows), '/', len(futures), row['name'], round(row['ann'] * 100, 2), round(row['mdd'] * 100, 2), flush=True)
    pd.DataFrame(rows).sort_values('name').to_csv(run / 'summary.csv', index=False)
    save(run / 'execution_complete.json', dict(accounts=len(rows), trades=sum(r['n_trades'] for r in rows),
        exact_parent_accounts=16, all_reconciled=True, live_orders=False))


if __name__ == '__main__': main()
