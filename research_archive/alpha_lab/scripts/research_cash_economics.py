# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
"""固定四个机制消融；复用已审查的恢复引擎与同日期同费用对照。"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import research_targeted_restore as parent
from cash_economics_state import EconomicCashState
from build_selection_stability import digest, save

CLOSE = None


def initialize(run):
    global CLOSE
    run = Path(run)
    protocol = json.loads((run/'protocol.json').read_text())
    parent.initialize(protocol['parent'])
    parent.research.prior.STATE['run'] = run
    CLOSE = pd.read_parquet(Path(protocol['selection'])/'signals/close.parquet')


def setup(case):
    parent.ENABLED = True
    parent.research.COST = case['cost_bps']/10000
    if case['label'] == 'control':
        parent.setup(case)
    else:
        parent.research.ACTIVE = EconomicCashState(CLOSE, case['label'])


def execute(case):
    setup(case)
    result = parent.research.prior.execute(case)
    folder = parent.research.prior.STATE['run']/'accounts'/case['name']
    pd.DataFrame(parent.research.ACTIVE.records).to_csv(folder/'risk_states.csv', index=False)
    if case['label'] == 'control':
        protocol = json.loads((folder.parent.parent/'protocol.json').read_text())
        source = Path(protocol['parent'])/'accounts'/case['source']
        for name in ['daily', 'trades', 'holdings', 'events']:
            pd.testing.assert_frame_equal(pd.read_parquet(folder/f'{name}.parquet'),
                                          pd.read_parquet(source/f'{name}.parquet'), check_exact=True)
        save(folder/'parent_parity.json', dict(all_four_ledgers_exact=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    run = args.run or ROOT/'outputs/cash_economics'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    old = ROOT/'outputs/targeted_restore/20260914T081149_007200Z'
    protocol = json.loads((old/'protocol.json').read_text())
    for name, value in json.loads((old/'completion.json').read_text())['artifact_sha256'].items():
        assert digest(old/name) == value, name
    cash = Path(protocol['cash_parent'])
    selection = json.loads((cash/'protocol.json').read_text())['parent']
    cases = []
    for case in protocol['cases']:
        if case['label'] != 'fast_restore':
            continue
        for mode in ['control', 'quick_release', 'basket_dd', 'trend_break', 'trend_half']:
            cases.append(dict(case, name=case['name']+'_'+mode, source=case['name'], label=mode))
    sources = list(protocol['source_sha256']) + ['src/cash_economics_state.py', 'scripts/research_cash_economics.py']
    save(run/'protocol.json', dict(parent=str(old), parent_seal_sha256=digest(old/'completion.json'),
        cash_parent=str(cash), selection=selection, cases=cases, capital=80000, max_positions=5,
        source_sha256={name:digest(ROOT/name) for name in sources},
        root_agents_sha256=digest(ROOT.parent/'AGENTS.md'), independent_oos=False,
        hypothesis='原回撤混合异步高点且恢复等待过长；先缩短恢复，再同步篮子回撤，再确认长期趋势，最后去除全退。',
        diagnosis='outputs/cash_economics/diagnosis_20260914_v2',
        rules=dict(quick_release='原触发；弱势权重<=40%一次周检查解除',
                   basket_dd='quick_release基础上用当前固定单位篮子的同步60日回撤',
                   trend_break='basket_dd基础上要求60日负收益且至少60%权重低于60日均线',
                   trend_half='trend_break基础上最多减至半仓，不触发全退'),
        screens=dict(median_ann_loss_max=.02, mdd_improved_min=6, median_trade_increase_max=.25),
        note='四机制一次预定，不按日历挑选；通过经济性筛选也非前向证明。'))
    print(run, flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn'), initializer=initialize, initargs=(str(run),)) as pool:
        for future in as_completed([pool.submit(execute, case) for case in cases]):
            row = future.result()
            rows.append(row)
            print(len(rows), '/', len(cases), row['name'], round(row['ann']*100,2), round(row['mdd']*100,2), flush=True)
    pd.DataFrame(rows).sort_values('name').to_csv(run/'summary.csv', index=False)
    save(run/'execution_complete.json', dict(accounts=len(rows), parent_parities=16,
         trades=sum(row['n_trades'] for row in rows), all_reconciled=True))


if __name__ == '__main__':
    main()
