"""预定长期趋势40/60/80日、回撤4/5/6%的单变量邻域；不搜索组合最优。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import research_cash_economics as research
from robust_cash_state import RobustCashState
from build_selection_stability import digest,save


def initialize(run):
    research.initialize(run)


def setup(case):
    research.parent.ENABLED=True
    research.parent.research.COST=case['cost_bps']/10000
    research.parent.research.ACTIVE=RobustCashState(research.CLOSE,case['trend_window'],case['trigger_dd'])


def execute(case):
    setup(case)
    prior=research.parent.research.prior
    row=prior.execute(case)
    folder=prior.STATE['run']/'accounts'/case['name']
    pd.DataFrame(research.parent.research.ACTIVE.records).to_csv(folder/'risk_states.csv',index=False)
    if case['label']=='base60_5':
        p=json.loads((folder.parent.parent/'protocol.json').read_text())
        for key in ['daily','trades','holdings','events']:
            pd.testing.assert_frame_equal(pd.read_parquet(folder/f'{key}.parquet'),
                pd.read_parquet(Path(p['mechanism_parent'])/'accounts'/case['source']/f'{key}.parquet'),check_exact=True)
        save(folder/'parent_parity.json',dict(all_four_ledgers_exact=True))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path)
    args=p.parse_args();run=args.run or ROOT/'outputs/cash_robustness'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True,exist_ok=False)
    mechanism=ROOT/'outputs/cash_economics/20260914T101455_810747Z'
    old=json.loads((mechanism/'protocol.json').read_text())
    assert json.loads((mechanism/'execution_complete.json').read_text())['accounts']==80
    cases=[]
    for case in old['cases']:
        if case['label']!='trend_break':continue
        for label,window,dd in [('base60_5',60,.05),('trend40',40,.05),('trend80',80,.05),('dd4',60,.04),('dd6',60,.06)]:
            cases.append(dict(case,name=case['name']+'_'+label,source=case['name'],label=label,trend_window=window,trigger_dd=dd))
    sources=list(old['source_sha256'])+['src/robust_cash_state.py','scripts/research_cash_robustness.py']
    save(run/'protocol.json',dict(parent=old['parent'],mechanism_parent=str(mechanism),cash_parent=old['cash_parent'],
        selection=old['selection'],cases=cases,capital=80000,max_positions=5,independent_oos=False,
        source_sha256={name:digest(ROOT/name) for name in sources},root_agents_sha256=digest(ROOT.parent/'AGENTS.md'),
        dependency_sha256={str(mechanism/name):digest(mechanism/name) for name in ['protocol.json','execution_complete.json','summary.csv']},
        screens=old['screens'],hypothesis='仅检验首轮通过候选的附近机制，不用邻域最优替换60/5主候选。',
        robustness_requirement='至少三个单变量邻域在两档费用均保留收益且不过度交易；完整公布防守幅度及逐账户分布。'))
    print(run,flush=True);rows=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn'),initializer=initialize,initargs=(str(run),)) as pool:
        for future in as_completed([pool.submit(execute,case) for case in cases]):
            row=future.result();rows.append(row)
            print(len(rows),'/',len(cases),row['name'],round(row['ann']*100,2),round(row['mdd']*100,2),flush=True)
    pd.DataFrame(rows).to_csv(run/'summary.csv',index=False)
    save(run/'execution_complete.json',dict(accounts=len(rows),parent_parities=16,trades=sum(r['n_trades'] for r in rows),all_reconciled=True))


if __name__=='__main__':main()
