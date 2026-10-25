"""定向恢复修复的月度/恢复周对照；真实减仓额度不能变成普遍加仓。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import argparse
import difflib
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import research_portfolio_cash as research
from research_cash_reentry import RecoveryState
from portfolio_cash_state import CashState
import targeted_restore_engine as engine
from build_selection_stability import digest,save

ENABLED=False


def wrapper(*args,**kwargs):
    return engine.run_backtest(*args,**kwargs,cash_overlay=research.ACTIVE,slippage_rate=research.COST,restore_enabled=ENABLED)


def initialize(run):
    run=Path(run);cfg=json.loads((run/'protocol.json').read_text())
    research.initialize(cfg['cash_parent']);research.prior.STATE['run']=run;research.prior.run_backtest=wrapper


def setup(case):
    global ENABLED
    ENABLED=case['restore']
    research.ACTIVE=(RecoveryState if case['fast'] else CashState)(research.FEATURES,'staged')
    research.COST=case['cost_bps']/10000


def execute(case):
    setup(case);result=research.prior.execute(case)
    folder=research.prior.STATE['run']/'accounts'/case['name']
    pd.DataFrame(research.ACTIVE.records).to_csv(folder/'risk_states.csv',index=False)
    if not case['restore']:
        cfg=json.loads((research.prior.STATE['run']/'protocol.json').read_text())
        parent=Path(cfg['fast_parent'] if case['fast'] else cfg['cash_parent'])
        source=case['source']+('_fast_reentry' if case['fast'] else '')
        for key in ['daily','trades','holdings','events']:
            pd.testing.assert_frame_equal(pd.read_parquet(folder/f'{key}.parquet'),pd.read_parquet(parent/'accounts'/source/f'{key}.parquet'),check_exact=True)
        save(folder/'parent_parity.json',dict(all_four_ledgers_exact=True))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path)
    args=p.parse_args();run=args.run or ROOT/'outputs/targeted_restore'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True,exist_ok=False)
    cash=ROOT/'outputs/portfolio_cash/20260914T075010_199292Z'
    fast=ROOT/'outputs/cash_reentry_ablation/20260914T075408_339162Z'
    for parent in [cash,fast]:
        for name,value in json.loads((parent/'completion.json').read_text())['artifact_sha256'].items():assert digest(parent/name)==value
    cases=[]
    for old in json.loads((cash/'protocol.json').read_text())['cases']:
        if old['label']!='staged':continue
        for speed in [False,True]:
            for enabled in [False,True]:
                label=('fast' if speed else 'monthly')+('_restore' if enabled else '_control')
                cases.append(dict(old,name=old['name']+'_'+label,source=old['name'],label=label,fast=speed,restore=enabled))
    sources=['scripts/research_targeted_restore.py','src/targeted_restore_engine.py','src/restore_quota.py',
        'scripts/research_portfolio_cash.py','scripts/research_cash_reentry.py','src/portfolio_cash_state.py',
        'scripts/run_selection_stability.py','src/selection_first.py','src/retail_rebalance.py']
    save(run/'protocol.json',dict(cash_parent=str(cash),fast_parent=str(fast),cases=cases,
        source_sha256={s:digest(ROOT/s) for s in sources},parent_seals={str(p):digest(p/'completion.json') for p in [cash,fast]},
        root_agents_sha256=digest(ROOT.parent/'AGENTS.md'),capital=80000,max_positions=5,independent_oos=False,
        contract='只补回本轮组合现金风控实际卖掉的仍持有旧股；新轮次/全退出/红股使额度清零；恢复预约优先、新仓只用剩余前日现金；月度hold仍允许且当前个股20日转强；不超过原真实卖量/单股风险上限/总仓位/成交量/整手。',
        corrections=['正式hold退出优先覆盖待减仓','风险恢复撤回已终结DAY后的现金减仓意图','未知恢复证据拒绝','持久化额度需成交证据余额'],
        screens=dict(median_ann_loss_max=.02,mdd_improved_min=6,median_trade_increase_max=.25),live_orders=False))
    (run/'engine_changes.diff').write_text(''.join(difflib.unified_diff((ROOT/'src/portfolio_cash_reentry_engine.py').read_text().splitlines(True),
        (ROOT/'src/targeted_restore_engine.py').read_text().splitlines(True),fromfile='sealed_reentry',tofile='targeted_restore')))
    print(run,flush=True);rows=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn'),initializer=initialize,initargs=(str(run),)) as pool:
        futures=[pool.submit(execute,c) for c in cases]
        for f in as_completed(futures):
            row=f.result();rows.append(row)
            print(len(rows),'/',len(cases),row['name'],round(row['ann']*100,2),round(row['mdd']*100,2),flush=True)
    pd.DataFrame(rows).to_csv(run/'summary.csv',index=False)
    save(run/'execution_complete.json',dict(accounts=len(rows),parent_parities=32,trades=sum(r['n_trades'] for r in rows),all_reconciled=True))


if __name__=='__main__':main()
