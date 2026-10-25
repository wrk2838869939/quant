"""固定组合共同风险、现金对照及恢复契约；保持旧引擎封存不变。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import argparse
import difflib
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_selection_stability as prior
import portfolio_cash_engine as engine
from portfolio_cash_state import basket_features, CashState
from build_selection_stability import digest, save

FEATURES=None
ACTIVE=None
COST=0.


def wrapper(*args,**kwargs):
    return engine.run_backtest(*args,**kwargs,cash_overlay=ACTIVE,slippage_rate=COST)


def initialize(run):
    global FEATURES
    run=Path(run); cfg=json.loads((run/'protocol.json').read_text())
    prior.initialize(cfg['parent']); prior.STATE['run']=run; prior.run_backtest=wrapper
    close=pd.read_parquet(Path(cfg['parent'])/'signals/close.parquet')
    FEATURES=basket_features(close)


def simulate(case, stop=None):
    global ACTIVE,COST
    variant=case['overlay']; COST=case['cost_bps']/10000
    ACTIVE=None if variant=='none' else CashState(FEATURES,variant,case['lower'],case['upper'])
    return prior.simulate(case,stop)


def execute(case):
    global ACTIVE,COST
    ACTIVE=None if case['overlay']=='none' else CashState(FEATURES,case['overlay'],case['lower'],case['upper'])
    COST=case['cost_bps']/10000
    result=prior.execute(case)
    folder=prior.STATE['run']/'accounts'/case['name']
    if ACTIVE is not None: pd.DataFrame(ACTIVE.records).to_csv(folder/'risk_states.csv',index=False)
    if case['overlay']=='none' and COST==0:
        cfg=json.loads((prior.STATE['run']/'protocol.json').read_text())
        old=Path(cfg['parent'])/'accounts'/f"{case['cohort']}_{case['offset']:02d}_{case['policy']}"
        for key in ['daily','trades','holdings','events']:
            pd.testing.assert_frame_equal(pd.read_parquet(folder/f'{key}.parquet'),pd.read_parquet(old/f'{key}.parquet'),check_exact=True)
        save(folder/'parent_parity.json',dict(all_four_ledgers_exact=True))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path)
    args=p.parse_args()
    run=args.run or ROOT/'outputs/portfolio_cash'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'outputs/selection_stability/20260914T070228_976007Z'
    seal=json.loads((parent/'completion.json').read_text())
    for name,value in seal['artifact_sha256'].items(): assert digest(parent/name)==value
    cases=[]
    variants=[('base','baseline','none',.05,.10),('rank','rank_buffer','none',.05,.10),
        ('pause','rank_buffer','pause',.05,.10),('half','rank_buffer','half',.05,.10),
        ('full','rank_buffer','full',.05,.10),('staged','rank_buffer','staged',.05,.10),
        ('fixed_half','rank_buffer','fixed_half',.05,.10),('base_staged','baseline','staged',.05,.10),
        ('staged_4_8','rank_buffer','staged',.04,.08),('staged_6_12','rank_buffer','staged',.06,.12)]
    for cohort,start in [('continued','2022-01-01'),('cold','2024-01-01')]:
        for day in [1,8,15,22]:
            for cost in [0,30]:
                for label,policy,overlay,lower,upper in variants:
                    if cost and label not in ['rank','staged']:continue
                    cases.append(dict(name=f'{cohort}_{day:02d}_{label}_cost{cost}',kind='long',cohort=cohort,
                        start=start,end='2026-09-08',offset=day,policy=policy,overlay=overlay,label=label,
                        lower=lower,upper=upper,cost_bps=cost))
    sources=['scripts/research_portfolio_cash.py','src/portfolio_cash_state.py','src/portfolio_cash_engine.py',
        'scripts/run_selection_stability.py','src/risk_budget_portfolio.py','src/selection_first.py','src/retail_rebalance.py']
    save(run/'protocol.json',dict(parent=str(parent),cases=cases,parent_seal_sha256=digest(parent/'completion.json'),
        source_sha256={s:digest(ROOT/s) for s in sources},root_agents_sha256=digest(ROOT.parent/'AGENTS.md'),
        capital=80000,max_positions=5,independent_oos=False,
        rules='周首收盘检查；股权权重至少60%在20日均线下，且加权个股60日回撤>=5%时半仓，>=10%时空仓；冻结原篮子权重。弱势权重<=40%且加权20日收益>0连续两周解除。重入仅原月度窗口/既有到账补位，不补足已有股票。',
        limitations=['阈值为机制实验预设，非拟合最优','含税分红','日线穿价/足量成交假设','重用历史窗口'],
        screens=dict(mdd_improved_min=6,median_ann_loss_max=.02,median_trade_increase_max=.25)))
    (run/'engine_changes.diff').write_text(''.join(difflib.unified_diff((ROOT/'src/risk_budget_portfolio.py').read_text().splitlines(True),
        (ROOT/'src/portfolio_cash_engine.py').read_text().splitlines(True),fromfile='frozen_risk_engine',tofile='research_cash_engine')))
    print(run,flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn'),initializer=initialize,initargs=(str(run),)) as pool:
        futures=[pool.submit(execute,c) for c in cases]
        for f in as_completed(futures):
            row=f.result();rows.append(row)
            print(len(rows),'/',len(cases),row['name'],round(row['ann']*100,2),round(row['mdd']*100,2),flush=True)
    pd.DataFrame(rows).sort_values('name').to_csv(run/'summary.csv',index=False)
    save(run/'execution_complete.json',dict(accounts=len(rows),trades=sum(r['n_trades'] for r in rows),all_reconciled=True))


if __name__=='__main__':main()
