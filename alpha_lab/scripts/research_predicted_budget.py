"""风险识别合格、空仓效用不合格模型的独立预算假设；不改写原准入结论。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from multiprocessing import get_context
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import research_targeted_restore as parent
import predicted_risk_budget_engine as engine
from predicted_risk_budget_state import PredictedRiskBudget
from forward_risk_evidence import matured_weekly_rows
from build_selection_stability import digest,save

PROB=None;ENTER=None;RELEASE=None


def wrapper(*args,**kwargs):
    return engine.run_backtest(*args,**kwargs,cash_overlay=parent.research.ACTIVE,
        slippage_rate=parent.research.COST,restore_enabled=parent.ENABLED)


def initialize(run):
    global PROB,ENTER,RELEASE
    run=Path(run);cfg=json.loads((run/'protocol.json').read_text())
    parent.initialize(cfg['targeted_parent']);parent.research.prior.STATE['run']=run
    parent.research.prior.run_backtest=wrapper
    evidence=Path(cfg['evidence'])
    PROB=pd.read_parquet(evidence/'predictions.parquet').volatility
    ENTER,RELEASE=cfg['enter'],cfg['release']


def setup(case):
    parent.ENABLED=case['restore'];parent.research.COST=case['cost_bps']/10000
    parent.research.ACTIVE=PredictedRiskBudget(parent.research.FEATURES,PROB,ENTER,RELEASE,case['risk_cap']) if case['restore'] else None


def execute(case):
    setup(case);row=parent.research.prior.execute(case)
    run=parent.research.prior.STATE['run'];folder=run/'accounts'/case['name']
    if parent.ENABLED:
        pd.DataFrame(parent.research.ACTIVE.records).to_csv(folder/'risk_states.csv',index=False)
    else:
        cfg=json.loads((run/'protocol.json').read_text())
        source=Path(cfg['cash_parent'])/'accounts'/case['source']
        for key in ['daily','trades','holdings','events']:
            pd.testing.assert_frame_equal(pd.read_parquet(folder/f'{key}.parquet'),pd.read_parquet(source/f'{key}.parquet'),check_exact=True)
        save(folder/'parent_parity.json',dict(all_four_ledgers_exact=True))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path)
    args=p.parse_args();run=args.run or ROOT/'outputs/predicted_budget'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'outputs/forward_risk/20260914T104245_763609Z'
    ep=json.loads((evidence/'protocol.json').read_text())
    gates=json.loads((evidence/'acceptance.json').read_text())['validation_gates']
    selected=[x['model'] for x in gates if all(x['checks'][k] for k in ['sample','auc','precision','exposure','calibration'])]
    assert selected==['volatility'],'不按2022+表现另选模型'
    features=pd.read_parquet(evidence/'features.parquet');labels=pd.read_parquet(evidence/'labels.parquet')
    train=matured_weekly_rows(features,labels,*ep['splits']['train'])
    prob=pd.read_parquet(evidence/'predictions.parquet').volatility
    targeted=ROOT/'outputs/targeted_restore/20260914T081149_007200Z'
    parent_cfg=json.loads((targeted/'protocol.json').read_text());cash=Path(parent_cfg['cash_parent'])
    cases=[]
    for case in json.loads((cash/'protocol.json').read_text())['cases']:
        if case['label']!='rank':continue
        for label,cap in [('rank_control',1.),('budget80',.8),('budget50',.5)]:
            cases.append(dict(case,name=case['name']+'_'+label,source=case['name'],label=label,risk_cap=cap,restore=label!='rank_control'))
    sources=list(parent_cfg['source_sha256'])+['src/predicted_risk_budget_engine.py','src/predicted_risk_budget_state.py','scripts/research_predicted_budget.py']
    save(run/'protocol.json',dict(evidence=str(evidence),targeted_parent=str(targeted),cash_parent=str(cash),cases=cases,
        enter=float(prob.loc[train].quantile(.8)),release=float(prob.loc[train].quantile(.5)),
        source_sha256={name:digest(ROOT/name) for name in sources},root_agents_sha256=digest(ROOT.parent/'AGENTS.md'),
        evidence_sha256={name:digest(evidence/name) for name in ['protocol.json','predictions.parquet','acceptance.json','features.parquet','labels.parquet']},
        hypothesis='风险预测不等于空仓alpha；只控制80%或50%预算，高风险也允许按预算买新仓。恢复沿用实际额度约束。',
        selection='仅按2019—2021风险识别五项检查选择volatility；不撤销纯现金效用失败结论。',
        screens=dict(median_ann_loss_max=.02,mdd_improved_min=6,median_trade_increase_max=.25),
        independent_oos=False,capital=80000,max_positions=5,live_orders=False))
    print(run,flush=True);rows=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn'),initializer=initialize,initargs=(str(run),)) as pool:
        for future in as_completed([pool.submit(execute,case) for case in cases]):
            row=future.result();rows.append(row)
            print(len(rows),'/',len(cases),row['name'],round(row['ann']*100,2),round(row['mdd']*100,2),flush=True)
    pd.DataFrame(rows).to_csv(run/'summary.csv',index=False)
    save(run/'execution_complete.json',dict(accounts=len(rows),parent_parities=16,trades=sum(x['n_trades'] for x in rows),all_reconciled=True))


if __name__=='__main__':main()
