"""后续单变量消融：风险恢复后允许一次周频补位，仍沿用当前月度名单。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import argparse
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import research_portfolio_cash as research
import portfolio_cash_reentry_engine as engine
from portfolio_cash_state import CashState
from build_selection_stability import digest,save


class RecoveryState(CashState):
    def on_close(self,date,values,equity):
        old=self.weights is not None
        result=super().on_close(date,values,equity)
        self.recovered_today=old and self.weights is None
        return result


def initialize(run,parent):
    research.initialize(parent)
    research.prior.STATE['run']=Path(run)
    research.engine=engine


def execute(case):
    research.ACTIVE=(CashState if case['control'] else RecoveryState)(research.FEATURES,'staged')
    research.COST=case['cost_bps']/10000
    result=research.prior.execute(case)
    folder=research.prior.STATE['run']/'accounts'/case['name']
    pd.DataFrame(research.ACTIVE.records).to_csv(folder/'risk_states.csv',index=False)
    if case['control']:
        cfg=json.loads((research.prior.STATE['run']/'protocol.json').read_text())
        old=Path(cfg['parent'])/'accounts'/case['source']
        for key in ['daily','trades','holdings','events']:
            pd.testing.assert_frame_equal(pd.read_parquet(folder/f'{key}.parquet'),pd.read_parquet(old/f'{key}.parquet'),check_exact=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    args=p.parse_args();parent=args.parent.resolve()
    assert (parent/'execution_complete.json').exists()
    run=ROOT/'outputs/cash_reentry_ablation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True,exist_ok=False)
    original=json.loads((parent/'protocol.json').read_text());cases=[]
    for old in original['cases']:
        if old['label']!='staged':continue
        cases.append(dict(old,name=old['name']+'_fast_reentry',source=old['name'],control=False,label='fast_reentry'))
    for name in ['cold_01_staged_cost0','continued_08_staged_cost30']:
        old=next(c for c in original['cases'] if c['name']==name)
        cases.append(dict(old,name=name+'_control',source=name,control=True))
    sources=['scripts/research_cash_reentry.py','src/portfolio_cash_reentry_engine.py',
        'scripts/research_portfolio_cash.py','src/portfolio_cash_engine.py','src/portfolio_cash_state.py']
    save(run/'protocol.json',dict(parent=str(parent),cases=cases,source_sha256={s:digest(ROOT/s) for s in sources},
        parent_protocol_sha256=digest(parent/'protocol.json'),independent_oos=False,
        purpose='观察首轮收益牺牲后登记的单变量机制消融，不是事前留出检验',
        contract='仅原风险状态从锁定转安全时增加一次次日建仓机会；名单/风险预算仍沿用最新已知月度信号，不改退出阈值，不补足已有股票，不增至5股以上。'))
    print(run,flush=True);rows=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn'),initializer=initialize,
            initargs=(str(run),str(parent))) as pool:
        futures=[pool.submit(execute,c) for c in cases]
        for f in as_completed(futures):
            row=f.result();rows.append(row)
            print(len(rows),'/',len(cases),row['name'],round(row['ann']*100,2),round(row['mdd']*100,2),flush=True)
    pd.DataFrame(rows).to_csv(run/'summary.csv',index=False)
    save(run/'execution_complete.json',dict(accounts=len(rows),controls=2,trades=sum(r['n_trades'] for r in rows),all_reconciled=True))


if __name__=='__main__':main()
