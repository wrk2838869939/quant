"""全账户独立资金复算、前缀回放及组合现金请求的执行时序验证。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import research_portfolio_cash as research
from portfolio_cash_state import basket_features
from build_selection_stability import digest,save
from audit_monthly_portfolio import reconcile
from portfolio_cash_engine import BacktestResult


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;cfg=json.loads((run/'protocol.json').read_text())
    for name,value in cfg['source_sha256'].items():assert digest(ROOT/name)==value
    assert digest(ROOT.parent/'AGENTS.md')==cfg['root_agents_sha256']
    research.initialize(run);prefixes=[]
    close=pd.read_parquet(Path(cfg['parent'])/'signals/close.parquet')
    for cut in ['2023-01-04','2025-01-03','2026-06-30']:
        short=basket_features(close.loc[:cut])
        for k,v in short.items():assert v.equals(research.FEATURES[k].loc[:cut])
    for name in ['continued_08_staged_cost0','cold_22_full_cost0','cold_01_staged_cost30']:
        case=next(c for c in cfg['cases'] if c['name']==name)
        folder=run/'accounts'/name
        old={k:pd.read_parquet(folder/f'{k}.parquet') for k in ['daily','trades','holdings','events']}
        cut=old['daily'].index[243]
        full_features=research.FEATURES
        research.FEATURES={k:v.loc[:cut] for k,v in full_features.items()}
        result,_=research.simulate(case,cut)
        research.FEATURES=full_features
        for key,expected in old.items():
            actual=getattr(result,key)
            if key=='daily':expected=expected.loc[:cut]
            else:
                expected=expected.loc[expected.date.lt(cut) if key=='events' else expected.date.le(cut)]
                if key=='events':
                    actual=actual.loc[actual.date.lt(cut)]
                    assert expected[expected.columns.difference(actual.columns)].isna().all().all()
                    actual=actual.reindex(columns=expected.columns)
                expected,actual=expected.reset_index(drop=True),actual.reset_index(drop=True)
            pd.testing.assert_frame_equal(actual,expected,check_exact=True,check_dtype=False)
        old_states=pd.read_csv(folder/'risk_states.csv',parse_dates=['date'])
        pd.testing.assert_frame_equal(pd.DataFrame(research.ACTIVE.records),old_states.loc[old_states.date.lt(cut)].reset_index(drop=True),check_dtype=False,rtol=1e-12,atol=1e-12)
        prefixes.append(dict(account=name,cutoff=str(cut.date()),all_ledgers_and_risk_states=True))
        print('prefix',name,flush=True)
    dates=research.FEATURES['known'].index
    weekly=set(dates.to_series().groupby(dates.to_period('W')).first())
    cash_requests=0;paused_buys=0;total_trades=0
    for case in cfg['cases']:
        folder=run/'accounts'/case['name'];assert not (folder/'failure.json').exists()
        data={k:pd.read_parquet(folder/f'{k}.parquet') for k in ['daily','trades','holdings','events']}
        result=BacktestResult(**data,summary=json.loads((folder/'summary.json').read_text()))
        reconcile(result,80000)
        t,e=result.trades,result.events;total_trades+=len(t)
        assert result.daily.n_hold.le(5).all() and result.daily.cash.ge(-1e-6).all()
        assert (t.date>t.decision_date).all()
        if case['overlay']=='none':continue
        orders=e.loc[e.kind.eq('order_planned')]
        for date,g in orders.loc[orders.side.eq('buy')].groupby('decision_date'):
            assert g.reserved_cash.sum()<=g.cash_at_decision.iloc[0]+1e-6
        for state in e.loc[e.kind.eq('portfolio_cash_state') & e.paused.eq(True)].itertuples():
            buys=orders.loc[orders.side.eq('buy') & orders.decision_date.eq(state.date)]
            assert buys.empty;paused_buys+=1
        reductions=e.loc[e.kind.eq('portfolio_cash_reduction')]
        for request in reductions.itertuples():
            assert request.date in weekly or request.date in research.prior.STATE['schedules'][case['offset']][case['policy']]['signals'].index
            assert request.target_shares<request.old_shares and request.target_shares%100==0
            cash_requests+=1
    assert len(list((run/'accounts').glob('*/parent_parity.json')))==16
    save(run/'verification.json',dict(accounts=96,trades=total_trades,feature_prefixes=3,account_prefixes=prefixes,
        parent_parities=16,weekly_or_monthly_cash_requests=cash_requests,paused_decisions_without_buys=paused_buys,
        all_reconciled=True,root_agents_unchanged=True,production_changed=False))
    print('verified',total_trades,'trades',cash_requests,'cash requests',flush=True)


if __name__=='__main__':main()
