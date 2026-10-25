"""逐笔独立重建恢复额度，检查真实账户截断与资金记账。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import research_targeted_restore as research
from targeted_restore_engine import BacktestResult
from audit_monthly_portfolio import reconcile
from build_selection_stability import digest,save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;cfg=json.loads((run/'protocol.json').read_text())
    for name,value in cfg['source_sha256'].items():assert digest(ROOT/name)==value
    research.initialize(run);prefixes=[];quota_checks=0;fills=0
    for case in cfg['cases']:
        folder=run/'accounts'/case['name'];assert not (folder/'failure.json').exists()
        data={k:pd.read_parquet(folder/f'{k}.parquet') for k in ['daily','trades','holdings','events']}
        result=BacktestResult(**data,summary=json.loads((folder/'summary.json').read_text()))
        reconcile(result,80000)
        assert data['daily'].n_hold.le(5).all() and data['daily'].cash.ge(-1e-6).all()
        t,e=data['trades'],data['events'];assert (t.date>t.decision_date).all()
        assert t.loc[t.side.eq('buy'),'shares'].mod(100).eq(0).all()
        orders=e.loc[e.kind.eq('order_planned') & e.side.eq('buy')]
        for date,g in orders.groupby('decision_date'):assert g.reserved_cash.sum()<=g.cash_at_decision.iloc[0]+1e-6
        if not case['restore']:continue
        epoch=0;quota={};positions={}
        for r in e.itertuples():
            if r.kind=='restore_episode_started':epoch=int(r.epoch);quota={}
            elif r.kind=='restore_quota_invalidated':quota.pop(r.ts_code,None)
            elif r.kind=='bonus_shares':positions[r.ts_code]=positions.get(r.ts_code,0)+int(r.shares)
            elif r.kind=='delist_writeoff':positions.pop(r.ts_code,None);quota.pop(r.ts_code,None)
            elif r.kind=='order_filled':
                q=int(r.shares);code=r.ts_code
                if r.side=='sell':
                    positions[code]-=q
                    if r.reason=='portfolio_cash_cap' and int(r.restore_epoch)==epoch:quota[code]=quota.get(code,0)+q
                    if positions[code]==0:quota.pop(code,None)
                else:
                    if r.reason=='targeted_restore':
                        assert int(r.restore_epoch)==epoch and positions.get(code,0)>0 and q<=quota.get(code,0)
                        quota[code]-=q;fills+=1
                    else:quota.pop(code,None)
                    positions[code]=positions.get(code,0)+q
            elif r.kind=='targeted_restore_planned':
                assert int(r.epoch)==epoch and int(r.remaining_quota)==quota.get(r.ts_code,0)
                assert 0<r.buy_shares<=quota.get(r.ts_code,0) and positions.get(r.ts_code,0)>0
                assert r.frozen_position_notional<=r.risk_cap+1e-6
                quota_checks+=1
        if case['name'] in ['continued_08_staged_cost0_fast_restore','cold_01_staged_cost0_monthly_restore','cold_22_staged_cost30_fast_restore']:
            cut=data['daily'].index[243]
            research.setup(case);new,_=research.research.prior.simulate(case,cut)
            for key,expected in data.items():
                actual=getattr(new,key)
                if key=='daily':expected=expected.loc[:cut]
                else:
                    expected=expected.loc[expected.date.lt(cut) if key=='events' else expected.date.le(cut)]
                    if key=='events':
                        actual=actual.loc[actual.date.lt(cut)]
                        assert expected[expected.columns.difference(actual.columns)].isna().all().all()
                        actual=actual.reindex(columns=expected.columns)
                    expected,actual=expected.reset_index(drop=True),actual.reset_index(drop=True)
                pd.testing.assert_frame_equal(actual,expected,check_exact=True,check_dtype=False)
            prefixes.append(dict(account=case['name'],cutoff=str(cut.date()),all_four_ledgers_exact=True))
            print('prefix',case['name'],flush=True)
    assert len(prefixes)==3
    assert len(list((run/'accounts').glob('*/parent_parity.json')))==32
    assert digest(ROOT.parent/'AGENTS.md')==cfg['root_agents_sha256']
    save(run/'verification.json',dict(accounts=64,parent_parities=32,prefixes=prefixes,
        independently_rebuilt_quota_plans=quota_checks,independently_rebuilt_restore_fills=fills,
        cash_and_share_reconciliation=True,root_agents_unchanged=True,live_approved=False))
    print(quota_checks,'quota plans,',fills,'restore fills verified')


if __name__=='__main__':main()
