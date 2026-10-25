"""复核提前重入消融及恢复后资金闲置，不能将允许新仓等同恢复旧仓。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import research_cash_reentry as branch
from build_selection_stability import digest,save
from audit_monthly_portfolio import reconcile
from portfolio_cash_reentry_engine import BacktestResult


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;cfg=json.loads((run/'protocol.json').read_text());parent=Path(cfg['parent'])
    for name,value in cfg['source_sha256'].items():assert digest(ROOT/name)==value
    assert digest(parent/'protocol.json')==cfg['parent_protocol_sha256']
    branch.initialize(run,parent);rows=[];prefixes=[]
    for case in cfg['cases']:
        folder=run/'accounts'/case['name'];data={k:pd.read_parquet(folder/f'{k}.parquet') for k in ['daily','trades','holdings','events']}
        summary=json.loads((folder/'summary.json').read_text())
        reconcile(BacktestResult(**data,summary=summary),80000)
        original=json.loads((parent/'accounts'/case['source']/'summary.json').read_text())
        if not case['control']:
            rows.append(dict(account=case['name'],cost_bps=case['cost_bps'],cohort=case['cohort'],offset=case['offset'],
                ann_delta=summary['ann']-original['ann'],mdd_delta=summary['mdd']-original['mdd'],
                trade_fraction=summary['n_trades']/original['n_trades']-1))
        if case['name'] in ['cold_01_staged_cost0_fast_reentry','continued_08_staged_cost30_fast_reentry']:
            cut=data['daily'].index[243]
            branch.research.ACTIVE=branch.RecoveryState(branch.research.FEATURES,'staged')
            branch.research.COST=case['cost_bps']/10000
            result,_=branch.research.prior.simulate(case,cut)
            for key,expected in data.items():
                actual=getattr(result,key)
                if key=='daily':expected=expected.loc[:cut]
                else:
                    expected=expected.loc[expected.date.lt(cut) if key=='events' else expected.date.le(cut)]
                    if key=='events':
                        actual=actual.loc[actual.date.lt(cut)]
                        assert expected[expected.columns.difference(actual.columns)].isna().all().all()
                        actual=actual.reindex(columns=expected.columns)
                    expected,actual=expected.reset_index(drop=True),actual.reset_index(drop=True)
                pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)
            prefixes.append(dict(account=case['name'],cutoff=str(cut.date()),all_four_ledgers_exact=True))
    paired=pd.DataFrame(rows);paired.to_csv(run/'paired.csv',index=False)
    med=paired.groupby('cost_bps').agg(ann_delta=('ann_delta','median'),mdd_delta=('mdd_delta','median'),
        trade_fraction=('trade_fraction','median'),ann_improved=('ann_delta',lambda x:int(x.gt(0).sum())),
        mdd_improved=('mdd_delta',lambda x:int(x.gt(0).sum())))
    med.to_csv(run/'paired_medians.csv')
    episodes=pd.read_csv(parent/'analysis/episodes.csv',parse_dates=['alarm','recovery'])
    capacity=[]
    for r in episodes.loc[episodes.label.eq('staged') & episodes.cost_bps.eq(0)&episodes.recovery.notna()].itertuples():
        daily=pd.read_parquet(parent/'accounts'/r.account/'daily.parquet');i=daily.index.get_loc(r.recovery)
        capacity.append(dict(account=r.account,recovery=r.recovery,n_hold=int(daily.n_hold.iloc[i]),
            exposure=daily.stock_value.iloc[i]/daily.equity.iloc[i],cash=daily.cash.iloc[i],
            mean_next20_exposure=(daily.stock_value.iloc[i:i+21]/daily.equity.iloc[i:i+21]).mean(),
            complete20=i+20<len(daily)))
    cap=pd.DataFrame(capacity);cap.to_csv(parent/'analysis/recovery_capacity.csv',index=False)
    save(run/'verification.json',dict(accounts=18,all_reconciled=True,controls=2,prefixes=prefixes,
        recovery_events=len(cap),five_slots_below60pct=int((cap.n_hold.eq(5)&cap.exposure.lt(.6)).sum()),
        interpretation='重入只增加新股建仓机会；未恢复减掉的旧仓，不能据本消融声称完整退出恢复机制无效',
        live_approved=False))
    print(med.to_string())


if __name__=='__main__':main()
