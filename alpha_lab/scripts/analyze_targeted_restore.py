"""定向恢复的交易闭环、历史效用与原问题实例核验。"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_selection_stability import save
from analyze_selection_stability import pnl_by_asset


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;cfg=json.loads((run/'protocol.json').read_text())
    out=run/'analysis';out.mkdir(exist_ok=True)
    df=pd.read_csv(run/'summary.csv');assert len(df)==64
    by=df.set_index(['cohort','offset','cost_bps','label'])
    pairs=[];restores=[];skips=[];metrics=[];years=[]
    for case in cfg['cases']:
        folder=run/'accounts'/case['name'];summary=json.loads((folder/'summary.json').read_text())
        item={k:pd.read_parquet(folder/f'{k}.parquet') for k in ['daily','trades','holdings','events']}
        d,e,t=item['daily'],item['events'],item['trades']
        assert abs(pnl_by_asset(item).sum()-(d.equity.iloc[-1]-80000))<1e-6
        previous=80000.
        for year,g in d.groupby(d.index.year):
            years.append(dict(account=case['name'],year=year,total=g.equity.iloc[-1]/previous-1))
            previous=g.equity.iloc[-1]
        if not case['restore']:continue
        control=('fast' if case['fast'] else 'monthly')+'_control'
        baseline=by.loc[(case['cohort'],case['offset'],case['cost_bps'],control)]
        rank_path=Path(cfg['cash_parent'])/'accounts'/f"{case['cohort']}_{case['offset']:02d}_rank_cost{case['cost_bps']}"/'summary.json'
        rank=json.loads(rank_path.read_text())
        pairs.append(dict(account=case['name'],label=case['label'],cost_bps=case['cost_bps'],cohort=case['cohort'],offset=case['offset'],
            ann_delta=summary['ann']-baseline.ann,mdd_delta=summary['mdd']-baseline.mdd,
            trade_fraction=summary['n_trades']/baseline.n_trades-1,
            ann_vs_rank=summary['ann']-rank['ann'],mdd_vs_rank=summary['mdd']-rank['mdd'],
            trade_vs_rank=summary['n_trades']/rank['n_trades']-1))
        planned=e.loc[e.kind.eq('targeted_restore_planned')]
        filled=e.loc[e.kind.eq('targeted_restore_filled')]
        metrics.append(dict(account=case['name'],planned=len(planned),fills=len(filled),
            restored_shares=float(filled.shares.sum()) if len(filled) else 0,
            mean_invested=summary['mean_invested'],mean_n_hold=d.n_hold.mean(),
            quota_invalidations=int(e.kind.eq('restore_quota_invalidated').sum()),
            stale_cash_intents_cancelled=int(e.kind.eq('cash_risk_intent_cancelled').sum())))
        for r in planned.itertuples():
            assert r.buy_shares<=r.remaining_quota and r.buy_shares%100==0
            assert r.frozen_position_notional<=r.risk_cap+1e-6
            restores.append(dict(account=case['name'],date=r.date,ts_code=r.ts_code,epoch=r.epoch,
                old_shares=r.old_shares,buy_shares=r.buy_shares,quota=r.remaining_quota,reserved_cash=r.reserved_cash))
        for reason,g in e.loc[e.kind.eq('restore_skipped')].groupby('reason'):
            skips.append(dict(account=case['name'],reason=reason,count=len(g)))
    paired=pd.DataFrame(pairs);paired.to_csv(out/'paired.csv',index=False)
    med=paired.groupby(['label','cost_bps']).agg(ann_delta=('ann_delta','median'),mdd_delta=('mdd_delta','median'),
        trade_fraction=('trade_fraction','median'),ann_improved=('ann_delta',lambda x:int(x.gt(0).sum())),
        mdd_improved=('mdd_delta',lambda x:int(x.gt(0).sum())),ann_vs_rank=('ann_vs_rank','median'),
        mdd_vs_rank=('mdd_vs_rank','median'),trade_vs_rank=('trade_vs_rank','median'))
    med.to_csv(out/'paired_medians.csv')
    pd.DataFrame(metrics).to_csv(out/'restoration_metrics.csv',index=False)
    pd.DataFrame(restores).to_csv(out/'restoration_plans.csv',index=False)
    pd.DataFrame(skips).to_csv(out/'restoration_skips.csv',index=False)
    pd.DataFrame(years).to_csv(out/'yearly.csv',index=False)
    # 原五席位闲置实例：固定已知窗口作状态核对，不用其表现调参。
    example=[]
    for label in ['monthly_control','monthly_restore','fast_control','fast_restore']:
        folder=run/'accounts'/f'cold_01_staged_cost0_{label}'
        d=pd.read_parquet(folder/'daily.parquet').loc['2024-10-08':'2024-10-25']
        e=pd.read_parquet(folder/'events.parquet')
        t=pd.read_parquet(folder/'trades.parquet')
        p=e.loc[e.kind.eq('targeted_restore_planned') & e.date.between('2024-10-08','2024-10-25')]
        example.append(dict(label=label,mean_exposure=(d.stock_value/d.equity).mean(),
            min_exposure=(d.stock_value/d.equity).min(),max_exposure=(d.stock_value/d.equity).max(),
            max_positions=d.n_hold.max(),restore_plans=len(p),
            buy_fills=int((t.side.eq('buy')&t.date.between('2024-10-08','2024-10-25')).sum())))
    pd.DataFrame(example).to_csv(out/'original_failure_example.csv',index=False)
    save(out/'assessment.json',dict(structural_fix='only_recorded_cash_cut_quota_can_restore_still_held_eligible_shares',
        independent_oos=False,live_approved=False,scope='研究实现；人工入口尚未接入风险状态与恢复额度事务'))
    print(med.to_string());print(pd.DataFrame(example).to_string(index=False))


if __name__=='__main__':main()
