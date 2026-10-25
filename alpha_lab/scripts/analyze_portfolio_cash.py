"""完整报告分级防守的回撤、收益牺牲、现金期限与原篮子机会成本。"""
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
    df=pd.read_csv(run/'summary.csv');assert len(df)==len(cfg['cases'])==96
    out=run/'analysis';out.mkdir(exist_ok=True)
    price=pd.read_parquet(Path(cfg['parent'])/'signals/close.parquet')
    accounts={};extended=[];years=[];episodes=[]
    for row in df.itertuples():
        folder=run/'accounts'/row.name
        item={k:pd.read_parquet(folder/f'{k}.parquet') for k in ['daily','trades','holdings','events']}
        accounts[row.name]=item
        daily=item['daily'];trades=item['trades'];events=item['events']
        pnl=pnl_by_asset(item);assert abs(pnl.sum()-(daily.equity.iloc[-1]-80000))<1e-6
        first=trades.loc[trades.side.eq('buy'),'date'].min()
        active=daily.loc[first:] if pd.notna(first) else daily
        flat=active.n_hold.eq(0);groups=flat.ne(flat.shift()).cumsum()
        longest=int(flat.groupby(groups).sum().max()) if len(flat) else 0
        d=dict(row._asdict(),fully_cash_days=int(flat.sum()),max_cash_run=longest,
            mean_n_hold=float(daily.n_hold.mean()),active_start=str(first),min_cash=float(daily.cash.min()))
        extended.append(d)
        previous=80000.
        for year,g in daily.groupby(daily.index.year):
            years.append(dict(account=row.name,label=row.label,cost_bps=row.cost_bps,cohort=row.cohort,
                offset=row.offset,year=year,total=g.equity.iloc[-1]/previous-1))
            previous=g.equity.iloc[-1]
        if row.overlay in ['none','fixed_half']:continue
        states=pd.read_csv(folder/'risk_states.csv',parse_dates=['date'])
        alarms=states.loc[states.cap_before.eq(1)&states.cap_after.lt(1)]
        for alarm in alarms.itertuples():
            next_alarm=alarms.loc[alarms.date.gt(alarm.date),'date'].min()
            later=states.loc[states.date.gt(alarm.date)]
            recovery=later.loc[later.cap_after.eq(1),'date'].min()
            end=recovery if pd.notna(recovery) else daily.index[-1]
            codes=alarm.basket.split('|');weights=np.array([float(x) for x in str(alarm.weights).split('|')])
            assert abs(weights.sum()-1)<1e-8
            prices=price.reindex(columns=codes)
            start=prices.index.get_loc(alarm.date)
            ep=dict(account=row.name,label=row.label,cost_bps=row.cost_bps,alarm=alarm.date,recovery=recovery,
                basket=alarm.basket,known_at_alarm=alarm.known,raw_weighted_dd=alarm.drawdown,
                cash_days=int(daily.loc[alarm.date:end].n_hold.eq(0).sum()))
            buy=trades.loc[trades.side.eq('buy') & trades.date.gt(recovery),'date'] if pd.notna(recovery) else pd.Series(dtype='datetime64[ns]')
            if pd.notna(next_alarm):buy=buy.loc[buy.lt(next_alarm)]
            ep['first_buy_after_recovery']=buy.min()
            ep['reentry_wait_sessions']=int(((daily.index>recovery)&(daily.index<=buy.min())).sum()) if len(buy) else np.nan
            for horizon in [5,20,60]:
                mark=start+horizon
                if mark<len(prices):
                    segment=prices.iloc[start:mark+1]
                    complete=segment.notna().all().all()
                    path=(segment/segment.iloc[0]-1).mul(weights,axis=1).sum(axis=1) if complete else None
                    ep[f'basket_{horizon}']=float(path.iloc[-1]) if complete else np.nan
                    ep[f'basket_mae_{horizon}']=min(0.,float(path.min())) if complete else np.nan
                    ep[f'complete_{horizon}']=bool(complete)
                else:ep[f'basket_{horizon}']=np.nan;ep[f'basket_mae_{horizon}']=np.nan;ep[f'complete_{horizon}']=False
            episodes.append(ep)
    detail=pd.DataFrame(extended);detail.to_csv(out/'accounts.csv',index=False)
    pd.DataFrame(years).to_csv(out/'yearly.csv',index=False)
    pd.DataFrame(episodes).to_csv(out/'episodes.csv',index=False)
    rows=[]
    lookup=detail.set_index(['cohort','offset','label','cost_bps'])
    for row in detail.itertuples():
        if row.label in ['base','rank']:continue
        comparator='base' if row.label=='base_staged' else 'rank'
        b=lookup.loc[(row.cohort,row.offset,comparator,row.cost_bps)]
        rows.append(dict(account=row.name,label=row.label,cost_bps=row.cost_bps,cohort=row.cohort,offset=row.offset,
            ann_delta=row.ann-b.ann,mdd_delta=row.mdd-b.mdd,sharpe_delta=row.sharpe-b.sharpe,
            total_delta=row.total-b.total,trade_fraction=row.n_trades/b.n_trades-1,
            invested_delta=row.mean_invested-b.mean_invested,cash_days=row.fully_cash_days,max_cash_run=row.max_cash_run))
    paired=pd.DataFrame(rows);paired.to_csv(out/'paired.csv',index=False)
    med=paired.groupby(['label','cost_bps']).agg(ann_delta=('ann_delta','median'),mdd_delta=('mdd_delta','median'),
        sharpe_delta=('sharpe_delta','median'),trade_fraction=('trade_fraction','median'),
        invested_delta=('invested_delta','median'),cash_days=('cash_days','median'),max_cash_run=('max_cash_run','max'),
        mdd_improved=('mdd_delta',lambda x:int(x.gt(0).sum())),ann_improved=('ann_delta',lambda x:int(x.gt(0).sum())))
    med.to_csv(out/'paired_medians.csv')
    screens=cfg['screens'];assessment=[]
    for (label,cost),r in med.iterrows():
        assessment.append(dict(label=label,cost_bps=cost,
            defense=r.mdd_improved>=screens['mdd_improved_min'],return_preservation=r.ann_delta>=-screens['median_ann_loss_max'],
            turnover=r.trade_fraction<=screens['median_trade_increase_max']))
    save(out/'assessment.json',dict(checks=assessment,independent_oos=False,live_approved=False,
        frozen_basket_note='警报原股票权重的复权价格代理，不是可执行收益或纯择时归因；缺报价标未知、不填安全；5/20/60天仅事后诊断'))
    print(med.to_string())


if __name__=='__main__':main()
