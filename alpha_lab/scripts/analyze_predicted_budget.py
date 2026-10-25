"""对所有预定账户复算配对效用，拒绝用最好日历作为策略结论。"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from analyze_selection_stability import pnl_by_asset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run
    protocol=json.loads((run/'protocol.json').read_text())
    summary=pd.read_csv(run/'summary.csv')
    assert len(summary)==len(protocol['cases'])
    out=run/'analysis';out.mkdir(exist_ok=True)
    paired=[];assets=[];years=[];episodes=[];frequency=[]
    for row in summary.itertuples():
        folder=run/'accounts'/row.name
        rank=Path(protocol['cash_parent'])/'accounts'/f'{row.cohort}_{row.offset:02d}_rank_cost{row.cost_bps}'
        baseline=json.loads((rank/'summary.json').read_text())
        item={k:pd.read_parquet(folder/f'{k}.parquet') for k in ['daily','trades','holdings','events']}
        other={k:pd.read_parquet(rank/f'{k}.parquet') for k in item}
        assert item['daily'].index.equals(other['daily'].index)
        scale=244/len(item['daily'])
        frequency.append(dict(account=row.name,label=row.label,cost_bps=row.cost_bps,
            orders_year=len(item['trades'])*scale,trade_days_year=item['trades'].date.nunique()*scale,
            base_orders_year=len(other['trades'])*scale,base_days_year=other['trades'].date.nunique()*scale,
            extra_orders_month=(len(item['trades'])-len(other['trades']))*scale/12))
        a,b=pnl_by_asset(item),pnl_by_asset(other)
        assert abs(a.sum()-(item['daily'].equity.iloc[-1]-80000))<1e-6
        delta=a.subtract(b,fill_value=0)
        assert abs(delta.sum()-(item['daily'].equity.iloc[-1]-other['daily'].equity.iloc[-1]))<1e-6
        assets.extend(dict(account=row.name,label=row.label,code=code,pnl_delta_yuan=value) for code,value in delta.items())
        paired.append(dict(account=row.name,label=row.label,cohort=row.cohort,offset=row.offset,cost_bps=row.cost_bps,
            ann_delta=row.ann-baseline['ann'],mdd_delta=row.mdd-baseline['mdd'],sharpe_delta=row.sharpe-baseline['sharpe'],
            trade_fraction=row.n_trades/baseline['n_trades']-1,fee_delta=row.total_fees-baseline['total_fees'],
            ann=row.ann,mdd=row.mdd,sharpe=row.sharpe,n_trades=row.n_trades,exposure=row.mean_invested))
        prev=prev_b=80000.
        for year,g in item['daily'].groupby(item['daily'].index.year):
            gb=other['daily'].loc[g.index]
            years.append(dict(account=row.name,label=row.label,cost_bps=row.cost_bps,year=year,
                total=g.equity.iloc[-1]/prev-1,rank_total=gb.equity.iloc[-1]/prev_b-1,
                pnl_delta_yuan=g.equity.iloc[-1]-prev-gb.equity.iloc[-1]+prev_b))
            prev=g.equity.iloc[-1];prev_b=gb.equity.iloc[-1]
        states=pd.read_csv(folder/'risk_states.csv') if (folder/'risk_states.csv').exists() else pd.DataFrame(columns=['cap_before','cap_after'])
        episodes.append(dict(account=row.name,label=row.label,cost_bps=row.cost_bps,
            alarms=int((states.cap_before.eq(1)&states.cap_after.lt(1)).sum()),
            full_cash_requests=int((states.cap_before.gt(0)&states.cap_after.eq(0)).sum()),
            limited_weeks=int(states.cap_after.lt(1).sum()),
            cash_days=int(item['daily'].n_hold.eq(0).sum())))
    paired=pd.DataFrame(paired);paired.to_csv(out/'paired.csv',index=False)
    med=paired.groupby(['label','cost_bps']).agg(ann_delta=('ann_delta','median'),mdd_delta=('mdd_delta','median'),
        sharpe_delta=('sharpe_delta','median'),trade_fraction=('trade_fraction','median'),
        ann=('ann','median'),mdd=('mdd','median'),sharpe=('sharpe','median'),
        mdd_improved=('mdd_delta',lambda x:int(x.gt(1e-10).sum())),ann_improved=('ann_delta',lambda x:int(x.gt(1e-10).sum())))
    med.to_csv(out/'paired_medians.csv')
    pd.DataFrame(assets).to_csv(out/'asset_contributions.csv',index=False)
    pd.DataFrame(years).to_csv(out/'yearly.csv',index=False)
    pd.DataFrame(episodes).to_csv(out/'episode_counts.csv',index=False)
    pd.DataFrame(frequency).to_csv(out/'absolute_frequency.csv',index=False)
    gates=[];screens=protocol['screens']
    for (label,cost),r in med.iterrows():
        checks=dict(return_preservation=bool(r.ann_delta>=-screens['median_ann_loss_max']),
            defense=bool(r.mdd_improved>=screens['mdd_improved_min']),
            trading_frequency=bool(r.trade_fraction<=screens['median_trade_increase_max']))
        gates.append(dict(label=label,cost_bps=int(cost),checks=checks,passes_all=all(checks.values())))
    (out/'acceptance.json').write_text(json.dumps(dict(gates=gates,independent_oos=False,live_approved=False),ensure_ascii=False,indent=2))
    print(med.to_string())
    print(json.dumps(gates,ensure_ascii=False))


if __name__=='__main__':main()
