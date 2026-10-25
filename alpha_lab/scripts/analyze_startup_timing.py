"""按路径与逐股损益解释日期敏感性，不把账户均值当成可交易组合。"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str))


def asset_pnl(item,end):
    trades,events,hold=item['trades'],item['events'],item['holdings']
    last=hold.loc[hold.date.eq(end)].set_index('ts_code').value
    flow=trades.assign(net=np.where(trades.side.eq('sell'),trades.gross,-trades.gross)-trades.fee).groupby('ts_code').net.sum()
    dividends=events.loc[events.kind.eq('dividend_receivable')].groupby('ts_code').amount.sum()
    recovery=events.loc[events.kind.eq('delist_writeoff')].groupby('ts_code').cash_recovery.sum() if 'cash_recovery' in events else pd.Series(dtype=float)
    return last.add(flow,fill_value=0).add(dividends,fill_value=0).add(recovery,fill_value=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run;out=run/'analysis';out.mkdir(exist_ok=False)
    manifest=json.loads((run/'manifest.json').read_text());capital=manifest['config']['capital']
    summary=pd.read_csv(run/'summary.csv');summary=summary.set_index('name',drop=False)
    accounts={}
    for name in summary.index:
        folder=run/'accounts'/name
        accounts[name]={key:pd.read_parquet(folder/(key+'.parquet')) for key in ['daily','trades','holdings','events']}
        pnl=asset_pnl(accounts[name],accounts[name]['daily'].index[-1])
        residual=float(pnl.sum()-(accounts[name]['daily'].equity.iloc[-1]-capital))
        assert abs(residual)<1e-6,(name,residual)
        accounts[name]['pnl']=pnl
        pnl.rename('net_pnl_yuan').to_csv(folder/'asset_pnl.csv')
    comparisons=[];stages=[];contributions=[];initial=[];parities=[];diagnostics=[]
    for cohort,group in summary.groupby('cohort'):
        wait=cohort+'_wait'
        for offset in manifest['config']['offset_days']:
            fresh=f'{cohort}_{offset:02d}_fresh_once';stale=f'{cohort}_{offset:02d}_stale_once';shift=f'{cohort}_{offset:02d}_shifted_calendar'
            boot=pd.Timestamp(summary.loc[fresh,'boot_date'])
            first=[]
            for name in [fresh,stale,shift]:
                t=accounts[name]['trades'];buy=t.loc[t.side.eq('buy') & t.decision_date.eq(boot)].sort_values('ts_code')
                first.append(buy)
                initial.extend(dict(account=name,cohort=cohort,offset=offset,**row) for row in buy.to_dict('records'))
            keys=['date','decision_date','ts_code','shares','price','fee']
            pd.testing.assert_frame_equal(first[0][keys].reset_index(drop=True),first[2][keys].reset_index(drop=True),check_exact=True)
            parities.append(dict(cohort=cohort,offset=offset,first_orders_exact=True))
            for label,left,right in [('fresh_minus_wait',fresh,wait),('stale_minus_wait',stale,wait),
                                      ('fresh_minus_stale',fresh,stale),('shift_minus_fresh',shift,fresh)]:
                a,b=accounts[left],accounts[right];ad,bd=a['daily'],b['daily']
                assert ad.index.equals(bd.index)
                delta=(ad.equity-bd.equity)/capital
                av=a['holdings'].pivot(index='date',columns='ts_code',values='value').reindex(ad.index).fillna(0)
                bv=b['holdings'].pivot(index='date',columns='ts_code',values='value').reindex(bd.index).fillna(0)
                columns=av.columns.union(bv.columns);av=av.reindex(columns=columns,fill_value=0);bv=bv.reindex(columns=columns,fill_value=0)
                holding_diff=(av-bv).abs().sum(axis=1)
                changed=holding_diff[holding_diff>.01]
                row=dict(cohort=cohort,offset=offset,comparison=label,left=left,right=right,
                    total_delta=float(delta.iloc[-1]),mdd_delta=float(summary.loc[left,'mdd']-summary.loc[right,'mdd']),
                    sharpe_delta=float(summary.loc[left,'sharpe']-summary.loc[right,'sharpe']),
                    trade_delta=int(summary.loc[left,'n_trades']-summary.loc[right,'n_trades']),
                    first_position_difference=None if changed.empty else str(changed.index[0].date()),
                    mean_l1_position_value_difference=float(holding_diff.mean()/capital))
                comparisons.append(row)
                for sessions in [20,60,126,244]:
                    stages.append(dict(cohort=cohort,offset=offset,comparison=label,sessions=sessions,
                        date=str(ad.index[sessions-1].date()),delta=float(delta.iloc[sessions-1])))
                diff=a['pnl'].subtract(b['pnl'],fill_value=0)
                assert abs(diff.sum()-delta.iloc[-1]*capital)<1e-6
                contributions.extend(dict(cohort=cohort,offset=offset,comparison=label,ts_code=c,
                    pnl_delta=float(v),return_delta=float(v/capital)) for c,v in diff.items())
            fset,sset=set(first[0].ts_code),set(first[1].ts_code)
            # 下一月的首个事件不一定是复核，明确从已知月初矩阵取日期。
            parent=ROOT/manifest['config']['parent']
            base_idx=pd.read_parquet(parent/'schedules/calendar_01/signals.parquet').index
            next_review=base_idx[base_idx>boot][0]
            ev=accounts[fresh]['events']
            reduced=ev.loc[ev.date.eq(next_review)&ev.kind.isin(['hold_risk_cap_reduction','hold_exit_requested'])]
            diagnostics.append(dict(cohort=cohort,offset=offset,
                initial_fresh_count=len(fset),initial_stale_count=len(sset),initial_overlap=len(fset&sset),
                initial_union=len(fset|sset),initial_jaccard=len(fset&sset)/len(fset|sset) if fset|sset else 1,
                first_regular_review=str(next_review.date()),
                first_regular_hold_exits=int((reduced.kind=='hold_exit_requested').sum()),
                first_regular_cap_reductions=int((reduced.kind=='hold_risk_cap_reduction').sum())))
    pairs=pd.DataFrame(comparisons);pairs.to_csv(out/'paired_results.csv',index=False)
    pd.DataFrame(stages).to_csv(out/'path_checkpoints.csv',index=False)
    pd.DataFrame(contributions).to_csv(out/'asset_contributions.csv',index=False)
    pd.DataFrame(initial).to_csv(out/'initial_fills.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'initial_diagnostics.csv',index=False)
    save(out/'same_first_fills.json',parities)
    spread=[]
    for (cohort,policy),group in summary.loc[summary.policy.ne('wait')].groupby(['cohort','policy']):
        spread.append(dict(cohort=cohort,policy=policy,total_range=float(group.total.max()-group.total.min()),
            mdd_range=float(group.mdd.max()-group.mdd.min()),sharpe_range=float(group.sharpe.max()-group.sharpe.min())))
    spread=pd.DataFrame(spread);spread.to_csv(out/'within_cohort_spreads.csv',index=False)
    aggregate=[]
    for label,group in pairs.groupby('comparison'):
        aggregate.append(dict(comparison=label,n=len(group),positive=int(group.total_delta.gt(1e-10).sum()),
            median_delta=float(group.total_delta.median()),min_delta=float(group.total_delta.min()),
            max_delta=float(group.total_delta.max()),median_abs_delta=float(group.total_delta.abs().median()),
            median_mdd_delta=float(group.mdd_delta.median()),median_trade_delta=float(group.trade_delta.median())))
    pd.DataFrame(aggregate).to_csv(out/'paired_aggregate.csv',index=False)
    spread_aggregate=spread.groupby('policy').total_range.agg(['median','min','max'])
    spread_aggregate.to_csv(out/'spread_aggregate.csv')
    # 展示单个最敏感窗口只解释机制；所有窗口仍完整列出，不据此选新默认。
    largest=spread.loc[spread.policy.eq('fresh_once')].sort_values('total_range',ascending=False).iloc[0]
    selected=summary.loc[(summary.cohort==largest.cohort)&summary.policy.eq('fresh_once')]
    low=selected.total.idxmin();high=selected.total.idxmax()
    top=accounts[high]['pnl'].subtract(accounts[low]['pnl'],fill_value=0).sort_values(ascending=False)
    top.rename('pnl_delta_yuan').to_csv(out/'largest_fresh_spread_assets.csv')
    save(out/'verification.json',dict(accounts=len(accounts),all_asset_pnl_reconciled=True,
        identical_first_fill_pairs=len(parities),largest_fresh_spread=dict(cohort=largest.cohort,
        low=low,high=high,range=float(largest.total_range)),overlapping_windows_not_independent=True))
    print(pd.DataFrame(aggregate).to_string(index=False));print(spread_aggregate.to_string())


if __name__=='__main__':main()
