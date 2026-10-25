"""对80账户作匹配比较、风险过程与机制触发检查，不筛选有利日历。"""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from research_path_mechanisms import ROOT,verify,save,digest
run=Path(sys.argv[1]).resolve();cfg,m=verify(run);parent=ROOT/cfg['parent'];out=run/'analysis';out.mkdir(exist_ok=False)
close=pd.read_parquet(parent/'inputs/daily_close.parquet')
rows=[];years=[];monthrows=[];assetmonths=[];recoveries=[];exits=[];checks=[];paired=[];orders=fills=0
for case in m['cases']:
    f=run/'execution'/case['name'];receipt=json.loads((f/'completion.json').read_text())
    for n,h in receipt['hashes'].items():assert digest(f/n)==h,n
    s=json.loads((f/'summary.json').read_text());d=pd.read_csv(f/'daily.csv',index_col='date',parse_dates=True);t=pd.read_csv(f/'trades.csv',parse_dates=['date','decision_date']);h=pd.read_csv(f/'holdings.csv',parse_dates=['date']);e=pd.read_csv(f/'events.csv',parse_dates=['date'])
    ret=d.equity.pct_change();ret.iloc[0]=d.equity.iloc[0]/cfg['capital']-1
    metrics=dict(total=d.equity.iloc[-1]/cfg['capital']-1,ann=(d.equity.iloc[-1]/cfg['capital'])**(244/len(d))-1,mdd=(d.equity/d.equity.cummax().clip(lower=cfg['capital'])-1).min(),sharpe=ret.mean()/ret.std(ddof=1)*np.sqrt(244))
    for k,v in metrics.items():assert abs(s[k]-v)<1e-10,(case['name'],k)
    assert d.cash.ge(-1e-7).all() and d.n_hold.le(5).all() and (t.date>t.decision_date).all()
    assert t[t.side.eq('buy')].shares.mod(100).eq(0).all()
    planned=e[e.kind.eq('order_planned')]
    for date,g in planned[planned.side.eq('buy')].groupby('decision_date'):
        assert g.reserved_cash.sum()<=d.loc[pd.Timestamp(date),'cash']+1e-7
    orders+=len(planned);fills+=len(t)
    # 用同日历基线首轮真实买单，核验所有机制均未改首次进入。
    base=parent/'execution'/f"{case['window']}_{case['schedule']}_hybrid"
    orig=pd.read_csv(base/'trades.csv');current=pd.read_csv(f/'trades.csv')
    first=lambda z:z[z.decision_date.eq(z.decision_date.min())].reset_index(drop=True)
    pd.testing.assert_frame_equal(first(orig),first(current),check_exact=True)
    first_buy=t[t.side.eq('buy')].date.min();s['active_cash_days']=int(d.loc[first_buy:,'n_hold'].eq(0).sum())
    s['trade_days']=t.date.nunique();s['initial_cash_days']=d.index.get_loc(first_buy)
    for kind in ['recovery_slot_wait','recovery_candidate','entry_slots_reserved','lot_cap_grace']:
        s[kind+'_events']=int(e.kind.eq(kind).sum())
    rec=e[e.kind.eq('recovery_candidate')][['date','ts_code']].drop_duplicates().rename(columns={'date':'decision_date'})
    bought=t[t.side.eq('buy')].merge(rec,on=['decision_date','ts_code'],how='inner',validate='one_to_one')
    s['actual_recovery_buys']=len(bought)
    for item in bought.itertuples():
        row=dict(name=case['name'],date=item.date,decision_date=item.decision_date,ts_code=item.ts_code,shares=item.shares,price=item.price)
        for horizon in [20,60]:
            loc=close.index.get_loc(item.date);future=loc+horizon
            row[f'postfill_close_return_{horizon}']=close.iloc[future][item.ts_code]/close.at[item.date,item.ts_code]-1 if future<len(close) else np.nan
        recoveries.append(row)
    reasons=planned[['order_id','reason']].drop_duplicates()
    for item in t[t.side.eq('sell')].merge(reasons,on='order_id',validate='one_to_one').itertuples():
        if item.reason!='hold_false':continue
        loc=close.index.get_loc(item.date);row=dict(name=case['name'],date=item.date,ts_code=item.ts_code,shares=item.shares)
        for horizon in [20,60]:row[f'postexit_close_return_{horizon}']=close.iloc[loc+horizon][item.ts_code]/close.at[item.date,item.ts_code]-1 if loc+horizon<len(close) else np.nan
        exits.append(row)
    trail=(d.equity/d.equity.shift(244)-1).dropna();s['worst_rolling_year']=trail.min();s['negative_rolling_year_fraction']=trail.lt(0).mean()
    periods=pd.read_csv(f/'holding_periods.csv');s['median_holding_days']=periods.holding_trading_days.median()
    s['end_equity']=d.equity.iloc[-1];rows.append(s);years.append(pd.read_csv(f/'yearly.csv'))
    month=d.equity.resample('ME').last().pct_change();month.iloc[0]=d.equity.resample('ME').last().iloc[0]/cfg['capital']-1
    monthrows.extend(dict(name=case['name'],month=str(date.to_period('M')),ret=value) for date,value in month.items())
    # 逐股逐日现金流+持仓市值变动+分红权利，与资产变动独立对账。
    values=h.pivot(index='date',columns='ts_code',values='value').reindex(d.index).fillna(0.)
    mark=values.diff();mark.iloc[0]=values.iloc[0]
    flow=t.assign(flow=np.where(t.side.eq('sell'),t.gross,-t.gross)-t.fee).pivot_table(index='date',columns='ts_code',values='flow',aggfunc='sum')
    ap=mark.add(flow.reindex(d.index).fillna(0.),fill_value=0.)
    div=e[e.kind.eq('dividend_receivable')]
    if len(div):ap=ap.add(div.pivot_table(index='date',columns='ts_code',values='amount',aggfunc='sum').reindex(d.index).fillna(0.),fill_value=0.)
    assert not e.kind.eq('delist_writeoff').any(),'退市回收需单列核验'
    expected=d.equity.diff();expected.iloc[0]=d.equity.iloc[0]-cfg['capital'];err=(ap.sum(axis=1)-expected).abs().max();assert err<1e-6
    am=ap.resample('ME').sum().stack().rename('pnl_yuan').reset_index();am['name']=case['name'];am['month']=am.date.dt.to_period('M').astype(str);assetmonths.append(am)
    checks.append(dict(name=case['name'],orders=len(planned),fills=len(t),daily_asset_pnl_error=err,cash_error=s['cash_max_error']))
summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False);pd.concat(years).to_csv(out/'yearly.csv',index=False)
pd.DataFrame(monthrows).to_csv(out/'monthly.csv',index=False);pd.concat(assetmonths).to_csv(out/'monthly_asset_pnl.csv',index=False)
pd.DataFrame(recoveries).to_csv(out/'actual_recovery_buys.csv',index=False);pd.DataFrame(exits).to_csv(out/'trend_exit_aftermath.csv',index=False)
for row in summary.itertuples():
    ref=summary[summary.window.eq(row.window)&summary.schedule.eq(row.schedule)&summary.policy.eq('baseline')].iloc[0]
    pair=dict(window=row.window,schedule=row.schedule,policy=row.policy,name=row.name)
    for key in ['total','ann','mdd','sharpe','n_trades','total_fees','mean_invested','max_marked_stock_weight','worst_rolling_year','active_cash_days']:
        pair[key+'_delta']=getattr(row,key)-ref[key]
    pair['both_return_drawdown_improve']=pair['ann_delta']>0 and pair['mdd_delta']>=0
    paired.append(pair)
pd.DataFrame(paired).to_csv(out/'paired_vs_baseline.csv',index=False)
# 恢复与单纯等待同样留名额，差额才支持评价恢复优先级。
rw=[]
for window in cfg['windows']:
    for schedule in cfg['schedules']:
        z=summary[summary.window.eq(window)&summary.schedule.eq(schedule)].set_index('policy')
        rw.append(dict(window=window,schedule=schedule,total_delta=z.at['recover','total']-z.at['wait','total'],mdd_delta=z.at['recover','mdd']-z.at['wait','mdd'],recovery_buys=z.at['recover','actual_recovery_buys']))
pd.DataFrame(rw).to_csv(out/'recovery_vs_wait.csv',index=False)
distributions=[]
for (window,policy),g in summary.groupby(['window','policy']):
    row=dict(window=window,policy=policy)
    for k in ['ann','total','mdd','sharpe','n_trades','active_cash_days','mean_invested','max_marked_stock_weight','worst_rolling_year']:
        row[k+'_median']=g[k].median();row[k+'_min']=g[k].min();row[k+'_max']=g[k].max()
    row['ann_range']=g.ann.max()-g.ann.min();distributions.append(row)
pd.DataFrame(distributions).to_csv(out/'distributions.csv',index=False)
save(out/'verification.json',dict(status='passed',cases=len(summary),orders=orders,fills=fills,checks=checks,initial_orders_exact=True))
save(out/'completion.json',dict(status='complete',source_sha256=digest(__file__),artifacts={f.name:digest(f) for f in out.iterdir()}))
print(pd.DataFrame(distributions)[['window','policy','ann_median','mdd_median','ann_range','active_cash_days_median']].round(4).to_string(index=False),flush=True)
