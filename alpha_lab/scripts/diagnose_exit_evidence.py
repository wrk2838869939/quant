"""固定预测的影响诊断、状态分组及恢复等待期间行情，不重选阈值。"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from research_exit_evidence import ROOT,verify,save,digest
run=Path(sys.argv[1]).resolve();cfg=verify(run);parent=ROOT/cfg['parent'];out=run/'diagnostics';out.mkdir(exist_ok=False)
pred=pd.read_csv(run/'models/predictions.csv',parse_dates=['date'])
g=pred[pred.cohort.eq('exit')&pred.horizon.eq(60)&pred.actual.notna()].copy()
def metrics(x):
    return dict(n=len(x),r2=float(1-((x.actual-x.prediction)**2).sum()/((x.actual-x.baseline)**2).sum()))
influence=[]
for key in ['ts_code','year']:
    for value in g[key].unique():influence.append(dict(omitted_kind=key,omitted=str(value),**metrics(g[g[key].ne(value)])))
g['halfyear']=g.date.dt.year.astype(str)+'H'+np.where(g.date.dt.month<=6,'1','2')
for value in g.halfyear.unique():influence.append(dict(omitted_kind='halfyear',omitted=value,**metrics(g[g.halfyear.ne(value)])))
pd.DataFrame(influence).to_csv(out/'fixed_prediction_influence.csv',index=False)
g['squared_error_gain']=(g.actual-g.baseline)**2-(g.actual-g.prediction)**2
g.groupby('ts_code').squared_error_gain.sum().sort_values(ascending=False).to_csv(out/'stock_error_contribution.csv')
rows=[];shape=[]
for cohort in cfg['cohorts']:
    data=pd.read_parquet(run/'data'/f'{cohort}_events.parquet');data=data[data.nonoverlap_primary]
    for period,sub in [('2022-2023',data[data.date.dt.year<2024]),('2024+',data[data.date.dt.year>=2024])]:
        for state,block in sub.groupby('state'):
            for horizon in cfg['horizons']:
                b=block[block[f'ret_{horizon}'].notna()]
                rows.append(dict(cohort=cohort,period=period,state=state,horizon=horizon,n=len(b),mean_return=b[f'ret_{horizon}'].mean(),median_return=b[f'ret_{horizon}'].median(),positive_fraction=b[f'ret_{horizon}'].gt(0).mean(),mean_mae=b[f'mae_{horizon}'].mean()))
        b=sub[sub.ret_60.notna()]
        for x in b.itertuples():shape.append(dict(cohort=cohort,period=period,date=x.date,ts_code=x.ts_code,ret60=x.ret_60,mae60=x.mae_60,path=('deep_then_positive' if x.mae_60<=-.1 and x.ret_60>0 else 'deep_nonpositive' if x.mae_60<=-.1 else 'positive' if x.ret_60>0 else 'mild_nonpositive')))
pd.DataFrame(rows).to_csv(out/'state_outcomes.csv',index=False);pd.DataFrame(shape).to_csv(out/'path_shapes.csv',index=False)
price=pd.read_parquet(parent/'inputs/daily_close.parquet');market=pd.read_parquet(parent/'inputs/market.parquet').set_index(['date','ts_code'])
recovery=pd.read_csv(run/'data/recovery_account_events.csv',parse_dates=['date','sale_completed'])
recovery['waiting_close_return']=[price.at[r.date,r.ts_code]/price.at[r.sale_completed,r.ts_code]-1 for r in recovery.itertuples()]
recovery['waiting_trading_days']=[price.index.get_loc(r.date)-price.index.get_loc(r.sale_completed) for r in recovery.itertuples()]
recovery.to_csv(out/'recovery_waiting_moves.csv',index=False)
# 原标签从次日收盘开始；单列遗漏的隔夜和首日日内变化，避免把标签视为实际卖单回报。
moves=[]
for cohort in cfg['cohorts']:
    events=pd.read_parquet(run/'data'/f'{cohort}_events.parquet');events=events[events.nonoverlap_primary]
    for r in events.itertuples():
        i=price.index.get_loc(r.date)
        if i+1>=len(price):continue
        nxt=price.index[i+1];quote=market.loc[(nxt,r.ts_code)]
        scale=price.at[nxt,r.ts_code]/quote['close'];op=quote['open']*scale
        moves.append(dict(cohort=cohort,date=r.date,ts_code=r.ts_code,next_date=nxt,decision_close_to_next_open=op/price.at[r.date,r.ts_code]-1,next_open_to_next_close=price.at[nxt,r.ts_code]/op-1))
pd.DataFrame(moves).to_csv(out/'first_day_moves.csv',index=False)
save(out/'completion.json',dict(status='complete',source_sha256=digest(__file__),artifacts={f.name:digest(f) for f in out.iterdir()}))
print(pd.DataFrame(influence).groupby('omitted_kind').r2.agg(['min','max']).round(4).to_string());print('Recovery waiting median return',recovery.waiting_close_return.median())
