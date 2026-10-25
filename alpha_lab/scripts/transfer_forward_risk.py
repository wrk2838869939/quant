"""将冻结市场风险预测映射到真实历史持仓篮子；仅作标签代理，不冒充交易回测。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;cfg=json.loads((run/'protocol.json').read_text())
    old=json.loads((Path(cfg['parent'])/'protocol.json').read_text())
    cash=Path(old['cash_parent']);cash_cfg=json.loads((cash/'protocol.json').read_text())
    selection=Path(cash_cfg['parent'])
    market=pd.read_parquet(selection/'inputs/market.parquet')
    assert not market.duplicated(['date','ts_code']).any()
    market['adjusted_open']=market.open*market.adj_factor
    market['adjusted_close']=market.close*market.adj_factor
    dates=pd.read_parquet(run/'features.parquet').index
    op=market.pivot(index='date',columns='ts_code',values='adjusted_open').reindex(dates)
    cl=market.pivot(index='date',columns='ts_code',values='adjusted_close').reindex(dates)
    raw_open=market.pivot(index='date',columns='ts_code',values='open').reindex(dates)
    predictions=pd.read_parquet(run/'predictions.parquet')
    models=list(cfg['families'])+['joint_utility']
    accepted=json.loads((run/'acceptance.json').read_text())['eligible']
    weekly=set(dates.to_series().groupby(dates.to_period('W')).first())
    rows=[];audits=[]
    for case in cash_cfg['cases']:
        if case['label']!='rank' or case['cost_bps']!=0:continue
        folder=cash/'accounts'/case['name']
        holdings=pd.read_parquet(folder/'holdings.parquet')
        daily=pd.read_parquet(folder/'daily.parquet')
        for date,held in holdings.loc[holdings.date.isin(weekly)].groupby('date'):
            i=dates.get_loc(date)
            row=dict(account=case['name'],cohort=case['cohort'],offset=case['offset'],date=date,
                stocks='|'.join(held.ts_code),n_hold=len(held),stock_weight=daily.at[date,'stock_value']/daily.at[date,'equity'])
            if i+21>=len(dates):
                row.update(known=False,reason='label_not_mature')
            else:
                codes=held.ts_code.to_list();q=held.set_index('ts_code').shares
                entry=op.iloc[i+1].reindex(codes)
                path=cl.iloc[i+1:i+21].reindex(columns=codes)
                terminal=op.iloc[i+21].reindex(codes)
                value=raw_open.iloc[i+1].reindex(codes)*q
                known=entry.notna().all() and path.notna().all().all() and terminal.notna().all() and value.notna().all()
                if not known:
                    row.update(known=False,reason='missing_future_basket_quote',label_end=dates[i+21])
                else:
                    weights=value/value.sum()
                    curve=path.div(entry).mul(weights).sum(axis=1,min_count=len(codes))-1
                    ret=float((terminal/entry-1).mul(weights).sum())
                    row.update(known=True,reason='complete',label_end=dates[i+21],worst=float(curve.min()),
                        terminal=ret,loss_event=float(curve.min()<=-.05),cash_benefit_proxy=-ret-.004)
            rows.append(row)
    samples=pd.DataFrame(rows)
    output=run/'portfolio_transfer';output.mkdir(exist_ok=False)
    samples.to_parquet(output/'basket_labels.parquet',index=False)
    for (cohort,offset),group in samples.groupby(['cohort','offset']):
        audits.append(dict(cohort=cohort,offset=int(offset),rows=len(group),complete=int(group.known.sum()),
            future_quote_unknown=int(group.reason.eq('missing_future_basket_quote').sum()),unmatured=int(group.reason.eq('label_not_mature').sum())))
    results=[]
    for (cohort,offset),g in samples.loc[samples.known].groupby(['cohort','offset']):
        for name in models:
            current=g.loc[g.date.isin(predictions.index)&g.date.map(predictions[name].notna())]
            prob=predictions.loc[current.date,name].to_numpy()
            alarm=predictions.loc[current.date,name+'_alarm'].to_numpy(dtype=bool)
            y=current.loss_event.to_numpy()
            results.append(dict(model=name,cohort=cohort,offset=int(offset),n=len(current),alarms=int(alarm.sum()),
                eligible_from_validation=name in accepted,base_loss_rate=float(y.mean()),
                alarm_loss_rate=float(y[alarm].mean()) if alarm.any() else np.nan,
                auc=float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else np.nan,
                brier=float(brier_score_loss(y,prob)),
                cash_benefit_mean=float(current.loc[alarm,'cash_benefit_proxy'].mean()) if alarm.any() else np.nan,
                account_cash_benefit_mean=float((current.loc[alarm,'cash_benefit_proxy']*current.loc[alarm,'stock_weight']).mean()) if alarm.any() else np.nan))
    pd.DataFrame(results).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(audits).to_csv(output/'coverage.csv',index=False)
    (output/'limitations.json').write_text(json.dumps(dict(
        contract='真实历史rank持仓与股数；次日开盘市值权重冻结；20日复权价格篮子标签。',
        limitations=['不是实际卖出/买回回测','复权价格代理含分红调整，不复用为可卖股数','保留缺报价与未成熟标签，不填零','重叠历史账户与目标，不是独立市场样本','市场风险概率不能假定在个人持仓同样校准'],
        selected_only_on_validation=accepted),ensure_ascii=False,indent=2))
    print(pd.DataFrame(audits).to_string(index=False))
    print(pd.DataFrame(results).groupby('model').agg(auc=('auc','median'),cash_benefit=('cash_benefit_mean','median'),alarms=('alarms','sum')).to_string())


if __name__=='__main__':main()
