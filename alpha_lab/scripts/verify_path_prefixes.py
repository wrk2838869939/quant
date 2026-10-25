"""四个机制的真实历史前缀重放；未来行情与未来复核不许改写旧订单和资产。"""
from pathlib import Path
from copy import deepcopy
from io import StringIO
import sys
import pandas as pd
import yaml
from research_path_mechanisms import ROOT,verify,run_backtest,save,digest,reconcile
run=Path(sys.argv[1]).resolve();cfg,m=verify(run);parent=ROOT/cfg['parent'];inp=parent/'inputs';stop=pd.Timestamp('2025-06-30');records=[]
frames=[pd.read_parquet(inp/f'{n}.parquet') for n in ['market','basics','st','limits','normalized_actions']]
portfolio=yaml.safe_load((inp/'execution_config.yaml').read_text())['portfolio'];portfolio.update(capital=cfg['capital'],top_n=5)
retail=yaml.safe_load((inp/'symmetric_phase_research.yaml').read_text())['retail'];ac=yaml.safe_load((inp/'corporate_action_components.yaml').read_text())
cc=yaml.safe_load((ROOT/'outputs/causal_execution/20260913T130810_374654Z/config.yaml').read_text())['causal']
for policy in ['duration','wait','recover','lot_buffer']:
    case=next(c for c in m['cases'] if c['window']=='continued' and c['schedule']=='rolling_00' and c['policy']==policy)
    source=parent/'schedules'/case['schedule'];sig,mu,hold=[pd.read_parquet(source/f'{n}.parquet').loc[:stop] for n in ['signals','forecasts','hold']]
    rule=cfg['policies'][policy]
    if rule['hold']=='duration':hold=pd.read_parquet(run/'holds'/f"{case['schedule']}.parquet").loc[:stop]
    needed=sorted(set().union(*(set(row.dropna().sort_values(ascending=False,kind='stable').head(20).index) for _,row in sig.iterrows())))
    local=[f[f.ts_code.isin(needed)].copy() for f in frames];local[0]=local[0][local[0].date<=stop];local[3]=local[3][local[3].date<=stop];local[4]=local[4][local[4].known_date<=stop]
    result=run_backtest(sig,*local,case['start'],stop,deepcopy(portfolio),ac.get('rights_issues',[]),retail_forecasts=mu,retail_config=retail,hold_allowed=hold,causal_config=cc,path_config={k:v for k,v in rule.items() if k!='hold'})
    checks=reconcile(result,cfg['capital']);folder=run/'execution'/case['name']
    for key,frame in [('daily',result.daily),('trades',result.trades),('holdings',result.holdings)]:
        actual=pd.read_csv(StringIO(frame.to_csv(index=key=='daily')))
        old=pd.read_csv(folder/f'{key}.csv');old=old[pd.to_datetime(old.date)<=stop].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,old,check_exact=True)
    # 末日不规划下一天；只比较截断日前的所有订单，包含等待和恢复改变的计划。
    e=pd.read_csv(folder/'events.csv');old=e[e.kind.eq('order_planned')&pd.to_datetime(e.date).lt(stop)]
    a=result.events[result.events.kind.eq('order_planned')&result.events.date.lt(stop)]
    fields=['date','ts_code','order_id','side','shares','limit_price','reserved_cash','reason']
    actual=pd.read_csv(StringIO(a[fields].to_csv(index=False)))
    pd.testing.assert_frame_equal(actual,old[fields].reset_index(drop=True),check_exact=True)
    records.append(dict(name=case['name'],stop=stop,daily_trades_holdings_orders_exact=True,checks=checks));print(case['name'],'prefix exact',flush=True)
save(run/'account_prefix_verification.json',dict(status='passed',checks=records,source_sha256=digest(__file__)))
