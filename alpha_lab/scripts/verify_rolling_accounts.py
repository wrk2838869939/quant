"""截断实际历史重跑两个非月初账户，核验持仓/成交/资产前缀。"""
from pathlib import Path
import sys
from copy import deepcopy
import json
import pandas as pd
import yaml
from research_rolling_review import ROOT,verify_manifest,run_backtest,digest,save
from audit_monthly_portfolio import reconcile
run=Path(sys.argv[1]);m=verify_manifest(run);cfg=yaml.safe_load((run/'config.yaml').read_text());inp=run/'inputs'
stop=pd.Timestamp('2025-06-30');rows=[]
for name in ['continued_calendar_15_hybrid','cold_rolling_00_hybrid']:
    case=next(c for c in m['cases'] if c['name']==name)
    frames=[pd.read_parquet(inp/f'{key}.parquet') for key in ['market','basics','st','limits','normalized_actions']]
    source=run/'schedules'/case['schedule']
    sig,mu,hold=[pd.read_parquet(source/f'{key}.parquet').loc[:stop] for key in ['signals','forecasts','hold']]
    needed=sorted(set().union(*(set(row.dropna().sort_values(ascending=False,kind='stable').head(20).index) for _,row in sig.iterrows())))
    frames=[f[f.ts_code.isin(needed)].copy() for f in frames]
    frames[0]=frames[0][frames[0].date<=stop];frames[3]=frames[3][frames[3].date<=stop]
    frames[4]=frames[4][frames[4].known_date<=stop]
    portfolio=yaml.safe_load((inp/'execution_config.yaml').read_text())['portfolio'];portfolio.update(capital=cfg['capital'],top_n=5)
    retail=yaml.safe_load((inp/'symmetric_phase_research.yaml').read_text())['retail']
    ac=yaml.safe_load((inp/'corporate_action_components.yaml').read_text())
    causal=yaml.safe_load((ROOT/cfg['parent']/'config.yaml').read_text())['causal']
    result=run_backtest(sig,*frames,case['start'],stop,deepcopy(portfolio),ac.get('rights_issues',[]),retail_forecasts=mu,retail_config=retail,hold_allowed=hold,causal_config=causal)
    checks=reconcile(result,cfg['capital'])
    folder=run/'execution'/name
    daily=pd.read_csv(folder/'daily.csv',index_col='date',parse_dates=True).loc[:stop]
    # CSV浮点序列化再读取保持与封存比较同口径；不放宽精度。
    from io import StringIO
    actual=pd.read_csv(StringIO(result.daily.to_csv()),index_col='date',parse_dates=True)
    pd.testing.assert_frame_equal(actual,daily,check_exact=True)
    for key,frame in [('trades',result.trades),('holdings',result.holdings)]:
        old=pd.read_csv(folder/f'{key}.csv');old=old[pd.to_datetime(old.date)<=stop].reset_index(drop=True)
        actual=pd.read_csv(StringIO(frame.to_csv(index=False)))
        pd.testing.assert_frame_equal(actual,old,check_exact=True)
    rows.append(dict(name=name,stop=stop,daily_exact=True,trades_exact=True,holdings_exact=True,checks=checks))
    print(name,'history-prefix exact',flush=True)
save(run/'account_prefix_verification.json',dict(status='passed',cases=rows,source_sha256=digest(__file__)))
