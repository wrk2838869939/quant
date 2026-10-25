"""同首次持有对照、恢复交易重复事件去重及固定案例逐股净利差。"""
from pathlib import Path
import sys
import pandas as pd
from research_path_mechanisms import ROOT,verify,save,digest
run=Path(sys.argv[1]).resolve();cfg,m=verify(run);parent=ROOT/cfg['parent'];out=run/'diagnostics';out.mkdir(exist_ok=False)
s=pd.read_csv(run/'analysis/summary.csv');old=pd.read_csv(parent/'analysis/summary.csv');rows=[]
for item in s.itertuples():
    ref=old[old.window.eq(item.window)&old.schedule.eq(item.schedule)&old.policy.eq('initial_buyhold')].iloc[0]
    rows.append(dict(name=item.name,window=item.window,schedule=item.schedule,policy=item.policy,total=item.total,initial_hold_total=ref.total,active_delta=item.total-ref.total))
pd.DataFrame(rows).to_csv(out/'vs_initial_buyhold.csv',index=False)
a=pd.read_csv(run/'analysis/actual_recovery_buys.csv');unique=a.drop_duplicates(['date','ts_code']);unique.to_csv(out/'unique_recovery_events.csv',index=False)
# 同日同股的事后行情标签必须相同，不把重复账户记为独立观察。
for key in ['postfill_close_return_20','postfill_close_return_60']:assert a.groupby(['date','ts_code'])[key].nunique().le(1).all()
frames=[]
for policy in ['baseline','duration','wait','recover','lot_buffer']:
    f=run/'execution'/f'continued_rolling_00_{policy}'
    frames.append(pd.read_csv(f/'asset_pnl.csv').set_index('ts_code').pnl_yuan.rename(policy))
z=pd.concat(frames,axis=1).fillna(0.);z['recover_minus_wait']=z['recover']-z['wait'];z.sort_values('recover_minus_wait').to_csv(out/'rolling00_stock_profit.csv')
save(out/'completion.json',dict(status='complete',source_sha256=digest(__file__),artifacts={p.name:digest(p) for p in out.iterdir()}))
print('Compared',len(rows),'to original buyhold;',len(a),'recovery buys map to',len(unique),'date-stock observations')
