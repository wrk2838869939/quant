"""从封存账本追踪复核日期的状态、名额与利润路径差异；不是机制独立消融。"""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,shutil
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conditional_hold import build_conditional_holds

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
run=ROOT/'outputs/rolling_review/20260913T141120_325588Z'
receipt=json.loads((run/'completion.json').read_text())
for name,h in receipt['artifact_sha256'].items():assert digest(run/name)==h,name
out=ROOT/'outputs/review_date_diagnosis'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
names=['continued_calendar_01_hybrid','continued_rolling_00_hybrid'];frames=[]
for name in names:
    frames.append(pd.read_csv(run/'execution'/name/'asset_pnl.csv').set_index('ts_code').pnl_yuan.rename(name))
pnl=pd.concat(frames,axis=1).fillna(0.);pnl['monthly_minus_rolling']=pnl.iloc[:,0]-pnl.iloc[:,1]
summaries=[json.loads((run/'execution'/name/'summary.json').read_text()) for name in names]
assert abs(pnl.monthly_minus_rolling.sum()-(summaries[0]['total']-summaries[1]['total'])*50000)<1e-6
pnl.sort_values('monthly_minus_rolling',ascending=False).to_csv(out/'stock_profit_differences.csv')
monthly=pd.read_csv(run/'analysis/monthly_asset_pnl.csv');z=monthly[monthly.name.isin(names)].groupby(['month','name']).pnl_yuan.sum().unstack().fillna(0);z['monthly_minus_rolling']=z[names[0]]-z[names[1]];z.to_csv(out/'monthly_profit_differences.csv')
cols=['601811.SH','601018.SH','600828.SH'];prices=pd.read_parquet(run/'inputs/daily_close.parquet')[cols]
trend=pd.read_parquet(run/'inputs/daily_trend.parquet')[cols];known=pd.read_parquet(run/'inputs/daily_observable.parquet')[cols]
spec=yaml.safe_load((run/'sources/configs/conditional_hold_research.yaml').read_text());records=[]
for schedule in ['calendar_01','rolling_00']:
    hold=pd.read_parquet(run/'schedules'/schedule/'hold.parquet');variants,e=build_conditional_holds(prices,trend,known,hold.index,spec)
    pd.testing.assert_frame_equal(variants['hybrid'],hold[cols],check_exact=True)
    for code in cols:
        f=pd.DataFrame({k:e[k][code] for k in ['monthly_failure_streak','daily_weak_count','weak_regime','long_return','long_ma_gap']})
        f['ret60']=(prices/prices.shift(60)-1).loc[hold.index,code];f['hold_allowed']=variants['hybrid'][code];f['ts_code']=code;f['schedule']=schedule;records.append(f.reset_index())
pd.concat(records).to_csv(out/'review_states.csv',index=False)
for name in names:
    e=pd.read_csv(run/'execution'/name/'events.csv');e[e.kind.isin(['hold_exit_requested','hold_risk_cap_reduction','entry_planning_attempt'])|e.ts_code.isin(['601811.SH','601138.SH','601398.SH'])].to_csv(out/f'{name}_events.csv',index=False)
    shutil.copy2(run/'execution'/name/'trades.csv',out/f'{name}_trades.csv')
shutil.copy2(__file__,out/Path(__file__).name)
(out/'completion.json').write_text(json.dumps(dict(status='complete',parent=str(run),parent_artifacts_verified=len(receipt['artifact_sha256']),profit_difference_yuan=pnl.monthly_minus_rolling.sum(),three_largest_stock_differences=pnl.monthly_minus_rolling.nlargest(3).sum(),interpretation='accounting attribution and observed state transitions; not isolated causal shares',artifacts={p.name:digest(p) for p in out.iterdir()}),ensure_ascii=False,indent=2))
print(out,flush=True)
