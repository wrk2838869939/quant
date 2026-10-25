"""主审复用上轮独立指标公式复算；本次不宣称新的独立智能体审查。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

import argparse
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
run=p.parse_args().run
(run/'metrics_audit').mkdir(exist_ok=True)
protocol=json.loads((run/'protocol.json').read_text());parent=Path(protocol['cash_parent'])
summary=pd.read_csv(run/'summary.csv').set_index('name');rows=[];cash_errors=[]
def read(folder):
 d=pd.read_parquet(folder/'daily.parquet');t=pd.read_parquet(folder/'trades.parquet')
 eq=d.equity.to_numpy();ret=np.r_[eq[0]/80000-1,eq[1:]/eq[:-1]-1]
 calc=dict(total=eq[-1]/80000-1,ann=(eq[-1]/80000)**(244/len(eq))-1,
  mdd=np.min(eq/np.maximum.accumulate(np.r_[80000,eq])[1:]-1),
  sharpe=np.mean(ret)/np.std(ret,ddof=1)*np.sqrt(244),n_trades=len(t),total_fees=t.fee.sum(),mean_invested=(d.stock_value/d.equity).mean())
 # 双边成交额/当日净资产累计后年化，不冒充传统单边换手。
 calc['annual_gross_nav']=sum(row.gross/d.at[row.date,'equity'] for row in t.itertuples())*244/len(d)
 cash_errors.append(float(np.max(np.abs(d.equity-d.cash-d.stock_value-d.receivables))))
 return calc,d,t
cache={}
for case in protocol['cases']:
 name=case['name'];calc,d,t=read(run/'accounts'/name)
 for k in ['total','ann','mdd','sharpe','n_trades','total_fees','mean_invested']:
  assert np.isclose(calc[k],summary.at[name,k],atol=1e-9), (name,k,calc[k],summary.at[name,k])
 if not case['restore']:continue
 rank_name=f"{case['cohort']}_{case['offset']:02d}_rank_cost{case['cost_bps']}"
 if rank_name not in cache:cache[rank_name]=read(parent/'accounts'/rank_name)
 rank,rankd,rankt=cache[rank_name];assert d.index.equals(rankd.index),(name,'mismatched dates')
 rank_summary=json.loads((parent/'accounts'/rank_name/'summary.json').read_text())
 for k in ['ann','mdd','sharpe','n_trades','total_fees']:assert np.isclose(rank[k],rank_summary[k],atol=1e-9)
 rows.append(dict(name=name,label=case['label'],cost_bps=case['cost_bps'],cohort=case['cohort'],offset=case['offset'],
  **calc,rank_ann=rank['ann'],rank_mdd=rank['mdd'],rank_sharpe=rank['sharpe'],rank_n_trades=rank['n_trades'],rank_annual_gross_nav=rank['annual_gross_nav'],
  ann_diff=calc['ann']-rank['ann'],mdd_diff=calc['mdd']-rank['mdd'],sharpe_diff=calc['sharpe']-rank['sharpe'],
  trade_fraction=calc['n_trades']/rank['n_trades']-1,gross_nav_fraction=calc['annual_gross_nav']/rank['annual_gross_nav']-1))
a=pd.DataFrame(rows);a.to_csv(run/'metrics_audit/accounts.csv',index=False)
out=[]
for (label,cost),g in a.groupby(['label','cost_bps']):
 out.append(dict(label=label,cost_bps=int(cost),n=len(g),ann_median=g.ann.median(),rank_ann_median=g.rank_ann.median(),ann_delta_median=g.ann_diff.median(),
  ann_worse=int(g.ann_diff.lt(-1e-10).sum()),mdd_median=g.mdd.median(),rank_mdd_median=g.rank_mdd.median(),mdd_delta_median=g.mdd_diff.median(),mdd_improved=int(g.mdd_diff.gt(1e-10).sum()),
  sharpe_median=g.sharpe.median(),rank_sharpe_median=g.rank_sharpe.median(),sharpe_delta_median=g.sharpe_diff.median(),sharpe_worse=int(g.sharpe_diff.lt(-1e-10).sum()),
  trade_fraction_median=g.trade_fraction.median(),annual_gross_nav_median=g.annual_gross_nav.median(),rank_annual_gross_nav_median=g.rank_annual_gross_nav.median(),gross_nav_fraction_median=g.gross_nav_fraction.median()))
result=dict(summary_rows_verified=len(summary),rank_accounts_verified=len(cache),max_cash_identity_error=max(cash_errors),independent_oos=protocol['independent_oos'],matched_results=out)
(run/'metrics_audit/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
