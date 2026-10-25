"""补核所有实际锁单只使用已知量能，并列出真实触发分支。"""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
R=Path('/root/autodl-tmp/alpha_lab');RUN=R/'outputs/causal_execution/20260913T130810_374654Z';OUT=Path(__file__).resolve().parent/RUN.name;OUT.mkdir(exist_ok=True)
manifest=json.loads((RUN/'manifest.json').read_text());market=pd.read_parquet(RUN/'inputs/market.parquet').set_index(['date','ts_code'])
rows=[];cancel=[]
for case in manifest['cases']:
 folder=RUN/'execution'/case['name'];e=pd.read_csv(folder/'events.csv',low_memory=False);t=pd.read_csv(folder/'trades.csv');cfg=json.loads((folder/'portfolio_config.json').read_text())['portfolio']
 plans=e[e.kind.eq('order_planned')]
 for o in plans.itertuples():
  vol=market.loc[(pd.Timestamp(o.decision_date),o.ts_code),'vol_share'];capacity=0 if not np.isfinite(vol) else vol*cfg['max_volume_participation']
  assert o.shares>0 and o.shares==int(o.shares) and o.shares<=capacity+1e-8,(case['name'],o.order_id,'not bounded by prior known volume')
 endings=e[e.kind.isin(['order_cancelled','order_deferred'])]
 if len(endings):
  for (kind,reason),g in endings.groupby(['kind','cancellation_reason']):cancel.append(dict(name=case['name'],kind=kind,reason=reason,count=len(g)))
 rows.append(dict(name=case['name'],planned_orders=len(plans),fills=len(t),
  open_fills=int(t.execution_kind.eq('open').sum()),intraday_fills=int(t.execution_kind.eq('strict_intraday_cross').sum()),
  cancelled_orders=int(e.kind.eq('order_cancelled').sum()),deferred_orders=int(e.kind.eq('order_deferred').sum()),
  cash_dividend_receipts=int(e.kind.eq('dividend_paid').sum()),bonus_events=int(e.kind.eq('bonus_shares').sum()),
  delist_events=int(e.kind.eq('delist_writeoff').sum()),st_exit_requests=int((e.kind.eq('hold_exit_requested')&e.reason.eq('st')).sum()),
  locked_volume_deferrals=int(e.kind.eq('sell_planning_deferred').sum()),
  known_next_day_ineligible=int(e.kind.eq('known_next_day_ineligible').sum())))
summary=pd.DataFrame(rows);summary.to_csv(OUT/'actual_branch_coverage.csv',index=False);pd.DataFrame(cancel).to_csv(OUT/'actual_cancellation_reasons.csv',index=False)
report={'cases':len(rows),'totals':summary.drop(columns='name').sum().astype(int).to_dict(),
 'cancellation_reasons':pd.DataFrame(cancel).groupby(['kind','reason'])['count'].sum().reset_index().to_dict('records'),
 'all_planned_quantities_bounded_by_decision_day_volume':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'actual_branch_coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
