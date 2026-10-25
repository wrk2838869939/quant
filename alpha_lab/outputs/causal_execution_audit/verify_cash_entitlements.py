"""直接以登记日股份与行动支付日期重建实际现金/应收款，拒绝未来分红预算。"""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
R=Path('/root/autodl-tmp/alpha_lab');RUN=R/'outputs/causal_execution/20260913T130810_374654Z';OUT=Path(__file__).resolve().parent/RUN.name
manifest=json.loads((RUN/'manifest.json').read_text());actions=pd.read_parquet(RUN/'inputs/normalized_actions.parquet');rows=[]
for case in manifest['cases']:
 f=RUN/'execution'/case['name'];d=pd.read_csv(f/'daily.csv',parse_dates=['date']).set_index('date');t=pd.read_csv(f/'trades.csv',parse_dates=['date']);h=pd.read_csv(f/'holdings.csv',parse_dates=['date']).set_index(['date','ts_code']);events=pd.read_csv(f/'events.csv',low_memory=False);cfg=json.loads((f/'portfolio_config.json').read_text())['portfolio']
 assert not events.kind.eq('delist_writeoff').any(),'This supplemental independent cash audit needs explicit delist recovery formula if new cases contain delisting'
 cash=cfg['capital'];create={};pay={};claims=0.;count=0
 for a in actions.itertuples():
  key=(a.record_date,a.ts_code)
  if key not in h.index:continue
  amount=float(h.loc[key,'shares'])*a.cash_div_tax*(1-cfg['cash_dividend_haircut'])
  if amount:
   assert a.known_date<=a.record_date<a.ex_date<=a.pay_date
   create[a.ex_date]=create.get(a.ex_date,0.)+amount;pay[a.pay_date]=pay.get(a.pay_date,0.)+amount
   if a.pay_date<=d.index[-1]:count+=1
 tg={day:list(g.itertuples()) for day,g in t.groupby('date',sort=False)};errcash=0.;errclaims=0.;paid=0.
 for day,row in d.iterrows():
  claims+=create.get(day,0.)
  # Dates are exchange-calendar effective records; pay may fall nontrading day, release by next account day.
  for daypay in [x for x in pay if x<=day]:
   amount=pay.pop(daypay);cash+=amount;claims-=amount;paid+=amount
  for tr in tg.get(day,[]):
   cash+=(-tr.gross if tr.side=='buy' else tr.gross)-tr.fee
   assert abs(cash-tr.cash_after)<1e-6,(case['name'],tr.order_id,'cash after trade uses unearned money')
  errcash=max(errcash,abs(cash-row.cash));errclaims=max(errclaims,abs(claims-row.receivables))
  assert abs(cash-row.cash)<1e-6 and abs(claims-row.receivables)<1e-6,(case['name'],day,'cash/receivable mismatch')
 rows.append(dict(name=case['name'],status='passed',cash_dividend_receipts=count,total_net_cash_dividends=paid,max_cash_error=errcash,max_receivable_error=errclaims))
report=dict(status='complete',cases=rows,case_count=len(rows),total_dividend_receipts=sum(r['cash_dividend_receipts'] for r in rows),
 max_cash_error=max(r['max_cash_error'] for r in rows),max_receivable_error=max(r['max_receivable_error'] for r in rows),
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'independent_cash_entitlements.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
