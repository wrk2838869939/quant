"""独立检查新因果订单路径的数据、行情撮合和逐笔权益，不导入策略实现。"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/root/autodl-tmp/alpha_lab')
DEFAULT=ROOT/'outputs/causal_execution/20260913T130810_374654Z'
OUT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):return ((a==b)|(a.isna()&b.isna())).to_numpy().all()
def near(a,b):return bool(np.isclose(a,b,atol=1e-7,rtol=1e-12))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=DEFAULT);a=ap.parse_args();run=a.run;manifest=json.loads((run/'manifest.json').read_text());inp=run/'inputs'
 assert all((run/'execution'/c['name']/'completion.json').exists() for c in manifest['cases']),'等待全部20组完成后运行'
 result=dict(status='running',run=str(run),manifest_sha256=sha(run/'manifest.json'),source_hashes={},checks=[],cases=[],scope='Local raw source consistency and daily-bar frozen-order fill conditions; not an independent exchange price certification or auction queue/fill-volume certification.')
 fills=[];marks=[];dividends=[]
 try:
  for name,h in manifest['inputs_sha256'].items():assert sha(inp/name)==h,('input_hash',name)
  market=pd.read_parquet(inp/'market.parquet');market=market.set_index(['date','ts_code']);limits=pd.read_parquet(inp/'limits.parquet').set_index(['date','ts_code']);codes=sorted(market.index.get_level_values('ts_code').unique())
  source_errors=[]
  for date,g in market.groupby(level='date'):
   day=date.strftime('%Y%m%d');g=g.droplevel('date')
   raw=pd.read_parquet(ROOT/f'data/raw/by_date/daily/{day}.parquet').set_index('ts_code').reindex(g.index);raw['vol_share']=raw.vol*100
   for col in ['open','high','low','close','pre_close','vol_share']:
    assert same(g[col],raw[col]),('raw_daily',date,col)
  result['checks'].append(dict(check='raw_daily_all_execution_universe_exact',rows=len(market),fields=['open','high','low','close','pre_close','vol_share']))
  cache=ROOT/'data/raw/execution_audit/20220101_20260908'
  rawlimits=pd.concat([pd.read_parquet(cache/'stk_limit'/f'{c}.parquet') for c in codes],ignore_index=True);rawlimits['date']=pd.to_datetime(rawlimits.trade_date);rawlimits=rawlimits.set_index(['date','ts_code'])
  assert same(limits[['up_limit','down_limit']].sort_index().astype(float),rawlimits[['up_limit','down_limit']].sort_index().astype(float))
  result['checks'].append(dict(check='official_limit_cache_exact',rows=len(limits)))
  rawactions=pd.read_parquet(inp/'raw_actions.parquet');sourceactions=pd.concat([pd.read_parquet(cache/'dividend'/f'{c}.parquet') for c in codes],ignore_index=True)
  assert same(rawactions,sourceactions)
  implemented=rawactions[rawactions.div_proc.eq('实施')].drop_duplicates().copy()
  for col in ['imp_ann_date','record_date','ex_date','pay_date','div_listdate']:implemented[col]=pd.to_datetime(implemented[col],errors='coerce')
  assert implemented.imp_ann_date.notna().all() and implemented.imp_ann_date.le(implemented.record_date).all()
  normalized=pd.read_parquet(inp/'normalized_actions.parquet').set_index(['ex_date','ts_code'])
  for case in manifest['cases']:
   name=case['name'];folder=run/'execution'/name;completion=json.loads((folder/'completion.json').read_text())
   assert completion['status']=='complete'
   for rel,h in completion['artifacts'].items():assert sha(folder/rel)==h,('case_hash',name,rel)
   cfg=json.loads((folder/'portfolio_config.json').read_text())['portfolio'];tr=pd.read_csv(folder/'trades.csv');hold=pd.read_csv(folder/'holdings.csv');ev=pd.read_csv(folder/'events.csv');daily=pd.read_csv(folder/'daily.csv')
   for frame in [tr,hold,ev,daily]:frame['date']=pd.to_datetime(frame.date)
   for col in ['decision_date','signal_date','reference_date']:tr[col]=pd.to_datetime(tr[col]);ev[col]=pd.to_datetime(ev[col],errors='coerce')
   planned=ev[ev.kind.eq('order_planned')].set_index('order_id');assert planned.index.is_unique
   fill_count=0;limit_count=0
   for t in tr.itertuples():
    m=market.loc[(t.date,t.ts_code)];l=limits.loc[(t.date,t.ts_code)];p=planned.loc[t.order_id]
    assert t.decision_date<t.date and t.reference_date==t.decision_date and t.signal_date<=t.decision_date
    assert near(t.reference_price,market.loc[(t.reference_date,t.ts_code),'close'])
    assert t.shares==p.shares and near(t.limit_price,p.limit_price) and t.side==p.side and t.ts_code==p.ts_code
    assert t.date==pd.Timestamp(p.execution_date),('DAY_date',name,t.order_id)
    assert l.down_limit-1e-8<=t.limit_price<=l.up_limit+1e-8
    assert l.down_limit-1e-8<=t.price<=l.up_limit+1e-8
    assert m.low-1e-8<=t.price<=m.high+1e-8
    if t.execution_kind=='open':
     assert near(t.price,m.open)
     if t.side=='buy':assert m.open<=t.limit_price+1e-8 and m.open<l.up_limit-cfg['price_tick']/2
     else:assert m.open>=t.limit_price-1e-8 and m.open>l.down_limit+cfg['price_tick']/2
    else:
     assert t.execution_kind=='strict_intraday_cross' and near(t.price,t.limit_price)
     if t.side=='buy':assert m.low<t.limit_price-1e-8
     else:assert m.high>t.limit_price+1e-8
     limit_count+=1
    assert near(t.gross,t.shares*t.price)
    expected_fee=max(cfg['min_fee'],t.gross*cfg['buy_cost' if t.side=='buy' else 'sell_cost'])
    if t.side=='sell':expected_fee=min(expected_fee,t.gross)
    assert near(t.fee,expected_fee)
    assert (t.date,t.ts_code) not in normalized.index,('ex_date_fill',name,t.order_id)
    fills.append(dict(case=name,date=str(t.date.date()),ts_code=t.ts_code,order_id=t.order_id,side=t.side,shares=t.shares,price=t.price,limit_price=t.limit_price,raw_open=m.open,raw_low=m.low,raw_high=m.high,down_limit=l.down_limit,up_limit=l.up_limit,execution_kind=t.execution_kind,passed=True));fill_count+=1
   # 原始收盘直接估值。若新路径包含停牌未知估值，先失败要求独立状态分析，不沿用策略兜底。
   assert not hold.duplicated(['date','ts_code']).any()
   joined=hold.merge(market[['close']].reset_index(),on=['date','ts_code'],how='left',validate='one_to_one')
   assert joined.close.notna().all() and joined.close.gt(0).all(),('unquoted_hold_needs_separate_audit',name)
   assert np.allclose(joined.price,joined.close,atol=1e-10,rtol=0)
   assert np.allclose(joined.value,joined.shares*joined.close,atol=1e-7,rtol=1e-12)
   value=joined.groupby('date').value.sum().reindex(daily.date,fill_value=0)
   assert np.allclose(value,daily.stock_value,atol=1e-7,rtol=1e-12)
   marks.append(dict(case=name,holding_rows=len(hold),raw_close_values_exact=True))
   case_div=[]
   for e in ev[ev.kind.eq('dividend_receivable')].itertuples():
    candidates=implemented[implemented.ts_code.eq(e.ts_code)&implemented.ex_date.eq(e.date)]
    # 未经认可的不同组件不能盲求和，所有实际新持仓均须唯一原始权益。
    assert len(candidates)==1,('raw_action_not_unique',name,e.ts_code,e.date,len(candidates))
    action=candidates.iloc[0];quantity=hold.loc[hold.date.eq(action.record_date)&hold.ts_code.eq(e.ts_code),'shares'].sum()
    assert quantity>0 and action.imp_ann_date<=action.record_date<action.ex_date<=action.pay_date
    expected=quantity*action.cash_div_tax*(1-cfg['cash_dividend_haircut'])
    assert near(e.amount,expected)
    record=dict(case=name,ts_code=e.ts_code,ex_date=str(e.date.date()),record_date=str(action.record_date.date()),pay_date=str(action.pay_date.date()),record_shares=int(quantity),cash_div_tax=float(action.cash_div_tax),haircut=cfg['cash_dividend_haircut'],expected=float(expected),ledger_amount=float(e.amount),passed=True)
    dividends.append(record);case_div.append(record)
   # 按股/支付日独立聚合应付股息，排除除息与支付重复入现金。
   expected_paid={}
   for row in case_div:
    key=(pd.Timestamp(row['pay_date']),row['ts_code'])
    if key[0]<=daily.date.max():expected_paid[key]=expected_paid.get(key,0)+row['expected']
   got=ev[ev.kind.eq('dividend_paid')].groupby(['date','ts_code']).amount.sum().to_dict()
   assert set(got)==set(expected_paid) and all(near(got[k],v) for k,v in expected_paid.items())
   daily_index=pd.DatetimeIndex(daily.date);cashflow=pd.Series(0.,index=daily_index)
   for t in tr.itertuples():cashflow.loc[t.date]+=(-t.gross-t.fee if t.side=='buy' else t.gross-t.fee)
   for (date,code),amount in expected_paid.items():cashflow.loc[date]+=amount
   writeoffs=ev[ev.kind.eq('delist_writeoff')]
   for e in writeoffs.itertuples():cashflow.loc[e.date]+=e.cash_recovery
   expected_cash=case['capital']+cashflow.cumsum()
   assert np.allclose(expected_cash,daily.cash,atol=1e-7,rtol=1e-12)
   receivables=pd.Series(0.,index=daily_index)
   for row in case_div:receivables.loc[(receivables.index>=pd.Timestamp(row['ex_date']))&(receivables.index<pd.Timestamp(row['pay_date']))]+=row['expected']
   assert np.allclose(receivables,daily.receivables,atol=1e-7,rtol=1e-12)
   assert np.allclose(expected_cash.to_numpy()+value.to_numpy()+receivables.to_numpy(),daily.equity,atol=1e-7,rtol=1e-12)
   result['cases'].append(dict(name=name,trades=fill_count,open_fills=fill_count-limit_count,strict_day_cross_fills=limit_count,holding_rows=len(hold),dividend_events=len(case_div),cash_equity_reconciled=True))
   print(name,'PASS',fill_count,'fills',len(hold),'marks',len(case_div),'dividends',flush=True)
  result['status']='complete';result['total_cases']=len(result['cases']);result['total_fills']=len(fills);result['total_holding_rows']=sum(x['holding_rows'] for x in marks);result['total_dividend_events']=len(dividends);result['source_hashes']={str(inp/n):sha(inp/n) for n in ['market.parquet','limits.parquet','raw_actions.parquet','normalized_actions.parquet']}
 except Exception as exc:
  result['status']='failed';result['error']=repr(exc);raise
 finally:
  result['verifier_sha256']=sha(__file__)
  (OUT/'causal_data_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
  pd.DataFrame(fills).to_csv(OUT/'actual_fill_source_checks.csv',index=False);pd.DataFrame(marks).to_csv(OUT/'holding_mark_checks.csv',index=False);pd.DataFrame(dividends).to_csv(OUT/'dividend_source_checks.csv',index=False)
if __name__=='__main__':main()
