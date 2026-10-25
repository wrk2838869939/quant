"""用历史夹具演练CLI闭环：部分成交、跨进程接续、对账及真实卖出后补位。"""
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from research_risk_budget import digest,save

run=Path(sys.argv[1]).resolve()
out=run/'operator_demo';out.mkdir()
db=out/'account.sqlite'
def load(path):return json.loads(Path(path).read_text())
def invoke(*args):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/shadow_execution.py'),'--db',str(db),*map(str,args)],
        capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    return json.loads(result.stdout)
snap=load(run/'example_snapshot.json')
bundle=load(run/'example_rebuilt_bundle.json')
calendar=load(run/'example_calendar.json')
calendar['sessions'].append('2026-09-03')
save(out/'calendar.json',calendar)
invoke('snapshot','--input',run/'example_snapshot.json')
plan=invoke('plan','--bundle',run/'example_rebuilt_bundle.json','--calendar',out/'calendar.json','--output',out/'plan_0901.json')
assert len(plan['orders'])==1 and plan['orders'][0]['shares']==700
again=invoke('plan','--bundle',run/'example_rebuilt_bundle.json','--calendar',out/'calendar.json','--output',out/'plan_0901.json')
assert again==plan
order=plan['orders'][0]
parent=ROOT/'outputs/risk_budget/20260913T160301_084080Z/execution/cold_calendar_01_full_risk'
inp=ROOT/'outputs/rolling_review/20260913T141120_325588Z/inputs'
limits=pd.read_parquet(inp/'limits.parquet').set_index(['date','ts_code'])
limit=limits.loc[(pd.Timestamp('2026-09-02'),order['ts_code'])]
market=dict(timestamp='2026-09-02T09:16:00+08:00',source='HISTORICAL_TEST_FIXTURE_NOT_LIVE',
    account_id=plan['account_id'],positions=plan['snapshot_positions'],open_orders=[],
    session='opening_auction',corporate_actions_checked=True,available_cash=snap['cash'],
    securities={order['ts_code']:dict(down_limit=float(limit.down_limit),up_limit=float(limit.up_limit),
        suspended=False,corporate_action=False,broker_accepts_limit=True,buy_allowed=True,
        account_sellable=snap['positions'][order['ts_code']]['sellable_next_day'])})
save(out/'preflight_fixture.json',dict(plan_id=plan['plan_id'],market=market))
invoke('preflight','--input',out/'preflight_fixture.json')
trades=pd.read_csv(parent/'trades.csv')
trade=trades[trades.date.eq('2026-09-02')&trades.ts_code.eq(order['ts_code'])].iloc[0]
fee1=round(float(trade.fee)*300/700,2)
for index,shares,fee in [(1,300,fee1),(2,400,float(trade.fee)-fee1)]:
    fill=dict(execution_id=f'FIXTURE-{index}',order_id=order['order_id'],date='2026-09-02',shares=shares,
        price=float(trade.price),fee=fee,evidence='SYNTHETIC_PARTIAL_SPLIT_OF_HISTORICAL_BACKTEST')
    save(out/f'fill_{index}.json',fill)
    status=invoke('fill','--input',out/f'fill_{index}.json')
    assert status['status']==('partial' if index==1 else 'filled')
    assert invoke('fill','--input',out/f'fill_{index}.json')['status']=='duplicate'
d=pd.read_csv(parent/'daily.csv',index_col='date')
h=pd.read_csv(parent/'holdings.csv')
row=d.loc['2026-09-02']
snap2=dict(account_id=snap['account_id'],asof='2026-09-02',source='HISTORICAL_FIXTURE_NOT_BROKER',
    cash=float(row.cash),equity=float(row.equity),receivables=float(row.receivables),frozen_cash=0.,open_orders=[],
    positions={r.ts_code:dict(shares=int(r.shares),sellable_next_day=int(r.shares),close=float(r.price))
               for r in h[h.date.eq('2026-09-02')].itertuples()})
save(out/'snapshot_0902.json',snap2)
assert invoke('snapshot','--input',out/'snapshot_0902.json')['status']=='reconciled'
quotes=pd.read_parquet(inp/'market.parquet');quotes=quotes[quotes.date.eq('2026-09-02')].set_index('ts_code')
bundle.update(asof='2026-09-02',is_review=False,provenance={'historical_daily_fixture':True})
bundle['quotes']={c:dict(date='2026-09-02',close=float(quotes.at[c,'close']),volume=float(quotes.at[c,'vol_share']))
                  for c in bundle['quotes'] if c in quotes.index}
save(out/'bundle_0902_fixture.json',bundle)
next_plan=invoke('plan','--bundle',out/'bundle_0902_fixture.json','--calendar',out/'calendar.json','--output',out/'plan_0902.json')
buy=next_plan['orders'][0]
assert len(next_plan['orders'])==1 and buy['ts_code']=='601288.SH' and buy['shares']==2800 and buy['side']=='buy'
assert buy['reason']=='sale_cash_supplement'
save(out/'status.json',invoke('status'))
save(out/'completion.json',dict(status='passed',separate_process_cli=True,duplicate_plan_exact=True,
    partial_fills=[300,400],duplicate_fills_ignored=True,snapshot_reconciled=True,
    supplement_after_actual_sale=dict(ts_code=buy['ts_code'],shares=buy['shares']),
    live_submission=False,fixture_only=True,source_sha256=digest(__file__)))
print('CLI lifecycle passed; synthetic historical fixture only')
