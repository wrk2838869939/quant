"""新计划器与冻结研究订单逐日对照，并重建全A当日信号；不视为真实成交。"""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from shadow_planner import make_plan, empty_memory, fingerprint
from shadow_inputs import build_bundle
from research_risk_budget import digest, save

out = ROOT/'outputs/shadow_execution'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
out.mkdir(parents=True)
print(out, flush=True)
cfg = yaml.safe_load((ROOT/'configs/shadow_execution.yaml').read_text())
parent = ROOT/'outputs/risk_budget/20260913T160301_084080Z'
rolling = ROOT/'outputs/rolling_review/20260913T141120_325588Z'
hash_counts = {}
for folder in [parent,rolling]:
    receipt = json.loads((folder/'completion.json').read_text())
    for name, expected in receipt['artifact_sha256'].items():
        assert digest(folder/name)==expected
    hash_counts[str(folder.relative_to(ROOT))]=len(receipt['artifact_sha256'])
inp = rolling/'inputs'
market = pd.read_parquet(inp/'market.parquet')
by_date = market.set_index(['date','ts_code'])
basics = pd.read_parquet(inp/'basics.parquet')
st = pd.read_parquet(inp/'st.parquet')
actions = pd.read_parquet(inp/'normalized_actions.parquet')
sig, mu, hold = [pd.read_parquet(rolling/'schedules/calendar_01'/f'{n}.parquet') for n in ['signals','forecasts','hold']]
risk = pd.read_parquet(parent/'risk/discounts.parquet')
annual = {x['year']:x for x in json.loads((parent/'risk/verification.json').read_text())['annual_checks']}
reference = pd.read_csv(parent/'risk/annual_reference.csv').set_index('year')
expected_return=float(np.unique(mu.to_numpy()[np.isfinite(mu.to_numpy())])[0])
calendar = dict(source='frozen historical exchange sessions',sessions=[str(d.date()) for d in sorted(market.date.unique())])
def model(year):
    z = annual[year]
    return dict(year=year,train_label_end=str(pd.Timestamp(reference.at[year,'max_label_end']).date()),
        expected_return=expected_return,
        **{key:z[key] for key in ['intercept','coefficients','center','scale','reference_risk','uniform_discount']})
def st_at(day):
    return set(st.loc[st.start.le(day)&(st.end.isna()|st.end.ge(day)), 'ts_code'])
def snapshot(date, daily, holdings):
    g=holdings[holdings.date.eq(date)]
    row=daily.loc[date]
    return dict(account_id='historical_fixture',asof=str(date.date()),source='frozen_backtest_fixture_not_broker',
        cash=float(row.cash),equity=float(row.equity),receivables=float(row.receivables),frozen_cash=0.,open_orders=[],
        positions={r.ts_code:dict(shares=int(r.shares),sellable_next_day=int(r.shares),close=float(r.price)) for r in g.itertuples()})
comparisons=[]
for window in ['cold','continued']:
    folder=parent/'execution'/f'{window}_calendar_01_full_risk'
    d=pd.read_csv(folder/'daily.csv',index_col='date',parse_dates=True)
    h=pd.read_csv(folder/'holdings.csv',parse_dates=['date'])
    e=pd.read_csv(folder/'events.csv',parse_dates=['date','signal_date'])
    t=pd.read_csv(folder/'trades.csv',parse_dates=['date','signal_date'])
    assert not e.kind.str.contains('bonus|stock_div|rights_applied').any(), '快照需要补充长期锁定股来源'
    state=empty_memory(); norders=0
    for date in d.index[:-1]:
        snap=snapshot(date,d,h)
        active=sig.index[sig.index<=date][-1]
        scores=sig.loc[active].dropna().sort_values(ascending=False,kind='stable').head(cfg['candidate_pool'])
        codes=set(scores.index)|set(snap['positions'])
        today=by_date.loc[date]
        next_date=pd.Timestamp(calendar['sessions'][calendar['sessions'].index(snap['asof'])+1])
        for trade in t[t.date.eq(date)&t.side.eq('sell')].itertuples():
            state.update(last_sale_date=snap['asof'],last_sale_signal=str(trade.signal_date.date()))
        m=model(active.year)
        bundle=dict(asof=snap['asof'],signal_date=str(active.date()),is_review=date in sig.index,
            eod_complete=True,provenance={'historical_fixture':True},model_year=active.year,model_cutoff=m['train_label_end'],
            expected_return=m['expected_return'],signals=scores.to_dict(),
            holds={c:bool(hold.at[active,c]) if pd.notna(hold.at[active,c]) else None for c in snap['positions']},
            risk_discounts={c:float(risk.at[active,c]) for c in codes},st_codes=sorted(st_at(date)),
            blocked_next_day=sorted(set(actions.loc[actions.known_date.le(date)&actions.ex_date.eq(next_date),'ts_code'])|st_at(next_date)),
            quotes={c:dict(date=snap['asof'],close=float(today.at[c,'close']),volume=float(today.at[c,'vol_share'])) for c in codes if c in today.index},
            eligible={c:c not in st_at(active)|st_at(date)|st_at(next_date) for c in codes})
        plan=make_plan(snap,bundle,calendar,cfg,state)
        fields=['ts_code','side','shares','limit_price','reserved_cash','signal_date','reason']
        actual=pd.DataFrame(plan['orders']).reindex(columns=fields)
        expected=e[e.kind.eq('order_planned')&e.date.eq(date)][fields].copy()
        expected['signal_date']=expected.signal_date.dt.strftime('%Y-%m-%d')
        for frame in [actual,expected]:
            for col in ['shares','limit_price','reserved_cash']:frame[col]=frame[col].astype(float)
        actual=actual.sort_values(['ts_code','side']).reset_index(drop=True)
        expected=expected.sort_values(['ts_code','side']).reset_index(drop=True)
        try:pd.testing.assert_frame_equal(actual,expected,check_dtype=False,atol=1e-7,rtol=0)
        except AssertionError:
            actual.to_csv(out/f'{window}_{date.date()}_actual.csv',index=False)
            expected.to_csv(out/f'{window}_{date.date()}_expected.csv',index=False)
            raise
        state=plan['next_memory'];norders+=len(actual)
        if window=='cold' and date==pd.Timestamp('2026-09-01'):
            save(out/'example_snapshot.json',snap);save(out/'example_frozen_bundle.json',bundle)
            save(out/'example_plan.json',plan);save(out/'example_memory.json',state)
    comparisons.append(dict(window=window,days=len(d)-1,orders=norders,exact_economic_orders=True))
    print('PLAN PARITY',comparisons[-1],flush=True)
save(out/'plan_parity.json',dict(checks=comparisons,planner_sha256=digest(ROOT/'src/shadow_planner.py'),
    config_sha256=digest(ROOT/'configs/shadow_execution.yaml')))
save(out/'model_2026.json',model(2026))
save(out/'calendar.json',calendar)
# 从当日之前的全A原始特征重建排名和状态，不能仅复用冻结信号冒充增量构建。
raw_path=ROOT/'outputs/recent_low_turnover/20260913T090336_282870Z/data_repair/repaired_feature_projection.parquet'
date='2026-09-01';stamp=pd.Timestamp(date)
raw=pd.read_parquet(raw_path,filters=[('date','<=',stamp)])
cal=dict(source='historical fixture calendar',sessions=[str(x.date()) for x in sorted(raw.date.unique())]+['2026-09-02'])
receipt=dict(asof=date,complete=True,source='historical reconstruction fixture, not fresh feed',
    downloaded_at=date+'T18:00:00+08:00',expected_rows=int(raw.date.eq(stamp).sum()))
snap=json.loads((out/'example_snapshot.json').read_text())
rebuilt=build_bundle(raw,market,basics,st,actions,date,cal,cfg,model(2026),snap['positions'],receipt)
old=json.loads((out/'example_frozen_bundle.json').read_text())
assert rebuilt['signals']==old['signals'] and rebuilt['holds']==old['holds']
for code,value in old['risk_discounts'].items():assert abs(rebuilt['risk_discounts'][code]-value)<1e-12
save(out/'example_rebuilt_bundle.json',rebuilt);save(out/'example_calendar.json',cal);save(out/'example_receipt.json',receipt)
save(out/'signal_parity.json',dict(date=date,ranking_exact=True,holds_exact=True,risk_tolerance=1e-12,
    universe=rebuilt['provenance']['full_universe'],source_panel_sha256=digest(raw_path)))
print('FULL UNIVERSE REBUILD PASSED',flush=True)
with (out/'tests.log').open('w') as log:
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_shadow*.py','-v'],
        env={**__import__('os').environ,'PYTHONPATH':str(ROOT/'src')},stdout=log,stderr=subprocess.STDOUT,check=True)
sources=['src/shadow_planner.py','src/shadow_account.py','src/shadow_inputs.py','scripts/shadow_execution.py',
         'scripts/validate_shadow_execution.py','tests/test_shadow_execution.py','tests/test_shadow_inputs.py','configs/shadow_execution.yaml']
for name in sources:
    target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
save(out/'verification.json',dict(status='passed',prior_artifacts_verified=hash_counts,plan_replays=comparisons,
    full_universe_signal_rebuilt=True,live_orders=False,broker_connected=False,
    source_hashes={n:digest(ROOT/n) for n in sources},root_agents_sha256=digest(ROOT.parent/'AGENTS.md')))
print('VALIDATED',out,flush=True)
