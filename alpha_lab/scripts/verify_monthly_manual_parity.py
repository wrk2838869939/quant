"""人工计划器对照封存月度50%研究订单；成交仅作历史夹具，非真实账户。"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shadow_planner import make_plan,empty_memory,fingerprint,ShadowError
from monthly_risk_live import load_model,probability,apply_fill
from portfolio_cash_state import basket_features
from restore_quota import RestoreQuota


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False);print(out,flush=True)
    base=ROOT/'outputs/selection_stability/20260914T070228_976007Z'
    run=ROOT/'outputs/monthly_predicted_budget/20260914T104757_015442Z'
    evidence=ROOT/'outputs/forward_risk/20260914T104245_763609Z'
    inputs=base/'inputs';market=pd.read_parquet(inputs/'market.parquet');bydate=market.set_index(['date','ts_code'])
    st=pd.read_parquet(inputs/'st.parquet');actions=pd.read_parquet(inputs/'normalized_actions.parquet')
    discounts=pd.read_parquet(base/'risk.parquet')
    close=pd.read_parquet(base/'signals/close.parquet');recovery=basket_features(close)
    features=pd.read_parquet(evidence/'features.parquet')
    calendar=dict(source='sealed research calendar',sessions=[str(d.date()) for d in features.index])
    cfg0=yaml.safe_load((ROOT/'configs/monthly_risk_manual.yaml').read_text())
    checks=[]
    for case in json.loads((run/'protocol.json').read_text())['cases']:
        if case['label']!='budget50' or case['cost_bps']!=0 or case['cohort']!='continued':continue
        cfg=json.loads(json.dumps(cfg0));cfg['schedule']['parameter']=case['offset']
        model=load_model(cfg);schedule=base/f"schedules/calendar_{case['offset']:02d}/rank_buffer"
        sig=pd.read_parquet(schedule/'signals.parquet');hold=pd.read_parquet(schedule/'hold.parquet');mu=pd.read_parquet(schedule/'forecasts.parquet')
        folder=run/'accounts'/case['name'];d=pd.read_parquet(folder/'daily.parquet');h=pd.read_parquet(folder/'holdings.parquet');t=pd.read_parquet(folder/'trades.parquet');e=pd.read_parquet(folder/'events.parquet')
        planned=e[e.kind.eq('order_planned')].set_index('order_id')
        state=empty_memory();count=0;restore_count=0;history=[];guarded=[]
        for date in d.index[:-1]:
            day=str(date.date());nextday=calendar['sessions'][calendar['sessions'].index(day)+1]
            active=sig.index[sig.index<=date][-1] if len(sig.index[sig.index<=date]) else None
            if active is None:continue
            # 真实研究成交，按时间顺序重建恢复额度；不从计划数量推算。
            for tr in t[t.date.eq(date)].itertuples():
                order=planned.loc[tr.order_id].to_dict();order['ts_code']=tr.ts_code
                order['restore_epoch']=int(order['restore_epoch']) if pd.notna(order['restore_epoch']) else state['monthly_risk']['quota']['epoch']
                position={r.ts_code:int(r.shares) for r in h[h.date.eq(date)].itertuples()}
                prior_position=dict(position)
                prior_position[tr.ts_code]=position.get(tr.ts_code,0)-(int(tr.shares) if tr.side=='buy' else -int(tr.shares))
                book=dict(memory=state,book_positions={c:q for c,q in prior_position.items() if q>0})
                apply_fill(book,order,dict(execution_id=tr.order_id,shares=int(tr.shares)),position.get(tr.ts_code,0))
                if tr.side=='sell':state.update(last_sale_date=day,last_sale_signal=str(tr.signal_date.date()))
            g=h[h.date.eq(date)];row=d.loc[date]
            snap=dict(account_id='historical_fixture',asof=day,source='sealed_backtest_not_broker',cash=float(row.cash),equity=float(row.equity),receivables=float(row.receivables),frozen_cash=0.,open_orders=[],
                positions={r.ts_code:dict(shares=int(r.shares),sellable_next_day=int(r.shares),close=float(r.price)) for r in g.itertuples()})
            if date in sig.index:
                x={k:float(features.at[date,k]) for k in model['features']}
                history.append(dict(date=day,features=x,probability=probability(x,model)))
            scores=sig.loc[active].dropna().sort_values(ascending=False,kind='stable').head(cfg['candidate_pool']);codes=set(scores.index)|set(snap['positions']);today=bydate.loc[date]
            def st_at(x):return set(st.loc[st.known_date.le(date)&st.start.le(x)&(st.end.isna()|st.end.ge(x)),'ts_code'])
            # 研究账本已结算公司行动；本项只比较策略经济订单。人工入口另测逐股暂停，不把研究流水当真实执行。
            blocked=st_at(pd.Timestamp(nextday))
            bundle=dict(asof=day,signal_date=str(active.date()),is_review=date in sig.index,eod_complete=True,provenance={'historical_fixture':True},model_year=active.year,model_cutoff=f'{active.year-1}-12-31',
                expected_return=float(mu.loc[active].dropna().iloc[0]),signals=scores.to_dict(),holds={c:bool(hold.at[active,c]) if pd.notna(hold.at[active,c]) else None for c in snap['positions']},
                risk_discounts={c:float(discounts.at[active,c]) for c in codes},st_codes=sorted(st_at(date)),blocked_next_day=sorted(blocked),
                quotes={c:dict(date=day,close=float(today.at[c,'close']),volume=float(today.at[c,'vol_share'])) for c in codes if c in today.index},eligible={c:c not in st_at(active)|st_at(date)|blocked for c in codes},
                strategy=cfg['strategy'],strategy_config=fingerprint(cfg),market_risk_history=list(history),
                stock_recovery={c:dict(known=bool(recovery['known'].at[date,c]),weak=bool(recovery['weak'].at[date,c]),ret20=float(recovery['ret20'].at[date,c]) if np.isfinite(recovery['ret20'].at[date,c]) else None) for c in snap['positions']})
            try:
                plan=make_plan(snap,bundle,calendar,cfg,state)
            except ShadowError:
                missing=set(snap['positions'])-set(bundle['quotes'])
                if not missing or date in sig.index or len(e[e.kind.eq('order_planned')&e.date.eq(date)]):raise
                guarded.append(dict(date=day,missing_quotes=sorted(missing),research_orders=0))
                continue
            fields=['ts_code','side','shares','limit_price','reserved_cash','signal_date','reason']
            actual=pd.DataFrame(plan['orders']).reindex(columns=fields)
            expected=e[e.kind.eq('order_planned')&e.date.eq(date)][fields].copy();expected['signal_date']=pd.to_datetime(expected.signal_date).dt.strftime('%Y-%m-%d')
            for frame in [actual,expected]:
                for col in ['shares','limit_price','reserved_cash']:frame[col]=frame[col].astype(float)
            actual=actual.sort_values(['ts_code','side']).reset_index(drop=True);expected=expected.sort_values(['ts_code','side']).reset_index(drop=True)
            try:pd.testing.assert_frame_equal(actual,expected,check_dtype=False,atol=1e-7,rtol=0)
            except AssertionError:
                actual.to_csv(out/f'{case["name"]}_{day}_actual.csv',index=False);expected.to_csv(out/f'{case["name"]}_{day}_expected.csv',index=False)
                raise
            count+=len(actual);restore_count+=sum(o['reason']=='targeted_restore' for o in plan['orders']);state=plan['next_memory']
        checks.append(dict(account=case['name'],days=len(d)-1,orders=count,restores=restore_count,exact_economic_orders=True,guarded_missing_quote_days=guarded));print(checks[-1],flush=True)
    (out/'verification.json').write_text(json.dumps(dict(checks=checks,historical_fixture=True,real_fills=False,corporate_actions='research_settled_fixture; live_manual_pause_tested_separately'),indent=2))


if __name__=='__main__':main()
