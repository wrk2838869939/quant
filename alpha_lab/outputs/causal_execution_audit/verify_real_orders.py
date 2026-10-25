"""从真实封存订单和独立股数状态核验因果资金与DAY限价，不调用回测引擎。"""
from pathlib import Path
import argparse,json,hashlib
from collections import defaultdict
import numpy as np,pandas as pd


def require(value,message):
    if not bool(value):raise AssertionError(message)


def verify(run,case):
    folder=run/'execution'/case['name'];inp=run/'inputs'
    daily=pd.read_csv(folder/'daily.csv',parse_dates=['date']).set_index('date')
    trades=pd.read_csv(folder/'trades.csv',parse_dates=['date','decision_date','signal_date'])
    events=pd.read_csv(folder/'events.csv',low_memory=False)
    for col in ['date','decision_date','execution_date','reference_date','signal_date']:
        if col in events:events[col]=pd.to_datetime(events[col],errors='coerce')
    holdings=pd.read_csv(folder/'holdings.csv',parse_dates=['date'])
    settings=json.loads((folder/'portfolio_config.json').read_text());cfg=settings['portfolio']
    market=pd.read_parquet(inp/'market.parquet').set_index(['date','ts_code']).sort_index()
    limits=pd.read_parquet(inp/'limits.parquet').set_index(['date','ts_code']).sort_index()
    basics=pd.read_parquet(inp/'basics.parquet').set_index('ts_code')
    actions=pd.read_parquet(inp/'normalized_actions.parquet')
    st=pd.read_parquet(inp/'st.parquet')
    require('known_date' in actions and actions.known_date.notna().all(),'corporate action knowledge date missing')
    require((actions.known_date<=actions.record_date).all(),'action known after record date')
    require('known_date' in st and st.known_date.notna().all(),'ST announcement date missing')
    plans=events[events.kind.eq('order_planned')].copy()
    require(not plans.order_id.duplicated().any(),'order id duplicated')
    require((plans.decision_date<plans.execution_date).all(),'order decision not before execution')
    require(plans.date.eq(plans.decision_date).all(),'order log written after its alleged decision')
    require(plans.reference_date.eq(plans.decision_date).all(),'reference not same known close')
    require(plans.signal_date.le(plans.decision_date).all(),'future strategy signal')
    require(plans.budget_source.eq('closing_cash').all(),'budget source must be known closing cash')
    calendar=list(daily.index);nextday=dict(zip(calendar[:-1],calendar[1:]))
    require(all(nextday[d]==e for d,e in zip(plans.decision_date,plans.execution_date)),'not next trading day DAY order')
    planmap=plans.set_index('order_id');buys=plans[plans.side.eq('buy')]
    for o in plans.itertuples():
        if o.side=='buy':
            active=st[(st.ts_code==o.ts_code)&(st.start<=o.execution_date)&(st.end.isna()|(st.end>=o.execution_date))]
            ambiguous=active[(active.known_date>o.decision_date)&(active.known_date<=o.execution_date)]
            require(ambiguous.empty,f'{o.order_id}: whole-day ST announcement applied without preopen timestamp')
        cross=actions[(actions.ts_code==o.ts_code)&(actions.ex_date==o.execution_date)]
        require((cross.known_date<=o.decision_date).all(),f'{o.order_id}: ex-date cancellation used not-yet-known action')
    maxbudgeterror=0.
    for date,g in plans.groupby('decision_date'):
        cash=float(daily.loc[date,'cash'])
        maxbudgeterror=max(maxbudgeterror,float(abs(g.cash_at_decision-cash).max()))
        require(np.allclose(g.cash_at_decision,cash,rtol=0,atol=1e-6),f'{date}: order cash differs actual close')
        buy=g[g.side.eq('buy')]
        require(buy.reserved_cash.sum()<=cash+1e-6,f'{date}: buy reservation spends future sale/dividend cash')
        require(len(buy)<=5-int(daily.loc[date,'n_hold']),f'{date}: order opens too many occupied names')
        held=set(holdings.loc[holdings.date.eq(date),'ts_code'])
        require(not set(buy.ts_code)&held,f'{date}: adding shares to retained holding')
        for o in buy.itertuples():
            q=int(o.shares);price=float(o.limit_price);gross=q*price
            expected=gross+max(cfg['min_fee'],gross*cfg['buy_cost'])
            require(abs(expected-o.reserved_cash)<1e-6,f'{o.order_id}: wrong fee reservation')
            require(q>0 and q%100==0,f'{o.order_id}: nonlegal mainboard lot')
            reference=market.loc[(date,o.ts_code),'close']
            require(abs(reference-o.reference_price)<1e-8,f'{o.order_id}: reference not known market close')
    finals=events[events.kind.isin(['order_filled','order_cancelled','order_deferred'])]
    require(set(finals.order_id)==set(plans.order_id),'order has no lifecycle terminal or foreign id')
    require(not finals.order_id.duplicated().any(),'multiple terminal outcomes for one DAY order')
    require(finals.date.eq(finals.execution_date).all(),'order resolved on wrong execution day')
    require(not trades.order_id.duplicated().any(),'multiple fills for one frozen order')
    require(set(trades.order_id)==set(finals.loc[finals.kind.eq('order_filled'),'order_id']),'trades and filled events disagree')
    intraday=0
    for t in trades.itertuples():
        p=planmap.loc[t.order_id]
        require(t.ts_code==p.ts_code and t.side==p.side and int(t.shares)==int(p.shares),f'{t.order_id}: fill changed locked name/quantity')
        require(t.date==p.execution_date and t.decision_date==p.decision_date and t.signal_date==p.signal_date,f'{t.order_id}: fill changed dates')
        require(abs(t.limit_price-p.limit_price)<1e-9,f'{t.order_id}: repriced frozen order')
        m=market.loc[(t.date,t.ts_code)];lim=limits.loc[(t.date,t.ts_code)]
        require(lim.down_limit-1e-8<=p.limit_price<=lim.up_limit+1e-8,f'{t.order_id}: illegal DAY limit')
        require(lim.down_limit-1e-8<=t.price<=lim.up_limit+1e-8,f'{t.order_id}: fill outside official range')
        if t.execution_kind=='open':
            require(abs(t.price-m.open)<1e-8,f'{t.order_id}: not actual opening fill')
            if t.side=='buy':require(m.open<=p.limit_price+1e-8 and m.open<lim.up_limit-cfg['price_tick']/2,f'{t.order_id}: opening buy not feasible')
            else:require(m.open>=p.limit_price-1e-8 and m.open>lim.down_limit+cfg['price_tick']/2,f'{t.order_id}: opening sell not feasible')
        else:
            require(t.execution_kind=='strict_intraday_cross',f'{t.order_id}: unknown matching semantics')
            intraday+=1;require(abs(t.price-p.limit_price)<1e-8,f'{t.order_id}: intraday fill not frozen limit')
            if t.side=='buy':require(m.low<p.limit_price-1e-8,f'{t.order_id}: DAY buy only touched limit')
            else:require(m.high>p.limit_price+1e-8,f'{t.order_id}: DAY sell only touched limit')
        if t.side=='buy':require(t.gross+t.fee<=p.reserved_cash+1e-6,f'{t.order_id}: spent beyond prior cash reservation')
        require(not ((actions.ts_code==t.ts_code)&(actions.ex_date==t.date)).any(),f'{t.order_id}: crossed ex-date without cancel/replan')
    # Entry supplement must be after an actual matching month's sale, at most once per original signal.
    attempts=events[events.kind.eq('entry_planning_attempt')].copy()
    extra=attempts[attempts.entry_attempt.eq(2)]
    require(not extra.signal_date.duplicated().any(),'more than one supplementary attempt for a monthly signal')
    for e in extra.itertuples():
        require(((trades.side=='sell')&(trades.date==e.decision_date)&(trades.signal_date==e.signal_date)).any(),f'{e.date}: supplement before actual matching sale')
        require(e.decision_date>e.signal_date,f'{e.date}: same-close supplementary sale budgeting')
    # Independent integer ownership + release dates, no Position class or existing verifier imported.
    shares=defaultdict(int);locked=defaultdict(list);entitlements={};bonus_events=0
    for date in daily.index:
        for j,a in actions[actions.ex_date.eq(date)].iterrows():
            held=entitlements.pop(j,0)
            if held and a.stk_div:
                amount=int(np.floor(held*a.stk_div+1e-8));shares[a.ts_code]+=amount
                locked[a.ts_code].append((amount,a.div_listdate));bonus_events+=1
        for code in list(shares):
            dead=basics.loc[code,'delist_date']
            if pd.notna(dead) and date>=dead:shares.pop(code,None);locked.pop(code,None)
        for t in trades[trades.date.eq(date)].itertuples():
            q=int(t.shares)
            if t.side=='sell':
                unavailable=sum(v for v,release in locked[t.ts_code] if date<release)
                require(q<=shares[t.ts_code]-unavailable,f'{t.order_id}: T+1 or bonus lock violation')
                shares[t.ts_code]-=q
            else:
                require(shares[t.ts_code]==0,f'{t.order_id}: unexpected same-name add')
                shares[t.ts_code]+=q
                release=nextday.get(date,date+pd.Timedelta(days=1));locked[t.ts_code].append((q,release))
        alive={k:v for k,v in shares.items() if v}
        actual=holdings[holdings.date.eq(date)].set_index('ts_code')
        require(alive==actual.shares.astype(int).to_dict(),f'{date}: ownership mismatch independent replay')
        require(len(alive)<=5,f'{date}: held more than five')
        for code,q in alive.items():
            available=q-sum(v for v,release in locked[code] if date<release)
            require(available==int(actual.loc[code,'available']),f'{date} {code}: sellable quantity mismatch')
        for j,a in actions[actions.record_date.eq(date)].iterrows():entitlements[j]=shares[a.ts_code]
    return dict(name=case['name'],status='passed',planned_orders=len(plans),filled_orders=len(trades),
        intraday_limit_fills=intraday,supplementary_attempts=len(extra),bonus_events=bonus_events,
        actual_days=len(daily),max_order_cash_difference=maxbudgeterror,
        missing_announcement_paths=0,same_day_announcement_ambiguity_paths=0,
        hashes={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['daily.csv','trades.csv','events.csv','holdings.csv']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case');a=p.parse_args()
    run=a.run.resolve();manifest=json.loads((run/'manifest.json').read_text());out=Path(__file__).resolve().parent/run.name;out.mkdir(exist_ok=True)
    cases=[c for c in manifest['cases'] if a.case is None or c['name']==a.case];require(bool(cases),'no matching case')
    rows=[]
    for c in cases:
        row=verify(run,c);rows.append(row);(out/(c['name']+'.json')).write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    result=dict(status='complete',run=str(run),cases=rows,case_count=len(rows),
                planned_orders=sum(r['planned_orders'] for r in rows),filled_orders=sum(r['filled_orders'] for r in rows),
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
                scope='independent known-cash/order lifecycle/DAY price/ownership/T+1/corporate actions; not queue/liquidity proof')
    (out/('verification_'+a.case+'.json' if a.case else 'independent_real_order_verification.json')).write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
