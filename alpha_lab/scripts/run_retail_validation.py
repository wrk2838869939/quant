#!/usr/bin/env python3
"""运行事前固定的小账户资金、风险、成本和执行压力矩阵。"""
import argparse,hashlib,json,sys
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from portfolio import run_backtest,daily_metrics
from audit_monthly_portfolio import load_inputs,reconcile
from check_cash_reserve import add_cash_reserve
from audit_active_cash import event_report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);a=ap.parse_args();out=a.prepared
    cfg=yaml.safe_load((out/'config.yaml').read_text());pc=yaml.safe_load((out/'execution_config.yaml').read_text())
    manifest_path=out/'execution_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('status')=='complete':raise ValueError('已完成的验证不能覆盖')
    sources=[Path(__file__),ROOT/'src/portfolio.py',ROOT/'src/retail_rebalance.py',ROOT/'src/risk_states.py',ROOT/'configs/corporate_action_components.yaml']
    cases=[]
    def add(name,capital,rule,cost=1.,delay=0,cap=None,slip=0.,active_fraction=1.):
        cases.append(dict(name=name,capital=capital,rule=rule,cost=cost,delay=delay,
            max_stock_fraction=cfg['base_max_stock_fraction'] if cap is None else cap,slippage_rate=slip,active_fraction=active_fraction))
    for capital in cfg['capitals']:
        for rule in cfg['rules']:add(f'base_{capital:g}_{rule}',capital,rule)
    reference=cfg['reference_capital']
    for rule in cfg['rules']:
        add(f'cost2_{rule}',reference,rule,cost=2.)
        add(f'delay1_{rule}',reference,rule,delay=1)
        add(f'cap40_{rule}',reference,rule,cap=cfg['stress_max_stock_fraction'])
        for slip in [.0005,.001]:add(f'slip{slip:g}_{rule}',reference,rule,slip=slip)
    add('fixed_half_none',reference,'none',active_fraction=.5)
    manifest=dict(status='registered',test_market_data_used=False,fresh_holdout=False,cases=cases,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'monthly_signals.parquet',out/'monthly_forecasts.parquet',out/'daily_signals.parquet',out/'daily_forecasts.parquet',out/'risk_plans.csv']})
    manifest_path.write_text(json.dumps(manifest,indent=2));print(f'执行矩阵冻结 {len(cases)} 个情景',flush=True)
    results={};summaries=[];years=[];event_rows=[]
    try:
        market,basics,st,limits,actions=load_inputs(out,pc)
        signals=pd.read_parquet(out/'monthly_signals.parquet');forecasts=pd.read_parquet(out/'monthly_forecasts.parquet')
        daily_signal=pd.read_parquet(out/'daily_signals.parquet');daily_forecast=pd.read_parquet(out/'daily_forecasts.parquet')
        plans=pd.read_csv(out/'risk_plans.csv',index_col=0,parse_dates=True)
        rights=yaml.safe_load((ROOT/'configs/corporate_action_components.yaml').read_text()).get('rights_issues',[])
        for k,case in enumerate(cases):
            name=case['name'];print(f'[{k+1}/{len(cases)}] {name}',flush=True)
            p=deepcopy(pc['portfolio']);p['capital']=case['capital']*case['active_fraction']
            for field in ['buy_cost','sell_cost','min_fee']:p[field]*=case['cost']
            kwargs=dict(retail_forecasts=forecasts,retail_config=dict(max_stock_fraction=case['max_stock_fraction'],candidate_pool=cfg['candidate_pool']),
                extra_execution_delay=case['delay'],slippage_rate=case['slippage_rate'])
            if case['rule']!='none':
                kwargs.update(risk_allowed=plans[case['rule']],risk_signal_max_age=cfg['risk_signal_max_age'],reentry_signals=daily_signal,reentry_forecasts=daily_forecast)
            r=run_backtest(signals,market,basics,st,limits,actions,*cfg['valid'],p,rights,**kwargs)
            if case['active_fraction']<1:r=add_cash_reserve(r,case['capital']-p['capital'],p['annual_days'])
            check=reconcile(r,case['capital'])
            for suffix,frame in [('daily',r.daily),('trades',r.trades),('holdings',r.holdings),('events',r.events)]:frame.to_csv(out/f'{name}_{suffix}.csv',index=suffix=='daily')
            h=r.holdings.copy()
            max_weight=float((h.value/h.date.map(r.daily.equity)).max()) if len(h) else 0.
            alloc=r.events[r.events.kind=='allocation_decision'] if len(r.events) else pd.DataFrame()
            buys=r.trades[r.trades.side=='buy'] if len(r.trades) else pd.DataFrame()
            # 超过平均每个候选预算的买单数量，仅证明未按50席限价筛选。
            above_old_seat=int((buys.gross>case['capital']/cfg['candidate_pool']).sum()) if len(buys) else 0
            row=dict({**r.summary,**case,**check},mean_holdings=r.daily.n_hold.mean(),max_holdings=int(r.daily.n_hold.max()),
                max_marked_stock_weight=max_weight,mean_cash_fraction=float((r.daily.cash/r.daily.equity).mean()),
                buys_above_legacy_seat=above_old_seat,n_cash_days=int(r.daily.n_hold.eq(0).sum()),
                n_forced_excess=int(alloc.unavoidable_locked_excess.gt(1e-6).sum()) if len(alloc) else 0)
            if case['rule']!='none' and case['name'].startswith('base_'):
                base=results[f"base_{case['capital']:g}_none"].daily
                er=event_report(r,base,name,[5,10,20]);er.to_csv(out/f'{name}_risk_episodes.csv',index=False)
                event_rows.append(er)
            summaries.append(row);results[name]=r
            pd.DataFrame(summaries).to_csv(out/'execution_summary.csv',index=False)
            previous=case['capital']
            for year,d in r.daily.groupby(r.daily.index.year):
                m,_=daily_metrics(d.equity,previous,p['annual_days']);years.append(dict(name=name,year=year,**m));previous=d.equity.iloc[-1]
            print(json.dumps({key:row[key] for key in ['name','ann','mdd','sharpe','mean_holdings','max_holdings','total_fees']},ensure_ascii=False),flush=True)
        pd.DataFrame(years).to_csv(out/'execution_yearly.csv',index=False)
        pd.concat(event_rows,ignore_index=True).to_csv(out/'retail_risk_episodes.csv',index=False)
        plot(results,out,reference)
        manifest.update(status='complete',completed_cases=len(cases))
    except Exception as e:
        manifest.update(status='failed',error_type=type(e).__name__,error=str(e),completed_cases=len(summaries));raise
    finally:manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))


def plot(results,out,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for rule in ['none','extreme_original','extreme_recovery']:
        d=results[f'base_{reference:g}_{rule}'].daily
        axes[0].plot(d.index,d.nav,label=rule)
        axes[1].plot(d.index,d.stock_value/d.equity)
        axes[2].plot(d.index,d.n_hold)
    axes[0].set_title('Retail research account: daily NAV, exposure and integer holdings')
    for ax,label in zip(axes,['NAV','Stock exposure','Holdings']):ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'retail_paths.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
