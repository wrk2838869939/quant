#!/usr/bin/env python3
"""固定风险规则的逐日成交与事件审计，限制在已使用开发窗口。"""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from risk_states import market_risk_features,fixed_risk_plan
from portfolio import run_backtest,daily_metrics
from audit_monthly_portfolio import load_inputs,reconcile
from check_cash_reserve import add_cash_reserve


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def event_report(result,baseline,rule,horizons):
    """逐个撤退请求的状态与同期完整对照账本；不假设两分支持仓相同。"""
    d=result.daily
    transitions=d.risk_target.ne(d.risk_target.shift(1).fillna(1))
    exits=d.index[transitions & d.risk_target.eq(0)]
    rows=[]
    for start in exits:
        loc=d.index.get_loc(start)
        future=d.loc[start:]
        next_exit=exits[exits>start]
        bound=next_exit[0] if len(next_exit) else d.index[-1]+pd.Timedelta(days=1)
        window=future[future.index<bound]
        recovered=window.index[window.risk_target.eq(1)]
        recovery=recovered[0] if len(recovered) else pd.NaT
        # 同一请求尚未完成前再次触发会被截断，明确记录为未完成事件。
        flat=window.index[window.n_hold.eq(0)]
        exit_done=flat[0] if len(flat) else pd.NaT
        buys=result.trades[result.trades.side.eq('buy') & result.trades.date.ge(start) & result.trades.date.lt(bound)]
        reentered=buys.date.min() if len(buys) else pd.NaT
        end=reentered if pd.notna(reentered) else window.index[-1]
        previous=baseline.equity.iloc[loc-1] if loc else baseline.equity.iloc[0]
        prior_risk=d.equity.iloc[loc-1] if loc else d.equity.iloc[0]
        row=dict(rule=rule,request_date=start,signal_date=d.loc[start,'risk_signal_date'],
            exit_complete_date=exit_done,recovery_execution_date=recovery,first_reentry_date=reentered,
            comparison_end=end,unfinished=bool(pd.isna(reentered)),
            exit_delay_trading_days=d.index.get_loc(exit_done)-loc if pd.notna(exit_done) else np.nan,
            active_days_until_reentry=d.index.get_loc(end)-loc+1,
            baseline_return_through_reentry=baseline.loc[end,'equity']/previous-1,
            risk_return_through_reentry=d.loc[end,'equity']/prior_risk-1,
            loss_before_signal_5d=baseline.equity.iloc[loc-1]/baseline.equity.iloc[max(0,loc-6)]-1 if loc else 0.)
        row['paired_return_difference']=row['risk_return_through_reentry']-row['baseline_return_through_reentry']
        for horizon in horizons:
            stop=loc+horizon-1
            row[f'baseline_next_{horizon}d_return']=baseline.equity.iloc[stop]/previous-1 if stop<len(d) else np.nan
        segment=d.loc[start:end]
        row['fees_through_reentry']=result.trades.loc[result.trades.date.between(start,end),'fee'].sum()
        row['cash_days_through_reentry']=int(segment.n_hold.eq(0).sum())
        row['exit_pending_days']=int(segment.risk_state.eq('exit_pending').sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    config_path=ROOT/'configs/active_cash_audit.yaml'
    spec=yaml.safe_load(config_path.read_text())
    hi=pd.Timestamp(spec['market_end']);lo=pd.Timestamp(spec['market_start'])
    if hi>=pd.Timestamp('2024-01-01') or pd.Timestamp(spec['valid'][1])>hi:
        raise ValueError('禁止读取TEST或窗口越界')
    parent=ROOT/spec['prepared']
    cfg=yaml.safe_load((parent/'config.yaml').read_text())
    if cfg['valid']!=spec['valid']:raise ValueError('父账本与风险审计窗口不同')
    out=ROOT/spec['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    (out/'experiment.yaml').write_text(config_path.read_text())
    sources=[Path(__file__),ROOT/'src/portfolio.py',ROOT/'src/risk_states.py',ROOT/'scripts/audit_monthly_portfolio.py',ROOT/'scripts/check_cash_reserve.py',ROOT/'configs/corporate_action_components.yaml']
    manifest=dict(status='preregistered',created_utc=datetime.now(timezone.utc).isoformat(),
        rules=spec['rules'],test_market_data_used=False,fresh_holdout=False,
        purpose='frozen_legacy_selection_risk_mechanism_not_retail_strategy',
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources},
        input_hashes={name:digest(parent/name) for name in ['config.yaml','candidate_signals.parquet','candidate_cost1_daily.csv','execution_codes.csv','execution_input_hashes.csv']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'预注册已落盘 {out}',flush=True)
    try:
        market,basics,st,limits,actions=load_inputs(parent,cfg)
        allcodes=basics[~basics.ts_code.str.startswith(('30','68')) & ~basics.ts_code.str.endswith('.BJ')].ts_code.tolist()
        raw=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',columns=['date','ts_code','pct_chg','vol_share'],
            filters=[('date','>=',lo),('date','<=',hi),('ts_code','in',allcodes)])
        raw=raw.merge(basics[['ts_code','list_date','delist_date']],on='ts_code',validate='many_to_one')
        raw=raw[(raw.date-raw.list_date).dt.days.ge(spec['min_listing_days']) &
            (raw.delist_date.isna()|raw.date.lt(raw.delist_date)) &raw.vol_share.gt(0)&np.isfinite(raw.pct_chg)].copy()
        # ST历史区间精确排除；不按股价、当前是否上市或因子是否好看筛市场广度。
        mask=np.zeros(len(raw),dtype=bool)
        date_values=raw.date.to_numpy();code_values=raw.ts_code.to_numpy()
        for code, intervals in st.groupby('ts_code'):
            pos=np.flatnonzero(code_values==code)
            for row in intervals.itertuples():
                mask[pos]|=(date_values[pos]>=row.start.to_datetime64()) & (date_values[pos]<=row.end.to_datetime64() if pd.notna(row.end) else True)
        raw=raw.loc[~mask]
        features=market_risk_features(raw,spec)
        features.to_parquet(out/'market_features.parquet')
        del raw
        print(f'风险特征 {len(features)} 日；VALID规则冻结',flush=True)
        valid_lo,valid_hi=map(pd.Timestamp,spec['valid'])
        plans={name:fixed_risk_plan(features,name,spec).loc[valid_lo:valid_hi] for name in spec['rules']}
        for name,plan in plans.items():
            if not plan.valid.all():raise ValueError(f'{name}开发窗口风险特征缺失')
            plan.to_csv(out/(name+'_plan.csv'))
        manifest.update(status='plans_frozen',feature_hash=digest(out/'market_features.parquet'),
            plan_hashes={name:digest(out/(name+'_plan.csv')) for name in plans})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        signals=pd.read_parquet(parent/'candidate_signals.parquet')
        rights=yaml.safe_load((ROOT/'configs/corporate_action_components.yaml').read_text()).get('rights_issues',[])
        reports=[];results={};all_events=[]
        for name in ['baseline','fixed_half']+spec['rules']:
            p=deepcopy(cfg['portfolio']);total=p['capital']
            p['capital']=total-spec['reserve_cash']
            if name=='fixed_half':p['capital']*=spec['fixed_exposure_fraction']
            kwargs={} if name not in plans else dict(risk_allowed=plans[name].allowed,risk_signal_max_age=spec['risk_signal_max_age'])
            result=run_backtest(signals,market,basics,st,limits,actions,*spec['valid'],p,rights,**kwargs)
            result=add_cash_reserve(result,total-p['capital'],p['annual_days'])
            checks=reconcile(result,total)
            if name=='baseline':
                old=pd.read_csv(parent/'candidate_cost1_daily.csv',index_col='date',parse_dates=True)
                pd.testing.assert_frame_equal(result.daily,old,check_freq=False,check_dtype=False,atol=1e-8,rtol=1e-12)
            for suffix,frame in [('daily',result.daily),('trades',result.trades),('events',result.events),('holdings',result.holdings)]:
                frame.to_csv(out/f'{name}_{suffix}.csv',index=suffix=='daily')
            result.daily['actual_exposure']=result.daily.stock_value/result.daily.equity
            row=dict(strategy=name,**result.summary,**checks)
            if name in plans:
                er=event_report(result,results['baseline'].daily,name,spec['event_horizons'])
                er.to_csv(out/f'{name}_risk_episodes.csv',index=False)
                all_events.append(er)
                row.update(risk_requests=len(er),requested_cash_days=int(result.daily.risk_target.eq(0).sum()),
                    actual_zero_stock_days=int(result.daily.n_hold.eq(0).sum()),
                    exit_pending_days=int(result.daily.risk_state.eq('exit_pending').sum()),
                    max_exit_delay=float(er.exit_delay_trading_days.max()) if len(er) else 0.,
                    completed_reentry_events=int(er.first_reentry_date.notna().sum()) if len(er) else 0)
            reports.append(row);results[name]=result
            pd.DataFrame(reports).to_csv(out/'summary.csv',index=False)
            print(json.dumps(row,ensure_ascii=False),flush=True)
        if all_events:pd.concat(all_events,ignore_index=True).to_csv(out/'risk_episodes.csv',index=False)
        years=[]
        for name,result in results.items():
            previous=cfg['portfolio']['capital']
            for year,d in result.daily.groupby(result.daily.index.year):
                metrics,_=daily_metrics(d.equity,previous,cfg['portfolio']['annual_days'])
                years.append(dict(strategy=name,year=year,**metrics));previous=d.equity.iloc[-1]
        pd.DataFrame(years).to_csv(out/'yearly.csv',index=False)
        plot(results,out)
        manifest.update(status='complete',baseline_daily_exactly_reproduced=True,results=reports)
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))


def plot(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for name,result in results.items():
        d=result.daily
        axes[0].plot(d.index,d.nav,label=name)
        axes[1].plot(d.index,d.equity/d.equity.cummax()-1)
        axes[2].plot(d.index,d.actual_exposure,alpha=.65)
    axes[0].set_title('Frozen development book: risk overlay diagnostics (2022-2023)')
    for ax,label in zip(axes,['NAV','Drawdown','Actual stock exposure']):ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(ncol=3);fig.tight_layout();fig.savefig(out/'risk_paths.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
