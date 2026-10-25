#!/usr/bin/env python3
"""预登记补充：旧市场风控接入耐心持仓，与固定现金整手对照。"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from portfolio import run_backtest
from audit_monthly_portfolio import load_inputs,reconcile
from run_patient_value_execution import process_report,json_write
from prepare_retail_validation import recovery_plan
from check_cash_reserve import add_cash_reserve
from audit_active_cash import event_report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--episodes-only',action='store_true');args=parser.parse_args()
    if args.episodes_only:
        write_episode_reports(args.run)
        return
    root=args.run; prepared=root/'execution';out=root/'risk_overlay';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()).get('status')=='complete':
        raise ValueError('完成的补充实验不能覆盖')
    cfg=yaml.safe_load((root/'config.yaml').read_text());pc=yaml.safe_load((prepared/'execution_config.yaml').read_text())
    rc=yaml.safe_load((ROOT/'configs/retail_validation.yaml').read_text())
    registration=json.loads((root/'risk_overlay_registration.json').read_text())
    features=pd.read_parquet(ROOT/rc['risk_features'],filters=[('date','<',pd.Timestamp('2024-01-01'))])
    if features.index.max()>=pd.Timestamp('2024-01-01'):raise ValueError('风险特征越过TEST')
    plan=recovery_plan(features,rc)
    previous=pd.read_csv(ROOT/registration['source'],index_col=0,parse_dates=True).extreme_recovery
    pd.testing.assert_series_equal(plan,previous,check_names=False,check_dtype=False)
    cutoff=pd.Timestamp('2021-12-31');pd.testing.assert_series_equal(recovery_plan(features.loc[:cutoff],rc),plan.loc[:cutoff])
    market,basics,st,limits,actions=load_inputs(prepared,pc)
    rights=yaml.safe_load((ROOT/'configs/corporate_action_components.yaml').read_text()).get('rights_issues',[])
    hold=pd.read_parquet(prepared/'monthly_hold_allowed.parquet')
    sources=[Path(__file__),ROOT/'src/risk_states.py',ROOT/'scripts/prepare_retail_validation.py',
        ROOT/'scripts/check_cash_reserve.py',ROOT/'configs/retail_validation.yaml',ROOT/'configs/active_cash_audit.yaml',
        ROOT/'src/portfolio.py',ROOT/'src/retail_rebalance.py',ROOT/'scripts/run_patient_value_execution.py',
        ROOT/'scripts/audit_active_cash.py',ROOT/'configs/corporate_action_components.yaml']
    manifest=dict(status='registered',fresh_holdout=False,test_market_data_used=False,
        registration=registration,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        cached_risk_plan_reproduced=True,prefix_invariant=True,results=[],
        parent_execution_manifest_sha256=hashlib.sha256((prepared/'manifest.json').read_bytes()).hexdigest(),
        risk_plan_sha256=hashlib.sha256(pd.util.hash_pandas_object(plan,index=True).values.tobytes()).hexdigest(),
        risk_features_sha256=hashlib.sha256(pd.util.hash_pandas_object(features,index=True).values.tobytes()).hexdigest())
    json_write(out/'manifest.json',manifest)
    (out/'parent_execution_manifest_at_start.json').write_bytes((prepared/'manifest.json').read_bytes())
    for path in sources:
        target=out/'sources'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    plan.to_csv(out/'risk_allowed.csv');summaries=[]
    for policy in cfg['execution_policies']:
        signals=pd.read_parquet(prepared/f'monthly_{policy}_signals.parquet')
        forecasts=pd.read_parquet(prepared/f'monthly_{policy}_point_forecasts.parquet')
        for kind in ['market_gate','fixed_half_cash']:
            name=kind+'_'+policy;print(name,flush=True);folder=out/name;folder.mkdir(exist_ok=True)
            capital=registration['capital'];p=deepcopy(pc['portfolio']);p['capital']=capital if kind=='market_gate' else capital*.5
            kwargs=dict(retail_forecasts=forecasts,retail_config=cfg['retail'],hold_allowed=hold)
            if kind=='market_gate':kwargs.update(risk_allowed=plan,risk_signal_max_age=rc['risk_signal_max_age'],reentry_on_schedule_only=True)
            r=run_backtest(signals,market,basics,st,limits,actions,*cfg['execution'],p,rights,**kwargs)
            if kind=='fixed_half_cash':r=add_cash_reserve(r,capital*.5,p['annual_days'])
            check=reconcile(r,capital);process,pnl,episodes,exits,metrics=process_report(r,capital)
            for suffix,frame in [('daily',r.daily),('trades',r.trades),('holdings',r.holdings),('events',r.events),
                ('process_daily',process),('asset_pnl',pnl),('holding_episodes',episodes),('exit_reasons',exits)]:
                frame.to_csv(folder/f'{suffix}.csv',index=suffix in ['daily','process_daily'])
            summary={**r.summary,**check,**metrics,'name':name,'policy':policy,'kind':kind,'capital':capital}
            json_write(folder/'summary.json',summary);summaries.append(summary)
            if kind=='market_gate':
                baseline_path=prepared/f'base_50000_monthly_{policy}'/'daily.csv'
                if baseline_path.exists():
                    baseline=pd.read_csv(baseline_path,index_col=0,parse_dates=True)
                    event_report(r,baseline,name,[5,10,20]).to_csv(folder/'risk_episodes.csv',index=False)
            pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
            manifest['results'].append(name);json_write(out/'manifest.json',manifest)
            print({k:summary[k] for k in ['name','ann','mdd','sharpe','n_cash_days']},flush=True)
    manifest['status']='complete';json_write(out/'manifest.json',manifest)


def write_episode_reports(root):
    """父账本完成后补齐对照事件；只读已有结果，不重新运行策略。"""
    cfg=yaml.safe_load((root/'config.yaml').read_text())
    for policy in cfg['execution_policies']:
        folder=root/'risk_overlay'/('market_gate_'+policy)
        baseline=pd.read_csv(root/'execution'/f'base_50000_monthly_{policy}'/'daily.csv',index_col=0,parse_dates=True)
        daily=pd.read_csv(folder/'daily.csv',index_col=0,parse_dates=True)
        trades=pd.read_csv(folder/'trades.csv',parse_dates=['date'])
        r=SimpleNamespace(daily=daily,trades=trades)
        event_report(r,baseline,'market_gate_'+policy,[5,10,20]).to_csv(folder/'risk_episodes.csv',index=False)
    json_write(root/'risk_overlay'/'episodes_complete.json',dict(status='complete',
        attribution='different evolving portfolios; paired paths are descriptive, not isolated timing alpha',
        unfinished_semantics='no reentry before next request or sample end; not necessarily failed liquidation'))


if __name__=='__main__':main()
