#!/usr/bin/env python3
"""完整矩阵结束后独立复核逐股损益、时间边界、求解误差与过程归因。"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from path_diagnostics import asset_daily_states
from prepare_retail_validation import recovery_plan
from ledger import ExperimentLedger
from portfolio import board_of


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);a=ap.parse_args();out=a.prepared.resolve()
    manifest=json.loads((out/'execution_manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['completed_cases']==25
    for name,digest in manifest['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in manifest['input_hashes'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
    cfg=yaml.safe_load((out/'config.yaml').read_text())
    features=pd.read_parquet(ROOT/cfg['risk_features'])
    risk=recovery_plan(features,cfg)
    for date in ['2020-06-30','2021-12-31','2022-06-30','2023-06-30']:
        short=recovery_plan(features.loc[:date],cfg)
        pd.testing.assert_series_equal(short,risk.loc[:date])
    obs=pd.read_csv(out/'train_rank_returns.csv',parse_dates=['date','label_end'])
    assert obs.label_end.max()<pd.Timestamp(cfg['valid'][0])
    model=pd.read_csv(out/'frozen_return_model.csv',index_col='bucket')
    np.testing.assert_allclose(obs.groupby('bucket').mean_return.mean(),model.expected_return)
    rolling=[]
    for year in cfg['walk_forward_years']:
        part=obs[obs.label_end<pd.Timestamp(f'{year}-01-01')]
        rolling.append(dict(year=year,last_training_label=str(part.label_end.max()),n_training_dates=part.date.nunique()))
    rows=[];worst=[];contributions=[]
    summary=pd.read_csv(out/'execution_summary.csv')
    for case in manifest['cases']:
        name=case['name'];capital=case['capital']
        daily=pd.read_csv(out/f'{name}_daily.csv',index_col='date',parse_dates=True)
        trades=pd.read_csv(out/f'{name}_trades.csv',parse_dates=['date','signal_date'])
        holdings=pd.read_csv(out/f'{name}_holdings.csv',parse_dates=['date'])
        events=pd.read_csv(out/f'{name}_events.csv',parse_dates=['date'])
        assert daily.index.max()<pd.Timestamp('2024-01-01')
        assert (trades.signal_date<trades.date).all()
        states=asset_daily_states(daily,trades,holdings,events,capital)
        states.to_parquet(out/f'{name}_asset_states.parquet',index=False)
        alloc=events[events.kind.eq('allocation_decision')]
        assert alloc.solver_gap.max()<=.001+1e-8
        checked=alloc.merge(trades[['date','ts_code','side','shares']],on=['date','ts_code'],how='left')
        actual=trades.pivot_table(index=['date','ts_code'],columns='side',values='shares',aggfunc='sum')
        planned=alloc.set_index(['date','ts_code'])
        mismatches=0
        for side in ['buy','sell']:
            realized=actual.get(side,pd.Series(dtype=float)).reindex(planned.index).fillna(0)
            mismatches+=int((realized!=planned[f'{side}_shares']).sum())
        buys=trades[trades.side.eq('buy')]
        p=capital*case['active_fraction']
        row=dict(name=name,pnl_reconciled=True,execution_days=len(daily),
            solver_max_relative_gap=alloc.solver_gap.max(),solver_max_currency_gap=alloc.solver_absolute_gap.max(),
            planned_actual_quantity_mismatches=mismatches,max_buy_price=buys.price.max(),median_buy_gross=buys.gross.median(),
            min_fee_buy_fraction=np.isclose(buys.fee,5*case['cost']).mean() if case['slippage_rate']==0 else np.nan,
            turnover_gross_per_year=trades.gross.sum()/capital/(len(daily)/244),
            empty_allocation_dates=int(alloc.groupby('date').target_shares.sum().eq(0).sum()),
            preserved_old_positions=int(((alloc.old_shares>0)&alloc.buy_shares.eq(0)&alloc.sell_shares.eq(0)).sum()),
            partial_sales=int(((alloc.sell_shares>0)&(alloc.sell_shares<alloc.old_shares)).sum()),
            min_cash=daily.cash.min(),mean_active_cash=(daily.cash-(capital-p)).mean())
        for board in ['main','chinext','star','beijing']:
            board_value=holdings[holdings.ts_code.map(board_of).eq(board)].groupby('date').value.sum().reindex(daily.index).fillna(0)
            row[f'mean_{board}_exposure']=float((board_value/daily.equity).mean())
        rows.append(row)
        peaks=np.maximum.accumulate(np.r_[capital,daily.equity.to_numpy()])[1:]
        trough=(daily.equity/peaks-1).idxmin();before=daily.loc[:trough].equity
        peak=before.idxmax() if before.max()>=capital else daily.index[0]-pd.Timedelta(days=1)
        draw=states[(states.date>peak)&(states.date<=trough)].groupby('ts_code').pnl.sum().sort_values()
        contributions.extend(dict(name=name,ts_code=code,pnl=value,peak=str(peak),trough=str(trough)) for code,value in draw.items())
        for date,r in daily.nsmallest(5,'ret').iterrows():
            st=states[states.date.eq(date)]
            worst.append(dict(name=name,date=date,ret=r.ret,pnl=st.pnl.sum(),losers=int(st.pnl.lt(0).sum()),winners=int(st.pnl.gt(0).sum()),n_hold=r.n_hold))
    # 成本压力会改变整数最优组合；单列路径分叉，禁止把收益上升归因为滑点。
    base_name=f"base_{cfg['reference_capital']:g}_none"
    base_d=pd.read_csv(out/f'{base_name}_daily.csv',index_col='date',parse_dates=True)
    base_h=pd.read_csv(out/f'{base_name}_holdings.csv',parse_dates=['date'])
    base_t=pd.read_csv(out/f'{base_name}_trades.csv',parse_dates=['date'])
    base_alloc=pd.read_csv(out/f'{base_name}_events.csv')
    base_alloc=base_alloc[base_alloc.kind.eq('allocation_decision')].set_index(['date','ts_code']).target_shares
    forks=[]
    for case in manifest['cases']:
        if case['rule']!='none' or not case['slippage_rate']:continue
        other=pd.read_csv(out/f"{case['name']}_events.csv")
        other=other[other.kind.eq('allocation_decision')].set_index(['date','ts_code']).target_shares
        joint=pd.concat([base_alloc,other],axis=1).fillna(0)
        changed=joint[joint.iloc[:,0]!=joint.iloc[:,1]]
        debit=base_t.groupby('date').gross.sum().reindex(base_d.index).fillna(0).cumsum()*case['slippage_rate']
        forks.append(dict(name=case['name'],first_changed_target_date=str(changed.index.get_level_values(0).min()),
            direct_extra_cost_on_frozen_orders=debit.iloc[-1],frozen_orders_shadow_min_cash=(base_d.cash-debit).min(),
            interpretation='shadow cash diagnostic only; negative cash means frozen orders infeasible; actual stress reruns change holdings'))
    pd.DataFrame(forks).to_csv(out/'cost_path_diagnostics.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'process_verification.csv',index=False)
    pd.DataFrame(worst).to_csv(out/'worst_days.csv',index=False)
    pd.DataFrame(contributions).to_csv(out/'drawdown_contributions.csv',index=False)
    trainrisk=pd.read_csv(out/'train_risk_events.csv')
    trainrisk.groupby(['rule','event']).forward5.agg(['count','mean','median']).to_csv(out/'train_risk_summary.csv')
    episodes=pd.read_csv(out/'retail_risk_episodes.csv')
    episodes['positive_pair']=episodes.paired_return_difference.gt(0)
    episodes.groupby('rule').agg(episodes=('rule','size'),completed=('unfinished',lambda x:(~x).sum()),
        positive_pairs=('positive_pair','sum'),mean_pair=('paired_return_difference','mean'),
        mean_market_branch_next5=('baseline_next_5d_return','mean'),exit_pending_days=('exit_pending_days','sum')).to_csv(out/'risk_episode_summary.csv')
    cache=Path((out/'execution_cache.txt').read_text().strip());codes=pd.read_csv(out/'execution_codes.csv').ts_code
    paths=[cache/k/f'{code}.parquet' for code in codes for k in ['stk_limit','dividend']]
    paths += [ROOT/'data/raw/stock_basic_all.parquet',ROOT/'data/raw/namechange_all.parquet']
    # 只哈希实际使用的开发行情切片，避免读取冻结TEST。
    market=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',columns=['ts_code','date','open','close','pre_close','adj_factor','vol_share'],
        filters=[('date','>=',pd.Timestamp(cfg['valid'][0])),('date','<=',pd.Timestamp(cfg['valid'][1])),('ts_code','in',codes.tolist())])
    market=market.sort_values(['date','ts_code']).reset_index(drop=True)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes['market_development_rows_sha256']=hashlib.sha256(pd.util.hash_pandas_object(market,index=False).values.tobytes()).hexdigest()
    (out/'execution_data_hashes.json').write_text(json.dumps(hashes,indent=2))
    report=dict(status='complete',cases=25,source_and_signal_hashes_verified=True,pnl_cases_reconciled=25,
        recovery_prefix_checks=4,training_last_label=str(obs.label_end.max()),rolling_training=rolling,
        no_test_market_data=True,fresh_holdout=False,statistical_admission='not_assessed_missing_independent_trials_and_fresh_evidence',
        max_cash_error=float(summary.cash_max_error.max()),max_shares_error=float(summary.shares_max_error.max()),
        execution_data_sufficient=bool(summary.execution_data_sufficient.all()),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    ledger=ExperimentLedger();experiment=str(out.relative_to(ROOT))
    if 'experiment_id' not in ledger.df or not ledger.df.experiment_id.eq(experiment).any():
        for row in summary.to_dict('records'):
            ledger.log('retail_cash_validation','daily_net_equity','VALID 2022-2023 reused development',np.nan,np.nan,np.nan,row['ann'],
                params=row,verdict='RESEARCH_ONLY_NOT_ADMITTED',note='25 scenarios are correlated execution checks, not 25 independent discoveries',
                experiment_id=experiment,source_hashes=manifest['source_hashes'],input_hashes=manifest['input_hashes'],
                selection_note='All preregistered cases retained; recovery hypothesis inspected in TRAIN; no winner promoted')
    (out/'verification_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
