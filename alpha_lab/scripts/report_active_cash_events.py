#!/usr/bin/env python3
"""在冻结回放后检查警报、压力日与实际敞口，不重选参数。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);args=ap.parse_args()
    out=args.prepared
    spec=yaml.safe_load((out/'experiment.yaml').read_text())
    baseline=pd.read_csv(out/'baseline_daily.csv',index_col='date',parse_dates=True)
    features=pd.read_parquet(out/'market_features.parquet')
    rows=[];stress=[]
    worst=baseline.nsmallest(10,'ret').index
    for rule in spec['rules']:
        d=pd.read_csv(out/f'{rule}_daily.csv',index_col='date',parse_dates=True)
        plan=pd.read_csv(out/f'{rule}_plan.csv',index_col='date',parse_dates=True)
        try:
            events=pd.read_csv(out/f'{rule}_risk_episodes.csv')
        except pd.errors.EmptyDataError:
            events=pd.DataFrame()
        report=dict(rule=rule,requests=len(events))
        for h in spec['event_horizons']:
            col=f'baseline_next_{h}d_return'
            values=events[col].dropna() if len(events) else pd.Series(dtype=float)
            report[f'next_{h}d_down_events']=int((values<0).sum())
            report[f'next_{h}d_observed_events']=len(values)
            report[f'next_{h}d_mean_return']=float(values.mean())
        report['mean_pre_signal_5d_return']=float(events.loss_before_signal_5d.mean()) if len(events) else np.nan
        complete=events[~events.unfinished] if len(events) else events
        report['completed_events']=len(complete)
        report['positive_paired_events']=int((complete.paired_return_difference>0).sum()) if len(complete) else 0
        report['mean_paired_return_difference']=float(complete.paired_return_difference.mean()) if len(complete) else np.nan
        report['max_cash_episode_days']=float(events.cash_days_through_reentry.max()) if len(events) else 0
        report['worst10_baseline_days_cash']=int(d.loc[worst,'n_hold'].eq(0).sum())
        report['worst10_baseline_days_requested_cash']=int(d.loc[worst,'risk_target'].eq(0).sum())
        rows.append(report)
        # 日末敞口不是当日开盘前暴露；两者都保留，避免先亏后卖被当成提前规避。
        prior_exposure=(d.stock_value/d.equity).shift(1).fillna(0)
        for date in worst:
            stress.append(dict(rule=rule,date=date,baseline_return=baseline.loc[date,'ret'],
                risk_return=d.loc[date,'ret'],risk_target=d.loc[date,'risk_target'],
                risk_state=d.loc[date,'risk_state'],previous_close_exposure=prior_exposure.loc[date],
                closing_exposure=d.loc[date,'stock_value']/d.loc[date,'equity'],holdings=d.loc[date,'n_hold']))
        joined=d.join(plan.add_prefix('decision_')).join(features.add_prefix('market_'))
        joined.to_csv(out/f'{rule}_decision_execution_states.csv')
    pd.DataFrame(rows).to_csv(out/'event_diagnostics.csv',index=False)
    pd.DataFrame(stress).to_csv(out/'stress_days.csv',index=False)
    summary=pd.read_csv(out/'summary.csv').set_index('strategy')
    fee_rows=[]
    for rule in spec['rules']:
        gap=(summary.loc['baseline','total']-summary.loc[rule,'total'])*summary.loc['baseline','capital']
        extra=summary.loc[rule,'total_fees']-summary.loc['baseline','total_fees']
        fee_rows.append(dict(rule=rule,terminal_profit_gap_yuan=gap,extra_recorded_fees_yuan=extra,
                            extra_fees_fraction_of_gap=extra/gap if gap else np.nan))
    pd.DataFrame(fee_rows).to_csv(out/'fee_gap_decomposition.csv',index=False)
    base5=baseline.equity.shift(-4)/baseline.equity.shift(1)-1
    valid_features=features.loc[baseline.index]
    notes=dict(baseline_next5_mean=float(base5.mean()),
        baseline_next5_negative_fraction=float((base5.dropna()<0).mean()),
        legacy_valid_median_min=float(valid_features.median_return.min()),
        legacy_valid_days_below_threshold=int((valid_features.median_return<spec['legacy_median_exit']).sum()),
        fixed_half_control_valid_for_half_exposure_comparison=False,
        reason='旧分席整手分配使资金减半后实际平均股票敞口从45.91%变8.38%，不是原敞口的一半',
        fee_decomposition='账面恒等式，不是无费用重新交易的反事实；数量与资金路径仍会受费用影响')
    (out/'mechanism_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    (out/'event_diagnostics_notes.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='worst baseline days are ex-post diagnostic slices, not signal inputs',
        counterfactual='independent full portfolio branches may have different holdings; paired event differences are descriptive',
        independence='overlapping forward windows are not independent observations',
        zero_requests='does not imply risk prediction success'),indent=2))


if __name__=='__main__':main()
