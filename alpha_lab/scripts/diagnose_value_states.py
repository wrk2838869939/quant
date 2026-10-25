#!/usr/bin/env python3
"""事后误差/尾部/入场状态归因；不生成或选择新策略。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);cfg=yaml.safe_load((out/'config.yaml').read_text())
    pred=pd.read_parquet(out/'development_predictions.parquet')
    cols=['date','ts_code','mom60','mom120','continuity','jump_share','extension','volatility','log_size']
    raw=pd.read_parquet(ROOT/cfg['source_panel'],columns=cols)
    pred=pred.merge(raw,on=['date','ts_code'],validate='one_to_one')
    bins=[]; states=[];tails=[];date_state=[]
    for date,g in pred[pred.monthly].groupby('date'):
        g=g.copy()
        g['rank_bin']=pd.qcut(g.full__expected_return.rank(method='first'),10,labels=False)
        for band,c in g.groupby('rank_bin'):
            bins.append(dict(date=date,bin=int(band),n=len(c),prediction=c.full__expected_return.mean(),
                realized=c.ret_60.mean(),median=c.ret_60.median(),win=c.ret_60.gt(0).mean(),
                mae=c.mae_60.mean(),mean_lower=c.full__mean_lower.mean()))
        confidence=(g.full__mean_lower>cfg['entry']['mean_lower_buffer'])&(g.full__expected_adverse<=cfg['entry']['max_expected_adverse'])
        groups={'pool':g,'full_top5':g.nlargest(5,'full__expected_return'),
                'confidence_top5':g[confidence].nlargest(5,'full__expected_return'),
                'low_turnover':g.nsmallest(5,'turnover')}
        for policy,c in groups.items():
            row=dict(date=date,policy=policy,n=len(c),ret=c.ret_60.mean(),mae=c.mae_60.mean())
            row.update({col:c[col].median() for col in cols[2:]+['turnover']})
            row['prediction']=c.full__expected_return.mean()
            row['future_top5_fraction']=c.ret_60.ge(g.ret_60.quantile(.95)).mean()
            states.append(row)
        # 未来尾部分层只用于解释误差，不用于筛股。
        cut_low,cut_high=g.ret_60.quantile([.05,.95])
        masks={'bottom5':g.ret_60<=cut_low,'middle90':g.ret_60.between(cut_low,cut_high,inclusive='neither'),
               'top5':g.ret_60>=cut_high}
        for label,mask in masks.items():
            c=g[mask]
            tails.append(dict(date=date,outcome_tail=label,n=len(c),return_contribution=c.ret_60.sum()/len(g),
                full_mse_contribution=((c.full__expected_return-c.ret_60)**2).sum()/len(g),
                constant_mse_contribution=((c.constant__expected_return-c.ret_60)**2).sum()/len(g),
                full_mean_prediction=c.full__expected_return.mean(),realized=c.ret_60.mean()))
        date_state.append(dict(date=date,n=len(g),pool_return=g.ret_60.mean(),
            model_date_mean=g.full__expected_return.mean(),confidence_fraction=confidence.mean(),
            momentum=g.mom60.median(),volatility=g.volatility.median()))
    for name,records,key in [('rank_calibration',bins,'bin'),('selected_states',states,'policy'),('outcome_tail_error',tails,'outcome_tail')]:
        table=pd.DataFrame(records);table.to_csv(out/(name+'_dates.csv'),index=False)
        table.groupby(key).mean(numeric_only=True).to_csv(out/(name+'_summary.csv'))
    dates=pd.DataFrame(date_state);dates.to_csv(out/'date_state_diagnostics.csv',index=False)
    dates.drop(columns='date').corr().to_csv(out/'date_state_correlations.csv')
    # 重采样重叠的日期块，展示同窗口差异的不确定性；不是新样本显著性检验。
    errors=pd.read_csv(out/'error_dates.csv');errors=errors[errors.frequency.eq('monthly')]
    mse=errors.pivot(index='date',columns='model',values='mse')
    paths=pd.read_csv(out/'entry_path_dates.csv');paths=paths[paths.frequency.eq('monthly')]
    returns=paths.pivot(index='date',columns='policy',values='cash_slot_return')
    differences={'full_minus_constant_mse':mse.full-mse.constant,
                 'confidence_minus_low_turnover_return':returns.full_confidence-returns.low_turnover,
                 'confidence_minus_point_return':returns.full_confidence-returns.full_point}
    uncertainty=[];rng=np.random.default_rng(cfg['forecast']['seed']);block=cfg['forecast']['block_dates']
    for name,values in differences.items():
        a=values.to_numpy();reps=[]
        for _ in range(1000):
            starts=rng.integers(0,len(a)-block+1,size=(len(a)+block-1)//block)
            ix=(starts[:,None]+np.arange(block)).ravel()[:len(a)]
            reps.append(a[ix].mean())
        uncertainty.append(dict(comparison=name,date_mean=a.mean(),q10=np.quantile(reps,.1),
            q90=np.quantile(reps,.9),n_dates=len(a),block_months=block,repetitions=1000))
    pd.DataFrame(uncertainty).to_csv(out/'paired_block_diagnostics.csv',index=False)
    (out/'process_addendum.json').write_text(json.dumps(dict(posthoc=True,strategy_changed=False,
        purpose='explain return calibration, tails and selected past states; descriptive only',
        future_outcomes_only_used_for_attribution=True),indent=2))
    print(pd.DataFrame(bins).groupby('bin')[['prediction','realized','win','mae']].mean().to_string())


if __name__=='__main__':main()
