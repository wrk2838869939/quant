#!/usr/bin/env python3
"""固定选择策略及分治对照，先登记、再读取开发标签，冻结执行模型。"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys
import joblib
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from divide_selection import fit_divided,phase_features,policy_predictions
from run_value_iterations import prediction_parts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path);args=ap.parse_args()
    cp=ROOT/'configs/divide_selection.yaml';cfg=yaml.safe_load(cp.read_text())
    out=args.run or ROOT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False);Path('/tmp/divide_selection_run_path').write_text(str(out))
    (out/'config.yaml').write_bytes(cp.read_bytes())
    manifest=dict(status='registered',fresh_holdout=False,test_market_data_used=False,policies=cfg['policies'],
        design='three prior-return phases; global vs independent vs shared experts; no group quotas; selection precedes sizing',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,Path(__file__),ROOT/'src/divide_selection.py',ROOT/'src/value_forecast.py']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out,flush=True)
    panel=pd.read_parquet(ROOT/cfg['source_panel']);assert panel.date.max()<pd.Timestamp('2022-01-01')
    train=panel[panel.monthly&panel.trend].copy()
    model=fit_divided(train,cfg,cfg['fit_end']);joblib.dump(model,out/'development_models.joblib')
    final=fit_divided(train,cfg,cfg['final_fit_end']);joblib.dump(final,out/'execution_models.joblib')
    (out/'model_metadata.json').write_text(json.dumps(dict(development=model.metadata,execution=final.metadata),indent=2))
    # 执行特征来自上一轮逐行复核过的全历史PIT计算，不接触TEST。
    features=pd.read_parquet(ROOT/cfg['execution_parent']/'pit_review_features.parquet')
    features=features[features.monthly].copy();assert features.date.max()<pd.Timestamp('2024-01-01')
    ep=policy_predictions(features,final,cfg);ep.to_parquet(out/'execution_predictions.parquet',index=False)
    print('execution predictions frozen',flush=True)
    evaluation=panel[panel.monthly&panel.trend&panel.date.between(*map(pd.Timestamp,cfg['development']))&panel.label_end_60.notna()].copy()
    pred=policy_predictions(evaluation,model,cfg)
    cols=['ret_60','mae_60','entry_missing_60','label_end_60']
    for col in cols:pred[col]=evaluation[col]
    pred.to_parquet(out/'development_predictions.parquet',index=False)
    rows=[];errors=[];phase_rows=[];selected=[]
    for date,g in pred.groupby('date'):
        for name in ['constant','global_return','phase_experts','shared_experts']:
            p=g[name+'__expected_return'];total,time,within=prediction_parts(g.ret_60,p)
            errors.append(dict(date=date,year=date.year,model=name,mse=total,time_mse=time,within_mse=within,
                rank_ic=p.corr(g.ret_60,method='spearman') if p.nunique()>1 else np.nan,mean_prediction=p.mean(),realized=g.ret_60.mean()))
        for phase,c in g.groupby('phase'):
            phase_rows.append(dict(date=date,phase=phase,n=len(c),ret=c.ret_60.mean(),relative_ret=c.ret_60.mean()-g.ret_60.mean(),
                global_prediction=c.global_return__expected_return.mean(),local_prediction=c.phase_experts__expected_return.mean(),shared_prediction=c.shared_experts__expected_return.mean()))
        for policy in cfg['policies']:
            score=policy+'__score';pool=g[g[score].notna()];chosen=pool.sort_values([score,'ts_code'],ascending=[False,True]).head(cfg['retail']['max_positions'])
            rows.append(dict(date=date,year=date.year,policy=policy,n=len(chosen),admitted=len(pool),
                opportunity_return=chosen.ret_60.mean(),cash_slot_return=chosen.ret_60.sum()/cfg['retail']['max_positions'],
                mae=chosen.mae_60.mean(),predicted=chosen[policy+'__expected_return'].mean(),
                pool_return=g.ret_60.mean(),groups=chosen.phase.nunique()))
            for _,stock in chosen.iterrows():selected.append(dict(date=date,policy=policy,ts_code=stock.ts_code,phase=stock.phase,score=stock[score],ret=stock.ret_60,mae=stock.mae_60))
    for name,records,key in [('entry',rows,'policy'),('error',errors,'model'),('phase',phase_rows,'phase')]:
        table=pd.DataFrame(records);table.to_csv(out/f'{name}_dates.csv',index=False)
        numeric=table.select_dtypes(include=np.number).columns.difference(['year']);table.groupby(key)[numeric].mean().to_csv(out/f'{name}_summary.csv')
        if 'year' in table:table.groupby(['year',key])[numeric].mean().to_csv(out/f'{name}_yearly.csv')
    pd.DataFrame(selected).to_csv(out/'selected_opportunities.csv',index=False)
    # 固定日期内重排行不应改变分组、模型预测或明确选股名单。
    shuffled=evaluation.sample(frac=1,random_state=cfg['forecast']['seed'])
    changed=policy_predictions(shuffled,model,cfg).reindex(pred.index)
    pd.testing.assert_frame_equal(changed,pred.drop(columns=cols))
    manifest.update(status='models_and_diagnostics_complete',evaluation_dates=pred.date.nunique(),input_row_permutation_invariant=True,
        note='overlapping 60d entry opportunities; not an executed portfolio; no new holdout')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(rows).groupby('policy')[['cash_slot_return','mae','admitted']].mean().to_string(),flush=True)


if __name__=='__main__':main()
