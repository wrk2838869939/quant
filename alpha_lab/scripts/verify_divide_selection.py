#!/usr/bin/env python3
"""独立复算分治模型正规方程、成熟边界与同日组间/组内收益归因。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from divide_selection import phase_features,policy_predictions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    out=args.run;cfg=yaml.safe_load((out/'config.yaml').read_text())
    panel=pd.read_parquet(ROOT/cfg['source_panel'])
    assert panel.date.max()<pd.Timestamp('2022-01-01')
    train=panel[panel.monthly&panel.trend].copy()
    expanded,_=phase_features(train,cfg['features'])
    checks=[];composition=[]
    for stage,filename,cut in [('development','development_models.joblib',cfg['fit_end']),('execution','execution_models.joblib',cfg['final_fit_end'])]:
        models=joblib.load(out/filename);cut=pd.Timestamp(cut)
        fit=expanded[expanded.label_end_60.le(cut)&expanded.entry_missing_60.eq(False)&expanded.ret_60.notna()&expanded.mae_60.notna()].copy()
        pairs=[('global',models.global_model,fit),('shared',models.shared_model,fit),('constant',models.constant_model,fit)]
        pairs.extend((phase,model,fit[fit.phase.eq(phase)]) for phase,model in models.local_models.items())
        for name,model,data in pairs:
            x=np.column_stack([np.ones(len(data)),data[list(model.feature_cols)].to_numpy()])
            y=data[['ret_60','mae_60']].to_numpy().copy();y[:,1]*=-1
            w=(1/data.groupby('date').date.transform('size')).to_numpy()
            penalty=np.eye(x.shape[1])*cfg['forecast']['alpha'];penalty[0,0]=0.
            coef=np.linalg.solve(x.T@(x*w[:,None])+penalty,x.T@(y*w[:,None]))
            error=float(np.max(np.abs(coef-model.coefficients.to_numpy())))
            assert error<1e-11 and model.last_label<=cut
            assert model.metadata['exclusion_counts']['fitted_rows']==len(data)
            checks.append(dict(stage=stage,model=name,rows=len(data),dates=data.date.nunique(),coefficient_max_error=error,last_label=str(model.last_label)))
        overall=fit.groupby('date').ret_60.mean()
        for phase,g in fit.groupby('phase'):
            dates=g.groupby('date').ret_60.mean()
            common=overall.reindex(dates.index)
            composition.append(dict(stage=stage,phase=phase,dates=len(dates),date_equal_mean=float(dates.mean()),
                common_date_pool_return=float(common.mean()),date_relative_mean=float((dates-common).mean()),
                missing_training_dates=int(len(overall)-len(dates)),
                single_stock_mean=float(g.ret_60.mean())))
    pd.DataFrame(composition).to_csv(out/'independent_training_phase_composition.csv',index=False)
    predictions=pd.read_parquet(out/'development_predictions.parquet')
    assert predictions.date.max()<pd.Timestamp('2022-01-01')
    errors=pd.read_csv(out/'error_dates.csv',parse_dates=['date']).set_index(['date','model'])
    attribution=[];phase_rows=[];mse_max_error=0.
    for date,g in predictions.groupby('date'):
        pool=g.ret_60.mean();phase_mean=g.groupby('phase').ret_60.mean()
        for name in ['constant','global_return','phase_experts','shared_experts']:
            residual=g[name+'__expected_return']-g.ret_60
            total=(residual**2).mean();time=residual.mean()**2;within=((residual-residual.mean())**2).mean()
            expected=errors.loc[(date,name),['mse','time_mse','within_mse']].to_numpy(float)
            error=float(np.max(np.abs(np.array([total,time,within])-expected)))
            mse_max_error=max(mse_max_error,error);assert error<1e-12
            for phase,c in g.groupby('phase'):
                p=c[name+'__expected_return'];actual=c.ret_60
                phase_rows.append(dict(date=date,year=date.year,phase=phase,model=name,n=len(c),
                    relative_mean_prediction=float(p.mean()-g[name+'__expected_return'].mean()),
                    relative_realized=float(actual.mean()-pool),
                    within_phase_rank_ic=p.corr(actual,method='spearman') if p.nunique()>1 else np.nan,
                    within_phase_mse=float(((p-p.mean()-(actual-actual.mean()))**2).mean())))
        for policy in cfg['policies']:
            col=policy+'__score';eligible=g[g[col].notna()]
            selected=eligible.sort_values([col,'ts_code'],ascending=[False,True]).head(cfg['retail']['max_positions'])
            shuffled=eligible.sample(frac=1,random_state=17).sort_values([col,'ts_code'],ascending=[False,True]).head(cfg['retail']['max_positions'])
            assert selected.ts_code.tolist()==shuffled.ts_code.tolist()
            if len(selected):
                own=selected.phase.map(phase_mean);between=float((own-pool).mean());within=float((selected.ret_60-own).mean());excess=float(selected.ret_60.mean()-pool)
                assert abs(between+within-excess)<1e-12
            else:between=within=excess=np.nan
            attribution.append(dict(date=date,year=date.year,policy=policy,n=len(selected),selected_excess=excess,
                between_phase_component=between,within_phase_component=within))
    at=pd.DataFrame(attribution);at.to_csv(out/'independent_selection_attribution_dates.csv',index=False)
    cols=['n','selected_excess','between_phase_component','within_phase_component']
    at.groupby('policy')[cols].mean().to_csv(out/'independent_selection_attribution_summary.csv')
    at.groupby(['year','policy'])[cols].mean().to_csv(out/'independent_selection_attribution_yearly.csv')
    ph=pd.DataFrame(phase_rows);ph.to_csv(out/'independent_phase_model_dates.csv',index=False)
    ph.groupby(['model','phase'])[['relative_mean_prediction','relative_realized','within_phase_rank_ic','within_phase_mse']].mean().to_csv(out/'independent_phase_model_summary.csv')
    model=joblib.load(out/'development_models.joblib')
    evaluation=panel[panel.monthly&panel.trend&panel.date.between(*map(pd.Timestamp,cfg['development']))&panel.label_end_60.notna()].copy()
    fresh=policy_predictions(evaluation.sample(frac=1,random_state=17),model,cfg).reindex(evaluation.index).reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh,predictions[fresh.columns].reset_index(drop=True))
    report=dict(status='passed',coefficient_checks=checks,mse_max_error=mse_max_error,
        saved_prediction_reproduction=True,selection_permutation_invariant=True,test_market_data_used=False,
        limitations=['Local models normalize dates within each phase; sparse phase dates are not the same population as global dates.',
            'Shared interactions use one reference phase and common Ridge alpha, not exchangeable Bayesian expert priors.',
            'Labels are marked paths and overlap; phase attribution is descriptive, not causal or fresh holdout evidence.',
            'Point return forecasts remain less calibrated than the date-constant control on inspected development data.'],
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'independent_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(at.groupby('policy')[cols].mean().to_string())
    print('Independent checks passed',len(checks),'model equations; MSE max error',mse_max_error)


if __name__=='__main__':main()
