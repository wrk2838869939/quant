"""独立由保存参数重建概率、原始指数价格重建目标及历史边界。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from forward_risk_evidence import FAMILIES,matured_weekly_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    run=p.parse_args().run;cfg=json.loads((run/'protocol.json').read_text())
    features=pd.read_parquet(run/'features.parquet');labels=pd.read_parquet(run/'labels.parquet')
    prices=pd.read_parquet(run/'index.parquet');pred=pd.read_parquet(run/'predictions.parquet')
    params=pd.read_csv(run/'coefficients.csv');train=matured_weekly_rows(features,labels,*cfg['splits']['train'])
    known=features[FAMILIES['joint']].notna().all(axis=1)
    count=0
    for name,columns in {**FAMILIES,'utility':FAMILIES['joint']}.items():
        rows=params.loc[params.model.eq(name)].set_index('feature').loc[columns]
        np.testing.assert_allclose(rows['mean'],features.loc[train,columns].mean(),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(rows['scale'],features.loc[train,columns].std(ddof=0),atol=1e-12,rtol=1e-12)
        z=features.loc[known,columns].subtract(rows['mean']).divide(rows['scale'])
        linear=z.to_numpy(dtype=float)@rows.coefficient.to_numpy()+rows.intercept.iloc[0]
        expected=linear if name=='utility' else expit(linear)
        np.testing.assert_allclose(expected,pred.loc[known,'predicted_terminal' if name=='utility' else name],atol=1e-12,rtol=1e-12)
        count+=1
    checked=0
    for split,bounds in cfg['splits'].items():
        dates=matured_weekly_rows(features,labels,*bounds)
        assert labels.loc[dates,'label_end'].le(pd.Timestamp(bounds[1])).all()
        for date in dates:
            i=prices.index.get_loc(date);entry=prices.open.iloc[i+1]
            worst=min(prices.close.iloc[i+1:i+21]/entry-1)
            terminal=prices.open.iloc[i+21]/entry-1
            assert np.isclose(worst,labels.at[date,'worst']) and np.isclose(terminal,labels.at[date,'terminal'])
            assert float(worst<=-.05)==labels.at[date,'loss_event']
            assert labels.at[date,'label_end']==prices.index[i+21]
            checked+=1
    original=ROOT/'outputs/forward_risk/20260914T103508_365515Z'
    prior_params=pd.read_csv(original/'coefficients.csv')
    pd.testing.assert_frame_equal(params,prior_params,check_exact=True)
    first=pd.Timestamp('2022-01-04')
    assert pred.loc[first,list(FAMILIES)].notna().all(),'2022年初概率必须有真实证据'
    result=dict(parameter_sets_reconstructed=count,weekly_labels_reconstructed=checked,
        all_targets_matured=True,training_coefficients_unchanged_after_2021_repair=True,
        early_2022_prediction_restored=True,independent_agent_review_this_turn=False)
    (run/'independent_formula_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
