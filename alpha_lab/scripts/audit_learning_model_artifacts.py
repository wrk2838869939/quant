#!/usr/bin/env python3
"""逐模型分支抽查归档加载：只读取已完成预测，不参与训练或参数选择。"""
import argparse
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args()
    run=args.run
    plan=yaml.safe_load((run/'sources/learning_strategy_search_v1.yaml').read_text())
    features=pd.read_parquet(run/'inputs/learning_features.parquet')
    eligible=pd.read_parquet(run/'inputs/eligible.parquet')
    checked=set();rows=[]
    for path in sorted(run.glob('predictions/*/*/*/fits.json')):
        records=[x for x in json.loads(path.read_text()) if x['status']=='ok']
        if not records:
            continue
        fit=records[0]
        params=fit['params']
        branch=(params['family'],params['variant'])
        if branch in checked:
            continue
        date=pd.Timestamp(fit['fit_date'])
        raw=features.xs(date,level='date').reindex(index=eligible.columns,columns=list(plan['features'])).to_numpy()
        missing=~np.isfinite(raw)
        valid=eligible.loc[date].to_numpy() & ((~missing).sum(axis=1)>=plan['training']['min_features'])
        ranks=np.nan_to_num(raw).astype(np.float32)
        x=np.concatenate([ranks,missing.astype(np.float32)],axis=1)
        saved=joblib.load(path.parent/'models'/fit['model_file'])
        if saved['weights'] is not None:
            pred=ranks[valid]@saved['weights']
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='X does not have valid feature names')
                pred=saved['model'].predict(x[valid])
        expected=pd.read_parquet(path.parent/'return_predictions.parquet').loc[date].to_numpy()[valid]
        np.testing.assert_allclose(pred,expected,rtol=1e-12,atol=1e-12)
        risk_error=None
        if saved['risk_model'] is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='X does not have valid feature names')
                risk=np.maximum(0.,saved['risk_model'].predict(x[valid]))
            expected_risk=pd.read_parquet(path.parent/'risk_predictions.parquet').loc[date].to_numpy()[valid]
            np.testing.assert_allclose(risk,expected_risk,rtol=1e-12,atol=1e-12)
            risk_error=float(np.max(np.abs(risk-expected_risk)))
        checked.add(branch)
        rows.append(dict(family=branch[0],variant=branch[1],fit_date=fit['fit_date'],
                         model_file=str((path.parent/'models'/fit['model_file']).relative_to(run)),
                         stocks=int(valid.sum()),return_max_error=float(np.max(np.abs(pred-expected))),
                         risk_max_error=risk_error,passed=True))
    expected={(family,variant) for family,variants in plan['families'].items() for variant in variants}
    assert checked==expected,(checked,expected)
    pd.DataFrame(rows).to_csv(run/'model_roundtrip_checks.csv',index=False)
    print(json.dumps(rows,ensure_ascii=False))


if __name__=='__main__':
    main()
