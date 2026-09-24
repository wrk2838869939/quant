# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""重放已保存模型与训练阈值，核对每年标签成熟和输入边界。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import sys
import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(PROJECT/'alpha_lab/src'))
from datahub import DataHub
from sector_dataset import digest
from model import TrainScaler, FuzzyNetwork, predict_network, fuzzy_score, purged_history


def main():
    source = Path(sys.argv[1]).resolve()
    cfg = yaml.safe_load((source/'config.yaml').read_text())
    manifest = json.loads((source/'manifest.json').read_text())
    assert manifest['status']=='complete'
    for name,h in manifest['artifacts'].items():
        assert digest(source/name)==h,name
    torch.set_num_threads(1)
    feature = pd.read_parquet(source/'causal_features.parquet')
    labels = pd.read_parquet(PROJECT/cfg['returns']/'sector_event_outcomes.parquet')
    target = labels[labels.horizon.eq(cfg['target_horizon'])][['event_id','end_date','end_return','complete']].rename(columns={'end_return':'target'})
    data=feature.merge(target,on='event_id',validate='one_to_one')
    predictions=pd.read_parquet(source/'predictions_before_labels.parquet')
    calendar=DataHub().calendar.cn_days(data.date.min()-pd.DateOffset(years=cfg['training_years']),data.date.max())
    checked=0
    for year in sorted(predictions.year.unique()):
        indices=json.loads((source/f'fold_{year}_indices.json').read_text())
        train=data.set_index('event_id').loc[indices['train']].reset_index()
        test=data.set_index('event_id').loc[indices['test']].reset_index()
        cutoff=pd.Timestamp(indices['outer_cutoff'])
        assert train.end_date.lt(cutoff).all()
        fit=data.set_index('event_id').loc[indices['fit']]
        assert fit.end_date.lt(pd.Timestamp(indices['inner_cutoff'])).all()
        assert set(indices['fit']).isdisjoint(indices['valid'])
        assert set(indices['train']).isdisjoint(indices['test'])
        scaler=TrainScaler(cfg).fit(train[cfg['features']])
        tx,ox=scaler.transform(train[cfg['features']]),scaler.transform(test[cfg['features']])
        ridge=np.load(source/'models'/f'{year}_ridge.npz')
        for key in ['median','mean','scale']:
            np.testing.assert_array_equal(getattr(scaler,key),ridge[key])
        train_scores={'ridge':(tx@ridge['coef']+ridge['intercept'])*cfg['target_scale']}
        test_scores={'ridge':(ox@ridge['coef']+ridge['intercept'])*cfg['target_scale']}
        medians=train[cfg['features']].median().fillna(0)
        train_scores['fuzzy']=fuzzy_score(train,medians,cfg)
        test_scores['fuzzy']=fuzzy_score(test,medians,cfg)
        train_fnn,test_fnn=[],[]
        for seed in cfg['fnn']['ensemble_seeds']:
            saved=torch.load(source/'models'/f'{year}_fnn_{seed}.pt',map_location='cpu')
            for key in ['median','mean','scale']:
                np.testing.assert_array_equal(getattr(scaler,key),saved[key])
            model=FuzzyNetwork(tx,saved['rules'],seed,cfg['fnn'])
            model.load_state_dict(saved['state_dict'])
            pred=predict_network(model,ox)
            np.testing.assert_allclose(pred[:1],predict_network(model,ox[:1]),atol=1e-6,rtol=1e-6)
            train_fnn.append(predict_network(model,tx)*cfg['target_scale'])
            test_fnn.append(pred*cfg['target_scale'])
        train_scores['fnn']=np.mean(train_fnn,axis=0)
        test_scores['fnn']=np.mean(test_fnn,axis=0)
        for kind in train_scores:
            for retention in cfg['retention_fractions']:
                expected=predictions[(predictions.year==year)&(predictions.model==kind)&(predictions.retention==retention)].set_index('event_id').loc[test.event_id]
                threshold=float(np.quantile(train_scores[kind],1-retention))
                np.testing.assert_allclose(expected.score,test_scores[kind],atol=1e-7,rtol=1e-6)
                np.testing.assert_allclose(expected.threshold,threshold,atol=1e-7,rtol=1e-6)
                np.testing.assert_array_equal(expected.selected,test_scores[kind]>=threshold)
                checked+=len(expected)
        # 改动测试年及之后收益，不得改变这一年的训练输入。
        changed=data.copy()
        changed.loc[changed.date>=pd.Timestamp(f'{year}-01-01'),'target']=999.0
        before,_=purged_history(data,pd.Timestamp(f'{year}-01-01'),calendar,cfg)
        after,_=purged_history(changed,pd.Timestamp(f'{year}-01-01'),calendar,cfg)
        pd.testing.assert_frame_equal(before,after)
    out=PROJECT/'alpha_lab/outputs/sector_fuzzy_audit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    for path in [Path(__file__),PROJECT/'alpha_lab/src/sector_dataset.py',PROJECT/'alpha_lab/src/sector_candidate_quality.py']:
        shutil.copy2(path,out/path.name)
    result=dict(status='complete',experiment=str(source),experiment_manifest_sha256=digest(source/'manifest.json'),
        prediction_rows_replayed=checked,saved_scalers_train_only=True,selected_flags_exact=True,
        mature_labels_and_disjoint_indices=True,future_target_shock_train_unchanged=True,pointwise_predictions=True,
        runtime=dict(torch=torch.__version__,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__),
        artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),**result),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
