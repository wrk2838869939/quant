# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""独立重放随机实验的划分、训练预处理、已保存模型和选择阈值。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import PROJECT, save_json, digest
from model import TrainScaler, FuzzyNetwork, predict_network, fuzzy_score
from random_split import make_assignments


def main():
    source = Path(sys.argv[1]).resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for path, expected in manifest['artifacts'].items():
        assert digest(source / path) == expected, path
    cfg = yaml.safe_load((source / 'random_config.yaml').read_text())
    features = pd.read_parquet(source / 'causal_features.parquet')
    assignments = pd.read_csv(source / 'split_assignments.csv')
    pd.testing.assert_frame_equal(assignments, make_assignments(features.event_id, cfg['split_seed']), check_dtype=False)
    labels = pd.read_parquet(PROJECT / cfg['returns'] / 'sector_event_outcomes.parquet')
    labels = labels[labels.horizon.eq(cfg['target_horizon'])]
    data = features.merge(labels[['event_id', 'complete', 'end_return']], on='event_id', validate='one_to_one')
    saved_predictions = pd.read_parquet(source / 'all_role_scores_before_evaluation.parquet')
    tests = pd.read_parquet(source / 'test_predictions_before_evaluation.parquet')
    pd.testing.assert_frame_equal(tests, saved_predictions[saved_predictions.role == 'test'].reset_index(drop=True))
    assert tests.drop_duplicates(['round', 'event_id']).event_id.is_unique
    torch.set_num_threads(1)
    replayed = 0
    search = pd.read_csv(source / 'validation_search.csv')
    for fold, assigned in assignments.groupby('round'):
        info = json.loads((source / f'round_{fold}_fit.json').read_text())
        observed = data.complete & data.end_return.notna()
        train_ids = assigned[assigned.role == 'train'].event_id
        val_ids = assigned[assigned.role == 'validation'].event_id
        test_ids = assigned[assigned.role == 'test'].event_id
        train = data[data.event_id.isin(train_ids) & observed]
        valid = data[data.event_id.isin(val_ids) & observed]
        assert set(train.event_id) == set(info['train_ids'])
        assert set(valid.event_id) == set(info['validation_ids'])
        assert set(train_ids).isdisjoint(val_ids) and set(train_ids).isdisjoint(test_ids) and set(val_ids).isdisjoint(test_ids)
        tuning = search[(search['round'] == fold) & (search.model == 'fnn')]
        choices = tuning.groupby(['rules', 'learning_rate']).agg(loss=('validation_loss', 'mean'), epochs=('epochs', 'median')).reset_index()
        best = choices.sort_values(['loss', 'rules', 'learning_rate', 'epochs']).iloc[0]
        assert info['rules'] == best.rules and info['learning_rate'] == best.learning_rate and info['epochs'] == int(best.epochs)
        scaler = TrainScaler(cfg).fit(train[cfg['features']])
        tx, ax = [scaler.transform(f[cfg['features']]) for f in [train, features]]
        train_scores, scores = {}, {}
        tn, an = [], []
        for seed in cfg['fnn']['ensemble_seeds']:
            saved = torch.load(source / 'models' / f'{fold}_fnn_{seed}.pt', map_location='cpu')
            for key in ['median', 'mean', 'scale']:
                np.testing.assert_array_equal(saved[key], getattr(scaler, key))
            assert saved['epochs'] == info['epochs'] and saved['rules'] == info['rules']
            network = FuzzyNetwork(tx, saved['rules'], seed, cfg['fnn'])
            network.load_state_dict(saved['state_dict'])
            tn.append(predict_network(network, tx) * cfg['target_scale'])
            an.append(predict_network(network, ax) * cfg['target_scale'])
        train_scores['fnn'], scores['fnn'] = np.mean(tn, axis=0), np.mean(an, axis=0)
        ridge = np.load(source / 'models' / f'{fold}_ridge.npz')
        for key in ['median', 'mean', 'scale']:
            np.testing.assert_array_equal(ridge[key], getattr(scaler, key))
        train_scores['ridge'] = (tx @ ridge['coef'] + ridge['intercept']) * cfg['target_scale']
        scores['ridge'] = (ax @ ridge['coef'] + ridge['intercept']) * cfg['target_scale']
        medians = train[cfg['features']].median().fillna(0)
        np.testing.assert_array_equal(medians.to_numpy(), pd.Series(info['fuzzy_medians']).loc[cfg['features']].to_numpy())
        train_scores['fuzzy'] = fuzzy_score(train, medians, cfg)
        scores['fuzzy'] = fuzzy_score(features, medians, cfg)
        for kind in scores:
            for retention in cfg['retention_fractions']:
                expected = saved_predictions[(saved_predictions['round'] == fold) & (saved_predictions.model == kind) & (saved_predictions.retention == retention)].set_index('event_id').loc[features.event_id]
                threshold = np.quantile(train_scores[kind], 1 - retention)
                np.testing.assert_allclose(expected.score, scores[kind], atol=1e-7, rtol=1e-6)
                np.testing.assert_allclose(expected.threshold, threshold, atol=1e-7, rtol=1e-6)
                np.testing.assert_array_equal(expected.selected, scores[kind] >= threshold)
                replayed += len(expected)
    out = PROJECT / 'alpha_lab/outputs/sector_fuzzy_random_audit' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__), out / 'audit_random.py')
    result = dict(status='complete', source=str(source), source_manifest_sha256=digest(source / 'manifest.json'),
                  verified_artifacts=len(manifest['artifacts']), replayed_rows=replayed,
                  test_rows=len(tests), test_events=tests.event_id.nunique(),
                  assignments_exact=True, roles_disjoint=True, preprocess_train_only=True,
                  validation_chosen_parameters_exact=True, selected_flags_exact=True)
    save_json(out / 'audit.json', result)
    print(json.dumps(dict(output=str(out), **result), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
