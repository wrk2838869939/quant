"""新增模型的时序隔离、联合标签成熟和保存重放。"""
from copy import deepcopy
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_rule_search import synthetic
from learning_search import build_features, build_labels, prepare_arrays, sample_parameters
from learning_extensions import attach_context, predict_walk_forward, infer_bundle, fit_bundle, score_predictions, positive_probability

PLAN = yaml.safe_load((ROOT / 'configs/learning_extensions_v1.yaml').read_text())
BRANCHES = [(f, v) for f, variants in PLAN['families'].items() for v in variants]


@pytest.fixture(scope='module')
def prepared():
    wide, eligible, calendar = synthetic()
    cfg = dict(PLAN['training'], min_dates=30, min_rows=80, filter_min_rows=50,
               min_features=3, min_ic_stocks=3, expert_min_rows=30, neural_batch_size=64,
               host_candidate_count=2, correlation_min_days=20)
    features = build_features(wide, eligible, calendar, PLAN['features'])
    data = prepare_arrays(features, eligible, build_labels(wide, eligible.index, [5,10,20]), cfg)
    attach_context(data, wide, eligible)
    return data, cfg


@pytest.mark.parametrize('family,variant', BRANCHES)
def test_future_poison_and_model_roundtrip(prepared, family, variant, tmp_path):
    data, cfg = prepared
    p = dict(family=family, variant=variant, **{k:v[0] for k,v in {**PLAN['common_axes'], **PLAN['families'][family][variant]}.items()})
    if family == 'correlation_experts':
        p['clusters'] = 2
    cut, start = data['dates'][160], data['dates'][100]
    a, ar, fits = predict_walk_forward(data, p, cfg, 19, start, cut, tmp_path)
    poisoned = deepcopy(data)
    for key in ['x','ranks','context','stock_returns']:
        poisoned[key][161:] = 999
    for label in poisoned['labels'].values():
        unknown = pd.isna(label['end']) | (label['end'] >= np.datetime64(cut))
        for key in ['returns','risk','ic']:
            label[key][unknown] = 999
    b, br, _ = predict_walk_forward(poisoned, p, cfg, 19, start, cut)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ar, br)
    assert a.notna().any().any()
    for record in fits:
        if record['status'] != 'ok':
            continue
        assert pd.Timestamp(record['max_label_end']) < pd.Timestamp(record['fit_date'])
        bundle = joblib.load(tmp_path / record['model_file'])
        date = pd.Timestamp(record['fit_date'])
        x, r = infer_bundle(bundle, data, data['dates'].get_loc(date))
        np.testing.assert_array_equal(a.loc[date], x)
        np.testing.assert_array_equal(ar.loc[date], r)
    if family == 'trade_filter':
        assert a.notna().sum(axis=1).max() <= cfg['host_candidate_count']


def test_joint_targets_wait_for_longest_label(prepared):
    data, cfg = prepared
    p = dict(family='multi_horizon',variant='relative',horizon=5,train_window=252,alpha=1.,blend='equal')
    _, record = fit_bundle(data,p,cfg,19,100)
    assert record['last_signal_date'] == str(data['dates'][79].date())


def test_gate_direction_and_single_class_probability():
    from sklearn.tree import DecisionTreeClassifier
    pred = pd.DataFrame([[.1,.2]])
    prob = pd.DataFrame([[.2,.8]])
    assert score_predictions(pred,prob,dict(family='trade_filter',probability_cut=.5)).notna().values.tolist() == [[False,True]]
    assert score_predictions(pred,prob,dict(family='market_gate',probability_cut=.5)).notna().values.tolist() == [[True,False]]
    for value in [0,1]:
        model = DecisionTreeClassifier().fit([[0],[1]],[value,value])
        assert positive_probability(model,[[2]])[0] == value


def test_local_and_global_sampling_each_family():
    for family in PLAN['families']:
        points = sample_parameters(PLAN,family,16,23)
        assert len(points) == 16
        assert {p['variant'] for p in points} == set(PLAN['families'][family])
        local = sample_parameters(PLAN,family,4,24,centers=points[:4])
        assert len(local) == 4
