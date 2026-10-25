"""监督学习的标签成熟、未来扰动、模型分支及联合采样测试。"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from learning_search import build_labels, build_features, prepare_arrays, training_mask, predict_walk_forward, sample_parameters, score_predictions
from rule_search import config_id
from test_rule_search import synthetic


def test_next_open_label_maturity_and_missing_path():
    dates = pd.bdate_range('2020-01-01', periods=8)
    close = pd.DataFrame({'A': [10., 11., 12., 13., 14., 15., 16., 17.]}, index=dates)
    wide = dict(hfq_close=close, close=close, open=close - .5, hfq_low=close - 1)
    labels = build_labels(wide, dates, [3])[3]
    assert labels['returns'].iloc[0, 0] == pytest.approx(13 / 10.5 - 1)
    assert labels['risk'].iloc[0, 0] == pytest.approx(1 - 10 / 10.5)
    assert labels['end'].iloc[0] == dates[3]
    assert labels['returns'].iloc[-3:].isna().all().all()
    wide['hfq_low'] = wide['hfq_low'].copy()
    wide['hfq_low'].iloc[2, 0] = np.nan
    assert pd.isna(build_labels(wide, dates, [3])[3]['returns'].iloc[0, 0])


@pytest.fixture(scope='module')
def prepared():
    plan = yaml.safe_load((ROOT / 'configs/learning_strategy_search_v1.yaml').read_text())
    wide, eligible, calendar = synthetic()
    cfg = dict(plan['training'], min_dates=30, min_rows=80, min_features=3, min_ic_stocks=3)
    features = build_features(wide, eligible, calendar, plan['features'])
    labels = build_labels(wide, eligible.index, [5, 10, 20])
    return plan, cfg, prepare_arrays(features, eligible, labels, cfg)


def test_training_excludes_label_ending_on_fit_day(prepared):
    _, cfg, data = prepared
    fit = data['dates'][100]
    p = dict(horizon=20, train_window=252)
    mask = training_mask(data, p, fit, cfg)
    used = np.flatnonzero(mask.any(axis=1))
    assert used.max() == 79
    assert (data['labels'][20]['end'][used] < np.datetime64(fit)).all()


@pytest.mark.parametrize('family,variant', [('factor_weight','ic'),('factor_weight','icir'),('linear','ridge'),
    ('linear','elastic_net'),('lgbm','regression'),('lgbm','ranking'),('dual','ridge'),('dual','regression')])
def test_future_perturbation_preserves_past_predictions(prepared, family, variant):
    plan, cfg, data = prepared
    p = dict(family=family, variant=variant, **{k:v[0] for k,v in plan['common_axes'].items()},
             **{k:v[0] for k,v in plan['families'][family][variant].items()})
    if 'min_child_samples' in p:
        p['min_child_samples'] = 5
    cut = data['dates'][160]
    start = data['dates'][100]
    baseline, baseline_risk, fits = predict_walk_forward(data, p, cfg, 19, start, cut)
    altered = deepcopy(data)
    altered['x'][161:] = 1000
    altered['ranks'][161:] = -1000
    # 连过去信号行中尚未到可得时点的标签一起破坏，确认模型不读取它们。
    for label in altered['labels'].values():
        future = pd.isna(label['end']) | (label['end'] >= np.datetime64(cut))
        label['returns'][future] = 999
        label['risk'][future] = 999
        label['ic'][future] = 999
    other, other_risk, _ = predict_walk_forward(altered, p, cfg, 19, start, cut)
    pd.testing.assert_frame_equal(baseline, other)
    pd.testing.assert_frame_equal(baseline_risk, other_risk)
    assert baseline.notna().any().any()
    assert all(pd.Timestamp(x['max_label_end']) < pd.Timestamp(x['fit_date']) for x in fits if x['status']=='ok')


def test_joint_sampling_and_risk_score(prepared):
    plan, _, _ = prepared
    a = sample_parameters(plan, 'dual', 16, 13)
    assert a == sample_parameters(plan, 'dual', 16, 13)
    assert len({config_id(p) for p in a}) == 16
    assert {p['variant'] for p in a} == {'ridge','regression'}
    b = sample_parameters(plan, 'dual', 8, 14, [config_id(p) for p in a], a[:4])
    assert not {config_id(p) for p in a} & {config_id(p) for p in b}
    pred = pd.DataFrame([[.10, .08]])
    risk = pd.DataFrame([[.20, .02]])
    scored = score_predictions(pred, risk, dict(family='dual', risk_aversion=1., risk_cap=.1))
    assert pd.isna(scored.iloc[0,0]) and scored.iloc[0,1] == pytest.approx(.06)
    pd.testing.assert_frame_equal(score_predictions(pred,risk,dict(family='dual'),remove_risk=True),pred)
