"""基线恒等、增量资格、成对训练成熟标签及未来扰动隔离。"""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from baseline_increment import prepare_data, training_mask, predict_pair, compose_score, select_paths
from baseline_combination import baseline_scores
from learning_search import build_labels
from test_rule_search import synthetic


def test_zero_increment_is_raw_identity_even_without_model():
    raw = pd.DataFrame([[10., np.nan, 1.], [20., 10., 3.]])
    pd.testing.assert_frame_equal(compose_score(raw, None, 'residual', 0., 1.), raw)
    pd.testing.assert_frame_equal(compose_score(raw, None, 'baseline', 0., 1.), raw)
    p = pd.DataFrame([[999., 1., -999.], [np.nan, 0., 1.]])
    result = compose_score(raw, p, 'residual', .2, 1.)
    expected = (raw.rank(axis=1, pct=True) + .2 * p.clip(-1, 1).fillna(0)).where(raw.notna())
    pd.testing.assert_frame_equal(result, expected)
    assert result.notna().equals(raw.notna())


@pytest.fixture(scope='module')
def prepared():
    plan = yaml.safe_load((ROOT / 'configs/baseline_increment_v1.yaml').read_text())
    plan['training'].update(min_dates=30, min_rows=80, min_label_stocks=3)
    wide, eligible, calendar = synthetic()
    specs = [dict(family='S01', window=20), dict(family='S02', variant='low_vol', window=20),
             dict(family='S04', variant='positive_ep', exclude_smallest_mv_quantile=.2)]
    scores = baseline_scores(wide, eligible, calendar, specs)
    labels = build_labels(wide, eligible.index, [5, 20])
    data = prepare_data(scores, scores[0], labels, 3)
    return plan, data, scores


def test_missing_other_features_does_not_remove_host_stocks(prepared):
    plan, data, scores = prepared
    assert np.array_equal(data['allowed'], scores[0].notna().to_numpy())
    assert (data['allowed'] & scores[2].isna().to_numpy()).any()
    assert np.isfinite(data['x']).all()


def test_exact_label_maturity_boundary(prepared):
    _, data, _ = prepared
    fit_date = data['dates'][100]
    mask = training_mask(data, {'horizon': 20}, fit_date, 504)
    used = np.flatnonzero(mask.any(axis=1))
    assert used[-1] == 79
    assert (data['labels'][20]['end'][used] < np.datetime64(fit_date)).all()


@pytest.mark.parametrize('model', ['ridge', 'lgbm'])
def test_paired_models_future_perturbation_and_reload(prepared, model, tmp_path):
    plan, data, _ = prepared
    spec = dict(model=model, horizon=5)
    spec.update(dict(alpha=1.) if model == 'ridge' else dict(num_leaves=3, n_estimators=8, learning_rate=.05, min_child_samples=5, reg_lambda=10.))
    start, end = data['dates'][100], data['dates'][180]
    expected, fits = predict_pair(data, spec, plan, start, end, tmp_path)
    altered = deepcopy(data)
    altered['x'][181:] = 999
    for label in altered['labels'].values():
        future = pd.isna(label['end']) | (label['end'] >= np.datetime64(end))
        label['direct'][future] = 999
        label['residual'][future] = -999
    actual, _ = predict_pair(altered, spec, plan, start, end)
    for mode in expected:
        pd.testing.assert_frame_equal(expected[mode], actual[mode])
        assert expected[mode].notna().any().any()
    assert all(r['reload_verified'] and pd.Timestamp(r['max_label_end']) < pd.Timestamp(r['fit_date']) for r in fits if r['status'] == 'ok')


def test_account_fallback_does_not_require_positive_model_increment():
    plan = yaml.safe_load((ROOT / 'configs/baseline_increment_v1.yaml').read_text())
    def row(kind, ret, amp=1.):
        return dict(id=kind, params=dict(kind=kind, amplitude=amp),
                    metrics=dict(median_ann=ret, turnover=1., feasible=True), neighbor_q25=None)
    rows = [row('baseline', .10, 0.), row('residual', .09, .1), row('direct', -.2)]
    chosen = select_paths(rows, plan)
    assert chosen['residual_best']['params']['kind'] == 'baseline'
    assert chosen['residual_stable']['params']['kind'] == 'baseline'
    assert chosen['direct_with_fallback']['params']['kind'] == 'baseline'
    assert chosen['direct_best']['params']['kind'] == 'direct'
