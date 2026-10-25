"""组合空间必须包含真实基线，并正确处理缺失、标签成熟和非单调选择。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from baseline_combination import (make_candidate, candidate_grid, blend_scores, baseline_scores,
                                  candidate_signals, select_methods, nonmonotonic_triplets, annotate_neighbors)
from learning_search import build_labels
from rule_search import config_id, decision_dates
from test_rule_search import synthetic


def params():
    return [dict(family='S01', window=19, frequency='weekly', top_n=3),
            dict(family='S02', variant='low_vol', window=32, frequency='monthly', top_n=3),
            dict(family='S04', variant='positive_ep', exclude_smallest_mv_quantile=.2, frequency='monthly', top_n=4)]


def test_complete_grid_and_endpoint_policies():
    p = params()
    grid = candidate_grid(p, 10)
    assert len(grid) == 195
    assert len({config_id(x) for x in grid}) == len(grid)
    for i in range(3):
        endpoints = [x for x in grid if x['endpoint'] == i]
        assert len(endpoints) == 1
        assert endpoints[0]['frequency'] == p[i]['frequency']
        assert endpoints[0]['top_n'] == p[i]['top_n']
    assert len(candidate_grid([dict(x, frequency='weekly', top_n=5) for x in p], 10)) == 67
    with pytest.raises(ValueError):
        make_candidate([1.1, -.1, 0], 'weekly', 3, p)


def test_zero_weight_missing_does_not_break_vertex():
    a = pd.DataFrame([[1., 2., np.nan], [2., 1., 3.]])
    b = pd.DataFrame([[np.nan, 2., 1.], [np.nan, 1., 2.]])
    c = b * np.nan
    pd.testing.assert_frame_equal(blend_scores([a, b, c], [1., 0., 0.]), a)
    blend = blend_scores([a, b, c], [.5, .5, 0.])
    pd.testing.assert_frame_equal(blend, .5 * a.rank(axis=1, pct=True) + .5 * b.rank(axis=1, pct=True))
    assert blend.iloc[0].notna().sum() == 1


def test_raw_vertex_and_score_prefix():
    wide, eligible, calendar = synthetic()
    p = params()
    scores = baseline_scores(wide, eligible, calendar, p)
    start, end, cut = eligible.index[0], eligible.index[-1], eligible.index[173]
    prefix_scores = baseline_scores({k: v.loc[:cut] for k, v in wide.items()}, eligible.loc[:cut], calendar, p)
    for i in range(3):
        candidate = make_candidate(np.eye(3)[i], 'daily', 1, p)
        signals, schedule = candidate_signals(scores, candidate, calendar, start, end)
        expected = scores[i].loc[decision_dates(eligible.index, p[i]['frequency'], calendar)]
        pd.testing.assert_frame_equal(signals, expected)
        assert schedule.eq(p[i]['top_n']).all()
    for w in [[.4, .3, .3], [0, .8, .2], [1, 0, 0]]:
        candidate = make_candidate(w, 'weekly', 3, p)
        full, n = candidate_signals(scores, candidate, calendar, start, end)
        short, m = candidate_signals(prefix_scores, candidate, calendar, start, cut)
        pd.testing.assert_frame_equal(short, full.loc[:cut])
        pd.testing.assert_series_equal(m, n.loc[:cut])


def fake_row(w, value, ic, feasible=True):
    p = make_candidate(w, 'weekly', 3, [dict(x, frequency='weekly', top_n=3) for x in params()])
    return dict(id=config_id(p), params=p, mean_ic=ic, metrics=dict(median_ann=value, turnover=1., feasible=feasible))


def test_account_floor_and_ic_fallback_are_explicit():
    rows = [fake_row([1, 0, 0], .10, .01), fake_row([0, 1, 0], .05, .02),
            fake_row([0, 0, 1], .08, .03), fake_row([.5, .5, 0], .12, .02),
            fake_row([.3, .4, .3], -.10, .09), fake_row([.2, .4, .4], 9., .99, False)]
    chosen = select_methods(rows, .005)
    assert chosen['account_best']['metrics']['median_ann'] == .12
    assert chosen['ic_best']['metrics']['median_ann'] == -.10
    assert chosen['ic_with_baseline_fallback']['params']['endpoint'] == 0
    assert chosen['baseline_selector']['params']['endpoint'] == 0
    for r in rows:
        r['metrics']['feasible'] = False
    assert all(v is None for v in select_methods(rows, .005).values())


def test_nonmonotonic_peak_and_measured_neighbors():
    rows = [fake_row([.6, .4, 0], .01, 0), fake_row([.5, .5, 0], .09, 0), fake_row([.4, .6, 0], .02, 0)]
    triples = nonmonotonic_triplets(rows, 10)
    assert len(triples) == 1 and triples[0]['nonmonotonic']
    annotate_neighbors(rows, 10, 3)
    assert rows[1]['neighbor_count'] == 2 and rows[1]['neighbor_q25'] is None


def test_training_labels_never_use_post_cutoff_prices():
    wide, eligible, _ = synthetic()
    cutoff = eligible.index[180]
    past = {k: v.loc[:cutoff] for k, v in wide.items()}
    labels = build_labels(past, eligible.loc[:cutoff].index, [20])[20]
    assert labels['returns'].iloc[-20:].isna().all().all()
    ends = labels['end'][labels['returns'].notna().any(axis=1)]
    assert ends.max() <= cutoff
    changed = {k: v.copy() for k, v in wide.items()}
    for value in changed.values():
        value.loc[value.index > cutoff] *= 100
    again = build_labels({k: v.loc[:cutoff] for k, v in changed.items()}, eligible.loc[:cutoff].index, [20])[20]
    pd.testing.assert_frame_equal(labels['returns'], again['returns'])
