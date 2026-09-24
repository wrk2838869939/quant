"""包含完整规则端点的组合学习；目标、资格和非单调邻域显式分离。"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from rule_search import RuleSignals, config_id, decision_dates


def make_candidate(weights, frequency, top_n, baseline_params):
    """正单纯形端点绑定原基线政策，禁止只保留评分而偷换账户动作。"""
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('需要三个非负且和为1的权重')
    active = np.flatnonzero(w > 0)
    endpoint = int(active[0]) if len(active) == 1 else None
    if endpoint is not None:
        frequency = baseline_params[endpoint]['frequency']
        top_n = baseline_params[endpoint]['top_n']
    if frequency not in ['daily', 'weekly', 'monthly'] or not 1 <= top_n <= 5:
        raise ValueError('组合交易政策无效')
    return dict(weights=w.tolist(), frequency=frequency, top_n=int(top_n), endpoint=endpoint)


def candidate_grid(baseline_params, denominator, include_equal=True):
    """穷举完整离散三角网格；去掉重复政策和端点，不插值收益。"""
    if not isinstance(denominator, int) or denominator < 1:
        raise ValueError('权重网格分母必须为正整数')
    weights = [[i / denominator, j / denominator, (denominator - i - j) / denominator]
               for i in range(denominator + 1) for j in range(denominator + 1 - i)]
    if include_equal:
        weights.append([1 / 3] * 3)
    policies = sorted({(p['frequency'], p['top_n']) for p in baseline_params})
    unique = {}
    for w, (freq, n) in itertools.product(weights, policies):
        p = make_candidate(w, freq, n, baseline_params)
        unique[config_id(p)] = p
    return [unique[k] for k in sorted(unique)]


def baseline_scores(wide, eligible, calendar, params):
    model = RuleSignals(wide, eligible, calendar)
    return [model.score(p) for p in params]


def blend_scores(scores, weights):
    """正权因子资格取交集；零权因子的缺失不能污染单规则端点。"""
    w = np.asarray(weights, dtype=float)
    if len(scores) != 3 or w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('非法组合权重')
    for score in scores[1:]:
        if not score.index.equals(scores[0].index) or not score.columns.equals(scores[0].columns):
            raise ValueError('因子轴不一致')
    active = np.flatnonzero(w > 0)
    if len(active) == 1:
        return scores[active[0]].copy()
    # pandas加法传播缺失；只对正权项求和，避免0*NaN使端点失效。
    result = scores[active[0]].rank(axis=1, pct=True) * w[active[0]]
    for i in active[1:]:
        result = result + scores[i].rank(axis=1, pct=True) * w[i]
    return result


def candidate_signals(scores, params, calendar, start, end):
    days = scores[0].index
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    if not len(days):
        raise ValueError('无信号日期')
    if params is None:
        signals = pd.DataFrame(np.nan, index=[days[0]], columns=scores[0].columns)
        return signals, pd.Series(1, index=signals.index, name='top_n')
    score = blend_scores(scores, params['weights']).loc[days]
    signals = score.loc[decision_dates(days, params['frequency'], calendar)]
    return signals, pd.Series(params['top_n'], index=signals.index, name='top_n')


def account_key(row):
    return (-row['metrics']['median_ann'], row['metrics']['turnover'],
            sum(w > 0 for w in row['params']['weights']), row['id'])


def annotate_neighbors(rows, denominator, minimum):
    """同政策下只使用实测的一步权重转移邻居，包括风险不合格点。"""
    for row in rows:
        p = row['params']
        neighbors = [other for other in rows if other['id'] != row['id']
                     and (p['frequency'], p['top_n']) == (other['params']['frequency'], other['params']['top_n'])
                     and np.isclose(np.abs(np.array(p['weights']) - other['params']['weights']).sum(), 2 / denominator)
                     and np.isclose(np.max(np.abs(np.array(p['weights']) - other['params']['weights'])), 1 / denominator)]
        row['neighbor_ids'] = [r['id'] for r in neighbors]
        row['neighbor_count'] = len(neighbors)
        row['neighbor_q25'] = (float(np.quantile([r['metrics']['median_ann'] for r in neighbors], .25))
                               if len(neighbors) >= minimum else None)


def select_methods(rows, tie_band):
    """只接收内层实测记录；账户选择显式包含且不劣于可行单基线。"""
    good = [r for r in rows if r['metrics']['feasible']]
    endpoints = [r for r in good if r['params']['endpoint'] is not None]
    pick = lambda values: min(values, key=account_key) if values else None
    baseline, account = pick(endpoints), pick(good)
    floor = baseline['metrics']['median_ann'] if baseline else -np.inf
    stable = None
    if account:
        near = [r for r in good if r['metrics']['median_ann'] >= max(floor, account['metrics']['median_ann'] - tie_band)]
        if all(r.get('neighbor_q25') is not None for r in near):
            stable = min(near, key=lambda r: (-r['neighbor_q25'], r['metrics']['turnover'], account_key(r)))
        else:
            stable = min(near, key=lambda r: (r['metrics']['turnover'], account_key(r)))
    ic_rows = [r for r in good if np.isfinite(r['mean_ic'])]
    ic = min(ic_rows, key=lambda r: (-r['mean_ic'], account_key(r))) if ic_rows else None
    selected = dict(baseline_selector=baseline, account_best=account, account_stable=stable, ic_best=ic,
                    ic_with_baseline_fallback=pick(endpoints + ([ic] if ic else [])),
                    equal_weight=pick([r for r in good if np.allclose(r['params']['weights'], [1 / 3] * 3)]))
    if baseline:
        for name in ['account_best', 'account_stable', 'ic_with_baseline_fallback']:
            assert selected[name]['metrics']['median_ann'] >= floor
    return selected


def nonmonotonic_triplets(rows, denominator):
    """保存同政策两权重转移的完整三点观察，不把线性权重误当线性绩效。"""
    lookup = {}
    for row in rows:
        p = row['params']
        units = np.array(p['weights']) * denominator
        if np.allclose(units, units.round()):
            lookup[(p['frequency'], p['top_n'], tuple(units.round().astype(int)))] = row
    result = []
    for (freq, n, units), middle in lookup.items():
        for a, b in itertools.combinations(range(3), 2):
            left, right = list(units), list(units)
            left[a] -= 1; left[b] += 1
            right[a] += 1; right[b] -= 1
            lo, hi = lookup.get((freq, n, tuple(left))), lookup.get((freq, n, tuple(right)))
            if lo and hi:
                x, y, z = [r['metrics']['median_ann'] for r in [lo, middle, hi]]
                result.append(dict(left=lo['id'], middle=middle['id'], right=hi['id'],
                                   frequency=freq, top_n=n, left_value=x, middle_value=y, right_value=z,
                                   nonmonotonic=bool((y - x) * (z - y) < 0),
                                   all_feasible=all(r['metrics']['feasible'] for r in [lo, middle, hi])))
    return result
