"""信号时点匹配对照；只接收当期暴露，不接收未来收益。"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_controls(universe, selected, features, seed):
    """行业/板块精确匹配，风格截面百分位最小距离、不放回分配。

    缺失行业用显式unknown组保留，不解释为行业已中性。
    无可配对象留空并记录；不跨行业回退，不使用未来成交或收益补选。
    """
    if not universe.index.is_unique or len(set(selected)) != len(selected):
        raise ValueError("股票代码必须唯一")
    if not set(selected).issubset(universe.index):
        raise ValueError("目标股票不在当期股票池")
    frame = universe.sort_index().copy()
    frame["industry"] = frame.industry.fillna("__unknown__")
    if features:
        ranks = frame[features].replace([np.inf, -np.inf], np.nan).rank(pct=True)
    else:
        ranks = pd.DataFrame(index=frame.index)
    rng = np.random.default_rng(seed)
    rows = []
    selected_set = set(selected)
    for keys, targets in frame.loc[selected].groupby(["board", "industry"], sort=True):
        candidates = frame[(frame.board == keys[0]) & (frame.industry == keys[1])
                           & ~frame.index.isin(selected_set)]
        valid_target = targets.index
        if features:
            valid_target = valid_target[ranks.loc[valid_target].notna().all(axis=1)]
            candidates = candidates[ranks.loc[candidates.index].notna().all(axis=1)]
        assigned = {}
        if len(valid_target) and len(candidates):
            if features:
                distance = np.square(ranks.loc[valid_target].to_numpy()[:, None, :]
                    - ranks.loc[candidates.index].to_numpy()[None, :, :]).mean(axis=2)
                cost = distance + rng.uniform(0, 1e-12, distance.shape)
            else:
                distance = np.zeros((len(valid_target), len(candidates)))
                cost = rng.uniform(size=distance.shape)
            left, right = linear_sum_assignment(cost)
            assigned = {valid_target[i]: (candidates.index[j], float(np.sqrt(distance[i, j])))
                        for i, j in zip(left, right)}
        for target in targets.index:
            control, dist = assigned.get(target, (None, np.nan))
            rows.append(dict(target=target, control=control, distance=dist,
                board=keys[0], industry=keys[1], industry_known=keys[1] != "__unknown__",
                status="matched" if control is not None else "unmatched"))
    return pd.DataFrame(rows).set_index("target").reindex(selected).reset_index()
