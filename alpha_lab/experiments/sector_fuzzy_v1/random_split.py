"""固定种子的事件级随机两半互换，划分不接收日期或标签。"""
import numpy as np
import pandas as pd


def make_assignments(event_ids, seed):
    """先随机二等分，再独立将各半区随机等分为验证和测试。"""
    ids = np.asarray(sorted(event_ids))
    if len(set(ids)) != len(ids) or len(ids) < 8:
        raise ValueError('事件编号须唯一且至少有 8 条')
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    halves = [shuffled[:len(ids) // 2], shuffled[len(ids) // 2:]]
    holdouts = [rng.permutation(half) for half in halves]
    rows = []
    for fold in range(2):
        holdout = holdouts[1 - fold]
        parts = {'train': halves[fold], 'validation': holdout[:len(holdout) // 2],
                 'test': holdout[len(holdout) // 2:]}
        for role, members in parts.items():
            rows.extend(dict(round=fold + 1, role=role, event_id=i) for i in members)
    return pd.DataFrame(rows)
