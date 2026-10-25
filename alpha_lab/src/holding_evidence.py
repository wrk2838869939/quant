"""C1研究变体：缺失保留已观察失败证据，不改变退出阈值或执行规则。"""
import numpy as np
import pandas as pd


def evidence_hold_states(trend, observable, confirmations, return_counts=False):
    """生成持有许可；未知不累计，明确恢复清零，明确失败累计。

    Parameters
    ----------
    trend, observable : pandas.DataFrame
        对齐的趋势和可观察状态；仅在observable为真时读取trend。
    confirmations : int
        沿用冻结政策的失败确认次数，研究登记值为2。
    return_counts : bool
        是否同时返回各次复核更新前、后的失败计数矩阵。

    Notes
    -----
    输出NaN表示未知而非持有成功。已经提交的退出由原portfolio不可撤销
    状态维护，本函数不管理订单，也不提供未知超时退出。
    """
    if (not trend.index.equals(observable.index)
            or not trend.columns.equals(observable.columns)
            or not isinstance(confirmations, int) or isinstance(confirmations, bool)
            or confirmations < 1 or observable.isna().any().any()
            or (trend.isna() & observable.astype(bool)).any().any()):
        raise ValueError('持有证据输入或确认次数非法')
    failures = np.zeros(len(trend.columns), dtype=int)
    states = np.full(trend.shape, np.nan)
    before, after = np.zeros(trend.shape, dtype=int), np.zeros(trend.shape, dtype=int)
    for i in range(len(trend)):
        known = observable.iloc[i].to_numpy(bool)
        good = trend.iloc[i].fillna(False).to_numpy(bool)
        before[i] = failures
        failures = np.where(known, np.where(good, 0, failures + 1), failures)
        after[i] = failures
        states[i, known] = (failures[known] < confirmations).astype(float)
    result = pd.DataFrame(states, index=trend.index, columns=trend.columns)
    if return_counts:
        return (result, pd.DataFrame(before, index=trend.index, columns=trend.columns),
                pd.DataFrame(after, index=trend.index, columns=trend.columns))
    return result
