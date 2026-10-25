"""低频复核中的入选持续性和排名退出缓冲，仅消费当时已知的截面。"""
import numpy as np
import pandas as pd


def smooth_scores(daily_scores, window):
    """将原[-.5,.5]分位分数平移至[0,1]；无资格日贡献0，今日仍须合格。

    多日计算不增加复核日。未知/不合格日不会被跳过后重新放大均值。
    """
    if type(window) is not int or window < 1:
        raise ValueError('平滑窗口必须为正整数')
    if not daily_scores.index.is_monotonic_increasing or daily_scores.index.has_duplicates:
        raise ValueError('观察日期必须唯一递增')
    finite = daily_scores.to_numpy()[np.isfinite(daily_scores.to_numpy())]
    if ((finite < -.5) | (finite > .5)).any():
        raise ValueError('需要原始分位排序分数，不能输入预测收益')
    support = (daily_scores + .5).fillna(0.)
    return support.rolling(window, min_periods=window).mean().where(daily_scores.notna())


def buffered_holds(scores, original_hold, buffer, confirmations, minimum_candidates=5):
    """连续月度复核仍有资格却排名落后时撤去保留许可；未知打断计数。

    这是相对排名退出代理，不声称把分位差换算成经济收益或保证替代股成交。
    """
    if any(type(x) is not int or x < 1 for x in [buffer, confirmations, minimum_candidates]):
        raise ValueError('排名缓冲和确认次数必须为正整数')
    if not scores.index.equals(original_hold.index) or not scores.columns.equals(original_hold.columns):
        raise ValueError('评分与持有矩阵必须对齐')
    if not scores.index.is_monotonic_increasing or scores.index.has_duplicates:
        raise ValueError('复核日期必须唯一递增')
    if buffer < minimum_candidates:
        raise ValueError('缓冲区不能小于席位数')
    # 并列按同一最优名次处理，不因代码顺序迫使旧仓退出。
    ranks = scores.rank(axis=1, ascending=False, method='min')
    weak = ranks.gt(buffer) & ranks.notna()
    weak = weak.mul(scores.notna().sum(axis=1).ge(minimum_candidates), axis=0)
    streak = np.zeros(len(scores.columns), dtype=int)
    exits = []
    for _, row in weak.iterrows():
        streak = np.where(row.to_numpy(), streak + 1, 0)
        exits.append(streak >= confirmations)
    exit_mask = pd.DataFrame(exits, index=scores.index, columns=scores.columns)
    result = original_hold.mask(exit_mask & original_hold.notna(), 0.)
    return result, ranks, exit_mask
