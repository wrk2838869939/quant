"""确认日到未来交易日的完整路径收益，缺失不跳日或填充。"""
import numpy as np


def window_metrics(prices, horizon):
    """列为不同资产，行从确认日开始；极值只取未来第1至h日。

    返回完整窗口的期末收益、最高／最低收盘收益与出现日。
    每个窗口独立核验全部价格，包括起点；缺失窗口不输出收益。
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim == 1:
        prices = prices[:, None]
    values = prices[:horizon + 1]
    n = values.shape[1]
    valid = np.isfinite(values) & (values > 0)
    complete = valid.all(axis=0) & (len(values) == horizon + 1)
    result = {'complete': complete, 'valid_prices': valid.sum(axis=0)}
    for key in ['end_return', 'max_return', 'min_return', 'max_day', 'min_day']:
        result[key] = np.full(n, np.nan)
    if complete.any():
        returns = values[1:, complete] / values[0, complete] - 1
        result['end_return'][complete] = returns[-1]
        result['max_return'][complete] = returns.max(axis=0)
        result['min_return'][complete] = returns.min(axis=0)
        result['max_day'][complete] = returns.argmax(axis=0) + 1
        result['min_day'][complete] = returns.argmin(axis=0) + 1
    return result


def cumulative_paths(prices):
    """缺口出现后不恢复该次事件的连续路径；不使用将来完整性筛今天。"""
    values = np.asarray(prices, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    valid = np.isfinite(values) & (values > 0)
    continuous = np.logical_and.accumulate(valid, axis=0)
    base = np.where(valid[0], values[0], np.nan)
    return np.where(continuous, values / base - 1, np.nan)
