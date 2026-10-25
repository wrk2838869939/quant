"""历史滚动市场暴露剔除与板块内部统计；不产生交易信号。"""
import numpy as np
import pandas as pd


def residualize(returns, factors, cfg):
    """按个股有效观测估计含截距的一/二因子 OLS，所有系数滞后一日。

    二因子分别为沪深300收益和中证500减沪深300收益。股票缺值时，
    所有矩估计统一使用该股票、全部因子同时有效的观测。
    """
    window, minimum = cfg['beta_window'], cfg['beta_minimum_days']
    factors = factors.reindex(returns.index)
    available = returns.notna() & factors.notna().all(axis=1).to_numpy()[:, None]
    y = returns.where(available)
    x = [pd.DataFrame(np.broadcast_to(factors[k].to_numpy()[:, None], returns.shape),
                      index=returns.index, columns=returns.columns).where(available) for k in factors]
    mean = lambda a: a.rolling(window, min_periods=minimum).mean()
    my = mean(y)
    mx = [mean(a) for a in x]
    cov_y = [mean(a*y)-b*my for a, b in zip(x, mx)]
    variance = [mean(a*a)-b*b for a, b in zip(x, mx)]
    eps = cfg['covariance_epsilon']
    if len(x) == 1:
        beta = [cov_y[0] / variance[0].where(variance[0] > eps)]
    elif len(x) == 2:
        cov = mean(x[0]*x[1])-mx[0]*mx[1]
        determinant = variance[0]*variance[1]-cov*cov
        denominator = determinant.where(determinant > eps)
        beta = [(cov_y[0]*variance[1]-cov_y[1]*cov)/denominator,
                (cov_y[1]*variance[0]-cov_y[0]*cov)/denominator]
    else:
        raise ValueError('仅支持一/二因子')
    intercept = my.copy()
    for b, m in zip(beta, mx):
        intercept -= b*m
    predicted = intercept.shift(1)
    for b, k in zip(beta, factors):
        predicted = predicted + b.shift(1).mul(factors[k], axis=0)
    residual = returns - predicted
    return residual, dict(intercept=intercept.shift(1), beta=[b.shift(1) for b in beta])


def correlation_stats(frame, minimum):
    """不含对角的等权配对分布；不够共同日期的配对不进入汇总。"""
    frame = frame.loc[:, frame.notna().sum().ge(minimum)]
    corr = frame.corr(min_periods=minimum)
    valid = frame.notna().to_numpy(np.int32)
    counts = valid.T @ valid
    ii, jj = np.triu_indices(len(frame.columns), 1)
    values = corr.to_numpy()[ii, jj]
    n = counts[ii, jj]
    ok = np.isfinite(values)
    values = values[ok]
    result = dict(eligible_stocks=len(frame.columns), pairs=int(ok.sum()),
                  mean=float(np.mean(values)) if len(values) else np.nan,
                  median=float(np.median(values)) if len(values) else np.nan,
                  q10=float(np.quantile(values, .1)) if len(values) else np.nan,
                  q90=float(np.quantile(values, .9)) if len(values) else np.nan,
                  median_days=float(np.median(n[ok])) if ok.any() else np.nan)
    return result


def series_correlation(a, b, minimum):
    """两条不重叠组合序列的共同日 Pearson 相关。"""
    joint = pd.concat([a, b], axis=1).dropna()
    return (float(joint.iloc[:, 0].corr(joint.iloc[:, 1]))
            if len(joint) >= minimum else np.nan), len(joint)
