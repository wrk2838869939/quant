"""日期等权的条件均值模型及训练月份块重采样不确定性。

区间只描述重采样训练集后条件收益均值估计的变动，不是单次收益的
预测区间，也不保证覆盖率。收益效用应使用点预测，不能用区间端点替代。
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd


def _positive_integer(value, name):
    """拒绝布尔值、非整数及非正参数。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} 必须为正整数")
    return int(value)


def _numeric_matrix(frame, columns):
    """读取有限数值，既不填充缺失也不进行全样本标准化。"""
    try:
        values = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("模型输入必须为有限数值") from exc
    if not np.isfinite(values).all():
        raise ValueError("模型输入必须为有限数值")
    return values


def _solve(xtx, xty, alpha):
    """求解日期等权 Ridge；首列截距不惩罚。"""
    penalty = np.eye(xtx.shape[0]) * alpha
    penalty[0, 0] = 0.0
    result = np.linalg.solve(xtx + penalty, xty)
    if not np.isfinite(result).all():
        raise ValueError("Ridge 求解产生非有限系数")
    return result


def _month_block_counts(n_months, block_dates, rng):
    """抽取连续日历月移动块，保留块内所有日期及股票。"""
    n_blocks = (n_months + block_dates - 1) // block_dates
    starts = rng.integers(0, n_months - block_dates + 1, size=n_blocks)
    sampled = (starts[:, None] + np.arange(block_dates)).ravel()[:n_months]
    return np.bincount(sampled, minlength=n_months)


@dataclass
class ValueForecastModel:
    """保留可审计训练信息的条件均值模型。

    coefficients 行为 ``intercept`` 和原始特征名，列为收益与正数不利幅度。
    fit_dates 是实际参与拟合的日期；training_input_hash 仅覆盖合格训练行，
    不含尚未成熟标签。metadata 可直接序列化为 JSON。
    """

    feature_cols: tuple
    coefficients: pd.DataFrame
    bootstrap_coefficients: np.ndarray
    fit_dates: pd.DatetimeIndex
    last_label: pd.Timestamp
    training_input_hash: str
    metadata: dict

    def predict(self, frame):
        """返回收益点预测、非负不利幅度及均值估计的 10/90% 分位区间。

        不修改输入，不在预测阶段重新归一化；缺失或无限特征直接拒绝。
        区间没有加入单次收益噪声，不能解释为持有收益的 80% 预测区间。
        """
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError("frame 必须是列名唯一的 DataFrame")
        missing = set(self.feature_cols) - set(frame.columns)
        if missing:
            raise ValueError(f"缺少预测特征: {sorted(missing)}")
        x = np.column_stack([np.ones(len(frame)), _numeric_matrix(frame, list(self.feature_cols))])
        point = x @ self.coefficients.to_numpy()
        bootstrap_means = x @ self.bootstrap_coefficients[:, :, 0].T
        lower, upper = np.quantile(bootstrap_means, [0.1, 0.9], axis=1)
        if not np.isfinite(point).all() or not np.isfinite(bootstrap_means).all():
            raise ValueError("预测产生非有限数值")
        return pd.DataFrame({
            "expected_return": point[:, 0],
            "expected_adverse": np.maximum(point[:, 1], 0.0),
            "mean_lower": lower,
            "mean_upper": upper,
        }, index=frame.index)


def fit_value_model(frame, feature_cols, *, asof, horizon=60, alpha=1.0,
                    bootstrap_reps=100, block_dates=6, seed=20260913, min_dates=36):
    """拟合严格按标签成熟日期截断的日期等权 Ridge。

    Parameters
    ----------
    frame : pandas.DataFrame
        包含 date、label_end_{horizon}、ret_{horizon}、mae_{horizon}、
        entry_missing_{horizon}；MAE 为相对入场价的非正不利收益。
    feature_cols : sequence of str
        已归一化的特征；空列表产生只含截距的日期等权基准。
    asof : datetime-like
        只使用 label_end <= asof 的成熟标签；比较采用 UTC 日历日期。
    alpha : float
        正的 Ridge 惩罚。目标为各日期股票平均平方误差之和加系数惩罚。
    block_dates : int
        历史接口名称；单位是连续日历月份。每次抽取完整月份块，
        保留块内日期簇，直到抽到与原训练日历跨度相同的月数。
    min_dates : int
        过滤后最少不同训练日期数。

    Returns
    -------
    ValueForecastModel
        10/90% 区间仅为条件均值估计不确定性，不承诺收益预测覆盖率。
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("frame 必须是列名唯一的 DataFrame")
    if isinstance(feature_cols, str):
        raise ValueError("feature_cols 必须为特征名序列")
    feature_cols = tuple(feature_cols)
    if any(not isinstance(c, str) for c in feature_cols) or len(set(feature_cols)) != len(feature_cols):
        raise ValueError("特征名必须为互不重复的字符串")
    if "intercept" in feature_cols:
        raise ValueError("intercept 为保留系数名")
    horizon = _positive_integer(horizon, "horizon")
    bootstrap_reps = _positive_integer(bootstrap_reps, "bootstrap_reps")
    block_dates = _positive_integer(block_dates, "block_dates")
    min_dates = _positive_integer(min_dates, "min_dates")
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError("alpha 必须为有限正数")
    try:
        alpha = float(alpha)
    except (ValueError, TypeError) as exc:
        raise ValueError("alpha 必须为有限正数") from exc
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha 必须为有限正数")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed 必须为非负整数")
    cutoff = pd.to_datetime(asof, utc=True)
    if not isinstance(cutoff, pd.Timestamp) or pd.isna(cutoff):
        raise ValueError("asof 必须为有效的单个日期")
    cutoff = cutoff.tz_convert(None).normalize()
    end_col, return_col = f"label_end_{horizon}", f"ret_{horizon}"
    mae_col, missing_col = f"mae_{horizon}", f"entry_missing_{horizon}"
    label_cols = ["date", end_col, return_col, mae_col, missing_col]
    if set(feature_cols) & set(label_cols):
        raise ValueError("日期及标签列不能作为预测特征")
    missing = set(label_cols + list(feature_cols)) - set(frame.columns)
    if missing:
        raise ValueError(f"缺少训练列: {sorted(missing)}")
    work = frame.loc[:, label_cols + list(feature_cols)].copy().reset_index(drop=True)
    ends = pd.to_datetime(work[end_col], utc=True).dt.tz_convert(None).dt.normalize()
    mature = ends.notna() & ends.le(cutoff)
    counts = {
        "input_rows": int(len(work)),
        "missing_label_end": int(ends.isna().sum()),
        "immature_label": int(ends.gt(cutoff).sum()),
    }
    entry = work[missing_col]
    if not entry.loc[mature].isin([True, False]).all():
        raise ValueError("成熟行的 entry_missing 必须是布尔值或 0/1")
    unavailable = mature & entry.eq(True)
    counts["entry_missing"] = int(unavailable.sum())
    eligible = mature & entry.eq(False)
    missing_target = work[[return_col, mae_col]].isna().any(axis=1)
    counts["missing_target"] = int((eligible & missing_target).sum())
    eligible &= ~missing_target
    work = work.loc[eligible].copy()
    work[end_col] = ends.loc[eligible]
    work["date"] = pd.to_datetime(work["date"], utc=True).dt.tz_convert(None).dt.normalize()
    if work["date"].isna().any() or work["date"].gt(work[end_col]).any():
        raise ValueError("训练日期必须有效且不得晚于标签结束日期")
    work = work.sort_values(["date", end_col] + list(feature_cols) + [return_col, mae_col], kind="stable")
    fit_dates = pd.DatetimeIndex(work["date"].drop_duplicates(), name="date")
    if len(fit_dates) < min_dates:
        raise ValueError(f"有效训练日期不足: {len(fit_dates)} < {min_dates}")
    x = np.column_stack([np.ones(len(work)), _numeric_matrix(work, list(feature_cols))])
    y = _numeric_matrix(work, [return_col, mae_col])
    if (y[:, 1] > 0).any():
        raise ValueError("MAE 标签必须为相对入场价的非正收益")
    y[:, 1] *= -1
    periods = pd.period_range(fit_dates[0].to_period("M"), fit_dates[-1].to_period("M"), freq="M")
    if len(periods) <= block_dates:
        raise ValueError("训练日历月数必须大于块长度，避免退化重采样")
    # 先计算日期簇充分统计量，再聚合月份；任何股票扩容都不改变日期总权重。
    date_ids = pd.factorize(work["date"], sort=False)[0]
    starts = np.r_[0, np.flatnonzero(np.diff(date_ids)) + 1, len(work)]
    date_xtx = np.empty((len(fit_dates), x.shape[1], x.shape[1]))
    date_xty = np.empty((len(fit_dates), x.shape[1], 2))
    for i, (start, stop) in enumerate(zip(starts[:-1], starts[1:])):
        date_xtx[i] = (x[start:stop].T @ x[start:stop]) / (stop - start)
        date_xty[i] = (x[start:stop].T @ y[start:stop]) / (stop - start)
    if not np.isfinite(date_xtx).all() or not np.isfinite(date_xty).all():
        raise ValueError("训练充分统计量产生非有限数值")
    month_ids = fit_dates.to_period("M").asi8 - periods[0].ordinal
    month_xtx = np.zeros((len(periods), x.shape[1], x.shape[1]))
    month_xty = np.zeros((len(periods), x.shape[1], 2))
    np.add.at(month_xtx, month_ids, date_xtx)
    np.add.at(month_xty, month_ids, date_xty)
    coef = _solve(date_xtx.sum(axis=0), date_xty.sum(axis=0), alpha)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((bootstrap_reps, x.shape[1], 2))
    for rep in range(bootstrap_reps):
        sampled_counts = _month_block_counts(len(periods), block_dates, rng)
        xtx = np.einsum("m,mij->ij", sampled_counts, month_xtx)
        xty = np.einsum("m,mij->ij", sampled_counts, month_xty)
        if xtx[0, 0] == 0:
            raise ValueError("重采样只抽到空月份；训练月份过于稀疏")
        bootstrap[rep] = _solve(xtx, xty, alpha)
    hash_header = json.dumps({"columns": list(work.columns), "feature_cols": feature_cols}, sort_keys=True)
    digest = hashlib.sha256(hash_header.encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(work, index=False).to_numpy(dtype="uint64").tobytes())
    input_hash = digest.hexdigest()
    counts["fitted_rows"] = int(len(work))
    metadata = {
        "asof": cutoff.strftime("%Y-%m-%d"), "horizon": horizon,
        "alpha": alpha, "bootstrap_reps": bootstrap_reps, "block_dates": block_dates,
        "block_unit": "calendar_month", "seed": int(seed), "min_dates": min_dates,
        "feature_cols": list(feature_cols), "fit_dates": [d.strftime("%Y-%m-%d") for d in fit_dates],
        "last_label": work[end_col].max().strftime("%Y-%m-%d"),
        "n_dates": len(fit_dates), "calendar_months": len(periods),
        "exclusion_counts": counts, "training_input_hash": input_hash,
        "date_total_weight": 1.0, "intercept_penalized": False,
        "interval_quantiles": [0.1, 0.9],
        "interval_semantics": "bootstrap_conditional_mean_estimation_not_predictive_no_coverage_guarantee",
        "ridge_objective": "sum_of_date_mean_squared_errors_plus_alpha_l2",
    }
    return ValueForecastModel(
        feature_cols=feature_cols,
        coefficients=pd.DataFrame(coef, index=["intercept"] + list(feature_cols),
                                  columns=["expected_return", "expected_adverse"]),
        bootstrap_coefficients=bootstrap, fit_dates=fit_dates,
        last_label=work[end_col].max(), training_input_hash=input_hash, metadata=metadata,
    )
