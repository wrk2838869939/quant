"""历史截面研究标签与末端隔离工具。

forward_returns 按每只股票后续 h 条观测计算收盘比值，不主动补齐交易所日历。
不验证中间停牌，不实现退市清算；使用者须先构造合适的日历面板。
close-to-close 标签用于描述性研究，不能当成同日收盘信号的可成交收益。
训练隔离须结合真实标签结束日，交易由独立执行引擎处理。
此说明修正旧文档超出实现的承诺，计算公式保持原样。
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

HORIZONS = {5: "fwd5_excess_rank", 20: "fwd20_excess_rank"}


def forward_returns(df: pd.DataFrame, horizons=(5, 20)) -> pd.DataFrame:
    """按股票分组计算前向收盘收益。

    Parameters
    ----------
    df : 需含 ts_code, date, close (不复权) 或 hfq_close, 按日期升序。

    Returns
    -------
    原 df 附加 fwd_ret_{h} 列。
    """
    df = df.sort_values(["ts_code", "date"]).copy()
    price_col = "hfq_close" if "hfq_close" in df.columns else "close"
    for h in horizons:
        fwd = df.groupby("ts_code")[price_col].shift(-h) / df[price_col] - 1.0
        df[f"fwd_ret_{h}"] = fwd
    return df


def excess_rank_labels(df: pd.DataFrame, horizons=(5, 20), rank_norm: bool = True,
                       min_cross_section: int = 100) -> pd.DataFrame:
    """构造超额收益截面 rank 标签。

    超额基准 = 当日全市场等权前向收益 (可交易股票: 当日有行情且标签非NaN)。
    横截面不足 min_cross_section 只时当日标签全部置 NaN (早期年份保护)。
    """
    df = forward_returns(df, horizons)
    dates = df["date"].drop_duplicates().sort_values()
    date_idx = pd.Series(np.arange(len(dates)), index=dates.values)
    df = df[df["date"].isin(date_idx.index)].copy()
    df["t"] = df["date"].map(date_idx)

    for h in horizons:
        # 等权基准: 当日截面 fwd_ret 均值 (NaN 已被 groupby mean 天然剔除)
        base = df.groupby("t")[f"fwd_ret_{h}"].transform("mean")
        exc = df[f"fwd_ret_{h}"] - base
        n = df.groupby("t")[f"fwd_ret_{h}"].transform("count")
        r = exc.groupby(df["t"]).rank(pct=True)
        if rank_norm:
            r = norm.ppf(np.clip(r, 1e-4, 1 - 1e-4))
        r[n < min_cross_section] = np.nan
        df[HORIZONS[h]] = r
    return df.drop(columns=["t"])


def purge_label_window(train: pd.DataFrame, label_col: str = "fwd5_excess_rank",
                       h: int = 5, embargo: int = 5) -> pd.DataFrame:
    """剔除训练窗末端 h+embargo 日样本 (标签窗口与后续测试期重叠 → 泄露)。

    用法: walk-forward 中 train = data[:split]; test = data[split:]
          train_purged = purge_label_window(train, h=5, embargo=5)
    """
    dates = pd.DatetimeIndex(sorted(train["date"].unique()))
    keep = purged_dates(dates, h, embargo)
    return train[train["date"].isin(keep)]


def purged_dates(dates, h: int, embargo: int = 5) -> pd.DatetimeIndex:
    """按交易日剔除末端 h+embargo；窗口不足时返回空集。"""
    if h < 1 or embargo < 0:
        raise ValueError("h 必须为正整数，embargo 不能为负")
    dates = pd.DatetimeIndex(dates)
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("日期必须严格递增且唯一")
    return dates[:max(0, len(dates) - h - embargo)]
