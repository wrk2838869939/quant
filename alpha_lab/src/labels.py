"""标签构造模块 —— 口径定死 (2026-09-08, 调研报告 Phase 1, 不再反复修改)。

主标签: fwd5_excess_rank  —— 未来5日超额收益截面rank (+反正态变换)
稳健性: fwd20_excess_rank —— 未来20日 (月频)

口径细则:
  1. 前向收益: close(T+h)/close(T) - 1, h=5/20 个交易日 (决策日T收盘信号, T收盘成交假设)
  2. 超额 = 个股前向收益 - 全市场等权前向收益 (同窗口, 可交易股票池)
  3. rank → [0,1] 均匀 → norm.ppf 反正态 (可选, rank_norm=True)
  4. 停牌: T+1~T+h 任一日缺行情 → 标签 NaN (不填充)
  5. 退市: T+h 前最后成交价结算 (悲观口径, 不假设清算价)
  6. 泄露防护: 训练集使用时必须配合 purge_label_window() 剔除训练窗末端
     h 日样本 + embargo; 本模块只负责生成, 剔除逻辑独立。
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
