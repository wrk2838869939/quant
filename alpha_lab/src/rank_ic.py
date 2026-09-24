"""统一截面 RankIC：先取成对有限样本，再分别平均排名及相关。"""
import numpy as np
import pandas as pd


def rank_ic_rows(factor: pd.DataFrame, target: pd.DataFrame,
                 min_obs: int = 100) -> pd.Series:
    """返回逐日 Spearman 相关；缺样本或任一序列为常量时返回 NaN。"""
    if not factor.index.equals(target.index) or not factor.columns.equals(target.columns):
        raise ValueError("RankIC 两个面板的日期和股票轴必须相同且同序")
    valid = np.isfinite(factor) & np.isfinite(target)
    f = factor.where(valid).rank(axis=1, method="average")
    y = target.where(valid).rank(axis=1, method="average")
    f = f.sub(f.mean(axis=1), axis=0)
    y = y.sub(y.mean(axis=1), axis=0)
    denom = np.sqrt((f * f).sum(axis=1) * (y * y).sum(axis=1))
    result = (f * y).sum(axis=1) / denom.where(denom > 0)
    return result.where(valid.sum(axis=1) >= max(2, min_obs))


def rank_ic_rows_torch(factor, target, min_obs: int = 100):
    """Torch 版相同契约；调用前由上层保证日期及股票轴对齐。"""
    import torch
    from gpu_ops import x_rank

    if factor.ndim != 2 or factor.shape != target.shape:
        raise ValueError("RankIC tensor 必须是相同形状的二维面板")
    valid = torch.isfinite(factor) & torch.isfinite(target)
    f = x_rank(torch.where(valid, factor, float("nan")))
    y = x_rank(torch.where(valid, target, float("nan")))
    # float32 常量行减均值可能留下舍入残差，显式检查原排名的跨度。
    spread_f = (torch.where(valid, f, float("-inf")).amax(dim=1)
                - torch.where(valid, f, float("inf")).amin(dim=1))
    spread_y = (torch.where(valid, y, float("-inf")).amax(dim=1)
                - torch.where(valid, y, float("inf")).amin(dim=1))
    f = f - torch.nanmean(f, dim=1, keepdim=True)
    y = y - torch.nanmean(y, dim=1, keepdim=True)
    denom = torch.sqrt(torch.nansum(f * f, dim=1) * torch.nansum(y * y, dim=1))
    result = torch.nansum(f * y, dim=1) / denom
    return torch.where((valid.sum(dim=1) >= max(2, min_obs)) & (spread_f > 0)
                       & (spread_y > 0) & (denom > 0),
                       result, float("nan"))
