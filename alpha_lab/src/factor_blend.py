"""固定因子池的截面合成；训练权重外部传入，不在评估窗口拟合。"""
import numpy as np
import pandas as pd


def blend_factors(factors, weights, method, families=None):
    """保留共同完整样本；排名合成限制极端值影响，族等权消除数量偏置。"""
    if not factors or set(factors) != set(weights):
        raise ValueError("因子与冻结权重必须一一对应")
    first = next(iter(factors.values()))
    for name, frame in factors.items():
        if not frame.index.equals(first.index) or not frame.columns.equals(first.columns):
            raise ValueError("因子日期/股票轴不一致")
        if not np.isfinite(weights[name]):
            raise ValueError("权重必须有限")
    if method not in {"z_icir", "rank_icir", "family_rank"}:
        raise ValueError(f"未知合成方法 {method}")
    transformed = {}
    for name, frame in factors.items():
        frame = frame.replace([np.inf, -np.inf], np.nan)
        if method == "z_icir":
            values = frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1).replace(0, np.nan), axis=0)
        else:
            values = frame.rank(axis=1, pct=True) - 0.5
        transformed[name] = values * (np.sign(weights[name]) if method == "family_rank" else weights[name])
    if method == "family_rank":
        members = [name for group in (families or {}).values() for name in group]
        if len(members) != len(set(members)) or set(members) != set(factors) or any(not g for g in families.values()):
            raise ValueError("因子族必须非空，且恰好覆盖每个因子一次")
        return sum(sum(transformed[name] for name in group) / len(group)
                   for group in families.values()) / len(families)
    return sum(transformed.values())
