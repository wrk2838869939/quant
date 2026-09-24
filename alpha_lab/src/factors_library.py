"""文献因子库 (Phase 2 管线验证用) —— 每条因子带预期符号与出处。

里程碑标准: 符号与文献一致、|RankIC| 0.02~0.08。复现不出文献结果之前不开始自动挖掘。
"""
import numpy as np
import pandas as pd

# (名称, 定义函数, 预期IC符号, 出处)
# 预期符号以 A 股研究为准 (短期反转/低换手/低波在 A 股强)。

LIBRARY = {}


def register(name, expect_sign, source):
    def deco(fn):
        LIBRARY[name] = dict(fn=fn, sign=expect_sign, source=source)
        return fn
    return deco


def _g(df, col):
    """按 ts_code 分组 helper。"""
    return df.sort_values(["ts_code", "date"]).groupby("ts_code")[col]


@register("ideal_amplitude_21d", -1, "开源证券微观结构系列: 理想振幅(简版)")
def f_ideal_amp(df):
    """理想振幅: 21日平均日振幅 (high-low)/pre_close。高振幅 → 低收益。"""
    amp = (df["high"] - df["low"]) / df["pre_close"]
    s = df.assign(_a=amp.values).sort_values(["ts_code", "date"])
    return s.groupby("ts_code")["_a"].transform(lambda x: x.rolling(21, min_periods=15).mean())


@register("reversal_5d", -1, "A股短期反转 (Jegadeesh 1990 A股版; 量价指纹/研报共识)")
def f_rev5(df):
    p = "hfq_close" if "hfq_close" in df.columns else "close"
    d = df.sort_values(["ts_code", "date"])
    return d.groupby("ts_code")[p].transform(lambda x: x / x.shift(5) - 1)


@register("reversal_20d", -1, "A股中期反转 (Liu-Stambaugh-Yuan 2019 中国版动量弱)")
def f_rev20(df):
    p = "hfq_close" if "hfq_close" in df.columns else "close"
    d = df.sort_values(["ts_code", "date"])
    return d.groupby("ts_code")[p].transform(lambda x: x / x.shift(20) - 1)


@register("momentum_60d_skip5", -1, "12-1动量A股版 (LSY2019: A股月频动量为负=反转; Phase2实证IC-0.034确认)")
def f_mom60(df):
    p = "hfq_close" if "hfq_close" in df.columns else "close"
    d = df.sort_values(["ts_code", "date"])
    return d.groupby("ts_code")[p].transform(lambda x: x.shift(5) / x.shift(60) - 1)


@register("amihud_20d", +1, "Amihud 2002 非流动性 |ret|/amount")
def f_amihud(df):
    illiq = df["pct_chg"].abs() / (df["amount_yuan"] / 1e8 + 1e-6)
    d = df.assign(_i=illiq.values).sort_values(["ts_code", "date"])
    return d.groupby("ts_code")["_i"].transform(lambda x: x.rolling(20, min_periods=15).mean())


@register("turnover_ma20", -1, "低换手率溢价 (A股研报共识; 华泰/东方多期验证)")
def f_turnover(df):
    d = df.sort_values(["ts_code", "date"])
    return d.groupby("ts_code")["turnover_rate"].transform(
        lambda x: x.rolling(20, min_periods=15).mean())


@register("volatility_20d", -1, "低波动异象 (Baker-Bradley-Wurgler 2011; A股显著)")
def f_vol(df):
    d = df.sort_values(["ts_code", "date"])
    return d.groupby("ts_code")["pct_chg"].transform(
        lambda x: x.rolling(20, min_periods=15).std())


@register("ep_ttm", +1, "盈利价值 1/PE_TTM (Fama-French 1992; CH-3)")
def f_ep(df):
    return 1.0 / df["pe_ttm"].where(df["pe_ttm"] > 0)


@register("bp_lf", +1, "账面价值 1/PB (CH-3 剔壳价值前的基准版)")
def f_bp(df):
    return 1.0 / df["pb"].where(df["pb"] > 0)


@register("size_log", +1, "规模效应 (LSY 2019; ⚠️勿中性化评估——log_mv即中性化变量; 2017后弱化+2024微盘危机, 仅作风险披露)", raw_eval=True)
def f_size(df):
    return np.log(df["circ_mv_yuan"])


def compute_all(df: pd.DataFrame, neutralize_fn=None) -> pd.DataFrame:
    """批量计算库内因子。neutralize_fn(df, col) -> Series 可选。"""
    out = df.copy()
    for name, spec in LIBRARY.items():
        try:
            out[name] = spec["fn"](out)
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
            out[name] = np.nan
    return out
