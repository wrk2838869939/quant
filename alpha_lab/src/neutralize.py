"""截面预处理: 去极值 + 行业市值中性化 (调研报告 Phase 2 标准流程)。

流程 (对每个交易日的截面):
  1. 去极值: MAD 法 (中位 ± 3×1.4826×MAD)
  2. 中性化: 残差 = y - (行业哑变量 + log总市值) 回归拟合值
  3. zscore 标准化

行业归属用 PIT 成分表 (sw_l1_member_hist: in_date ≤ T < out_date)。
"""
import numpy as np
import pandas as pd


def winsorize_cross(df: pd.DataFrame, col: str, n_mad: float = 3.0) -> pd.Series:
    """按日截面 MAD 去极值。"""
    def _w(s: pd.Series) -> pd.Series:
        med = s.median()
        mad = (s - med).abs().median() * 1.4826
        if mad <= 0 or np.isnan(mad):
            return s
        return s.clip(med - n_mad * mad, med + n_mad * mad)
    return df.groupby("date")[col].transform(_w)


def neutralize(df: pd.DataFrame, col: str, industry_col: str = "sw_l1",
               mv_col: str = "log_mv", *, min_samples: int = 50,
               return_diagnostics: bool = False):
    """按日回归取残差；无效行/失败截面返回NaN，绝不退回原信号。

    return_diagnostics=True返回(残差Series, 日期诊断DataFrame)。
    诊断区分缺失暴露、样本不足、设计矩阵秩亏及数值失败。
    """
    if min_samples < 3 or not df.index.is_unique or df["date"].isna().any():
        raise ValueError("要求唯一行索引、非空日期和至少3个样本")
    out = pd.Series(np.nan, index=df.index)
    diagnostics = []
    for date, g in df.groupby("date"):
        valid = (np.isfinite(g[col]) & np.isfinite(g[mv_col])
                 & g[industry_col].notna())
        info = dict(date=date, n_rows=len(g), n_valid=int(valid.sum()),
                    n_output=0, n_industries=int(g.loc[valid, industry_col].nunique()),
                    n_parameters=0, design_rank=0, status="insufficient_samples")
        if valid.sum() >= min_samples:
            gv = g.loc[valid]
            X = pd.get_dummies(gv[industry_col].astype(str), drop_first=True).astype(float)
            # 只改条件数，不改变带截距回归的残差定义。
            mv = gv[mv_col].to_numpy(dtype=float)
            mv_sd = mv.std()
            X["__mv"] = (mv - mv.mean()) / mv_sd if mv_sd > 0 else 0.
            X.insert(0, "__const", 1.)
            info["n_parameters"] = X.shape[1]
            try:
                beta, _, rank, _ = np.linalg.lstsq(X.to_numpy(), gv[col].to_numpy(), rcond=None)
                info["design_rank"] = int(rank)
                if rank < X.shape[1]:
                    info["status"] = "rank_deficient"
                elif len(gv) <= rank:
                    info["status"] = "no_residual_degrees_of_freedom"
                else:
                    resid = gv[col].to_numpy() - X.to_numpy() @ beta
                    if np.isfinite(resid).all():
                        out.loc[gv.index] = resid
                        info.update(status="ok", n_output=len(gv))
                    else:
                        info["status"] = "nonfinite_residuals"
            except np.linalg.LinAlgError:
                info["status"] = "linear_algebra_failure"
        diagnostics.append(info)
    report = pd.DataFrame(diagnostics, columns=["date", "n_rows", "n_valid", "n_output",
        "n_industries", "n_parameters", "design_rank", "status"])
    return (out, report) if return_diagnostics else out


def zscore_cross(df: pd.DataFrame, col: str) -> pd.Series:
    """按日截面 zscore。"""
    def _z(s: pd.Series) -> pd.Series:
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else s * np.nan
    return df.groupby("date")[col].transform(_z)


def standard_factor_pipeline(df: pd.DataFrame, col: str,
                             industry_col: str = "sw_l1",
                             mv_col: str = "log_mv", *, min_samples: int = 50,
                             return_diagnostics: bool = False):
    """一站式: 去极值 → 行业市值中性化 → zscore。返回处理后序列 (新列名建议 {col}_neu)。"""
    s = winsorize_cross(df, col)
    tmp = df[[industry_col, mv_col]].copy()
    tmp["_v"] = s.values
    r, diagnostics = neutralize(tmp.assign(date=df["date"].values), "_v", industry_col, mv_col,
                                min_samples=min_samples, return_diagnostics=True)
    r.index = df.index
    tmp["_r"] = r.values
    result = zscore_cross(tmp.assign(date=df["date"].values), "_r")
    return (result, diagnostics) if return_diagnostics else result


def build_sw_membership(member_hist: pd.DataFrame) -> pd.DataFrame:
    """把进出记录展开为 (ts_code, sw_l1, in_date, out_date)。

    member_hist: fetch_sw_member.py 的输出 (con_code/index_code/in_date/out_date)。
    """
    m = member_hist.rename(columns={"con_code": "ts_code", "index_code": "sw_l1"})[
        ["ts_code", "sw_l1", "in_date", "out_date"]].copy()
    for c in ["in_date", "out_date"]:
        m[c] = pd.to_datetime(m[c], format="%Y%m%d", errors="coerce")
    m["out_date"] = m["out_date"].fillna(pd.Timestamp("2099-12-31"))
    return m.sort_values(["ts_code", "in_date"]).reset_index(drop=True)


def attach_industry_pit(df: pd.DataFrame, member_hist: pd.DataFrame) -> pd.DataFrame:
    """为面板按 date 匹配 PIT 行业归属 (区间 [in_date, out_date))。

    向量化实现: 面板按 (ts_code, date) 排序后每只股票行连续,
    对每条成分记录 searchsorted 定位区间内行号。
    成分记录按 (ts_code, in_date) 排序遍历 → 重叠区间时 in_date 更晚者覆盖
    (即"取最近生效的行业")。1227万行 ~秒级。
    """
    m = build_sw_membership(member_hist)
    df = df.sort_values(["ts_code", "date"]).reset_index(drop=True).copy()
    codes = df["ts_code"].values
    dates = df["date"].values.astype("datetime64[ns]")
    uniq, first = np.unique(codes, return_index=True)  # 排序后每只股票首行位置
    n_per = np.diff(np.append(first, len(codes)))
    pos_of = {c: i for i, c in enumerate(uniq)}
    sw = np.full(len(df), None, dtype=object)

    for row in m.itertuples(index=False):
        i = pos_of.get(row.ts_code)
        if i is None:
            continue
        s = first[i]
        dd = dates[s:s + n_per[i]]
        lo = np.searchsorted(dd, np.datetime64(row.in_date), side="left")
        hi = np.searchsorted(dd, np.datetime64(row.out_date), side="left")
        if hi > lo:
            sw[s + lo:s + hi] = row.sw_l1
    df["sw_l1"] = sw
    return df
