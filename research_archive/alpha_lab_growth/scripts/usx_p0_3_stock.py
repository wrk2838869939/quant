# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
# -*- coding: utf-8 -*-
"""P0.3 个股层因子测试 (预注册 docs/USX_DESIGN_20260918.md §五)。

因子 (全部只用 ≤ D-1 数据, 板块内 z-score; 板块缺失时用全市场):
  mf_5d    5日主力(大单+特大单)净流入 / 流通市值     方向 +
  pos_250d close / 250日最高收盘 (52周高点近度)      方向 +
  turn_lv  5日均换手率                               方向 − (预注册)
  turn_chg turn_5d / turn_60d − 1                    方向双向 (数据定)

标签: fwd_oo_h = open(D)→open(D+h) − 全市场等权同口径 (excess), h=5 主口径。
评法: 每周三的截面 rank IC (Newey-West), D10−D1 / D10−D8..10 多头价差 (年化),
      两段子样本 (2017-2021 / 2022-2026) 同号性 + 多头尾毒性 (D10 最差年)。
过滤: 上市≥120日, 非ST, circ_mv 横截面 30-95 分位, amount_yuan 横截面 ≥20 分位。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usx.panel_slice import build_core  # noqa: E402

OUT = Path("alpha_lab_growth/outputs/usx")
START, END, H = "2017-01-01", "2026-09-08", 5


def zscore_by(g: pd.Series, groups: pd.Series) -> pd.Series:
    """组内 z-score。"""
    m = g.groupby(groups).transform("mean")
    s = g.groupby(groups).transform("std")
    return (g - m) / s.replace(0, np.nan)


def nw_t(x, lags=5):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 20:
        return np.nan, n
    x_c = x - x.mean()
    S = (x_c * x_c).sum()
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        g = (x_c[l:] * x_c[:-l]).sum()
        S += 2 * w * g
    var = S / n ** 2
    se = np.sqrt(max(var, 1e-18))
    return (x.mean() / se).item(), n


def main():
    core = build_core()
    core = core[(core["date"] >= "2015-06-01") & (core["date"] <= END)]
    core = core.sort_values(["code", "date"]).reset_index(drop=True)
    print("core:", core.shape)

    # ---- 因子 (逐股滚动) ----
    g = core.groupby("code", sort=False)
    core["turn_lv"] = g["turnover_rate"].transform(lambda s: s.rolling(5).mean())
    t60 = g["turnover_rate"].transform(lambda s: s.rolling(60).mean())
    core["turn_chg"] = core["turn_lv"] / t60 - 1
    hh = g["hfq_close"].transform(lambda s: s.rolling(250).max())
    core["pos_250d"] = core["hfq_close"] / hh
    # 5日主力净流入/流通市值
    net5 = g["net_lgelg_yuan"].transform(lambda s: s.rolling(5).sum())
    core["mf_5d"] = net5 / core["circ_mv_yuan"]

    # ---- 前向 open→open excess (标签, 仅未来开盘价; 可交易口径: 因子∈close(D) → 买入∈open(D+1)) ----
    dates = np.sort(core["date"].unique())
    ow = core.pivot_table(index="date", columns="code", values="hfq_open", aggfunc="last").reindex(dates)
    fwd = (ow.shift(-1 - H) / ow.shift(-1) - 1)
    mkt = fwd.mean(axis=1)  # 全市场等权同口径
    core["fwd_oo"] = fwd.stack().reindex(
        pd.MultiIndex.from_frame(core[["date", "code"]])).values - core["date"].map(mkt).values

    # ---- 过滤器 (D-1 时点状态) ----
    core["mkt_cap_rank"] = core.groupby("date")["circ_mv_yuan"].rank(pct=True)
    core["amt_rank"] = core.groupby("date")["amount_yuan"].rank(pct=True)
    list_age = (core["date"] - core["list_date"]).dt.days
    ok = ((list_age >= 180) & (~core["is_st"])
          & core["mkt_cap_rank"].between(0.30, 0.95)
          & (core["amt_rank"] >= 0.20)
          & core["mf_5d"].notna() & core["pos_250d"].notna()
          & core["turn_lv"].notna() & core["turn_chg"].notna()
          & core["sector"].notna() & core["fwd_oo"].notna())
    uni = core[ok].copy()
    print("universe rows:", len(uni), "days:", uni["date"].nunique())

    # ---- 板块内 z-score ----
    for f in ["mf_5d", "pos_250d", "turn_lv", "turn_chg"]:
        uni[f + "_z"] = zscore_by(uni[f], uni["sector"])

    # ---- 评法: 每周三截面 ----
    uni["dow"] = pd.to_datetime(uni["date"]).dt.dayofweek
    weds = uni[uni["dow"] == 2]
    print("weekly cross-sections:", weds["date"].nunique())

    facs = ["mf_5d_z", "pos_250d_z", "turn_lv_z", "turn_chg_z"]
    rows = []
    for f in facs:
        ic_by_day = weds.groupby("date").apply(
            lambda d: d[f].corr(d["fwd_oo"], method="spearman"), include_groups=False)
        t, n = nw_t(ic_by_day)
        rows.append({"factor": f, "ic_mean": ic_by_day.mean(), "ic_ir": ic_by_day.mean() / ic_by_day.std(),
                     "nw_t": t, "n_days": n})
    ic_tab = pd.DataFrame(rows)
    print("\n=== P0.3 周频 rank IC (fwd open→open 5d excess) ===")
    print(ic_tab.round(4).to_string(index=False))

    # 分年 IC
    print("\n=== 分年 IC 均值 ===")
    yr_tab = {}
    for f in facs:
        ic_by_day = weds.groupby("date").apply(
            lambda d: d[f].corr(d["fwd_oo"], method="spearman"), include_groups=False)
        yr_tab[f] = ic_by_day.groupby(ic_by_day.index.year).mean()
    yr_df = pd.DataFrame(yr_tab)
    print(yr_df.round(4).to_string())
    yr_df.to_csv(OUT / "p0_3_yearly_ic.csv")

    # 十分组多头价差 (年化, 全样本): D10 vs D1, D10 vs D8-10 (毒尾检查)
    print("\n=== 十分组 (per 周截面, 等权) ===")
    def decile(s: pd.Series, q: int = 10) -> pd.Series:
        if len(s) < 80:
            return pd.Series(np.nan, index=s.index)
        return np.ceil(s.rank(pct=True, method="first") * q).clip(1, q)
    dec_rows = []
    for f in facs:
        weds2 = weds.dropna(subset=[f, "fwd_oo"]).copy()
        weds2["dec"] = weds2.groupby("date")[f].transform(decile)
        dm = weds2.groupby(["date", "dec"])["fwd_oo"].mean().unstack()
        d10 = dm[10]
        row = {"factor": f,
               "D10_wk": d10.mean(), "D1_wk": dm[1].mean(),
               "D10-D1_wk": d10.mean() - dm[1].mean(),
               "D10-D810_wk": d10.mean() - dm[[8, 9, 10]].mean(axis=1).mean(),
               "D10最差年": d10.groupby(d10.index.year).mean().min(),
               "D1最好年": dm[1].groupby(dm.index.year).mean().max()}
        dec_rows.append(row)
    dec_tab = pd.DataFrame(dec_rows)
    dec_tab["D10-D1年化"] = dec_tab["D10-D1_wk"] * 52
    print(dec_tab.round(4).to_string(index=False))
    ic_tab.to_csv(OUT / "p0_3_ic.csv", index=False)
    dec_tab.to_csv(OUT / "p0_3_deciles.csv", index=False)
    print("\nsaved -> outputs/usx/p0_3_*.csv")


if __name__ == "__main__":
    main()
