# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
# -*- coding: utf-8 -*-
"""P0.1 指数层吸收测试 (预注册见 docs/USX_DESIGN_20260918.md §五)。

问题: 美股隔夜收益对 A股指数的
  Y1 = 开盘 gap (open/pre_close - 1)
  Y2 = 日内 open→close
  Y3 = close→close
  Y4 = open(D)→open(D+5)          ← 开盘买入者可得部分
的解释力分布。若 Y1 吸收了绝大部分而 Y4 无残差 → 美股信号在指数层不可交易 (H4 权重上调)。

对齐契约: A股 D 日只能用美股 session ≤ D-1 (alignment.build_us_feature_date_map)。
附带对齐自检: (a) 映射严格 U < D; (b) 已知事件日复盘 (2020-03-09 美股熔断夜)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usx import us_data, alignment  # noqa: E402

ALAB = Path("alpha_lab")
OUT = Path("alpha_lab_growth/outputs/usx")


def nw_t(x: np.ndarray, y: np.ndarray, lags: int = 5) -> tuple:
    """Newey-West t 值 (HAC, Bartlett)。"""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 30 or x.std() == 0:
        return np.nan, np.nan, n
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    h = X * resid[:, None]
    S = h.T @ h
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = h[l:].T @ h[:-l]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return beta[1], (beta[1] / se[1] if se[1] > 0 else np.nan), n


def main():
    # A股指数日线
    idx = pd.read_parquet(ALAB / "data/raw/index_daily.parquet")
    hs300 = idx[idx["ts_code"] == "000300.SH"].copy()
    hs300["date"] = pd.to_datetime(hs300["trade_date"], format="%Y%m%d")
    hs300 = hs300.sort_values("date").set_index("date")
    a_dates = hs300.index

    # 美股
    spx = us_data.load("SPX")
    ixic = us_data.load("IXIC")

    # 对齐: 每个A股交易日 → 最近可得美股 session (< D)
    m_spx = alignment.build_us_feature_date_map(a_dates, pd.DatetimeIndex(spx["date"]))
    m_ixic = alignment.build_us_feature_date_map(a_dates, pd.DatetimeIndex(ixic["date"]))

    # 自检 1: 严格 U < D
    bad = (m_spx.dropna() >= pd.Series(m_spx.dropna().index, index=m_spx.dropna().index)).sum()
    print(f"[check1] 映射违反 U<D 的天数: {bad}")

    # 自检 2: 事件复盘 (2020-03-09 周一美股 -7.6%, 收盘北京 03-10 清晨)
    ev = spx.set_index("date")["ret_1d"]
    d_event = pd.Timestamp("2020-03-10")  # A股应 gap 大跌
    u_prev = m_spx.loc[d_event]
    print(f"[check2] 2020-03-10 A股使用的美股 session = {u_prev.date()}, "
          f"SPX ret = {ev.loc[u_prev]:+.4f}, "
          f"HS300 gap = {hs300['open'].loc[d_event]/hs300['pre_close'].loc[d_event]-1:+.4f}")

    # US 特征: 最近1 session 收益
    feat = pd.DataFrame(index=a_dates)
    feat["spx_1d"] = m_spx.map(ev)
    feat["ixic_1d"] = m_ixic.map(ixic.set_index("date")["ret_1d"])

    # A股 Y 变量
    o, c, pc = hs300["open"], hs300["close"], hs300["pre_close"]
    feat["gap"] = o / pc - 1
    feat["intraday"] = c / o - 1
    feat["cc"] = c / pc - 1
    feat["oo5"] = o.shift(-5) / o - 1  # 注意: 这是"当日开盘买入5日后开盘卖"的总收益(含5个隔夜)

    feat = feat.loc["2014":].dropna(subset=["spx_1d"])
    rows = []
    for x in ["spx_1d", "ixic_1d"]:
        for y in ["gap", "intraday", "cc", "oo5"]:
            b, t, n = nw_t(feat[x].values, feat[y].values)
            r = np.corrcoef(feat[x].fillna(0), feat[y].fillna(0))[0, 1]
            rows.append({"x": x, "y": y, "beta": b, "nw_t": t, "corr": r, "n": n})
    res = pd.DataFrame(rows)
    print("\n=== P0.1 全样本 (2014-2026) ===")
    print(res.to_string(index=False))

    # 分年 corr (spx_1d vs 各Y)
    print("\n=== 分年 corr(spx_1d, Y) ===")
    yr = feat.groupby(feat.index.year)[["spx_1d", "gap", "intraday", "cc", "oo5"]].corr()[
        "spx_1d"].unstack()[["gap", "intraday", "cc", "oo5"]]
    print(yr.round(3).to_string())
    yr.to_csv(OUT / "p0_1_yearly_corr.csv")

    # 关键第一性对比: Y3(close-close) vs Y4(open-open) 的解释力
    # 开盘买入者可得的是 oo5 中"gap 之后"的部分。比较:
    #   oo5 对 spx 的回归系数  vs  (oo5 - gap 对 spx) 的回归系数
    feat["oo5_exgap"] = feat["oo5"] - feat["gap"]  # 剔除已被开盘吸收的部分
    for y in ["oo5", "oo5_exgap"]:
        b, t, n = nw_t(feat["spx_1d"].values, feat[y].values)
        print(f"SPX→{y}: beta={b:+.4f} nw_t={t:+.2f}")
    res.to_csv(OUT / "p0_1_index.csv", index=False)
    print("\nsaved -> outputs/usx/p0_1_index.csv, p0_1_yearly_corr.csv")


if __name__ == "__main__":
    main()
