# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
# -*- coding: utf-8 -*-
"""P0.2 板块层传导测试 (预注册 docs/USX_DESIGN_20260918.md §五)。

主线存活条件 (预注册, 不可回改):
  美股锚 ETF 隔夜收益 → 对应 A股 SW 板块 **开盘买入者可得收益**:
    oo_h(D) = mean_m [ hfq_open_m(D+h) / hfq_open_m(D) - 1 ]   (板块内等权, h ∈ {3,5})
    oo5_rel(D) = oo5_i(D) - 全市场等权 oo5(D)                  (轮动=相对结构)
  至少 6/10 个映射板块: us_1d→oo_h 的 NW-t > 2 且 分年符号一致 ≥7/10年。
  否则主线降级 H4 (美股信号只配做风控门)。

辅助诊断: gap 吸收 (X→gap)、日内 fade (X→intraday)、cc 的虚假解释力 (X→cc, 含 gap,
开盘买入者拿不到 —— 若 X→cc 显著而 X→oo_h 不显著, 正是"纸面收益"的证据形态)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usx import us_data, alignment, sector_map  # noqa: E402
from usx.panel_slice import build_core  # noqa: E402

OUT = Path("alpha_lab_growth/outputs/usx")
START, END = "2017-01-01", "2026-09-08"


def nw_t(x, y, lags=5):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
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
    V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return beta[1], (beta[1] / se[1] if se[1] > 0 else np.nan), n


def main():
    core = build_core()
    core = core[(core["date"] >= START) & (core["date"] <= END)]
    core = core.dropna(subset=["hfq_open", "hfq_close", "sector"])
    print(f"core rows {len(core)}, dates {core['date'].nunique()}")

    # ---- 板块日度组合量 (等权, 按 D 日成分) ----
    core = core.sort_values(["code", "date"])
    core["close_hfq_prev"] = core.groupby("code")["hfq_close"].shift(1)
    core["gap"] = core["hfq_open"] / core["close_hfq_prev"] - 1
    core["intraday"] = core["hfq_close"] / core["hfq_open"] - 1
    core["ret_cc"] = core["hfq_close"] / core["close_hfq_prev"] - 1
    # 开盘买入者 oo_h: 逐股向前 h 个**该股交易日**——停牌股会错位, 用日期宽表做, 精确对齐交易日历
    dates = np.sort(core["date"].unique())
    date_pos = pd.Series(np.arange(len(dates)), index=dates)

    print("building open wide matrix ...")
    open_w = core.pivot_table(index="date", columns="code", values="hfq_open", aggfunc="last")
    open_w = open_w.reindex(dates)

    def sector_oo(h: int) -> pd.DataFrame:
        ow = open_w
        fwd = ow.shift(-h) / ow - 1  # date D 行 = open(D)->open(D+h), 仅用未来开盘价
        # 板块内等权
        tmp = core[["date", "code", "sector"]].copy()
        tmp["oo"] = fwd.stack().reindex(pd.MultiIndex.from_frame(tmp[["date", "code"]])).values
        g = tmp.dropna(subset=["oo"]).groupby(["date", "sector"])["oo"].mean().unstack()
        return g

    oo = {}
    for h in (3, 5):
        oo[h] = sector_oo(h)
        print(f"oo{h} done")

    # gap/intraday/cc 板块等权
    sec_g = core.groupby(["date", "sector"])[["gap", "intraday", "ret_cc"]].mean().unstack()

    # ---- 美股锚特征 ----
    anchors = sorted(set(sector_map.SECTOR_ANCHOR.values()))
    us_feat = {}
    for a in anchors:
        df = us_data.load(a)
        s = df.set_index("date")["ret_1d"]
        us_feat[a] = s
    # 全市场等权 oo5 (相对基准)
    mkt_oo5 = oo[5].mean(axis=1)

    a_dates = pd.DatetimeIndex(dates)
    rows = []
    yearly = {}
    for sector, anchor in sector_map.SECTOR_ANCHOR.items():
        if sector not in sec_g["gap"].columns:
            continue
        s = us_feat[anchor]
        m1 = alignment.build_us_feature_date_map(a_dates, pd.DatetimeIndex(s.index))
        x1 = m1.map(s)  # us_1d
        s5 = s.groupby(pd.Grouper(freq="5D")).sum()  # 占位, 真正用滚动 session 窗
        # 5 session 累计: 滚动对 session 序列求和后按映射取值
        cum5 = s.rolling(5).sum()
        x5 = m1.map(cum5)
        idx = pd.DatetimeIndex(dates)
        sub = pd.DataFrame({
            "us1": x1.reindex(idx), "us5": x5.reindex(idx),
            "gap": sec_g["gap"][sector].reindex(idx),
            "intraday": sec_g["intraday"][sector].reindex(idx),
            "cc": sec_g["ret_cc"][sector].reindex(idx),
            "oo3": oo[3][sector].reindex(idx),
            "oo5": oo[5][sector].reindex(idx),
        })
        sub["oo5_rel"] = sub["oo5"] - mkt_oo5.reindex(idx)
        sub = sub.loc[START:END]
        for x in ["us1", "us5"]:
            for y in ["gap", "intraday", "cc", "oo3", "oo5", "oo5_rel"]:
                b, t, n = nw_t(sub[x].values, sub[y].values)
                if y in ("oo3", "oo5", "oo5_rel"):
                    # 分年符号
                    yr_sign = sub[[x, y]].dropna().groupby(sub[[x, y]].dropna().index.year).apply(
                        lambda g: np.sign(nw_t(g[x].values, g[y].values)[0]))
                    yearly[(sector, x, y)] = int(yr_sign[yr_sign == np.sign(t if t == t else 0)].sum())
                rows.append({"sector": sector, "anchor": anchor, "x": x, "y": y,
                             "beta": b, "nw_t": t, "n": n})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "p0_2_sector.csv", index=False)

    main_tab = res[res["x"] == "us1"].pivot_table(index="sector", columns="y", values="nw_t")
    main_tab = main_tab[["gap", "intraday", "cc", "oo3", "oo5", "oo5_rel"]]
    print("\n=== P0.2 NW-t: 锚ETF us_1d → 板块各口径 (2017-2026) ===")
    print(main_tab.round(2).to_string())

    # 预注册判据统计
    crit = res[(res["x"] == "us1") & (res["y"].isin(["oo3", "oo5", "oo5_rel"]))]
    pass_t = crit[crit["nw_t"] > 2].groupby("sector")["y"].count()
    print("\n=== 主线存活统计 (判据: ≥6/10板块 t>2) ===")
    print(f"映射板块数: {crit['sector'].nunique()},  t>2 的 (板块,口径) 数: {(crit['nw_t'] > 2).sum()}")
    print(f"t>2 的板块数: {len(pass_t)}")
    print(crit.sort_values("nw_t", ascending=False).head(15).to_string(index=False))
    print("\nsaved -> outputs/usx/p0_2_sector.csv")


if __name__ == "__main__":
    main()
