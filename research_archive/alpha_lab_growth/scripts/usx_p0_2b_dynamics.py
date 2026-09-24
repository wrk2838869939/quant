# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
# -*- coding: utf-8 -*-
"""P0.2b 美股历史动态扩展 (预注册 docs/USX_P0_2B_PREREG_20260918.md, 判据不可回改)。

H_a 多周期 {1,5,20,60} → 板块 oo5_rel (FM 跨板块)
H_b 条件吸收: us_1d × HS300 趋势态 → oo5_rel
H_c 长假累积: 假日后首日 oo5 vs 平日
H_d SPX_20d 状态 → HS300 oo5 (全局门)
H_e 确认/冲突: us_1d × 板块自身 mom20 同向 → oo5_rel
全部格子输出, 不挑选。
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


def nw_t(x, y=None, lags=5):
    x = np.asarray(x, dtype=float)
    if y is None:
        v = x[~np.isnan(x)]
        n = len(v)
        if n < 15:
            return np.nan, n
        mu, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
        return (mu / se if se > 0 else np.nan), n
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], np.asarray(y, dtype=float)[ok]
    n = len(x)
    if n < 30 or x.std() == 0:
        return np.nan, n
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
    return (beta[1] / se[1] if se[1] > 0 else np.nan), n


def fm(xs: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """逐日截面回归 y ~ x (各列), 返回 beta 序列均值 NW-t (跨日 HAC)。"""
    betas = {}
    common = xs.index.intersection(y.index)
    for d in common:
        x = xs.loc[d].dropna()
        if len(x) < 8:
            continue
        yy = y.loc[d].reindex(x.index)
        ok = yy.notna()
        if ok.sum() < 8:
            continue
        X = np.column_stack([np.ones(ok.sum()), x[ok].values])
        b = np.linalg.lstsq(X, yy[ok].values, rcond=None)[0][1]
        betas[d] = b
    b = pd.Series(betas).sort_index()
    t, n = nw_t(b.values)
    return t, n, b


def main():
    core = build_core()
    core = core[(core["date"] >= START) & (core["date"] <= END)]
    core = core.dropna(subset=["hfq_open", "hfq_close", "sector"])
    core = core.sort_values(["code", "date"])
    dates = np.sort(core["date"].unique())
    idx = pd.DatetimeIndex(dates)

    # ---- 板块 oo5 (开盘买入者可得) ----
    open_w = core.pivot_table(index="date", columns="code", values="hfq_open",
                              aggfunc="last").reindex(dates)
    fwd5 = open_w.shift(-5) / open_w - 1
    tmp = core[["date", "code", "sector"]].copy()
    tmp["oo5"] = fwd5.stack().reindex(pd.MultiIndex.from_frame(tmp[["date", "code"]])).values
    sec_oo5 = tmp.dropna(subset=["oo5"]).groupby(["date", "sector"])["oo5"].mean().unstack()
    mkt_oo5 = tmp.dropna(subset=["oo5"]).groupby("date")["oo5"].mean()
    sec_rel = sec_oo5.sub(mkt_oo5, axis=0)

    # ---- HS300 状态 (≤D-1 趋势态: close vs MA60, 用昨收状态) ----
    idxdf = pd.read_parquet("alpha_lab/data/raw/index_daily.parquet")
    hs = idxdf[idxdf["ts_code"] == "000300.SH"].copy()
    hs["date"] = pd.to_datetime(hs["trade_date"], format="%Y%m%d")
    hs = hs.sort_values("date").set_index("date")
    hs["ma60"] = hs["close"].rolling(60).mean()
    state = (hs["close"] > hs["ma60"]).astype(float).shift(1)  # 昨日状态用于今日
    hs_oo5 = hs["open"].shift(-5) / hs["open"] - 1

    # ---- 美股特征 ----
    spx = us_data.load("SPX").set_index("date")["ret_1d"]
    cum = {h: spx.rolling(h).sum() for h in (1, 5, 20, 60)}

    anchors = sorted(set(sector_map.SECTOR_ANCHOR.values()))
    anchor_cum = {h: {} for h in (1, 5, 20, 60)}
    anchor_1d = {}
    for a in anchors:
        s = us_data.load(a).set_index("date")["ret_1d"]
        anchor_1d[a] = s
        for h in (1, 5, 20, 60):
            anchor_cum[h][a] = s.rolling(h).sum()

    mapped = {s: a for s, a in sector_map.SECTOR_ANCHOR.items() if s in sec_rel.columns}
    cells = []

    # ===== H_a: 多周期 =====
    for h in (1, 5, 20, 60):
        xs = pd.DataFrame({s: alignment.build_us_feature_date_map(idx, pd.DatetimeIndex(
            anchor_cum[h][a].index)).map(anchor_cum[h][a]).reindex(idx)
            for s, a in mapped.items()})
        t, n, b = fm(xs, sec_rel)
        half1 = b[b.index < "2022-01-01"]
        half2 = b[b.index >= "2022-01-01"]
        cells.append({"hyp": "H_a", "cell": f"us{h}d->oo5_rel", "t": t, "n_days": n,
                      "t_h1": nw_t(half1.values)[0], "t_h2": nw_t(half2.values)[0],
                      "beta_mean": b.mean()})
    # 稳健性: H_a 主格子 us20 剔除 |t| 最大 2 板块 (单变量逐板块)
    xs20 = pd.DataFrame({s: alignment.build_us_feature_date_map(idx, pd.DatetimeIndex(
        anchor_cum[20][a].index)).map(anchor_cum[20][a]).reindex(idx) for s, a in mapped.items()})
    per_sec = []
    for s in mapped:
        tt, _ = nw_t(xs20[s].values, sec_rel[s].reindex(idx).values)
        per_sec.append((s, tt))
    per_sec_df = pd.DataFrame(per_sec, columns=["sector", "t"]).sort_values(
        "t", key=abs, ascending=False)
    drop2 = per_sec_df.head(2)["sector"].tolist()
    t_rob, n_rob, _ = fm(xs20.drop(columns=drop2), sec_rel.drop(columns=drop2))
    cells.append({"hyp": "H_a", "cell": f"us20d->oo5_rel_去top2{drop2}", "t": t_rob,
                  "n_days": n_rob, "t_h1": np.nan, "t_h2": np.nan, "beta_mean": np.nan})

    # ===== H_b: 条件吸收 =====
    xs1 = pd.DataFrame({s: alignment.build_us_feature_date_map(idx, pd.DatetimeIndex(
        anchor_1d[a].index)).map(anchor_1d[a]).reindex(idx) for s, a in mapped.items()})
    st = state.reindex(idx)
    for tag, mask in [("state_up", st > 0.5), ("state_down", st <= 0.5)]:
        xs_m = xs1[mask.reindex(xs1.index).fillna(False).values]
        y_m = sec_rel[mask.reindex(sec_rel.index).fillna(False).values]
        t, n, b = fm(xs_m, y_m)
        h1 = b[b.index < "2022-01-01"]
        h2 = b[b.index >= "2022-01-01"]
        cells.append({"hyp": "H_b", "cell": f"us1->oo5_rel@{tag}", "t": t, "n_days": n,
                      "t_h1": nw_t(h1.values)[0], "t_h2": nw_t(h2.values)[0],
                      "beta_mean": b.mean()})

    # ===== H_c: 长假累积 =====
    gaps = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    post_hol = np.concatenate([[False], gaps >= 4])
    ph_oo5 = hs_oo5.reindex(idx)
    t_c, n_c = nw_t(ph_oo5[post_hol].dropna().values - ph_oo5[~post_hol].mean())
    pre_hol_mean = ph_oo5[~post_hol].mean()
    ph_vals = ph_oo5[post_hol].dropna()
    t_c, n_c = nw_t((ph_vals - pre_hol_mean).values)
    # 附: 累积美股(假日窗) → 假日后 oo5
    prev_used = alignment.build_us_feature_date_map(idx, pd.DatetimeIndex(spx.index))
    acc = []
    for i, d in enumerate(idx):
        if not post_hol[i]:
            acc.append(np.nan)
            continue
        prev_d = idx[i - 1]
        u_prev = prev_used.loc[prev_d]
        u_now = prev_used.loc[d]
        acc.append(spx.loc[(spx.index > u_prev) & (spx.index <= u_now)].sum())
    acc = pd.Series(acc, index=idx)
    tt, nn = nw_t(acc[post_hol].values, ph_oo5[post_hol].values)
    cells.append({"hyp": "H_c", "cell": "假日后首日oo5-平日均值", "t": t_c, "n_days": n_c,
                  "t_h1": nw_t((ph_vals[ph_vals.index < "2022-01-01"] - pre_hol_mean).values)[0],
                  "t_h2": nw_t((ph_vals[ph_vals.index >= "2022-01-01"] - pre_hol_mean).values)[0],
                  "beta_mean": (ph_vals - pre_hol_mean).mean()})
    cells.append({"hyp": "H_c附", "cell": "假日累积SPX->HS300oo5", "t": tt, "n_days": nn,
                  "t_h1": np.nan, "t_h2": np.nan, "beta_mean": np.nan})

    # ===== H_d: SPX 20d 状态 → HS300 oo5 =====
    spx20 = alignment.build_us_feature_date_map(idx, pd.DatetimeIndex(cum[20].index)).map(cum[20]).reindex(idx)
    t_d, n_d = nw_t(spx20.values, hs_oo5.reindex(idx).values)
    h1 = pd.Series(hs_oo5.reindex(idx).values, index=idx)[idx < "2022-01-01"]
    h2 = pd.Series(hs_oo5.reindex(idx).values, index=idx)[idx >= "2022-01-01"]
    x1 = spx20[idx < "2022-01-01"]
    x2 = spx20[idx >= "2022-01-01"]
    cells.append({"hyp": "H_d", "cell": "SPX20d->HS300oo5", "t": t_d, "n_days": n_d,
                  "t_h1": nw_t(x1.values, h1.values)[0], "t_h2": nw_t(x2.values, h2.values)[0],
                  "beta_mean": np.nan})
    # 附: 分状态均值差 (门控视角)
    up = spx20 > 0
    v_up = hs_oo5.reindex(idx)[up].dropna()
    v_dn = hs_oo5.reindex(idx)[~up.fillna(True)].dropna()
    t_gate, n_gate = nw_t(v_up.values - v_dn.mean())
    cells.append({"hyp": "H_d附", "cell": "oo5@SPX20d>0 − oo5@SPX20d<0", "t": t_gate,
                  "n_days": n_gate, "t_h1": np.nan, "t_h2": np.nan,
                  "beta_mean": v_up.mean() - v_dn.mean()})

    # ===== H_e: 确认/冲突交互 =====
    # 板块自身 mom20 (≤D-1): 板块等权 ret_cc 20日和
    core["ret_cc"] = core.groupby("code")["hfq_close"].transform(lambda s: s.pct_change())
    sec_cc = core.dropna(subset=["ret_cc"]).groupby(["date", "sector"])["ret_cc"].mean().unstack()
    sec_mom20 = sec_cc.rolling(20).sum().shift(1).reindex(idx)
    conf = pd.DataFrame({s: (np.sign(xs1[s]) == np.sign(sec_mom20[s])) & xs1[s].notna()
                         for s in mapped}).astype(float)
    conf[conf.isin([0.0])] = np.nan  # 缺信号/缺动量 → 不进回归
    t_e1, n_e1, b_up = fm(xs1.mul(conf), sec_rel)          # 同向
    t_e0, n_e0, b_dn = fm(xs1.mul(1 - conf), sec_rel)      # 反向
    h1 = b_up[b_up.index < "2022-01-01"]
    h2 = b_up[b_up.index >= "2022-01-01"]
    cells.append({"hyp": "H_e", "cell": "us1->oo5_rel@同向", "t": t_e1, "n_days": n_e1,
                  "t_h1": nw_t(h1.values)[0], "t_h2": nw_t(h2.values)[0], "beta_mean": b_up.mean()})
    h1 = b_dn[b_dn.index < "2022-01-01"]
    h2 = b_dn[b_dn.index >= "2022-01-01"]
    cells.append({"hyp": "H_e", "cell": "us1->oo5_rel@反向", "t": t_e0, "n_days": n_e0,
                  "t_h1": nw_t(h1.values)[0], "t_h2": nw_t(h2.values)[0], "beta_mean": b_dn.mean()})

    res = pd.DataFrame(cells)
    res.to_csv(OUT / "p0_2b_cells.csv", index=False)
    print("\n=== P0.2b 全部格子 (判据: 主格子|t|>3 且两段同号) ===")
    print(res.round(3).to_string(index=False))

    # H_a 逐板块明细 (报告义务)
    print("\n=== H_a us20d → oo5_rel 逐板块 NW-t ===")
    print(per_sec_df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
