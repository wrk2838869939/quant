# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
# -*- coding: utf-8 -*-
"""合成策略回测 (USX 终局)。

组成 (全部经过 P0 预注册裁决):
  - 板块层: A股板块自身 20 日动量 (P0.2: 美股轮动被拒绝, 用国内动量替代 — 行业动量文献先验)
  - 个股层: mf_5d(+1) + pos_250d(+1) + turn_lv(−1) 等权 (P0.3 裁决; turn_chg 死亡剔除)
  - 执行: 开盘买入; 当日开盘 gap > +5% 或开盘涨停 → 放弃 (fade 物理 + 不可成交)
  - 风控门 (非 alpha 声明): ASHR ≤ −3% 或 SPX ≤ −2.5% → 当日不开新仓
  - 持有 5 日, 开盘卖出; 开盘跌停/停牌顺延

运行: python scripts/usx_composite.py [--hold 5] [--gate 1] [--gap-filter 1] [--no-mf]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usx import us_data, alignment  # noqa: E402
from usx.panel_slice import build_core  # noqa: E402
from usx.backtest import LedgerBacktest, BTConfig  # noqa: E402

ROOT = Path("alpha_lab_growth")
OUT = ROOT / "outputs/usx"
START, END = "2017-01-01", "2026-09-08"


def build_universe(core: pd.DataFrame) -> pd.DataFrame:
    core = core.sort_values(["code", "date"]).copy()
    g = core.groupby("code", sort=False)
    core["turn_lv"] = g["turnover_rate"].transform(lambda s: s.rolling(5).mean())
    hh = g["hfq_close"].transform(lambda s: s.rolling(250).max())
    core["pos_250d"] = core["hfq_close"] / hh
    net5 = g["net_lgelg_yuan"].transform(lambda s: s.rolling(5).sum())
    core["mf_5d"] = net5 / core["circ_mv_yuan"]
    core["ret_cc"] = core["hfq_close"].pct_change()

    core["mkt_cap_rank"] = core.groupby("date")["circ_mv_yuan"].rank(pct=True)
    core["amt_rank"] = core.groupby("date")["amount_yuan"].rank(pct=True)
    age = (core["date"] - core["list_date"]).dt.days
    ok = ((age >= 180) & (~core["is_st"])
          & core["mkt_cap_rank"].between(0.30, 0.95)
          & (core["amt_rank"] >= 0.20)
          & core["turn_lv"].notna() & core["pos_250d"].notna()
          & core["mf_5d"].notna() & core["sector"].notna())
    uni = core[ok]
    return uni


def make_picks(uni: pd.DataFrame, cfg) -> dict:
    """决策日 D 的 picks: 用 D-1 及更早的因子排名 (shift 1 天)。"""
    use_mf = not cfg.get("no_mf")
    facs = ["mf_5d", "pos_250d", "turn_lv"] if use_mf else ["pos_250d", "turn_lv"]
    signs = [1, 1, -1] if use_mf else [1, -1]
    u = uni.copy()
    zsum = 0.0
    for f, s_ in zip(facs, signs):
        mu = u.groupby("date")[f].transform("mean")
        sd = u.groupby("date")[f].transform("std")
        zsum = zsum + s_ * ((u[f] - mu) / sd)
    u["score"] = zsum
    # 板块动量 (≤D-1): 板块等权 ret_cc 20日和, shift 1
    sec_mom = u.dropna(subset=["ret_cc"]).groupby(["date", "sector"])["ret_cc"].mean().unstack()
    sec_mom = sec_mom.rolling(20).sum().shift(1)
    sec_rank = sec_mom.rank(axis=1, ascending=False)

    # 风控门: ASHR/SPX 隔夜 (对齐契约: A股 D 用美股 session < D)
    dates = pd.DatetimeIndex(np.sort(u["date"].unique()))
    gate = pd.Series(False, index=dates)
    if cfg.get("gate", True):
        ashr = us_data.load("ASHR").set_index("date")["ret_1d"]
        spx = us_data.load("SPX").set_index("date")["ret_1d"]
        m1 = alignment.build_us_feature_date_map(dates, pd.DatetimeIndex(ashr.index)).map(ashr)
        m2 = alignment.build_us_feature_date_map(dates, pd.DatetimeIndex(spx.index)).map(spx)
        gate = (m1.reindex(dates) <= -0.03) | (m2.reindex(dates) <= -0.025)
        gate = gate.fillna(False)

    u = u.set_index(["date", "code"])
    picks = {}
    prev_dates = pd.Series(dates, index=dates).shift(1)
    for d, d_prev in prev_dates.items():
        if pd.isna(d_prev):
            continue
        day = u.xs(d_prev, level="date")  # D-1 截面
        if len(day) < 100:
            continue
        top_secs = sec_rank.loc[d].dropna()
        top_secs = top_secs[top_secs <= 6].index  # 动量前6板块为池
        day = day[day["sector"].isin(top_secs)]
        if day.empty:
            continue
        ranked = day.sort_values("score", ascending=False)
        # 板块上限 3
        picks_list, cnt = [], {}
        max_pos = cfg.get("max_positions", 9)
        for code, row in ranked.iterrows():
            s = row["sector"]
            if cnt.get(s, 0) >= 3:
                continue
            picks_list.append(code)
            cnt[s] = cnt.get(s, 0) + 1
            if len(picks_list) >= max_pos * 3:  # 超配候选, 执行层再淘汰
                break
        if gate.loc[d]:
            picks_list = []  # 风控门: 不开新仓
        picks[d] = picks_list
    return picks


def matrices(uni_core_all: pd.DataFrame, dates: pd.DatetimeIndex):
    """全市场(含未入宇宙股, 用于开盘价/涨跌停判断)宽矩阵。"""
    def pv(col):
        w = uni_core_all.pivot_table(index="date", columns="code", values=col, aggfunc="last")
        return w.reindex(dates).astype("float64")
    open_hfq = pv("hfq_open")
    close_hfq = pv("hfq_close")
    raw_open = pv("open")
    raw_pc = pv("pre_close")
    board = uni_core_all.drop_duplicates(["date", "code"]).set_index(["date", "code"])["board"]
    lim = board.map({"main": 0.10, "chinext": 0.20, "star": 0.20}).unstack()
    lim = lim.reindex(index=dates, columns=open_hfq.columns)
    # 创业板 2020-08-24 起 20%
    d_arr = dates
    chinext_mask = (lim.columns.str.startswith(("300", "301")))
    for i, d in enumerate(d_arr):
        if d < pd.Timestamp("2020-08-24"):
            lim.iloc[i, lim.columns.isin(lim.columns[chinext_mask])] = 0.10
    return open_hfq, close_hfq, raw_open, raw_pc, lim


def main():
    cfg = {"hold": 5, "gate": True, "gap_filter": True, "no_mf": False,
           "max_positions": 9}
    argv = sys.argv
    if "--hold" in argv:
        cfg["hold"] = int(argv[argv.index("--hold") + 1])
    if "--no-gate" in argv:
        cfg["gate"] = False
    if "--no-gapfilter" in argv:
        cfg["gap_filter"] = False
    if "--no-mf" in argv:
        cfg["no_mf"] = True

    print("cfg:", cfg)
    core = build_core()
    core = core[(core["date"] >= "2015-06-01") & (core["date"] <= END)]
    uni = build_universe(core)
    uni = uni[(uni["date"] >= START)]
    print("universe rows:", len(uni))

    picks = make_picks(uni, cfg)
    print("picks days:", len(picks), "avg candidates:",
          np.mean([len(v) for v in picks.values()]).round(1))

    dates = pd.DatetimeIndex(np.sort(core["date"].unique()))
    dates = dates[(dates >= START) & (dates <= END)]
    open_hfq, close_hfq, raw_open, raw_pc, lim = matrices(core, dates)

    # gap 过滤在执行层实现: 开盘价/昨收比 > 1.05 → 放弃; 直接改 picks 在运行时处理不了,
    # 在引擎外预处理: 把当日 gap>5% 的候选剔除 (open(D) 与 pre_close(D) 已知于竞价后)
    if cfg["gap_filter"]:
        gap_today = raw_open / raw_pc - 1
        picks = {d: [c for c in lst
                     if np.isfinite(gap_today.at[d, c]) and gap_today.at[d, c] <= 0.05]
                 for d, lst in picks.items()}

    bt = LedgerBacktest(
        BTConfig(hold_days=cfg["hold"], max_positions=cfg["max_positions"]),
        open_hfq=open_hfq, raw_open=raw_open, raw_pre_close=raw_pc,
        limit_pct=lim, close_hfq=close_hfq, picks=picks)
    res = bt.run()
    st = res["stats"]
    nav = res["nav"]
    trades = res["trades"]
    st["n_trades"] = len(trades)
    st["win_rate"] = (trades["ret_net"] > 0).mean() if len(trades) else np.nan
    st["avg_trade"] = trades["ret_net"].mean() if len(trades) else np.nan
    print("\n=== 合成策略 (h=%d, gate=%s, gapfilter=%s, mf=%s) ===" %
          (cfg["hold"], cfg["gate"], cfg["gap_filter"], not cfg["no_mf"]))
    for k, v in st.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 分年
    yr = nav["nav"].groupby(nav.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    print("\n分年收益:")
    print(yr.round(4).to_string())

    tag = f"h{cfg['hold']}_gate{int(cfg['gate'])}_gap{int(cfg['gap_filter'])}_mf{int(not cfg['no_mf'])}"
    nav.to_csv(OUT / f"bt_nav_{tag}.csv")
    if len(trades):
        trades.to_csv(OUT / f"bt_trades_{tag}.csv", index=False)
    with open(OUT / "reports" / f"bt_summary_{tag}.txt", "w") as f:
        f.write(str(st) + "\n" + yr.to_string())


if __name__ == "__main__":
    main()
