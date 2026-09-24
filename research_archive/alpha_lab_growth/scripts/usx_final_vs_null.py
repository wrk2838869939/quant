# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
# -*- coding: utf-8 -*-
"""终局对比: 策略变体 vs 同约束随机零假设 (预注册判据见 USX_DESIGN §六)。

变体:
  V1 全合成: mf+pos−turn, 板块动量top6, gate+gapfilter, h=5
  V2 去mf:   pos−turn, 板块动量top6
  V3 纯规避: −turn_lv −turn_chg, 无板块过滤 (存活因子的真实形态)
  V0 随机:   同宇宙同执行约束均匀随机 9 只, 200 种子
判据: 变体 CAGR 净值 ≥ 随机分布 p95, 且年度 ≥7/10 为正。
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
H = 5
N_POS = 9
SEEDS = 200


def build_all():
    core = build_core()
    core = core[(core["date"] >= "2015-06-01") & (core["date"] <= END)]
    core = core.sort_values(["code", "date"]).reset_index(drop=True)
    g = core.groupby("code", sort=False)
    core["turn_lv"] = g["turnover_rate"].transform(lambda s: s.rolling(5).mean())
    t60 = g["turnover_rate"].transform(lambda s: s.rolling(60).mean())
    core["turn_chg"] = core["turn_lv"] / t60 - 1
    hh = g["hfq_close"].transform(lambda s: s.rolling(250).max())
    core["pos_250d"] = core["hfq_close"] / hh
    net5 = g["net_lgelg_yuan"].transform(lambda s: s.rolling(5).sum())
    core["mf_5d"] = net5 / core["circ_mv_yuan"]
    core["ret_cc"] = core["hfq_close"].pct_change()

    core["cap_rank"] = core.groupby("date")["circ_mv_yuan"].rank(pct=True)
    core["amt_rank"] = core.groupby("date")["amount_yuan"].rank(pct=True)
    age = (core["date"] - core["list_date"]).dt.days
    ok = ((age >= 180) & (~core["is_st"]) & core["cap_rank"].between(0.30, 0.95)
          & (core["amt_rank"] >= 0.20) & core["sector"].notna()
          & core["turn_lv"].notna() & core["turn_chg"].notna()
          & core["pos_250d"].notna() & core["mf_5d"].notna())
    uni = core[ok & (core["date"] >= START)].copy()
    print("universe rows:", len(uni), "days:", uni["date"].nunique())

    dates = pd.DatetimeIndex(np.sort(core["date"].unique()))
    dates = dates[(dates >= START) & (dates <= END)]

    def pv(col):
        w = core.pivot_table(index="date", columns="code", values=col, aggfunc="last")
        return w.reindex(dates)

    open_hfq, close_hfq = pv("hfq_open"), pv("hfq_close")
    raw_open, raw_pc = pv("open"), pv("pre_close")
    board = core.drop_duplicates(["date", "code"]).set_index(["date", "code"])["board"]
    lim = board.map({"main": 0.10, "chinext": 0.20, "star": 0.20}).unstack().reindex(
        index=dates, columns=open_hfq.columns)
    chinext = lim.columns.str.startswith(("300", "301"))
    lim = lim.where(~(chinext & (dates < pd.Timestamp("2020-08-24")).reshape(-1, 1)), 0.10)
    return uni, dates, open_hfq.ffill(), close_hfq.ffill(), raw_open, raw_pc, lim


def execution_ok_picks(universe_day_codes, d, raw_open, raw_pc, lim, cap=None):
    """执行层过滤后的可选池 (开盘有价/非涨停/非巨高开)。cap=None 不过截断(随机池用)。"""
    ok_codes = []
    for c in universe_day_codes:
        op = raw_open.at[d, c] if c in raw_open.columns else np.nan
        pc = raw_pc.at[d, c] if c in raw_pc.columns else np.nan
        if not (np.isfinite(op) and np.isfinite(pc)):
            continue
        lu = round(pc * 1.10, 2)
        if op >= lu - 1e-9:
            continue
        if op / pc - 1 > 0.05:
            continue
        ok_codes.append(c)
        if cap is not None and len(ok_codes) >= cap:
            break
    return ok_codes


def run_variant(name, picks, mats):
    open_hfq, close_hfq, raw_open, raw_pc, lim = mats
    bt = LedgerBacktest(BTConfig(hold_days=H, max_positions=N_POS),
                        open_hfq=open_hfq, raw_open=raw_open, raw_pre_close=raw_pc,
                        limit_pct=lim, close_hfq=close_hfq, picks=picks)
    res = bt.run()
    nav = res["nav"]["nav"]
    trades = res["trades"]
    years = nav.groupby(nav.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    return {"name": name, "cagr": res["stats"]["cagr"], "mdd": res["stats"]["mdd"],
            "sharpe": res["stats"]["sharpe"],
            "n_trades": len(trades),
            "win": (trades["ret_net"] > 0).mean() if len(trades) else np.nan,
            "pos_years": int((years > 0).sum()), "n_years": len(years),
            "final": nav.iloc[-1] / nav.iloc[0]}, years


def main():
    uni, dates, open_hfq, close_hfq, raw_open, raw_pc, lim = build_all()
    mats = (open_hfq, close_hfq, raw_open, raw_pc, lim)

    # 因子截面 rank (≤D-1)
    def xz(col):
        mu = uni.groupby("date")[col].transform("mean")
        sd = uni.groupby("date")[col].transform("std")
        return ((uni[col] - mu) / sd).values
    uni = uni.assign(z_turn=xz("turn_lv"), z_turnchg=xz("turn_chg"),
                     z_pos=xz("pos_250d"), z_mf=xz("mf_5d"))
    uni_i = uni.set_index(["date", "code"])

    # 板块动量 top6 (≤D-1)
    sec_mom = uni.dropna(subset=["ret_cc"]).groupby(["date", "sector"])["ret_cc"].mean().unstack()
    sec_rank = sec_mom.rolling(20).sum().shift(1).rank(axis=1, ascending=False)
    top_secs_by_d = {d: set(sec_rank.loc[d].dropna()[sec_rank.loc[d] <= 6].index)
                     for d in dates if d in sec_rank.index}

    # 风控门
    ashr = us_data.load("ASHR").set_index("date")["ret_1d"]
    spx = us_data.load("SPX").set_index("date")["ret_1d"]
    m1 = alignment.build_us_feature_date_map(dates, pd.DatetimeIndex(ashr.index)).map(ashr)
    m2 = alignment.build_us_feature_date_map(dates, pd.DatetimeIndex(spx.index)).map(spx)
    gate = ((m1.reindex(dates) <= -0.03) | (m2.reindex(dates) <= -0.025)).fillna(False)

    prev = pd.Series(dates, index=dates).shift(1)
    uni_codes_by_prev = {d: grp.index for d, grp in uni_i.groupby(level=0)}

    # 每日执行可行集合 (开盘有价/非涨停开盘/非巨高开) — 变体与随机共用同一道闸
    ok_set_by_day = {}
    cand_by_day = {}
    for d, dp in prev.items():
        if pd.isna(dp) or dp not in uni_codes_by_prev:
            continue
        grp = uni_i.loc[dp]
        codes = grp.index.to_numpy()
        if len(codes) < 30:
            continue
        okl = execution_ok_picks(codes, d, raw_open, raw_pc, lim)
        ok_set_by_day[d] = set(okl)
        cand_by_day[d] = okl

    def picks_from_scores(score_of_day, sector_filter, use_gate):
        picks = {}
        for d, dp in prev.items():
            if pd.isna(dp) or dp not in uni_codes_by_prev:
                continue
            grp = uni_i.loc[dp]
            if len(grp) < 100:
                continue
            if sector_filter:
                grp = grp[grp["sector"].isin(top_secs_by_d.get(d, set()))]
                if grp.empty:
                    continue
            okset = ok_set_by_day.get(d, set())
            ranked = grp.assign(s=score_of_day(grp)).sort_values("s", ascending=False)
            lst, cnt = [], {}
            for code, row in ranked.iterrows():
                if code not in okset:
                    continue
                s = row["sector"]
                if cnt.get(s, 0) >= 3:
                    continue
                lst.append(code)
                cnt[s] = cnt.get(s, 0) + 1
                if len(lst) >= N_POS:
                    break
            if use_gate and gate.loc[d]:
                lst = []
            picks[d] = lst
        return picks

    variants = {
        "V1_full": picks_from_scores(lambda g: g["z_mf"] + g["z_pos"] - g["z_turn"], True, True),
        "V2_no_mf": picks_from_scores(lambda g: g["z_pos"] - g["z_turn"], True, True),
        "V3_avoid_only": picks_from_scores(lambda g: -g["z_turn"] - g["z_turnchg"], False, True),
    }

    rows = []
    for name, picks in variants.items():
        r, _ = run_variant(name, picks, mats)
        rows.append(r)
        print(r)

    # ---- 随机零假设 (与变体完全同约束: 同宇宙 + 同执行闸 + 引擎涨停/停牌处理) ----
    rng = np.random.default_rng(42)
    null_cagrs = []
    all_days = [d for d, c in cand_by_day.items() if len(c) >= N_POS]
    for seed in range(SEEDS):
        picks = {d: list(rng.choice(cand_by_day[d], size=N_POS, replace=False))
                 for d in all_days}
        r, years = run_variant(f"rand_{seed}", picks, mats)
        null_cagrs.append(r["cagr"])
        if seed % 50 == 0:
            print(f"seed {seed}: cagr {r['cagr']:+.3f}", flush=True)
    nl = pd.Series(null_cagrs)
    print("\n=== 随机零假设 CAGR 分布 (同执行约束/同成本/同持有) ===")
    print(nl.describe(percentiles=[.05, .25, .5, .75, .95]).round(4).to_string())
    for r in rows:
        pctl = (nl < r["cagr"]).mean()
        print(f"{r['name']}: cagr={r['cagr']:+.4f} null分位={pctl:.3f} "
              f"pos_years={r['pos_years']}/{r['n_years']}")
    pd.DataFrame(rows).to_csv(OUT / "final_variants.csv", index=False)
    nl.to_csv(OUT / "final_null_cagrs.csv", index=False)
    print("saved -> outputs/usx/final_*.csv")


if __name__ == "__main__":
    main()
