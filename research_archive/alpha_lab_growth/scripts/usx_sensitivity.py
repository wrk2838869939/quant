# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
# -*- coding: utf-8 -*-
"""敏感性: {V2, V3} × {h=3,5,10} × {全成本, 零成本} — 信号与成本拖累分解。"""
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


src = Path(__file__).resolve().parent.joinpath("usx_final_vs_null.py").read_text()
src = src.split("if __name__")[0]
mod = types.ModuleType("fnl")
mod.__dict__["__file__"] = str(Path(__file__).resolve().parent / "usx_final_vs_null.py")
exec(compile(src, "fnl", "exec"), mod.__dict__)

from usx.backtest import LedgerBacktest, BTConfig  # noqa: E402

OUT = Path("alpha_lab_growth/outputs/usx")


def main():
    uni, dates, open_hfq, close_hfq, raw_open, raw_pc, lim = mod.build_all()
    mats = (open_hfq, close_hfq, raw_open, raw_pc, lim)

    def xz(col):
        mu = uni.groupby("date")[col].transform("mean")
        sd = uni.groupby("date")[col].transform("std")
        return ((uni[col] - mu) / sd).values
    uni = uni.assign(z_turn=xz("turn_lv"), z_turnchg=xz("turn_chg"), z_pos=xz("pos_250d"))
    uni_i = uni.set_index(["date", "code"])

    prev = pd.Series(dates, index=dates).shift(1)
    uni_codes_by_prev = {d: grp.index for d, grp in uni_i.groupby(level=0)}
    ok_set_by_day, cand_by_day = {}, {}
    for d, dp in prev.items():
        if pd.isna(dp) or dp not in uni_codes_by_prev:
            continue
        grp = uni_i.loc[dp]
        codes = grp.index.to_numpy()
        if len(codes) < 30:
            continue
        okl = mod.execution_ok_picks(codes, d, raw_open, raw_pc, lim)
        ok_set_by_day[d] = set(okl)
        cand_by_day[d] = okl

    # V3 picks: 按 -z_turn - z_turnchg 排序, 无板块过滤
    # V2 picks: z_pos - z_turn, 板块动量top6 + 风控门 (与终局脚本一致)
    from usx import us_data, alignment
    sec_mom = uni.dropna(subset=["ret_cc"]).groupby(["date", "sector"])["ret_cc"].mean().unstack()
    sec_rank = sec_mom.rolling(20).sum().shift(1).rank(axis=1, ascending=False)
    dates_all = pd.DatetimeIndex(np.sort(uni["date"].unique()))
    ashr = us_data.load("ASHR").set_index("date")["ret_1d"]
    spx = us_data.load("SPX").set_index("date")["ret_1d"]
    m1 = alignment.build_us_feature_date_map(dates_all, pd.DatetimeIndex(ashr.index)).map(ashr)
    m2 = alignment.build_us_feature_date_map(dates_all, pd.DatetimeIndex(spx.index)).map(spx)
    gate = ((m1.reindex(dates_all) <= -0.03) | (m2.reindex(dates_all) <= -0.025)).fillna(False)
    prev_dates = set(uni_i.index.get_level_values(0))
    v3_picks, v2_picks = {}, {}
    for d, dp in prev.items():
        if pd.isna(dp) or dp not in prev_dates:
            continue
        grp = uni_i.loc[dp]
        if len(grp) < 100:
            continue
        okset = ok_set_by_day.get(d, set())
        r3 = grp.assign(s=-grp["z_turn"] - grp["z_turnchg"]).sort_values("s", ascending=False)
        v3_picks[d] = [c for c in r3.index if c in okset][:9]
        top6 = set(sec_rank.loc[d].dropna()[sec_rank.loc[d] <= 6].index) if d in sec_rank.index else set()
        g2 = grp[grp["sector"].isin(top6)]
        if len(g2):
            r2 = g2.assign(s=g2["z_pos"] - g2["z_turn"]).sort_values("s", ascending=False)
            lst, cnt = [], {}
            for c in r2.index:
                if c not in okset:
                    continue
                s_ = r2.at[c, "sector"]
                if cnt.get(s_, 0) >= 3:
                    continue
                lst.append(c)
                cnt[s_] = cnt.get(s_, 0) + 1
                if len(lst) >= 9:
                    break
            if gate.loc[d]:
                lst = []
            v2_picks[d] = lst
        else:
            v2_picks[d] = []

    rows = []
    for name, picks in [("V3_avoid", v3_picks), ("V2_pos_turn_sector", v2_picks)]:
        for h in (3, 5, 10):
            for cost_tag, cost_kw in [("net", {}), ("gross", {"cost_rate": 0.0, "stamp_rate": 0.0, "slippage": 0.0})]:
                bt = LedgerBacktest(BTConfig(hold_days=h, max_positions=9, **cost_kw),
                                    open_hfq=open_hfq, raw_open=raw_open, raw_pre_close=raw_pc,
                                    limit_pct=lim, close_hfq=close_hfq, picks=picks)
                res = bt.run()
                nav = res["nav"]["nav"]
                st = res["stats"]
                years = nav.groupby(nav.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
                rows.append({"variant": name, "h": h, "cost": cost_tag,
                             "cagr": st["cagr"], "mdd": st["mdd"], "sharpe": st["sharpe"],
                             "pos_years": int((years > 0).sum()), "n_years": len(years)})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)
    print("saved -> outputs/usx/sensitivity.csv")


if __name__ == "__main__":
    main()
