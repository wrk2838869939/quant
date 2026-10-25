#!/usr/bin/env python3
"""候选因子池体检 (入库前, 不碰 TEST)。

1. 因子间相关矩阵 (TRAIN+VALID 2014-2023)
2. 逐年度 RankIC (2014~2023, 重点 2015/2018 危机年稳定性)
3. Harvey 门槛对照: |t| vs 3 (多重检验提醒: 台账 n_trials 已累计)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path.insert(0, str(ROOT / "src"))
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]

from run_gp_mining import build_wide, parse_expr          # noqa: E402
from expr_engine import eval_node                          # noqa: E402
from rank_ic import rank_ic_rows                          # noqa: E402
from labels import purged_dates                           # noqa: E402

CANDIDATES = [
    "sqrt(zscore(to))",
    "sqrt(ts_delta(vol,5))",
    "sqrt(ts_zscore(amt,60),20)",
    "ts_delta(amt,60)",
    "ts_min(vwap_dev,5)",
    "corr(hclose,rng,10)",
]
OUT = ROOT / "outputs/gp_mining"


def main():
    cols_map = dict(ret1="pct_chg", vol="vol_share", amt="amount_yuan", to="turnover_rate",
                    high="high", low="low", close="close", hclose="hfq_close",
                    buy_lg="buy_lg_amount_yuan", sell_lg="sell_lg_amount_yuan",
                    buy_elg="buy_elg_amount_yuan", net_mf="net_mf_amount_yuan",
                    circ_mv="circ_mv_yuan", volume_ratio="volume_ratio")
    print("构建宽表...", flush=True)
    wide = build_wide(cols_map, end_date="2023-12-31")
    wide["vwap_dev"] = (wide["amt"] / wide["vol"]) / wide["close"] - 1.0
    wide["ret1"] = wide["ret1"] / 100.0
    wide["rng"] = (wide["high"] - wide["low"]) / wide["close"]

    lab = wide["hclose"].shift(-5) / wide["hclose"] - 1.0
    lab = lab.sub(lab.mean(axis=1), axis=0)
    lab = lab.rank(axis=1, pct=True)

    dates = wide["close"].index
    tr = purged_dates(dates[dates <= pd.Timestamp("2021-12-31")], h=5)
    va = purged_dates(dates[(dates >= pd.Timestamp("2022-01-01"))
                            & (dates <= pd.Timestamp("2023-12-31"))], h=5)
    step_dates = list(tr[::5]) + list(va[::5])

    facs = {}
    for e in CANDIDATES:
        tree = parse_expr(e)
        with np.errstate(all="ignore"):
            facs[e] = eval_node(tree, wide)

    # 1. 相关矩阵 (采样截面平均 |corr|)
    n = len(CANDIDATES)
    M = np.zeros((n, n))
    sub = [f.loc[step_dates] for f in facs.values()]
    for i in range(n):
        for j in range(n):
            M[i, j] = rank_ic_rows(sub[i].loc[step_dates[::4]],
                                   sub[j].loc[step_dates[::4]]).mean()
    print("\n== 因子间平均截面相关 ==")
    print(pd.DataFrame(M, index=[c[:18] for c in CANDIDATES], columns=[c[:10] for c in CANDIDATES]).round(2).to_string())

    # 2. 年度 IC
    print("\n== 年度 RankIC (TRAIN+VALID, 周采样) ==")
    rows = {}
    for e, f in facs.items():
        fr = f.loc[step_dates]
        yr_ics = {}
        for y in range(2014, 2024):
            yy = [d for d in step_dates if d.year == y]
            if not yy:
                continue
            g = fr.loc[yy]
            yl = lab.loc[yy]
            ic = rank_ic_rows(g, yl)
            yr_ics[y] = ic.mean()
        rows[e] = yr_ics
    yr = pd.DataFrame(rows).T
    sig = np.sign(yr.iloc[:, :10])
    yr["符号一致率"] = sig.eq(sig.apply(lambda r: np.sign(r.median()), axis=1)).mean(axis=1)
    yr["全期IC"] = yr.iloc[:, :10].mean(axis=1)
    print(yr.round(4).to_string())

    yr.to_csv(OUT / "pool_check_yearly_ic.csv")
    pd.DataFrame(M, index=CANDIDATES, columns=CANDIDATES).to_csv(OUT / "pool_check_corr.csv")
    print(f"\n-> {OUT}/pool_check_*.csv")


if __name__ == "__main__":
    main()
