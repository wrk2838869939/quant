#!/usr/bin/env python3
"""Phase 3 终局: 因子合成 (TRAIN+VALID 内定权重) + TEST 窗一次性终评。

纪律:
  - 合成权重只用 2014-2023 (ICIR 加权)
  - 历史 TEST 已被使用，当前冻结；本脚本保留作历史终评入口，本轮不执行
  - t 值报 Newey-West(lag=1) 校正值; DSR 用实验台账累计尝试次数折减

输出: outputs/gp_mining/TEST_REPORT.md + CSV
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path.insert(0, str(ROOT / "src"))
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]

from run_gp_mining import build_wide, parse_expr   # noqa: E402
from expr_engine import eval_node                   # noqa: E402
from rank_ic import rank_ic_rows                    # noqa: E402
from labels import purged_dates                     # noqa: E402

OUT = ROOT / "outputs/gp_mining"

# 终评对象
GP_FACTORS = [
    "sqrt(zscore(to))",
    "sqrt(ts_delta(vol,5))",
    "sqrt(ts_zscore(amt,60),20)",
    "ts_delta(amt,60)",
    "ts_min(vwap_dev,5)",
    "corr(hclose,rng,10)",
]

# 月频因子 (m1 轮)
GP_FACTORS_M20 = [
    "ts_min(vwap_dev,5)",
    "ts_std(lg_share,10)",
    # decay(sqrt(zscore(gap)),10) 已剔除: GPU/CPU decay NaN语义不一致的产物 (2026-09-09)
]


def build_all_factors(wide, horizon=5):
    facs = {}
    # 文献因子 (Phase 2 口径, 宽表版)
    facs["lit:low_turnover"] = -wide["to"].rolling(20, min_periods=15).mean()
    amp = (wide["high"] - wide["low"]) / wide["close"].shift(1)
    facs["lit:low_amp"] = -amp.rolling(21, min_periods=15).mean()
    facs["lit:low_vol"] = -wide["ret1"].rolling(20, min_periods=15).std()
    facs["lit:rev20"] = -(wide["hclose"] / wide["hclose"].shift(20) - 1)
    facs["lit:rev5"] = -(wide["hclose"] / wide["hclose"].shift(5) - 1)
    illiq = wide["ret1"].abs() / (wide["amt"] / 1e8 + 1e-6)
    facs["lit:amihud"] = illiq.rolling(20, min_periods=15).mean()
    facs["lit:mom60s5"] = wide["hclose"].shift(5) / wide["hclose"].shift(60) - 1
    # GP 新因子
    src = GP_FACTORS_M20 if horizon == 20 else GP_FACTORS
    for e in src:
        tree = parse_expr(e)
        with np.errstate(all="ignore"):
            facs[f"gp:{e[:24]}"] = eval_node(tree, wide)
    return facs


def ic_series(f: pd.DataFrame, lab: pd.DataFrame, dates: list) -> pd.Series:
    return rank_ic_rows(f.loc[dates], lab.loc[dates]).dropna()


def nw_t(x: pd.Series, lags: int = 1) -> float:
    """Newey-West t 值。"""
    x = x.dropna().values
    n = len(x)
    mu = x.mean()
    g0 = np.var(x, ddof=1)
    gv = g0
    for l in range(1, lags + 1):
        g = np.mean((x[l:] - mu) * (x[:-l] - mu))
        gv += 2 * g
    se = np.sqrt(max(gv, 1e-12) / n)
    return mu / se


def main():
    t0 = time.time()
    print("构建宽表...", flush=True)
    cols_map = dict(ret1="pct_chg", vol="vol_share", amt="amount_yuan", to="turnover_rate",
                    high="high", low="low", close="close", hclose="hfq_close",
                    buy_lg="buy_lg_amount_yuan", sell_lg="sell_lg_amount_yuan",
                    buy_elg="buy_elg_amount_yuan", net_mf="net_mf_amount_yuan",
                    circ_mv="circ_mv_yuan", volume_ratio="volume_ratio",
                    pe="pe_ttm", pb="pb")
    wide = build_wide(cols_map)
    wide["vwap_dev"] = (wide["amt"] / wide["vol"]) / wide["close"] - 1.0
    wide["ret1"] = wide["ret1"] / 100.0
    wide["rng"] = (wide["high"] - wide["low"]) / wide["close"]
    wide["gap"] = wide["close"] / wide["close"].shift(1) - 1.0
    wide["mf_imb"] = (wide["buy_lg"] - wide["sell_lg"]) / (wide["buy_lg"] + wide["sell_lg"] + 1e-6)
    wide["lg_share"] = (wide["buy_lg"] + wide["sell_lg"]) / (wide["amt"] + 1e-6)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    args2, _ = ap.parse_known_args()
    H = args2.horizon
    lab = wide["hclose"].shift(-H) / wide["hclose"] - 1.0
    lab = lab.sub(lab.mean(axis=1), axis=0)
    lab = lab.rank(axis=1, pct=True)

    facs = build_all_factors(wide, H)
    if "pe" in wide:
        facs["lit:ep"] = 1.0 / wide["pe"].where(wide["pe"] > 0)
        facs["lit:bp"] = 1.0 / wide["pb"].where(wide["pb"] > 0)

    all_dates = wide["close"].index
    tv_dates = purged_dates(all_dates[all_dates <= pd.Timestamp("2023-12-31")], H)
    te_dates = [d for d in all_dates if d >= pd.Timestamp("2024-01-01")]
    tv5 = [d for i, d in enumerate(tv_dates) if i % H == 0]
    te5 = [d for i, d in enumerate(te_dates) if i % H == 0]

    print(f"因子 {len(facs)} 条 | TRAIN+VALID {len(tv5)} 截面 | TEST {len(te5)} 截面 "
          f"({te_dates[0].date()}~{te_dates[-1].date()})", flush=True)

    # ---- 合成权重 (仅 TRAIN+VALID) ----
    print("\n合成权重 (ICIR 加权, TRAIN+VALID)...", flush=True)
    weights = {}
    for name, f in facs.items():
        ic = ic_series(f, lab, tv5)
        w = ic.mean() / (ic.std() + 1e-12)
        weights[name] = w
        print(f"  {name:32s} IC {ic.mean():+.4f} ICIR {w:+.2f}")

    # 合成因子: 截面 zscore 加权
    def composite(date_list):
        out = None
        for name, f in facs.items():
            sub = f.loc[date_list]
            m, s = sub.mean(axis=1), sub.std(axis=1)
            z = sub.sub(m, axis=0).div(s.replace(0, np.nan), axis=0)
            out = z * weights[name] if out is None else out + z * weights[name]
        return out

    comp_tv = composite(tv5)
    comp_te = composite(te5)
    facs["SYNTH:icir_weighted"] = pd.concat(
        [comp_tv, comp_te]).reindex(facs["lit:rev20"].index)
    # 落盘合成因子 (全史, 回测器用)
    facs["SYNTH:icir_weighted"].to_parquet(OUT / f"synth_factor_h{H}.parquet")
    weights_s = pd.Series(weights).sort_values()
    weights_s.to_csv(OUT / f"synth_weights_h{H}.csv")
    print(f"合成因子已落盘 -> {OUT}/synth_factor_h{H}.parquet")

    # ---- TEST 终评 ----
    print("\n===== TEST 窗终评 (2024-01 ~ 2026-09) =====", flush=True)
    rows = []
    for name, f in facs.items():
        ic_te = ic_series(f, lab, te5)
        ic_tv = ic_series(f, lab, tv5)
        t_nw = nw_t(ic_te)
        rows.append(dict(factor=name, tv_ic=round(ic_tv.mean(), 4),
                         te_ic=round(ic_te.mean(), 4),
                         te_icir=round(ic_te.mean() / (ic_te.std() + 1e-12), 2),
                         te_t=round(ic_te.mean() / (ic_te.std() / np.sqrt(len(ic_te))), 1),
                         te_t_nw=round(t_nw, 1),
                         te_n=len(ic_te),
                         sign_stable=bool(np.sign(ic_te.mean()) == np.sign(ic_tv.mean()))))
    rep = pd.DataFrame(rows)

    # DSR (对合成因子)
    ic_synth = ic_series(facs["SYNTH:icir_weighted"], lab, te5)
    from ledger import deflated_sharpe_report
    dsr = deflated_sharpe_report(ic_synth.mean() / (ic_synth.std() + 1e-12),
                            len(ic_synth), rets=ic_synth.values)

    # TEST 分层 (合成因子, 等权持有5日, 双边成本)
    print("\n合成因子 TEST 分层...", flush=True)
    f = facs["SYNTH:icir_weighted"]
    fwd = (wide["hclose"].shift(-H) / wide["hclose"] - 1.0)
    q = f.loc[te5].rank(axis=1, pct=True)
    groups = {}
    for g in range(10):
        mask = (q > g / 10) & (q <= (g + 1) / 10)
        groups[g + 1] = fwd.loc[te5][mask].mean(axis=1).mean()
    ls = groups[10] - groups[1]
    top_set = f.loc[te5][q > 0.9].notna()
    daily_top = f.loc[te5][q > 0.9].count(axis=1)
    hold = 5 / 244  # 周持有近似
    gross_top = groups[10] / hold * 0.2  # 每期收益按组合频率年化(粗)
    lines = []
    lines.append("# TEST 窗终评报告 (2026-09-09, 一次性)")
    lines.append(f"\n- 面板: {len(all_dates)} 日 | TEST 截面 {len(te5)} | 合成权重定于 TRAIN+VALID\n"
                 f"- DSR: {dsr['status']} ({dsr['reason']})；IC序列不是账户收益，未估计独立试验数")
    lines.append("\n## 单因子 TEST IC\n")
    lines.append(rep.to_string(index=False))
    lines.append("\n## 合成因子十分组 (TEST, 等权持有5日/期)\n")
    lines.append(" | ".join(f"Q{g}:{v:+.4f}" for g, v in groups.items()))
    lines.append(f"\n多空(Q10-Q1) 每期: {ls:+.4f} | 年化≈{ls*244/H:+.1%}")
    lines.append(f"\n生成: factor_test_report.csv / pool CSV")
    print("\n".join(lines))

    rep.to_csv(OUT / f"TEST_factor_report_h{H}.csv", index=False)
    (OUT / f"TEST_REPORT_h{H}.md").write_text("\n".join(lines))
    print(f"\n用时 {(time.time()-t0)/60:.0f}min -> {OUT}/TEST_REPORT.md")


if __name__ == "__main__":
    main()
