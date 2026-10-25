#!/usr/bin/env python3
"""Phase 2: 复现文献因子, 验证研究管线 (管线 audit)。

对 factors_library 全部因子: 计算 → 去极值/行业市值中性化/zscore → RankIC 诊断 →
与文献预期符号对照 → 实验台账记录 + DSR。

判定标准 (调研报告): 符号一致 且 |RankIC| ∈ [0.01, 0.10] → 管线通过。
任一文献因子方向复现失败 → 停止, 检查数据/口径, 不进入自动挖掘。

用法: python replicate_factors.py [--panel data/panel/daily_panel.parquet]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path.insert(0, str(ROOT / "src"))

from factors_library import LIBRARY, compute_all          # noqa: E402
from labels import excess_rank_labels                      # noqa: E402
from neutralize import standard_factor_pipeline, attach_industry_pit  # noqa: E402
from eval_factor import evaluate_factor                    # noqa: E402
from ledger import ExperimentLedger, deflated_sharpe_report       # noqa: E402

PANEL = ROOT / "data/panel/daily_panel.parquet"
SW_HIST = ROOT / "data/raw/sw_l1_member_hist.parquet"
OUT = ROOT / "outputs/phase2_replication"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    led = ExperimentLedger()

    print("读面板...", flush=True)
    cols = ["ts_code", "date", "open", "high", "low", "close", "pre_close", "pct_chg",
            "hfq_close", "amount_yuan", "turnover_rate", "pe_ttm", "pb", "circ_mv_yuan"]
    df = pd.read_parquet(PANEL, columns=cols)
    print(f"  {len(df):,} 行 | {df['ts_code'].nunique()} 只 | "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")

    print("PIT 行业归属...", flush=True)
    sw = pd.read_parquet(SW_HIST)
    df = attach_industry_pit(df, sw)
    df["log_mv"] = np.log(df["circ_mv_yuan"].where(df["circ_mv_yuan"] > 0))

    print("标签...", flush=True)
    df = excess_rank_labels(df, horizons=(5, 20))

    print("因子计算...", flush=True)
    df = compute_all(df)

    rows = []
    for name, spec in LIBRARY.items():
        t1 = time.time()
        neu, neutralization = standard_factor_pipeline(df, name, return_diagnostics=True)
        neutralization.to_csv(OUT / f"{name}_neutralization_status.csv", index=False)
        d2 = df.copy()
        d2[f"{name}_neu"] = neu.values
        rep = evaluate_factor(d2, f"{name}_neu", label="fwd5_excess_rank")
        sign_ok = np.sign(rep.ic_mean) == spec["sign"] if rep.ic_mean != 0 else False
        mag_ok = 0.005 <= abs(rep.ic_mean) <= 0.10
        verdict = "PASS" if (sign_ok and mag_ok) else ("SIGN_FLIP" if not sign_ok else "WEAK/TOTAL")
        n = led.log(name, "fwd5_excess_rank",
                    f"{df['date'].min().date()}~{df['date'].max().date()}",
                    rep.ic_mean, rep.icir, rep.ic_t, rep.ls_ret.mean() * 244,
                    params={"neutralized": "sw_l1+log_mv", "source": spec["source"]},
                    verdict=verdict)
        dsr = deflated_sharpe_report(rep.icir, len(rep.ic), rets=rep.ic.values)
        rows.append(dict(factor=name, source=spec["source"], expect_sign=spec["sign"],
                         rank_ic=round(rep.ic_mean, 4), icir=round(rep.icir, 2),
                         ic_t=round(rep.ic_t, 1), dsr_p=dsr["p_value"],
                         dsr_status=dsr["status"], dsr_reason=dsr["reason"],
                         sign_ok=sign_ok, verdict=verdict))
        print(f"  {name}: IC {rep.ic_mean:+.4f} (预期{'+' if spec['sign']>0 else '-'}) "
              f"t={rep.ic_t:+.1f} → {verdict}  [{time.time()-t1:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "replication_summary.csv", index=False)
    print("\n" + res.to_string(index=False))
    n_pass = (res["verdict"] == "PASS").sum()
    print(f"\n通过 {n_pass}/{len(res)} | 用时 {(time.time()-t0)/60:.0f}min")
    print(f"-> {OUT/'replication_summary.csv'}")


if __name__ == "__main__":
    main()
