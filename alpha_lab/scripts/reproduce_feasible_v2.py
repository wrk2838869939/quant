#!/usr/bin/env python3
"""高可行性策略复现 v2 — 基于 datakit (预注册: docs/REPRODUCE_FEASIBLE_PREREG_20260919.md)。

与 v1 的差异: 数据接入全部走 datakit (构建期已验证: 缺日修复/公司行动核验/守卫等价隔离),
策略层零数据修复逻辑; 隔离代码在 load 层默认不可见 (候选/基准对称)。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]

from datakit import api  # noqa: E402
from portfolio import run_backtest  # noqa: E402
from style_matching import match_controls  # noqa: E402
from ledger import ExperimentLedger  # noqa: E402

OUT = ROOT / "outputs/reproduce_feasible_v2"
CFG = yaml.safe_load((ROOT / "configs/monthly_portfolio_audit.yaml").read_text())
START, END = "2014-01-01", "2023-12-31"
RIGHTS = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())


def nw_t(x, lags=5):
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < 30:
        return np.nan
    mu = x.mean()
    S = ((x - mu) ** 2).sum()
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        S += 2 * w * ((x[l:] - mu) * (x[:-l] - mu)).sum()
    return mu / np.sqrt(max(S / n ** 2, 1e-18))


def backtest(tag, plan, run_dir, capital):
    """datakit.load → run_backtest; 返回 daily.ret。"""
    plan = plan.dropna(axis=1, how="all")
    codes = sorted(plan.columns[plan.notna().any()])
    market, basics, st, limits, actions = api.load(START, END, codes=codes)
    cfg_p = dict(CFG["portfolio"], capital=capital,
                 top_n=max(1, int(plan.iloc[0].notna().sum())))
    res = run_backtest(plan, market, basics, st, limits, actions,
                       START, END, cfg_p, RIGHTS.get("rights_issues", []))
    res.daily.to_csv(run_dir / f"{tag}_daily.csv")
    return res.daily.ret, res.summary


def main():
    run_dir = OUT / pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"产出 {run_dir}", flush=True)
    grids = api.month_ends(START, END)
    cap0 = CFG["portfolio"]["capital"]

    # ---- 公共: 信号日资格掩码 + 属性快照 + EP/市值/行业 ----
    print("构建资格/属性...", flush=True)
    core = api._core()
    ep = (1.0 / core.pe_ttm.where(core.pe_ttm > 0)).to_frame("ep")
    ep["date"], ep["ts_code"] = core.date.values, core.ts_code.values
    ep = ep.set_index(["date", "ts_code"])["ep"]
    mv = core.set_index(["date", "ts_code"])["circ_mv_yuan"]
    mem = core[core.industry.notna()][["date", "ts_code", "industry"]]
    rets = core.set_index(["date", "ts_code"])["pct_chg"].div(100)
    ind_ret = core.dropna(subset=["industry"]).groupby(["date", "industry"])[
        "pct_chg"].mean().div(100).unstack().sort_index()
    mom20 = ind_ret.rolling(20, min_periods=15).mean()
    mom60 = ind_ret.rolling(60, min_periods=45).mean()

    allowed = api.investable(grids, capital=35_000_000.0, top_n=10 ** 6)
    snap = api.attrs(grids).set_index(["date", "ts_code"])

    results = {}

    def add(tag, ret, summ):
        results[tag] = (ret, summ)
        print(f"  {tag}: total {summ.get('total'):+.4f} ann {summ.get('ann'):+.4f} "
              f"mdd {summ.get('mdd'):+.4f}", flush=True)

    # ============ R1: CH-3 小盘价值 ============
    print("\n=== R1 CH-3 小盘价值 ===", flush=True)
    for scale, capital, k_cap in [("portfolio", 35_000_000.0, None),
                                  ("account", cap0, 50)]:
        sig_v_rows, sig_u_rows = {}, {}
        for d in grids:
            ok = allowed.loc[d]
            codes_d = ok.index[ok.fillna(False)]
            mvq = mv.reindex(codes_d).dropna()
            mvq = mvq[mvq > 0]
            keep = mvq[mvq >= mvq.quantile(0.30)].index          # 剔壳: 最小30%市值
            e = ep.reindex(keep).dropna()
            e = e[e > 0].sort_values(ascending=False, kind="stable")
            n30 = max(1, int(len(e) * 0.30))
            val = list(e.index[:n30])
            if k_cap:
                val = val[:k_cap]
            sig_v_rows[d] = pd.Series(np.arange(len(val), 0, -1, dtype=float), index=val)
            sig_u_rows[d] = pd.Series(1.0, index=keep)
        sig_v = pd.DataFrame(sig_v_rows).T.reindex(columns=allowed.columns)
        sig_u = pd.DataFrame(sig_u_rows).T.reindex(columns=allowed.columns)
        r, s = backtest(f"R1v_{scale}", sig_v, run_dir, capital)
        add(f"R1v_{scale}", r, s)
        rb, sb = backtest(f"R1u_{scale}", sig_u, run_dir, capital)
        add(f"R1u_{scale}", rb, sb)
        # 附报: 规模+流动性匹配对照 (组合级)
        if scale == "portfolio":
            ctrl_rows = {}
            for d in grids:
                sel = [c for c in sig_v_rows[d].index if (d, c) in snap.index]
                uni = snap.loc[d]
                uni = uni[uni.index.isin(allowed.loc[d][allowed.loc[d].fillna(False)].index)]
                if len(uni) < 100 or not sel:
                    continue
                m = match_controls(uni, sel, ["log_mv", "log_amount20"], 20260919)
                ctl = m.control.dropna().tolist()
                ctrl_rows[d] = pd.Series(np.arange(len(ctl), 0, -1, dtype=float), index=ctl)
            sig_c = pd.DataFrame(ctrl_rows).T.reindex(columns=allowed.columns)
            rc, sc = backtest("R1c_portfolio", sig_c, run_dir, capital)
            add("R1c_portfolio", rc, sc)

    # ============ R2: 行业动量 ============
    print("\n=== R2 行业动量 ===", flush=True)
    for scale, capital in [("portfolio", 35_000_000.0), ("account", cap0)]:
        for hz, mom in [("m20", mom20), ("m60", mom60)]:
            if scale == "account" and hz == "m60":
                pass  # 预注册: 60日为稳健性配置, 两个规模都跑
            rows = {}
            for d in grids:
                mrow = mom.loc[:d].iloc[-1].dropna()
                if len(mrow) < 5:
                    continue
                top3 = mrow.sort_values(ascending=False).head(3).index
                mm = mem[mem.date == d].set_index("ts_code")["industry"]
                members = [c for c in mm[mm.isin(top3)].index
                           if bool(allowed.loc[d].get(c, False))]
                if scale == "account" and len(members) > 50:
                    members = sorted(members)[:50]
                if members:
                    rows[d] = pd.Series(np.arange(len(members), 0, -1, dtype=float),
                                        index=members)
            sig = pd.DataFrame(rows).T.reindex(columns=allowed.columns)
            r, s = backtest(f"R2{hz}_{scale}", sig, run_dir, capital)
            add(f"R2{hz}_{scale}", r, s)
    # R2 基准: 全行业成员等权 (组合级)
    rows = {}
    for d in grids:
        members = [c for c in allowed.loc[d].index
                   if bool(allowed.loc[d].get(c, False)) and (d, c) in mem.set_index(
                       ["date", "ts_code"]).index]
        rows[d] = pd.Series(1.0, index=members)
    sigb = pd.DataFrame(rows).T.reindex(columns=allowed.columns)
    rb, sb = backtest("R2bench_portfolio", sigb, run_dir, 35_000_000.0)
    add("R2bench_portfolio", rb, sb)

    # ============ R3: v14 择时门 ============
    print("\n=== R3 v14 择时门 ===", flush=True)
    hs = api.index_close("000300.SH").loc[START:END]
    r = hs.pct_change()
    med60 = r.rolling(60, min_periods=40).median()
    up = core[core.date.isin(hs.index)].groupby("date")["pct_chg"].apply(lambda s: (s > 0).mean())
    gate = (((med60 < -0.002) & (up.reindex(hs.index) < 0.45)).fillna(False)).shift(1).fillna(False)
    flips = int((gate != gate.shift(1)).sum())
    cost = pd.Series(0.0, index=hs.index)
    cost[gate != gate.shift(1)] = 0.0006
    strat = pd.Series(np.where(gate, 0.0, r), index=hs.index).fillna(0) - cost.fillna(0)

    def stats(x):
        nav = (1 + x).cumprod()
        yrs = len(x) / 244
        return nav.iloc[-1] ** (1 / yrs) - 1, (nav / nav.cummax() - 1).min()
    c1, m1 = stats(r.fillna(0))
    c2, m2 = stats(strat)
    r3 = dict(bh_cagr=c1, bh_mdd=m1, gate_cagr=c2, gate_mdd=m2, flips=flips,
              mdd_improve_rel=float(1 - m2 / m1), cagr_loss_pp=float((c1 - c2) * 100),
              pass_gate=bool((1 - m2 / m1) >= 0.25 and (c1 - c2) <= 0.01))
    results["R3"] = (strat, r3)
    print(f"  BH cagr {c1:+.4f} mdd {m1:+.4f} | gate cagr {c2:+.4f} mdd {m2:+.4f} "
          f"flips {flips} | pass {r3['pass_gate']}", flush=True)

    # ============ 汇总 ============
    rows = []
    bench = results.get("R1u_portfolio", (None, None))[0]
    for k, (ret, summ) in results.items():
        if isinstance(summ, dict) and "ann" in summ:
            ex = ret.reindex(bench.index) - bench if bench is not None and k.startswith("R1") else None
            rows.append(dict(strategy=k, ann=summ.get("ann"), mdd=summ.get("mdd"),
                             sharpe=summ.get("sharpe"),
                             ex_vs_R1u_ann=float(ex.mean() * 244) if ex is not None else np.nan,
                             ex_t=float(nw_t(ex)) if ex is not None else np.nan))
        else:
            rows.append(dict(strategy=k, **summ))
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "summary.csv", index=False)
    print("\n=== 汇总 (excess 列 = vs R1u_portfolio 即同过滤等权宇宙, 仅 R1 系) ===")
    print(out.round(4).to_string(index=False))
    ExperimentLedger().log(
        "reproduce_feasible_v2", "R1_CH3/R2_indmom/R3_v14gate", "2014-2023",
        np.nan, np.nan, np.nan, np.nan, params=dict(n=len(rows)),
        verdict="PREREGISTERED", experiment_id=str(run_dir),
        selection_note="datakit 版; 协议 REPRODUCE_FEASIBLE_PREREG_20260919.md, 记1次试验")
    print(f"\n产出 -> {run_dir}")


if __name__ == "__main__":
    main()
