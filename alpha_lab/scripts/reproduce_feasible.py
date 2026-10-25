#!/usr/bin/env python3
"""高可行性策略复现 (预注册: docs/REPRODUCE_FEASIBLE_PREREG_20260919.md)。

R1 CH-3 小盘价值 (LSY 2019) | R2 申万行业动量 (Lin 2022) | R3 v14 规则择时 (audit/12 复现)。
定义全部执行前固定; 窗口 2014-2023; 引擎诚实口径; 不触 TEST。
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from investable import investable_signals  # noqa: E402
from portfolio import board_of, run_backtest, AccountingError  # noqa: E402
from style_matching import match_controls  # noqa: E402
from neutralize import attach_industry_pit  # noqa: E402
from check_cash_reserve import add_cash_reserve  # noqa: E402
from prepare_monthly_audit import fetch_execution_data  # noqa: E402
from audit_monthly_portfolio import load_inputs  # noqa: E402
from walkforward_monthly_oos import (  # noqa: E402
    screen_actions, screen_prices, patch_missing_days)
from ledger import ExperimentLedger  # noqa: E402

OUT = ROOT / "outputs/reproduce_feasible"
CFG = yaml.safe_load((ROOT / "configs/monthly_portfolio_audit.yaml").read_text())
PANEL = ROOT / "data/panel/daily_panel.parquet"
START, END = pd.Timestamp("2014-01-01"), pd.Timestamp("2023-12-31")
SEED = 20260919


def month_ends(dates: pd.DatetimeIndex) -> list:
    s = pd.Series(dates, index=dates)
    return list(s.groupby([dates.year, dates.month]).last())


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


def load_common():
    basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
    for c in ("list_date", "delist_date"):
        basics[c] = pd.to_datetime(basics[c], errors="coerce")
    names = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    names["start"] = pd.to_datetime(names.start_date, errors="coerce")
    names["end"] = pd.to_datetime(names.end_date, errors="coerce")
    st = names[names.name.str.contains("ST", case=False, na=False)][["ts_code", "start", "end"]]
    sw = pd.read_parquet(ROOT / "data/raw/sw_l1_member_hist.parquet")
    return basics, st, sw


EXCLUDED = set()   # 与引擎守卫等价的预核验隔离 (全局对称, 候选/基准同时剔除)


def run_plan(tag, plan, codes, y0, y1, cfg_p, run_dir, market_cache={}):
    """带缓存与预核验隔离的诚实回测; 返回 daily.ret。"""
    key = (y0, y1)
    if key not in market_cache:
        cache = fetch_execution_data(sorted(codes), (y0, y1), CFG)
        prepared = run_dir / f"prepared_{y0:%Y}"
        prepared.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(dict(ts_code=sorted(codes))).to_csv(prepared / "execution_codes.csv", index=False)
        (prepared / "execution_cache.txt").write_text(str(cache))
        cfg_y = dict(CFG, valid=[str(y0.date()), str(y1.date())])
        market, b2, st2, limits, actions = load_inputs(prepared, cfg_y)
        market, missing = patch_missing_days(market, y0, y1)
        if missing:
            print(f"  应用修复投影补缺失交易日: {[str(x.date()) for x in missing]}", flush=True)
        bad_a = screen_actions(market, actions, CFG["portfolio"]["action_price_atol"],
                               CFG["portfolio"]["action_price_rtol"])
        bad_p = screen_prices(market, actions)
        newly = {k: v for k, v in ({**bad_a, **bad_p}).items() if k not in EXCLUDED}
        EXCLUDED.update(newly)
        print(f"  预核验隔离 {len(newly)} 只 (累计 {len(EXCLUDED)})", flush=True)
        (run_dir / "excluded.json").write_text(json.dumps(EXCLUDED, ensure_ascii=False, indent=1))
        market_cache[key] = (market, b2, st2, limits, actions)
    market, b2, st2, limits, actions = market_cache[key]
    if EXCLUDED:
        plan = plan.drop(columns=[c for c in EXCLUDED if c in plan.columns])
    res = run_backtest(plan, market, b2, st2, limits, actions, y0, y1, cfg_p,
                       yaml.safe_load((ROOT / "configs/corporate_action_components.yaml")
                                      .read_text()).get("rights_issues", []))
    res.daily.to_csv(run_dir / f"{tag}_daily.csv")
    return res.daily.ret, res.summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / time.strftime("%Y%m%dT%H%M%S") if False else OUT / pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir()
    print(f"产出 {run_dir}", flush=True)
    basics, st, sw = load_common()
    cap0 = CFG["portfolio"]["capital"]

    # ---------- 公共面板切片 ----------
    print("加载面板...", flush=True)
    df = pd.read_parquet(PANEL, columns=["date", "ts_code", "close", "vol_share", "pe_ttm",
                                         "circ_mv_yuan", "amount_yuan", "pct_chg"],
                         filters=[("date", ">=", pd.Timestamp("2013-06-01")),
                                  ("date", "<=", END)])
    df = attach_industry_pit(df, sw).rename(columns={"sw_l1": "industry"})
    df["board"] = df.ts_code.map(board_of)
    # 不可核验公司行动代码: 预核验发现的同日多分红组 (logs/dividend_conflicts.json),
    # 本地无法核验真伪 → 对候选与基准对称整段剔除 (000564 为精确重复, 引擎自动去重, 保留)
    EXCLUDE = ["000418.SZ", "000528.SZ", "000958.SZ", "002506.SZ", "600074.SH",
               "600188.SH", "600332.SH", "600600.SH", "600938.SH", "601066.SH",
               "601088.SH", "601318.SH", "601601.SH", "601828.SH"]
    n0 = df.ts_code.nunique()
    df = df[~df.ts_code.isin(EXCLUDE)]
    print(f"公司行动不可核验剔除 {len(EXCLUDE)} 只 ({n0 - df.ts_code.nunique()} 只实际移除)", flush=True)
    dates_all = pd.DatetimeIndex(np.sort(df.date.unique()))
    grids = month_ends(dates_all[dates_all >= START])

    close = df.pivot_table(index="date", columns="ts_code", values="close", aggfunc="last")
    volume = df.pivot_table(index="date", columns="ts_code", values="vol_share", aggfunc="last")
    pe = df.pivot_table(index="date", columns="ts_code", values="pe_ttm", aggfunc="last")
    ep = 1.0 / pe.where(pe > 0)

    mc = {}
    results = {}

    # ---------- R1: CH-3 小盘价值 ----------
    print("\n=== R1 CH-3 小盘价值 ===", flush=True)
    for scale, capital, top_n_cap in [("portfolio", 35_000_000.0, 10 ** 6),
                                      ("account", cap0, 50)]:
        p1 = dict(CFG["portfolio"], capital=capital)
        sig_v, sig_u = [], []
        picks = {}
        for d in grids:
            allowed = None
            s1 = pd.DataFrame(1.0, index=[d], columns=close.columns)
            try:
                _, allowed = investable_signals(s1, close.loc[[d]], volume.loc[[d]],
                                                basics, st, p1, 1.05)
            except Exception as e:
                print(f"  investable {d.date()} fail: {e}", flush=True)
                continue
            ok = allowed.iloc[0].fillna(False)
            codes_d = ok[ok].index
            # 剔除市值最小 30% (壳)
            mv = df[df.date == d].set_index("ts_code")["circ_mv_yuan"].reindex(codes_d)
            mv = mv[mv > 0]
            keep = mv[mv >= mv.quantile(0.30)].index
            e = ep.loc[d].reindex(keep)
            e = e[e.notna() & (e > 0)]
            top = e.sort_values(ascending=False, kind="stable")
            n30 = max(1, int(len(top) * 0.30))
            val = top.iloc[:n30].index
            if scale == "account":
                val = val[:top_n_cap]
            picks[d] = list(val)
            sig_v.append(pd.Series(np.arange(len(val), 0, -1, dtype=float),
                                   index=val, name=d))
            # 基准: 同过滤等权 (账户级取 EP 排名前50池的对齐基准: 过滤池内 EP 前 150 随机? —
            # 预注册基准=同过滤等权全池; 账户级也用全池基准, 规模差异已由 active capital 控制)
            sig_u.append(pd.Series(1.0, index=keep, name=d))
        sig_v = pd.DataFrame(sig_v).fillna(0.0).replace(0.0, np.nan)
        sig_u = pd.DataFrame(sig_u)
        for name, sigx, topn in [("R1v_" + scale, sig_v, len(sig_v.columns)),
                                 ("R1u_" + scale, sig_u, len(sig_u.columns))]:
            codes = sorted(set(sigx.columns[sigx.notna().any()]))
            plan = sigx.copy()
            ret, summ = run_plan(name, plan, codes, START, END,
                                 dict(p1, top_n=max(1, min(topn, int(sigx.notna().iloc[0].sum()) or 1))),
                                 run_dir, mc)
            results[name] = (ret, summ)
            print(f"  {name}: total {summ.get('total'):+.4f} ann {summ.get('ann'):+.4f} "
                  f"mdd {summ.get('mdd'):+.4f}", flush=True)

    # ---------- R2: 行业动量 ----------
    print("\n=== R2 行业动量 ===", flush=True)
    rets = df.set_index(["date", "ts_code"])["pct_chg"].div(100)
    ind_ret = df.dropna(subset=["industry"]).groupby(["date", "industry"])["pct_chg"].mean().div(100)
    ind_ret = ind_ret.unstack().sort_index()
    mom20 = ind_ret.rolling(20, min_periods=15).mean()
    mom60 = ind_ret.rolling(60, min_periods=45).mean()
    # 成员表 (PIT)
    mem = df.dropna(subset=["industry"])[["date", "ts_code", "industry"]]

    for scale, capital in [("portfolio", 35_000_000.0), ("account", cap0)]:
        for hz, mom in [("m20", mom20), ("m60", mom60)]:
            if scale == "portfolio" and hz == "m60":
                continue  # 预注册: 60日仅账户级稳健性
            p2 = dict(CFG["portfolio"], capital=capital)
            sig_rows, bench_rows = [], []
            for d in grids:
                s1 = pd.DataFrame(1.0, index=[d], columns=close.columns)
                try:
                    _, allowed = investable_signals(s1, close.loc[[d]], volume.loc[[d]],
                                                    basics, st, p2, 1.05)
                except Exception:
                    continue
                ok = allowed.iloc[0].fillna(False)
                m = mom.loc[:d].iloc[-1].dropna()
                if len(m) < 5:
                    continue
                top3 = m.sort_values(ascending=False).head(3).index
                mm = mem[mem.date == d].set_index("ts_code")["industry"]
                members = mm[mm.isin(top3)].index
                members = [c for c in members if ok.get(c, False)]
                if scale == "account":
                    # 账户级: top-3 行业内按 EP 取 50 (动量行业+价值选股的可实现混合? —
                    # 预注册未定义选股, 保持纯行业: 随机50? 不好。改为: 全体成员等权最多50只按代码序)
                    members = sorted(members)[:50] if len(members) > 50 else members
                if not members:
                    continue
                sig_rows.append(pd.Series(np.arange(len(members), 0, -1, dtype=float),
                                          index=members, name=d))
                bench_rows.append(pd.Series(1.0, index=ok[ok].index, name=d))
            sig = pd.DataFrame(sig_rows)
            bench = pd.DataFrame(bench_rows)
            name = f"R2{hz}_" + scale
            codes = sorted(set(sig.columns[sig.notna().any()]))
            ret, summ = run_plan(name, sig, codes, START, END, p2, run_dir, mc)
            results[name] = (ret, summ)
            print(f"  {name}: total {summ.get('total'):+.4f} ann {summ.get('ann'):+.4f}", flush=True)
            if scale == "portfolio" and hz == "m20":
                codes_b = sorted(set(bench.columns[bench.notna().any()]))
                retb, sumb = run_plan("R2bench_portfolio", bench, codes_b, START, END, p2,
                                      run_dir, mc)
                results["R2bench_portfolio"] = (retb, sumb)
                print(f"  R2bench_portfolio: ann {sumb.get('ann'):+.4f}", flush=True)

    # ---------- R3: v14 择时门 ----------
    print("\n=== R3 v14 择时门 ===", flush=True)
    idx = pd.read_parquet(ROOT / "data/raw/index_daily.parquet")
    hs = idx[idx.ts_code == "000300.SH"].copy()
    hs["date"] = pd.to_datetime(hs.trade_date, format="%Y%m%d")
    hs = hs.sort_values("date").set_index("date").loc[START:END]
    r = hs.close.pct_change()
    med60 = r.rolling(60, min_periods=40).median()
    up_frac = df[df.date.isin(hs.index)].groupby("date")["pct_chg"].apply(
        lambda s: (s > 0).mean())
    bear = ((med60 < -0.002) & (up_frac.reindex(hs.index) < 0.45)).fillna(False)
    gate = bear.shift(1).fillna(False)  # t-1 信息决定 t 日持仓
    flips = (gate != gate.shift(1)).sum()
    strat = np.where(gate, 0.0, r)
    cost = pd.Series(0.0, index=hs.index)
    cost[gate != gate.shift(1)] = 0.0006  # 单边 0.06%
    strat_net = strat - cost
    # 买入持有对照
    def stats(x):
        nav = (1 + x).cumprod()
        yrs = len(x) / 244
        cagr = nav.iloc[-1] ** (1 / yrs) - 1
        mdd = (nav / nav.cummax() - 1).min()
        return cagr, mdd
    c1, m1 = stats(r.fillna(0))
    c2, m2 = stats(pd.Series(strat_net, index=hs.index).fillna(0))
    r3 = dict(hs300_buyhold_cagr=c1, hs300_buyhold_mdd=m1,
              gate_cagr=c2, gate_mdd=m2, flips=int(flips),
              mdd_improve_rel=float(1 - m2 / m1), cagr_loss_pp=float((c1 - c2) * 100),
              pass_gate=bool((1 - m2 / m1) >= 0.25 and (c1 - c2) <= 0.01))
    results["R3"] = (pd.Series(strat_net, index=hs.index), r3)
    print(f"  BH cagr {c1:+.4f} mdd {m1:+.4f} | gate cagr {c2:+.4f} mdd {m2:+.4f} "
          f"flips {flips} | pass {r3['pass_gate']}", flush=True)

    # ---------- 汇总 ----------
    rows = []
    bench_p = results.get("R1u_portfolio", (None, None))[0]
    for k, (ret, summ) in results.items():
        if isinstance(summ, dict) and "ann" in summ:
            ex_bench = ret.reindex(bench_p.index) - bench_p if bench_p is not None else None
            rows.append(dict(strategy=k, ann=summ.get("ann"), mdd=summ.get("mdd"),
                             sharpe=summ.get("sharpe"),
                             excess_vs_universe_ann=(ex_bench.mean() * 244
                                                     if ex_bench is not None else np.nan),
                             excess_t=(nw_t(ex_bench.dropna()) if ex_bench is not None else np.nan)))
        else:
            rows.append(dict(strategy=k, **summ))
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    print("\n=== 汇总 ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    ledger = ExperimentLedger()
    ledger.log("reproduce_feasible", "R1_CH3/R2_indmom/R3_v14gate", "2014-2023",
               np.nan, np.nan, np.nan, np.nan,
               params=dict(n=len(rows)), verdict="PREREGISTERED",
               experiment_id=str(run_dir),
               selection_note="复现协议 REPRODUCE_FEASIBLE_PREREG_20260919.md, 记1次试验")
    print(f"\n产出 -> {run_dir}")


if __name__ == "__main__":
    main()
