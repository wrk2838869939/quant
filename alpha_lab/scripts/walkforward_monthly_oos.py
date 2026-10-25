#!/usr/bin/env python3
"""Walk-forward 逐年样本外重验 (预注册: docs/WALKFORWARD_OOS_PREREG_20260918.md)。

每年: trailing 5 年 ICIR 权重 → 年内 20 日截面合成信号 → investable 资格过滤 →
candidate(top50) vs size_liquidity_matched 对照 → run_backtest 诚实执行 → 合并判定。

不触碰 TEST (全部窗 ≤ 2023-12-31); 不改任何生产配置。
用法: python scripts/walkforward_monthly_oos.py [--years 2019 2020 2021 2022 2023]
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from run_gp_mining import build_wide, parse_expr  # noqa: E402
from expr_engine import eval_node  # noqa: E402
from investable import investable_signals  # noqa: E402
from portfolio import AccountingError, board_of, run_backtest  # noqa: E402
from style_matching import match_controls  # noqa: E402
from neutralize import attach_industry_pit  # noqa: E402
from check_cash_reserve import add_cash_reserve  # noqa: E402
from check_portfolio_improvement import asset_pnl  # noqa: E402
from prepare_monthly_audit import fetch_execution_data  # noqa: E402
from audit_monthly_portfolio import load_inputs  # noqa: E402
from ledger import ExperimentLedger  # noqa: E402

OUT = ROOT / "outputs/walkforward_oos"
CFG = yaml.safe_load((ROOT / "configs/monthly_portfolio_audit.yaml").read_text())
H = 20
SEED = 20260918
PRICE_MULT = 1.05  # 与 investability_followup 链一致
GP_FACTORS_M20 = ["ts_min(vwap_dev,5)", "ts_std(lg_share,10)"]  # 与 synth_weights_h20 相同集合


def build_wide_and_factors():
    cols_map = dict(ret1="pct_chg", vol="vol_share", amt="amount_yuan", to="turnover_rate",
                    high="high", low="low", close="close", hclose="hfq_close",
                    buy_lg="buy_lg_amount_yuan", sell_lg="sell_lg_amount_yuan",
                    buy_elg="buy_elg_amount_yuan", net_mf="net_mf_amount_yuan",
                    circ_mv="circ_mv_yuan", volume_ratio="volume_ratio",
                    pe="pe_ttm", pb="pb")
    wide = build_wide(cols_map)
    wide["vwap_dev"] = (wide["amt"] / wide["vol"]) / wide["close"] - 1.0
    wide["ret1"] = wide["ret1"] / 100.0
    wide["lg_share"] = (wide["buy_lg"] + wide["sell_lg"]) / (wide["amt"] + 1e-6)

    facs = {}
    facs["lit:low_turnover"] = -wide["to"].rolling(20, min_periods=15).mean()
    amp = (wide["high"] - wide["low"]) / wide["close"].shift(1)
    facs["lit:low_amp"] = -amp.rolling(21, min_periods=15).mean()
    facs["lit:low_vol"] = -wide["ret1"].rolling(20, min_periods=15).std()
    facs["lit:rev20"] = -(wide["hclose"] / wide["hclose"].shift(20) - 1)
    facs["lit:rev5"] = -(wide["hclose"] / wide["hclose"].shift(5) - 1)
    illiq = wide["ret1"].abs() / (wide["amt"] / 1e8 + 1e-6)
    facs["lit:amihud"] = illiq.rolling(20, min_periods=15).mean()
    facs["lit:mom60s5"] = wide["hclose"].shift(5) / wide["hclose"].shift(60) - 1
    for e in GP_FACTORS_M20:
        with np.errstate(all="ignore"):
            facs[f"gp:{e[:24]}"] = eval_node(parse_expr(e), wide)
    facs["lit:ep"] = 1.0 / wide["pe"].where(wide["pe"] > 0)
    facs["lit:bp"] = 1.0 / wide["pb"].where(wide["pb"] > 0)
    frozen = set(pd.read_csv(ROOT / "outputs/gp_mining/synth_weights_h20.csv",
                             index_col=0).index)
    assert set(facs) == frozen, f"因子集与冻结版不一致: {set(facs) ^ frozen}"
    return wide, facs


def fit_weights(facs, lab, t_start, t_end):
    """trailing ICIR 权重; 20 日间隔截面, 末端 purge H (标签不越过 trailing 末端)。"""
    dates = facs["lit:rev20"].index
    win = dates[(dates >= pd.Timestamp(t_start)) & (dates <= pd.Timestamp(t_end))]
    win = win[:max(0, len(win) - H)]
    grid = [d for i, d in enumerate(win) if i % H == 0]
    assert len(grid) >= 24, f"trailing 截面不足: {len(grid)}"
    weights = {}
    for name, f in facs.items():
        vals = [f.loc[d].corr(lab.loc[d]) for d in grid]
        ic = pd.Series(vals, index=grid).dropna()
        weights[name] = ic.mean() / (ic.std() + 1e-12)
    return weights


def synth_signals(facs, weights, grid):
    """截面 zscore × 权重, 逐元素求和; 任一分量 NaN → NaN (与 synthesis 语义一致)。"""
    out = None
    for name, f in facs.items():
        sub = f.loc[grid]
        m, s = sub.mean(axis=1), sub.std(axis=1)
        z = sub.sub(m, axis=0).div(s.replace(0, np.nan), axis=0)
        term = z * weights[name]
        out = term if out is None else out + term
    return out


def screen_actions(market, actions, atol, rtol):
    """研究层预核验: 分红金额 vs 交易所除权参考价 (panel pre_close)。

    与 run_backtest 内核验同公式; 不过者返回代码集合, 由调用方整年隔离。
    """
    bad = {}
    for code, g in actions.groupby("ts_code"):
        px = market[market.ts_code == code]
        px = px.sort_values("date").set_index("date")
        for _, a in g.iterrows():
            ex = a["ex_date"]
            if ex not in px.index:
                continue
            i = px.index.get_loc(ex)
            if i == 0:
                continue
            prev_close = float(px["close"].iloc[i - 1])
            pc = float(px["pre_close"].iloc[i])
            div = float(a["cash_div_tax"]) if np.isfinite(a["cash_div_tax"]) else 0.0
            bonus = float(a["stk_div"]) if np.isfinite(a["stk_div"]) else 0.0
            ref = (prev_close - div) / (1 + bonus)
            if np.isfinite(pc) and not np.isclose(ref, pc, atol=atol, rtol=rtol):
                bad[code] = dict(ex_date=str(pd.Timestamp(ex).date()),
                                 reference=round(ref, 4), pre_close=round(pc, 4))
    return bad


def screen_prices(market, actions, atol_cont=0.005 + 1e-9, atol_act=0.02, rtol_act=0.002):
    """研究层预核验 (与 run_backtest 守卫逐条等价, 连续持有口径):

    非除权日: |pre_close(t) − close(t−1)| ≤ price_tick/2 (引擎 425 行, rtol=0)
    除权日:   |(close(t−1) − div)/(1+bonus) − pre_close(t)| ≤ 0.02 + 0.002×|pc| (引擎 366 行)
    不过者整年隔离。"""
    act = {}
    for _, a in actions.iterrows():
        ex = pd.to_datetime(a.get("ex_date"), errors="coerce")
        if pd.notna(ex):
            act[(a["ts_code"], ex)] = (a.get("cash_div_tax"), a.get("stk_div"))
    bad = {}
    for code, g in market.groupby("ts_code"):
        g = g.sort_values("date")
        pc = g["pre_close"].to_numpy()
        cl = g["close"].to_numpy()
        dt = pd.DatetimeIndex(g["date"])
        for i in range(1, len(g)):
            prev, cur = cl[i - 1], pc[i]
            if not (np.isfinite(prev) and np.isfinite(cur)):
                continue
            div, bonus = act.get((code, dt[i]), (np.nan, np.nan))
            div = float(div) if div is not None and np.isfinite(div) else 0.0
            bonus = float(bonus) if bonus is not None and np.isfinite(bonus) else 0.0
            if div or bonus:
                ref = (prev - div) / (1 + bonus)
                if abs(ref - cur) > atol_act + rtol_act * abs(cur):
                    bad[code] = dict(date=str(dt[i].date()), kind="action_ref",
                                     reference=round(float(ref), 4), pre_close=float(cur))
                    break
            else:
                if abs(cur - prev) > atol_cont:
                    bad[code] = dict(date=str(dt[i].date()), kind="continuity",
                                     prev_close=float(prev), pre_close=float(cur))
                    break
    # 盲区三: 除权日恰逢停牌 (该日无行情行) — 引擎在除权日照常给持仓调账,
    # 复牌首日的 pre_close 必须等于除权参考价; 厂商有时不调整 (如 600781 2019-07)。
    dates_by_code = {c: pd.DatetimeIndex(g["date"]) for c, g in market.groupby("ts_code")}
    for (code, ex), (div, bonus) in act.items():
        dts = dates_by_code.get(code)
        if dts is None or len(dts) == 0 or ex in dts:
            continue
        later = dts[dts > ex]
        if len(later) == 0:
            continue
        t_resume = later[0]
        g = market[market.ts_code == code].sort_values("date")
        g = g.set_index("date")
        i_resume = g.index.get_loc(t_resume)
        if i_resume == 0:
            continue
        prev_close = float(g["close"].iloc[i_resume - 1])
        pc = float(g["pre_close"].iloc[i_resume])
        div = float(div) if div is not None and np.isfinite(div) else 0.0
        bonus = float(bonus) if bonus is not None and np.isfinite(bonus) else 0.0
        ref = (prev_close - div) / (1 + bonus)
        if abs(ref - pc) > atol_cont + 0.5 * prev_close * 0 + 1e-9 and abs(ref - pc) > atol_act + rtol_act * abs(pc):
            bad[code] = dict(date=str(t_resume.date()), kind="resume_after_exdiv_suspension",
                             reference=round(ref, 4), pre_close=pc)
    return bad



def patch_missing_days(market, y0, y1):
    """显式应用 2021-12-13 缺日修复投影 (outputs/forward_risk/repair_20211213, 带来源哈希)。

    close/adj_factor 取投影值; pre_close := 该股前一日收盘 (无公司行动时的定义值,
    有除权者由引擎 action 交叉核验兜底); open/vol_share 置 NaN 且断言缺失日不在信号
    网格 (当日无成交需求)。原始面板与投影文件均不修改。
    """
    cal = pd.read_parquet(ROOT / "data/raw/calendar.parquet")
    cal["cal_date"] = pd.to_datetime(cal["cal_date"].astype(str), format="%Y%m%d")
    open_days = cal.loc[(cal.cal_date >= y0) & (cal.cal_date <= y1)
                        & (cal.is_open.astype(int) == 1), "cal_date"]
    have = set(pd.DatetimeIndex(market.date.unique()))
    missing = [d for d in open_days if d not in have]
    if not missing:
        return market, []
    patch = pd.read_parquet(ROOT / "outputs/forward_risk/repair_20211213/projection.parquet")
    patch["date"] = pd.to_datetime(patch["date"])
    frames = []
    for d in missing:
        p = patch[patch.date == d]
        if p.empty:
            raise AccountingError(f"{d.date()} 日历交易日缺失且无修复投影")
        last_close = market[market.date < d].sort_values("date").groupby("ts_code").close.last()
        p = p.merge(last_close.rename("pre_close"), left_on="ts_code",
                    right_index=True, how="left")
        p["open"] = np.nan
        p["vol_share"] = np.nan
        frames.append(p[["date", "ts_code", "open", "close", "pre_close",
                         "adj_factor", "vol_share"]])
    out = pd.concat([market] + frames, ignore_index=True).sort_values(["date", "ts_code"])
    return out, [pd.Timestamp(d) for d in missing]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir()
    print(f"产出目录 {run_dir}", flush=True)

    print("构建宽表与因子 (一次)...", flush=True)
    t0 = time.time()
    wide, facs = build_wide_and_factors()
    print(f"  {time.time()-t0:.0f}s", flush=True)
    all_dates = facs["lit:rev20"].index
    lab = wide["hclose"].shift(-H) / wide["hclose"] - 1.0
    lab = lab.sub(lab.mean(axis=1), axis=0).rank(axis=1, pct=True)
    fwd20 = wide["hclose"].shift(-H) / wide["hclose"] - 1.0

    panel = ROOT / "data/panel/daily_panel.parquet"
    basics = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
    for col in ("list_date", "delist_date"):
        basics[col] = pd.to_datetime(basics[col], errors="coerce")
    names = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    names["start"] = pd.to_datetime(names.start_date, errors="coerce")
    names["end"] = pd.to_datetime(names.end_date, errors="coerce")
    st = names[names.name.str.contains("ST", case=False, na=False)][["ts_code", "start", "end"]]
    sw = pd.read_parquet(ROOT / "data/raw/sw_l1_member_hist.parquet")
    cap = CFG["portfolio"]["capital"]
    p_half = dict(CFG["portfolio"], capital=cap * 0.5)
    rights = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())

    yearly_rows, daily_cols, decile_rows, ind_rows = [], [], [], []
    for Y in args.years:
        assert Y <= 2023, "本实验禁止触碰 TEST"
        y0, y1 = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y}-12-31")
        print(f"\n===== 评估年 {Y} =====", flush=True)
        w = fit_weights(facs, lab, f"{Y-5}-01-01", f"{Y-1}-12-31")
        print("trailing ICIR:", {k: round(v, 2) for k, v in sorted(w.items())}, flush=True)
        pd.Series(w).to_csv(run_dir / f"weights_{Y}.csv")

        year_dates = all_dates[(all_dates >= y0) & (all_dates <= y1)]
        grid = [d for i, d in enumerate(year_dates) if i % H == 0]
        sig = synth_signals(facs, w, grid)

        mk = pd.read_parquet(panel, columns=["date", "ts_code", "close", "vol_share"],
                             filters=[("date", ">=", y0), ("date", "<=", y1)])
        close = mk.pivot(index="date", columns="ts_code", values="close").reindex(
            index=grid, columns=sig.columns)
        volume = mk.pivot(index="date", columns="ts_code", values="vol_share").reindex(
            index=grid, columns=sig.columns)
        plan, allowed = investable_signals(sig, close, volume, basics, st, p_half, PRICE_MULT)

        # 匹配宇宙快照 (行业/规模/成交额20日; lookback 自上年 10 月)
        stat = pd.read_parquet(panel, columns=["date", "ts_code", "circ_mv_yuan", "amount_yuan"],
                               filters=[("date", ">=", pd.Timestamp(f"{Y-1}-10-01")),
                                        ("date", "<=", y1)])
        stat = stat.sort_values(["ts_code", "date"])
        amt = stat.pivot(index="date", columns="ts_code", values="amount_yuan").sort_index()
        amount20 = amt.where(amt > 0).rolling(20, min_periods=20).mean().loc[grid]
        snap = stat[stat.date.isin(grid)].copy()
        snap["log_mv"] = np.log(snap.circ_mv_yuan.where(snap.circ_mv_yuan > 0))
        lookup = pd.MultiIndex.from_frame(snap[["date", "ts_code"]])
        snap["log_amount20"] = np.log(amount20.stack()).reindex(lookup).to_numpy()
        snap = attach_industry_pit(snap, sw).rename(columns={"sw_l1": "industry"})
        snap["board"] = snap.ts_code.map(board_of)
        snap = snap.set_index(["date", "ts_code"]).sort_index()
        snap["score"] = sig.stack().reindex(snap.index)

        def build_plans(excl):
            cand = pd.DataFrame(np.nan, index=grid, columns=sig.columns)
            ctrl = pd.DataFrame(np.nan, index=grid, columns=sig.columns)
            codes_ = set()
            for d in grid:
                uni = snap.loc[d]
                uni = uni[uni.score.notna() & allowed.loc[d].reindex(uni.index).fillna(False)]
                uni = uni[~uni.index.isin(excl)]
                if len(uni) < CFG["min_cross_section"]:
                    continue
                selected = uni.score.sort_values(ascending=False, kind="stable").head(
                    CFG["portfolio"]["top_n"]).index.tolist()
                cand.loc[d, selected] = np.arange(len(selected), 0, -1, dtype=float)
                m = match_controls(uni, selected, ["log_mv", "log_amount20"], SEED)
                ctl = m.control.dropna().tolist()
                ctrl.loc[d, ctl] = np.arange(len(ctl), 0, -1, dtype=float)
                codes_.update(selected)
                codes_.update(ctl)
            return cand, ctrl, codes_

        excluded, market, basics2, st2, limits, actions = {}, None, None, None, None, None
        cand_plan = ctrl_plan = None
        for attempt in range(3):
            cand_plan, ctrl_plan, codes = build_plans(excluded)
            if not codes:
                break
            cache = fetch_execution_data(sorted(codes), (y0, y1), CFG)
            prepared = run_dir / f"prepared_{Y}"
            prepared.mkdir(exist_ok=True)
            pd.DataFrame(dict(ts_code=sorted(codes))).to_csv(prepared / "execution_codes.csv",
                                                             index=False)
            (prepared / "execution_cache.txt").write_text(str(cache))
            cfg_y = dict(CFG, valid=[str(y0.date()), str(y1.date())])
            market, basics2, st2, limits, actions = load_inputs(prepared, cfg_y)
            market, missing = patch_missing_days(market, y0, y1)
            if missing:
                assert not (set(grid) & set(missing)), "缺失日恰为信号日, 无法诚实补日"
                print(f"[{Y}] 应用修复投影补缺失交易日: {[str(d.date()) for d in missing]}",
                      flush=True)
            bad_a = screen_actions(market, actions, CFG["portfolio"]["action_price_atol"],
                                   CFG["portfolio"]["action_price_rtol"])
            bad_p = screen_prices(market, actions)
            newly = {k: v for k, v in ({**bad_a, **bad_p}).items() if k not in excluded}
            if not newly:
                break
            excluded.update(newly)
            print(f"[{Y}] 第{attempt+1}轮核验隔离 {len(newly)}: 行动 {sorted(bad_a)} 价格 {sorted(bad_p)}",
                  flush=True)
        if excluded:
            (run_dir / f"{Y}_excluded.json").write_text(
                json.dumps(excluded, indent=2, ensure_ascii=False))
        print(f"信号日 {len(grid)}: cand {(cand_plan.notna().any(axis=1)).sum()}, "
              f"ctrl {(ctrl_plan.notna().any(axis=1)).sum()}, codes {len(codes)}, "
              f"隔离 {len(excluded)}", flush=True)
        if not codes or cand_plan is None:
            print(f"[{Y}] 无有效信号日, 跳过 (预注册: 显式说明)", flush=True)
            yearly_rows.append(dict(year=Y, strategy="excess", skipped=True))
            continue

        res_by = {}
        for name, planx in [("candidate", cand_plan), ("control", ctrl_plan)]:
            res = run_backtest(planx, market, basics2, st2, limits, actions, y0, y1,
                               p_half, rights.get("rights_issues", []))
            res = add_cash_reserve(res, cap - p_half["capital"], CFG["portfolio"]["annual_days"])
            res.daily.to_csv(run_dir / f"{Y}_{name}_daily.csv")
            res.trades.to_csv(run_dir / f"{Y}_{name}_trades.csv", index=False)
            res_by[name] = res
            yearly_rows.append(dict(year=Y, strategy=name, skipped=False,
                                    **{k: res.summary.get(k) for k in
                                       ("total", "ann", "mdd", "sharpe")}))
            print(f"  {name}: total {res.summary.get('total'):+.4f} "
                  f"ann {res.summary.get('ann'):+.4f} mdd {res.summary.get('mdd'):+.4f}", flush=True)
        exc = (res_by["candidate"].daily.ret - res_by["control"].daily.ret).dropna()
        yearly_rows.append(dict(year=Y, strategy="excess", skipped=False,
                                total=float((1 + exc).prod() - 1),
                                ann=float(exc.mean() * 244), mdd=np.nan, sharpe=np.nan))
        daily_cols.append(exc.rename(Y))

        # 行业归因 (candidate 净利, 首买日 PIT 行业)
        pnl = asset_pnl(res_by["candidate"].trades, res_by["candidate"].events,
                        res_by["candidate"].holdings, res_by["candidate"].daily)
        buys = res_by["candidate"].trades[res_by["candidate"].trades.side == "buy"]
        first_buy = buys.sort_values("date").drop_duplicates("ts_code").set_index("ts_code")
        ind_of = {}
        for d in grid:
            uni = snap.loc[d]
            for c in uni.index:
                ind_of.setdefault(c, uni.loc[c, "industry"])
        joined = pnl.to_frame("pnl_yuan").join(first_buy[["signal_date"]], how="left")
        joined["industry"] = [ind_of.get(c, "__unknown__") for c in joined.index]
        for industry, frame in joined.groupby("industry"):
            ind_rows.append(dict(year=Y, industry=industry, pnl_yuan=float(frame.pnl_yuan.sum()),
                                 n_stocks=len(frame)))

        # 十分组 (描述性, 无成本, 20日前瞻; 前瞻窗不得越过 2023-12-31, 不触 TEST)
        usable = all_dates[all_dates <= pd.Timestamp("2023-12-31")]
        cutoff = usable[max(0, len(usable) - 1 - H)]
        grid_d = [d for d in grid if d <= cutoff]
        q = sig.rank(axis=1, pct=True)
        elig = sig.notna() & allowed
        for g in range(10):
            mask = (q > g / 10) & (q <= (g + 1) / 10) & elig
            vals = fwd20.loc[grid_d][mask].stack().dropna()
            decile_rows.append(dict(year=Y, decile=g + 1, mean_fwd20=float(vals.mean()),
                                    n=len(vals)))

    # ---- 汇总与判定 ----
    pd.DataFrame(yearly_rows).to_csv(run_dir / "yearly_summary.csv", index=False)
    pd.DataFrame(decile_rows).to_csv(run_dir / "deciles.csv", index=False)
    pd.DataFrame(ind_rows).to_csv(run_dir / "industry_pnl.csv", index=False)
    if not daily_cols:
        print("无任何评估年产出")
        return
    exc_all = pd.concat(daily_cols, axis=1)
    pooled = exc_all.stack()
    vals = pooled.values
    n, block = len(vals), 20
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(2000, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(2000, -1)[:, :n]
    boots = vals[idx].mean(axis=1) * 244
    year_exc = exc_all.mean() * 244
    verdict = dict(
        pooled_excess_ann=float(pooled.mean() * 244),
        boot_lo=float(np.quantile(boots, 0.025)), boot_hi=float(np.quantile(boots, 0.975)),
        years_positive=int((year_exc > 0).sum()), n_years=int(len(year_exc)),
        median_year_excess=float(year_exc.median()),
        per_year={str(int(c)): float(v) for c, v in year_exc.items()})
    verdict["pass_main"] = bool(verdict["pooled_excess_ann"] > 0 and verdict["years_positive"] >= 3)
    verdict["pass_secondary"] = bool(verdict["median_year_excess"] > 0)
    (run_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n=== WALK-FORWARD 判定 (预注册判据) ===")
    print(json.dumps(verdict, indent=2))
    ledger = ExperimentLedger()
    ledger.log("walkforward_lowvol_oos", "monthly_top50_vs_style_matched",
               f"{args.years[0]}~{args.years[-1]} walk-forward",
               np.nan, np.nan, np.nan, verdict["pooled_excess_ann"],
               params=verdict, verdict="PASS" if verdict["pass_main"] else "FAIL",
               experiment_id=str(run_dir),
               selection_note="预注册协议 docs/WALKFORWARD_OOS_PREREG_20260918.md, 记1次试验")
    print(f"\n产出 -> {run_dir}")


if __name__ == "__main__":
    main()
