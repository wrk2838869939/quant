# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""消融实验 (预注册: docs/WALKFORWARD_OOS_PREREG_20260918.md 消融附录)。

所有变体共用同一协议与统一宇宙 (investable 通过 且 11 因子全非缺失),
仅改变打分方式; 每变体配各自的 size_liquidity_matched 对照。
年外层循环复用行情装载; 全部结果落盘一并对齐。
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("alpha_lab")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from walkforward_monthly_oos import (  # noqa: E402
    CFG, H, SEED, build_wide_and_factors, fit_weights, synth_signals,
    screen_actions, screen_prices, patch_missing_days)
from investable import investable_signals  # noqa: E402
from portfolio import board_of, run_backtest  # noqa: E402
from style_matching import match_controls  # noqa: E402
from neutralize import attach_industry_pit  # noqa: E402
from check_cash_reserve import add_cash_reserve  # noqa: E402
from prepare_monthly_audit import fetch_execution_data  # noqa: E402
from audit_monthly_portfolio import load_inputs  # noqa: E402
from ledger import ExperimentLedger  # noqa: E402

OUT = ROOT / "outputs/walkforward_ablation"
FAMILIES = {
    "lowrisk": ["lit:low_turnover", "lit:low_amp", "lit:low_vol"],
    "rev": ["lit:rev5", "lit:rev20"],
    "value": ["lit:ep", "lit:bp"],
    "liq": ["lit:amihud"],
    "mom": ["lit:mom60s5"],
    "gp": ["gp:ts_min(vwap_dev,5)", "gp:ts_std(lg_share,10)"],
}
ALL_FACTORS = [f for fs in FAMILIES.values() for f in fs]
SEEDS_RANDOM = ["20260919s1", "20260919s2", "20260919s3"]


def det_seed(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")


def variant_specs():
    specs = [("full_icir", {"mode": "icir", "factors": ALL_FACTORS})]
    specs.append(("eq_weight", {"mode": "eq", "factors": ALL_FACTORS}))
    for fam, fs in FAMILIES.items():
        rest = [f for f in ALL_FACTORS if f not in fs]
        specs.append((f"drop_{fam}", {"mode": "icir", "factors": rest}))
    specs.append(("only_low_turnover", {"mode": "single", "factors": ["lit:low_turnover"]}))
    for s in SEEDS_RANDOM:
        specs.append((f"avoidq1_{s}", {"mode": "avoidq1", "factors": ALL_FACTORS, "seed": s}))
    for s in SEEDS_RANDOM:
        specs.append((f"random50_{s}", {"mode": "random", "factors": ALL_FACTORS, "seed": s}))
    return specs


def score_variant(spec, facs, weights_by_name, uni_codes_by_date, rng_seed):
    """返回 {date: {code: score}}; 已限定统一宇宙 (uni_codes_by_date)。"""
    names = spec["factors"]
    mode = spec["mode"]
    picks = {}
    for d, codes in uni_codes_by_date.items():
        if len(codes) < CFG["min_cross_section"]:
            continue
        if mode == "random":
            rng = np.random.default_rng(det_seed(rng_seed))
            sel = rng.choice(np.sort(np.array(codes)), size=min(50, len(codes)), replace=False)
            picks[d] = {c: 1.0 for c in sel}
            continue
        if mode == "avoidq1":
            zsum = None
            for n in names:
                f = facs[n].loc[d].reindex(codes)
                z = (f - f.mean()) / (f.std() or np.nan)
                w = weights_by_name[n]
                term = z * w
                zsum = term if zsum is None else zsum + term
            rank = zsum.rank(pct=True)
            pool = rank[rank > 0.10].dropna()
            rng = np.random.default_rng(det_seed(rng_seed))
            k = min(50, len(pool))
            sel = rng.choice(np.sort(np.array(pool.index)), size=k, replace=False)
            picks[d] = {c: float(zsum[c]) for c in sel}
            continue
        zsum = None
        for n in names:
            f = facs[n].loc[d].reindex(codes)
            z = (f - f.mean()) / (f.std() or np.nan)
            w = 1.0 if mode in ("eq", "single") else weights_by_name[n]
            term = z * w
            zsum = term if zsum is None else zsum + term
        zsum = zsum.dropna()
        top = zsum.sort_values(ascending=False, kind="stable").head(CFG["portfolio"]["top_n"])
        picks[d] = {c: float(v) for c, v in top.items()}
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir()
    print(f"产出目录 {run_dir}", flush=True)

    specs = variant_specs()
    print(f"变体数 {len(specs)}", flush=True)

    print("构建宽表与因子 (一次)...", flush=True)
    t0 = time.time()
    wide, facs = build_wide_and_factors()
    print(f"  {time.time()-t0:.0f}s", flush=True)
    all_dates = facs["lit:rev20"].index
    lab = wide["hclose"].shift(-H) / wide["hclose"] - 1.0
    lab = lab.sub(lab.mean(axis=1), axis=0).rank(axis=1, pct=True)

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

    rows = []
    for Y in args.years:
        assert Y <= 2023, "禁止触碰 TEST"
        y0, y1 = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y}-12-31")
        print(f"\n===== 评估年 {Y} =====", flush=True)
        weights_all = fit_weights(facs, lab, f"{Y-5}-01-01", f"{Y-1}-12-31")

        year_dates = all_dates[(all_dates >= y0) & (all_dates <= y1)]
        grid = [d for i, d in enumerate(year_dates) if i % H == 0]

        mk = pd.read_parquet(panel, columns=["date", "ts_code", "close", "vol_share"],
                             filters=[("date", ">=", y0), ("date", "<=", y1)])
        close = mk.pivot(index="date", columns="ts_code", values="close").reindex(index=grid)
        volume = mk.pivot(index="date", columns="ts_code", values="vol_share").reindex(index=grid)
        sig_all = pd.DataFrame(1.0, index=grid, columns=close.columns)
        _, allowed = investable_signals(sig_all, close, volume, basics, st, p_half, 1.05)

        # 统一宇宙: investable 且 11 因子全非缺失
        uni_codes_by_date = {}
        for d in grid:
            avail = allowed.loc[d].reindex(close.columns).fillna(False)
            for n in ALL_FACTORS:
                avail = avail & facs[n].loc[d].reindex(close.columns).notna()
            uni_codes_by_date[d] = list(close.columns[avail])
        n_uni = int(np.mean([len(v) for v in uni_codes_by_date.values()]))
        print(f"统一宇宙均值 {n_uni} 只/信号日", flush=True)

        # 每变体打分
        picks_by_variant = {}
        for name, spec in specs:
            picks_by_variant[name] = score_variant(
                spec, facs, weights_all, uni_codes_by_date, spec.get("seed", name))
            print(f"  scored {name}", flush=True)

        # 匹配宇宙快照
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

        # 全变体 candidate ∪ control 代码并集 (一次抓取/装载)
        def build_plans(picks):
            cand = pd.DataFrame(np.nan, index=grid, columns=close.columns)
            ctrl = pd.DataFrame(np.nan, index=grid, columns=close.columns)
            codes_ = set()
            for d in grid:
                sel = picks.get(d, {})
                if not sel:
                    continue
                uni = snap.loc[d]
                # 匹配宇宙 = 统一宇宙中该日有快照特征的股票 (含已选; match_controls 自身排除候选)
                uni = uni[uni.index.isin(sel.keys()) | uni.index.isin(uni_codes_by_date.get(d, []))]
                uni = uni[~uni.index.duplicated(keep="last")]
                if len(uni) < 2:
                    continue
                names_sel = [c for c in sel.keys() if c in uni.index]
                cand.loc[d, names_sel] = [sel[c] for c in names_sel]
                m = match_controls(uni, names_sel, ["log_mv", "log_amount20"], SEED)
                ctl = m.control.dropna().tolist()
                ctrl.loc[d, ctl] = np.arange(len(ctl), 0, -1, dtype=float)
                codes_.update(names_sel)
                codes_.update(ctl)
            return cand, ctrl, codes_

        excluded, market = {}, None
        plans, codes_union = None, None
        for attempt in range(3):
            plans, codes_union = {}, set()
            for name, _ in specs:
                c, k, cs = build_plans(picks_by_variant[name])
                plans[name] = (c, k)
                codes_union |= cs
            if not codes_union:
                raise RuntimeError(f"[{Y}] attempt{attempt}: 全变体无候选 (宇宙/匹配为空)")
            cache = fetch_execution_data(sorted(codes_union), (y0, y1), CFG)
            prepared = run_dir / f"prepared_{Y}"
            prepared.mkdir(exist_ok=True)
            pd.DataFrame(dict(ts_code=sorted(codes_union))).to_csv(
                prepared / "execution_codes.csv", index=False)
            (prepared / "execution_cache.txt").write_text(str(cache))
            cfg_y = dict(CFG, valid=[str(y0.date()), str(y1.date())])
            market, b2, st2, limits, actions = load_inputs(prepared, cfg_y)
            market, missing = patch_missing_days(market, y0, y1)
            if missing:
                assert not (set(grid) & set(missing)), "缺失日恰为信号日"
                print(f"[{Y}] 应用修复投影补缺失交易日: {[str(x.date()) for x in missing]}", flush=True)
            bad_a = screen_actions(market, actions, CFG["portfolio"]["action_price_atol"],
                                   CFG["portfolio"]["action_price_rtol"])
            bad_p = screen_prices(market, actions)
            newly = {k: v for k, v in ({**bad_a, **bad_p}).items() if k not in excluded}
            if not newly:
                break
            excluded.update(newly)
            # 重新打分时从宇宙剔除 (随机/避免类同样剔除, 保持对称)
            uni_codes_by_date = {d: [c for c in cs if c not in excluded]
                                 for d, cs in uni_codes_by_date.items()}
            for name, spec in specs:
                picks_by_variant[name] = score_variant(
                    spec, facs, weights_all, uni_codes_by_date, spec.get("seed", name))
            print(f"[{Y}] 第{attempt+1}轮核验隔离 {len(newly)}: "
                  f"行动 {sorted(bad_a)} 价格 {sorted(bad_p)}", flush=True)
        if excluded:
            (run_dir / f"{Y}_excluded.json").write_text(
                json.dumps(excluded, indent=2, ensure_ascii=False))
        if market is None:
            raise RuntimeError(f"[{Y}] 无可用行情 (宇宙为空?)")

        # 回测全部变体 (candidate + control)
        for name, _ in specs:
            cand, ctrl = plans[name]
            rets = {}
            for side, planx in [("cand", cand), ("ctrl", ctrl)]:
                res = run_backtest(planx, market, b2, st2, limits, actions, y0, y1,
                                   p_half, rights.get("rights_issues", []))
                res = add_cash_reserve(res, cap - p_half["capital"],
                                       CFG["portfolio"]["annual_days"])
                res.daily.to_csv(run_dir / f"{Y}_{name}_{side}_daily.csv")
                rets[side] = res.daily.ret
                if side == "cand":
                    rows.append(dict(year=Y, variant=name, side="cand",
                                     total=res.summary.get("total"),
                                     ann=res.summary.get("ann"), mdd=res.summary.get("mdd"),
                                     sharpe=res.summary.get("sharpe")))
            exc = (rets["cand"] - rets["ctrl"]).dropna()
            rows.append(dict(year=Y, variant=name, side="excess",
                             total=float((1 + exc).prod() - 1),
                             ann=float(exc.mean() * 244), mdd=np.nan, sharpe=np.nan))
            print(f"  {name}: cand {rets['cand'].sum()*100:+.1f}% "
                  f"excess {exc.mean()*244*100:+.2f}pp", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "ablation_raw.csv", index=False)

    # 汇总: 每变体合并超额 + bootstrap
    exc_all = df[df.side == "excess"].pivot_table(index="year", columns="variant", values="ann")
    summary = []
    rng = np.random.default_rng(SEED)
    for v in exc_all.columns:
        per_year = exc_all[v].dropna()
        # 合并日度超额: 从年度日度文件重建
        daily = []
        for Y in per_year.index:
            f = run_dir / f"{Y}_{v}_excess_daily.csv"
            if not f.exists():
                c = pd.read_csv(run_dir / f"{Y}_{v}_cand_daily.csv", index_col=0, parse_dates=True)
                k = pd.read_csv(run_dir / f"{Y}_{v}_ctrl_daily.csv", index_col=0, parse_dates=True)
                d = (c.ret - k.ret).dropna()
            else:
                d = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
            daily.append(d.rename(Y))
        d = pd.concat(daily).dropna()
        vals = d.values
        n, block = len(vals), 20
        starts = rng.integers(0, n, size=(2000, int(np.ceil(n / block))))
        idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(2000, -1)[:, :n]
        boots = vals[idx].mean(axis=1) * 244
        summary.append(dict(variant=v, pooled_ann=d.mean() * 244,
                            boot_lo=float(np.quantile(boots, 0.025)),
                            boot_hi=float(np.quantile(boots, 0.975)),
                            years_pos=int((per_year > 0).sum()), n_years=len(per_year),
                            median_year=float(per_year.median())))
    sm = pd.DataFrame(summary).sort_values("pooled_ann", ascending=False)
    sm.to_csv(run_dir / "ablation_summary.csv", index=False)
    print("\n=== 消融合并超额 (vs 各自 matched control) ===")
    print(sm.round(4).to_string(index=False))
    ledger = ExperimentLedger()
    ledger.log("walkforward_ablation", "variant_vs_matched_control",
               f"{args.years[0]}~{args.years[-1]}",
               np.nan, np.nan, np.nan, np.nan,
               params=dict(n_variants=len(specs)), verdict="PREREGISTERED",
               experiment_id=str(run_dir),
               selection_note="消融附录预注册, 记1次试验; 全变体保留不选冠军")
    print(f"\n产出 -> {run_dir}")


if __name__ == "__main__":
    main()
