# 历史 v3：含稀疏净值/年化/除权记账缺陷，禁止用作生产或收益依据。
#!/usr/bin/env python3
"""Phase 4: 周频组合回测器 v3 —— A股制度约束完整版 (诚实口径, 线性状态机)。

持仓状态: positions[c] = {"value": 货币估值, "px": 最近估值价格}
  卖出: cash += value × (op[t1]/px) × (1-sell_cost)
  买入: positions[c] = {"value": w×(1-buy_cost), "px": op[t1]}
  估值: value ×= cl[t1]/px; px = cl[t1]

制度: T收盘信号 → T+1开盘成交; 开盘涨停禁买(候补); 开盘跌停/停牌禁卖(顺延);
      新股<60日/ST剔除; 退市自然退场(最后价)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/alpha_lab")
sys.path.insert(0, str(ROOT / "src"))


def build_filters():
    nc = pd.read_parquet(ROOT / "data/raw/namechange_all.parquet")
    basic = pd.read_parquet(ROOT / "data/raw/stock_basic_all.parquet")
    nc["start"] = pd.to_datetime(nc["start_date"], format="%Y%m%d", errors="coerce")
    nc["end"] = pd.to_datetime(nc["end_date"], format="%Y%m%d", errors="coerce")
    st_iv = nc[nc["name"].str.contains("ST", case=False, na=False)][["ts_code", "start", "end"]]
    basic["list_dt"] = pd.to_datetime(basic["list_date"], format="%Y%m%d", errors="coerce")
    return st_iv, dict(zip(basic["ts_code"], basic["list_dt"]))


def run_backtest(fac, op, cl, pc, idx, rebal, topn, st_iv, list_dt,
                 buy_cost=0.001, sell_cost=0.002, band_mult=2.0, period_days=5):
    """band_mult: 持仓保留带 —— 跌出 top (topn×band_mult) 才卖; 新买入仅限新进 top topn。
    period_days: 调仓周期交易日数 (5=周频, 20=月频) — 用于年化。"""
    def limit_of(c):
        return 0.19 if str(c)[:2] in ("30", "68") else 0.095

    def st_set_at(d):
        m = st_iv[(st_iv["start"] <= d) & (st_iv["end"].isna() | (st_iv["end"] >= d))]
        return set(m["ts_code"])

    cash = 1.0
    positions = {}   # c -> [value, px]
    eq_rows = []
    n_skip_up = n_skip_st = n_skip_new = 0
    n_buy_events = []
    n_sell_events = []

    for ri, t in enumerate(rebal):
        t1 = op.index[idx[t] + 1]

        # ---- 1) 卖出跌出保留带的持仓 @ t1 开盘 ----
        row0 = fac.loc[t].dropna()
        rank0 = row0.rank(ascending=False)
        keep_zone = set(rank0[rank0 <= topn * band_mult].index) if band_mult else None
        for c in list(positions.keys()):
            if keep_zone is not None and c in keep_zone:
                continue                       # 在保留带内 → 不卖 (降换手)
            val, px = positions[c]
            o1 = op.at[t1, c] if (t1 in op.index and c in op.columns) else np.nan
            p1 = pc.at[t1, c] if (t1 in pc.index and c in pc.columns) else np.nan
            if np.isnan(o1):
                continue                       # 停牌 → 冻结
            lim_dn = p1 * (1 - limit_of(c) + 0.002) if not np.isnan(p1) else np.nan
            if not np.isnan(lim_dn) and o1 <= lim_dn:
                continue                       # 开盘跌停 → 冻结
            cash += val * (o1 / px if px > 0 else 1.0) * (1 - sell_cost)
            n_sell_events.append(1)
            del positions[c]

        # ---- 2) 选股与换仓带 ----
        row = fac.loc[t].dropna()
        st_set = st_set_at(t)
        rank = row.rank(ascending=False)   # 1=最强
        keep_zone = set(rank[rank <= topn * band_mult].index)   # 保留带
        buy_zone = rank[rank <= topn].index.tolist()            # 目标池

        # 旧仓: 跌出保留带 → 卖 (停牌/跌停顺延已在上方处理)
        # (上方卖出循环改为: 只卖跌出带的)
        # ---- 3) 买入: 目标池中未持有的 (涨停/停牌/ST/新股过滤) ----
        cand = []
        for c in buy_zone:
            if len(cand) >= topn:
                break
            if c in positions:
                continue
            ld = list_dt.get(c)
            if ld is None or pd.isna(ld):
                continue
            if (t - ld).days < 60:
                n_skip_new += 1
                continue
            if c in st_set:
                n_skip_st += 1
                continue
            if c not in op.columns or np.isnan(op.at[t1, c]):
                continue
            cand.append(c)
        bought = []
        for c in cand:
            if len(bought) >= topn - len(positions):
                break
            o1, p1 = op.at[t1, c], pc.at[t1, c]
            lim_up = p1 * (1 + limit_of(c) - 0.002) if not np.isnan(p1) else np.nan
            if not np.isnan(lim_up) and o1 >= lim_up:
                n_skip_up += 1
                continue
            bought.append((c, o1))
        n_slots = max(topn - len(positions), 0)
        w = min(cash / max(n_slots, 1), cash / max(len(bought), 1)) if bought else 0.0
        for c, o1 in bought:
            cash -= w
            positions[c] = [w * (1 - buy_cost), o1]
        n_buy_events.append(len(bought))

        # ---- 4) 估值 @ t1 收盘 ----
        v = cash
        for c, (val, px) in positions.items():
            c1 = cl.at[t1, c] if (t1 in cl.index and c in cl.columns) else np.nan
            if np.isnan(c1) or px <= 0:
                v += val
            else:
                v += val * c1 / px
        eq_rows.append(dict(date=t1, equity=v, n_hold=len(positions)))

    eq = pd.DataFrame(eq_rows).set_index("date")
    eq["ret"] = eq["equity"].pct_change().fillna(eq["equity"] - 1)
    total = eq["equity"].iloc[-1] - 1
    ppy = 244 / period_days
    ann = (1 + total) ** (244 / (period_days * len(eq))) - 1 if total > -1 else -1  # 注: eq每期=horizon日
    mdd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    sharpe = eq["ret"].mean() / (eq["ret"].std() + 1e-12) * np.sqrt(244 / 5)
    win = (eq["ret"] > 0).mean()
    turnover = (sum(n_sell_events) / max(len(eq), 1))   # 每期卖出事件数
    ret_period_days = period_days
    return dict(top_n=topn, total=total, ann=ann, mdd=mdd, sharpe=sharpe,
                win_rate=win, n_periods=len(eq), skip_up=n_skip_up,
                skip_st=n_skip_st, skip_new=n_skip_new, band=band_mult,
                sells_per_period=turnover), eq


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=str, default=str(ROOT / "outputs/gp_mining/synth_factor.parquet"))
    ap.add_argument("--top-n", type=int, nargs="+", default=[20, 30, 50])
    ap.add_argument("--start", type=str, default="2024-01-01")
    ap.add_argument("--cost-scan", action="store_true")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--out", type=str, default=str(ROOT / "outputs/gp_mining/phase4_backtest.csv"))
    args = ap.parse_args()

    print("读面板...", flush=True)
    fac = pd.read_parquet(args.factor)
    px = pd.read_parquet(ROOT / "data/panel/daily_panel.parquet",
                         columns=["ts_code", "date", "open", "close", "pre_close"])
    st_iv, list_dt = build_filters()
    op = px.pivot_table(index="date", columns="ts_code", values="open", aggfunc="first").sort_index()
    cl = px.pivot_table(index="date", columns="ts_code", values="close", aggfunc="first").sort_index()
    pc = px.pivot_table(index="date", columns="ts_code", values="pre_close", aggfunc="first").sort_index()
    del px

    idx = {d: i for i, d in enumerate(op.index)}
    dates = [d for d in fac.index if d >= pd.Timestamp(args.start) and d in idx]
    rebal = [d for i, d in enumerate(dates) if i % args.horizon == 0 and idx[d] + 1 < len(op.index)]
    print(f"回测: {rebal[0].date()} ~ {rebal[-1].date()} | {len(rebal)} 期", flush=True)

    results, curves = [], {}
    plans = [(n, 0.001, 0.002, b) for n in args.top_n for b in ([1.0, 1.5, 2.0] if args.cost_scan else [1.0])]
    if args.cost_scan:
        plans += [(30, 0.0005, 0.001, 2.0), (30, 0.002, 0.004, 2.0)]
    for topn, bc, sc, bm in plans:
        r, eq = run_backtest(fac, op, cl, pc, idx, rebal, topn, st_iv, list_dt, bc, sc, bm, args.horizon)
        r.update(buy_cost=bc, sell_cost=sc)
        results.append(r)
        curves[(topn, bc, sc, bm)] = eq
        print(f"  top{topn} cost{bc:.3%}/{sc:.3%} band{bm}: 总收益 {r['total']:+.2%} | 年化 {r['ann']:+.2%} | "
              f"MDD {r['mdd']:.2%} | Sharpe {r['sharpe']:.2f} | 胜率 {r['win_rate']:.0%} | "
              f"每期卖{r['sells_per_period']:.1f}只 | 禁买{r['skip_up']}", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(args.out, index=False)
    key0 = list(curves)[0]
    curves[key0].to_csv(str(args.out).replace(".csv", "_eq.csv"))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    raise SystemExit("旧版仅供审计复现；请使用 scripts/audit_monthly_portfolio.py")
