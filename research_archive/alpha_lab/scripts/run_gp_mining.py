# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""Phase 3a: GP 因子挖掘入口 (三段式: TRAIN 挖掘 / VALID 入库 / TEST 一次性终评)。

宽表构建 → 现有因子库(Phase2) → GP 进化 → 收割。
用法: python run_gp_mining.py [--pop 100] [--gen 10] [--tag smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("alpha_lab")
sys.path.insert(0, str(ROOT / "src"))

PANEL = ROOT / "data/panel/daily_panel.parquet"
SW = ROOT / "data/raw/sw_l1_member_hist.parquet"
OUT = ROOT / "outputs/gp_mining"

# 三段式切分 (TEST 窗在挖掘中绝不触碰)
SPLIT = dict(train=("2014-01-01", "2021-12-31"),
             valid=("2022-01-01", "2023-12-31"),
             test=("2024-01-01", "2026-09-08"))


def build_wide(cols_map: dict, end_date=None) -> dict:
    filters = [("date", "<=", pd.Timestamp(end_date))] if end_date is not None else None
    df = pd.read_parquet(PANEL, columns=["ts_code", "date"] + list(cols_map.values()),
                         filters=filters)
    dates = pd.DatetimeIndex(sorted(df["date"].unique()), name="date")
    codes = pd.Index(sorted(df["ts_code"].unique()), name="ts_code")
    wide = {}
    for name, col in cols_map.items():
        w = df.pivot_table(index="date", columns="ts_code", values=col, aggfunc="first")
        wide[name] = w.reindex(index=dates, columns=codes)
    return wide


def parse_expr(s: str):
    """前缀表达式解析: op(a,b,w) → Node (支持嵌套/数字窗口/叶子)。"""
    import re
    from expr_engine import Node
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|.", s)
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def next_tok():
        t = tokens[pos[0]]; pos[0] += 1; return t

    def parse():
        t = next_tok()
        if t is None:
            raise ValueError("unexpected end")
        nxt = peek()
        if nxt == "(":
            next_tok()  # consume (
            args = []
            while peek() != ")":
                args.append(parse())
                if peek() == ",":
                    next_tok()
            next_tok()  # consume )
            window = None
            if args and args[-1].leaf and args[-1].leaf.isdigit():
                window = int(args.pop().leaf)
            return Node(t, args, window)
        return Node(op=None, leaf=t)

    node = parse()
    if pos[0] != len(tokens):
        raise ValueError(f"trailing tokens: {tokens[pos[0]:]}")
    return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=10)
    ap.add_argument("--tag", type=str, default="run1")
    ap.add_argument("--train-years", type=str, default="2018-2021")
    ap.add_argument("--gpu", action="store_true", help="GPU 后端 (实测表达式级 9~53x)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra-libs", type=str, default="", help="逗号分隔的历届harvest.csv, passed因子纳入库惩罚")
    ap.add_argument("--horizon", type=int, default=5, help="标签前向窗口: 5=周频, 20=月频")
    ap.add_argument("--embargo", type=int, default=5, help="窗口末端额外剔除交易日数")
    args = ap.parse_args()
    y0, y1 = map(int, args.train_years.split("-"))
    if not 2014 <= y0 <= y1 <= 2021 or args.horizon not in (5, 20) or args.embargo < 0:
        ap.error("TRAIN 必须位于 2014-2021；horizon=5/20；embargo>=0")
    OUT.mkdir(parents=True, exist_ok=True)

    from gp_miner import GPMiningTask, GPConfig
    if args.gpu:
        from gp_miner_gpu import GPGPMiningTask

    t0 = time.time()
    print("构建宽表...", flush=True)
    cols_map = dict(
        ret1="pct_chg", vol="vol_share", amt="amount_yuan",
        to="turnover_rate", high="high", low="low", close="close",
        hclose="hfq_close",
        buy_lg="buy_lg_amount_yuan", sell_lg="sell_lg_amount_yuan",
        buy_elg="buy_elg_amount_yuan", net_mf="net_mf_amount_yuan",
        circ_mv="circ_mv_yuan", volume_ratio="volume_ratio",
    )
    cols_map = {k: v for k, v in cols_map.items() if v}
    wide = build_wide(cols_map, end_date=SPLIT["valid"][1])
    # 衍生序列 (量价日内结构)
    wide["vwap_dev"] = (wide["amt"] / wide["vol"]) / wide["close"] - 1.0
    wide["ret1"] = wide["ret1"] / 100.0
    wide["rng"] = (wide["high"] - wide["low"]) / wide["close"]            # 日振幅
    wide["cmo"] = (wide["close"] - wide["close"].shift(1)) / wide["close"].shift(1)
    wide["gap"] = wide["close"] / wide["close"].shift(1) - 1.0
    wide["mf_imb"] = (wide["buy_lg"] - wide["sell_lg"]) / (wide["buy_lg"] + wide["sell_lg"] + 1e-6)
    wide["lg_share"] = (wide["buy_lg"] + wide["sell_lg"]) / (wide["amt"] + 1e-6)
    wide["log_mv"] = np.log(wide["circ_mv"].where(wide["circ_mv"] > 0))
    print(f"  {len(wide)} 序列 × {wide['close'].shape}, {time.time()-t0:.0f}s", flush=True)

    # 标签宽表 (fwd{h} 超额 rank, h=5周频/20月频)
    print(f"标签宽表 (fwd{args.horizon})...", flush=True)
    lab = wide["hclose"].shift(-args.horizon) / wide["hclose"] - 1.0
    mkt = lab.mean(axis=1)
    lab = lab.sub(mkt, axis=0)
    lab = lab.rank(axis=1, pct=True)

    # 现有因子库 (Phase 2 的最强 5 条, 中性化版直接从复现脚本口径重建太贵 —— 用原始因子宽表)
    print("因子库宽表...", flush=True)
    lib = {}
    lib["low_turnover"] = -wide["to"].rolling(20, min_periods=15).mean()
    amp = (wide["high"] - wide["low"]) / wide["close"].shift(1)
    lib["low_amp"] = -amp.rolling(21, min_periods=15).mean()
    lib["low_vol"] = -wide["ret1"].rolling(20, min_periods=15).std()
    lib["rev20"] = -(wide["hclose"] / wide["hclose"].shift(20) - 1)
    illiq = wide["ret1"].abs() / (wide["amt"] / 1e8 + 1e-6)
    lib["amihud"] = illiq.rolling(20, min_periods=15).mean()

    # 历届 VALID_PASS 因子纳入库 (增量挖掘去相关)
    if args.extra_libs:
        import expr_engine as E2
        for path in args.extra_libs.split(","):
            hp = pd.read_csv(path)
            for _, r in hp[hp["passed"] == True].iterrows():
                try:
                    from expr_engine import Node as _N
                    tree = parse_expr(r["expr"])
                    with np.errstate(all="ignore"):
                        lib[f"gp:{r['expr'][:24]}"] = E2.eval_node(tree, wide)
                    print(f"  + lib: {r['expr'][:40]}")
                except Exception as e:
                    print(f"  [WARN] 库重建失败 {r['expr'][:30]}: {e}")

    from labels import purged_dates
    tr = lab.loc[f"{y0}-01-01":f"{y1}-12-31"].index
    va = lab.loc[SPLIT["valid"][0]:SPLIT["valid"][1]].index
    tr = purged_dates(tr, args.horizon, args.embargo)
    va = purged_dates(va, args.horizon, args.embargo)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("剔除标签窗口后 TRAIN/VALID 无样本")
    print(f"TRAIN {tr[0].date()}~{tr[-1].date()} ({len(tr)}d) | VALID {va[0].date()}~{va[-1].date()} ({len(va)}d)", flush=True)

    cfg = GPConfig(n_population=args.pop, n_generations=args.gen, seed=args.seed)
    sample_weeks = max(args.horizon // 5, 1)
    if args.gpu:
        task = GPGPMiningTask(wide, lab, lib, tr, va, cfg=cfg, sample_weeks=sample_weeks)
        task.prep()
    else:
        task = GPMiningTask(wide, lab, lib, tr, va, cfg=cfg, sample_weeks=sample_weeks)
    print(f"GP 启动: pop={cfg.n_population} gen={cfg.n_generations} terms={len(wide)} "
          f"backend={'GPU' if args.gpu else 'CPU'}", flush=True)
    scored, hist = task.run()

    print("\n收割 (VALID 复核)...", flush=True)
    harv = task.harvest(scored, top_n=30)
    harv.to_csv(OUT / f"harvest_{args.tag}.csv", index=False)
    pd.DataFrame(hist).to_csv(OUT / f"history_{args.tag}.csv", index=False)
    # 落盘标签列名供台账
    print(harv.to_string(index=False) if len(harv) else "(无个体通过训练门槛)")
    print(f"\n用时 {(time.time()-t0)/60:.0f}min -> {OUT}/harvest_{args.tag}.csv")
    # 台账补 horizon 标记
    led = task.led
    led.df.loc[led.df.index[-len(harv):] if len(harv) else [], "label"] = f"fwd{args.horizon}_excess_rank"
    led.df.to_csv(led.path, index=False)


if __name__ == "__main__":
    main()
