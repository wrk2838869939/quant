#!/usr/bin/env python3
"""固定原月频十一因子，TRAIN 定权，准备 VALID 信号和所需执行数据。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_gp_mining import build_wide
from synthesis_and_test import build_all_factors
from labels import purged_dates
from rank_ic import rank_ic_rows


def fetch_execution_data(codes, dates, cfg):
    """按固定候选代码补分红和涨跌停；不下载 TEST 行情，空响应与失败区分。"""
    from ts_client import get_pro
    start, end = dates
    cache = ROOT / "data/raw/execution_audit" / f"{start:%Y%m%d}_{end:%Y%m%d}"
    cache.mkdir(parents=True, exist_ok=True)
    failures = []

    def worker(code):
        pro = get_pro()
        paths = {}
        for api in ("dividend", "stk_limit"):
            path = cache / api / f"{code}.parquet"
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                paths[api] = path
                continue
            kwargs = {"ts_code": code}
            if api == "stk_limit":
                kwargs.update(start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
            try:
                frame = getattr(pro, api)(**kwargs)
                if frame is None:
                    raise ValueError("接口返回 None")
                if len(frame) >= (2000 if api == "dividend" else cfg["api_row_limit"]):
                    raise ValueError("达到接口行数上限，可能截断")
                if api == "dividend" and len(frame):
                    # API 只支持按代码查全史；在进入研究产物前剔除窗口外实施事件。
                    ex = pd.to_datetime(frame["ex_date"], errors="coerce")
                    frame = frame[(ex >= start) & (ex <= end)].copy()
                temp = path.with_suffix(".tmp.parquet")
                frame.to_parquet(temp, index=False)
                temp.replace(path)
                paths[api] = path
            except Exception as exc:
                return code, type(exc).__name__  # 不输出可能含凭证的远端异常原文
            finally:
                time.sleep(cfg["api_interval_seconds"])
        return code, None

    with ThreadPoolExecutor(max_workers=cfg["data_workers"]) as pool:
        pending = [pool.submit(worker, code) for code in sorted(codes)]
        for n, future in enumerate(as_completed(pending), 1):
            code, error = future.result()
            if error:
                failures.append({"ts_code": code, "error": error})
            if n % 25 == 0 or n == len(codes):
                print(f"执行数据 {n}/{len(codes)}, 失败 {len(failures)}", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(cache / "failures.csv", index=False)
        raise RuntimeError(f"执行数据未齐全：{len(failures)} 个代码，重跑可补齐")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/monthly_portfolio_audit.yaml")
    ap.add_argument("--resume", type=Path, help="复用已固定的信号目录，仅补执行数据")
    args = ap.parse_args()
    cfg = yaml.safe_load((args.resume / "config.yaml" if args.resume else args.config).read_text())
    lo, hi = map(pd.Timestamp, cfg["valid"])
    if hi >= pd.Timestamp("2024-01-01") or pd.Timestamp(cfg["train"][1]) >= lo:
        raise ValueError("TRAIN/VALID 必须隔离，且 VALID 不得进入 TEST")
    if args.resume:
        output = args.resume
        codes = set(pd.read_csv(output / "execution_codes.csv")["ts_code"])
    else:
        output = ROOT / cfg["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.yaml").write_text(args.config.read_text())
        print(f"准备目录 {output}", flush=True)
        cols = dict(ret1="pct_chg", vol="vol_share", amt="amount_yuan", to="turnover_rate",
                    high="high", low="low", close="close", hclose="hfq_close",
                    buy_lg="buy_lg_amount_yuan", sell_lg="sell_lg_amount_yuan",
                    pe="pe_ttm", pb="pb", mv="circ_mv_yuan")
        wide = build_wide(cols, end_date=hi)
        wide["ret1"] = wide["ret1"] / 100
        wide["vwap_dev"] = wide["amt"] / wide["vol"] / wide["close"] - 1
        wide["rng"] = (wide["high"] - wide["low"]) / wide["close"]
        wide["lg_share"] = (wide["buy_lg"] + wide["sell_lg"]) / (wide["amt"] + 1e-6)
        factors = build_all_factors(wide, horizon=cfg["horizon"])
        factors["lit:ep"] = 1 / wide["pe"].where(wide["pe"] > 0)
        factors["lit:bp"] = 1 / wide["pb"].where(wide["pb"] > 0)
        prices = wide["hclose"]
        labels = (prices.shift(-cfg["horizon"]) / prices - 1).rank(axis=1, pct=True)
        train = prices.loc[cfg["train"][0]:cfg["train"][1]].index
        train = purged_dates(train, cfg["horizon"], cfg["embargo"])[::cfg["horizon"]]
        valid = prices.loc[lo:hi].index
        signal_dates = valid[::cfg["horizon"]]
        weights, composite = {}, None
        for name, factor in factors.items():
            ic = rank_ic_rows(factor.loc[train], labels.loc[train], cfg["min_cross_section"]).dropna()
            weight = ic.mean() / ic.std()
            if not np.isfinite(weight):
                raise ValueError(f"{name} 无有限 TRAIN 权重")
            weights[name] = dict(weight=float(weight), train_ic=float(ic.mean()), n=len(ic))
            sub = factor.loc[signal_dates]
            z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
            composite = weight * z if composite is None else composite + weight * z
        composite.to_parquet(output / "signals.parquet")
        benchmark = wide["mv"].loc[signal_dates]
        benchmark.to_parquet(output / "benchmark_signals.parquet")
        pd.DataFrame(weights).T.to_csv(output / "weights.csv")
        # 仅 topN 可买，所有可能持仓均来自这份事前信号候选并集。
        codes = set()
        for frame in (composite, benchmark):
            for _, row in frame.iterrows():
                codes.update(row.dropna().sort_values(ascending=False, kind="stable").head(cfg["portfolio"]["top_n"]).index)
        pd.DataFrame({"ts_code": sorted(codes)}).to_csv(output / "execution_codes.csv", index=False)
        sources = [Path(__file__), ROOT / "scripts/synthesis_and_test.py", ROOT / "src/rank_ic.py",
                   ROOT / "src/expr_engine.py", ROOT / "scripts/run_gp_mining.py"]
        manifest = dict(train=cfg["train"], valid=cfg["valid"], weight_last_label_end_before=str(lo.date()),
                        training_signal_last=str(train[-1]), signal_periods=len(signal_dates),
                        test_market_data_used=False, fresh_holdout=False,
                        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        (output / "preparation.json").write_text(json.dumps(manifest, indent=2))
        print(f"固定十一因子权重完成，VALID {len(signal_dates)} 个信号日，候选并集 {len(codes)} 只", flush=True)
        del factors, labels, wide, prices
    cache = fetch_execution_data(codes, (lo, hi), cfg)
    (output / "execution_cache.txt").write_text(str(cache))
    print(f"准备完成 {output}", flush=True)


if __name__ == "__main__":
    main()
