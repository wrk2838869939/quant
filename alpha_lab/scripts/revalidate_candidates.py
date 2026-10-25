#!/usr/bin/env python3
"""固定候选的 TRAIN/VALID 重验；读取截断、末端剔除、双后端与旧 IC 口径对照。"""
import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_gp_mining as entry
import gpu_eval
from expr_engine import eval_node
from labels import purged_dates
from ledger import ExperimentLedger
from rank_ic import rank_ic_rows, rank_ic_rows_torch


def old_ic_rows(f, lab, min_obs):
    """仅作归因：固定新因子值，复现旧合成评估公式，不复现旧算子。"""
    fr = f.rank(axis=1)
    fm = fr.sub(fr.mean(axis=1), axis=0)
    ym = lab.sub(lab.mean(axis=1), axis=0)
    result = (fm * ym).mean(axis=1) / (fr.std(axis=1) * lab.std(axis=1) + 1e-12)
    return result.where((f.notna() & lab.notna()).sum(axis=1) >= min_obs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/revalidate_candidates.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train_start, train_end = map(pd.Timestamp, cfg["train"])
    valid_start, valid_end = map(pd.Timestamp, cfg["valid"])
    if not train_start <= train_end < valid_start <= valid_end < pd.Timestamp("2024-01-01"):
        raise ValueError("只允许不重叠的 TRAIN/VALID，且截止日期必须早于 2024")
    if not torch.cuda.is_available():
        raise RuntimeError("本重验要求实际 CUDA 设备，不能静默跳过 GPU 核对")
    output = ROOT / cfg["output_root"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(args.config.read_text())
    entry.PANEL = ROOT / cfg["panel"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [*sorted((ROOT / "src").glob("*.py")), Path(__file__),
                        ROOT / "scripts/run_gp_mining.py", args.config.resolve()]}
    stat = entry.PANEL.stat()
    manifest = dict(status="running", panel=str(entry.PANEL), panel_bytes=stat.st_size,
                    panel_mtime_ns=stat.st_mtime_ns, read_end=str(valid_end.date()),
                    source_hashes=hashes, torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(), test_accessed=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"输出：{output}", flush=True)
    started = time.monotonic()
    wide = entry.build_wide(cfg["columns"], end_date=valid_end)
    if wide["close"].index.max() > valid_end:
        raise AssertionError("读取边界失效")
    wide["vwap_dev"] = wide["amt"] / wide["vol"] / wide["close"] - 1
    wide["ret1"] = wide["ret1"] / 100
    wide["rng"] = (wide["high"] - wide["low"]) / wide["close"]
    wide["gap"] = wide["close"] / wide["close"].shift(1) - 1
    wide["lg_share"] = (wide["buy_lg"] + wide["sell_lg"]) / (wide["amt"] + 1e-6)
    library = {
        "low_turnover": -wide["to"].rolling(20, min_periods=15).mean(),
        "low_amp": -((wide["high"] - wide["low"]) / wide["close"].shift(1)).rolling(21, min_periods=15).mean(),
        "low_vol": -wide["ret1"].rolling(20, min_periods=15).std(),
        "rev20": -(wide["hclose"] / wide["hclose"].shift(20) - 1),
        "amihud": (wide["ret1"].abs() / (wide["amt"] / 1e8 + 1e-6)).rolling(20, min_periods=15).mean(),
    }
    print(f"读取 {wide['close'].shape}，末日 {wide['close'].index.max().date()}，构建标签及 GPU 输入", flush=True)
    tensors = gpu_eval.prep_wide(wide)
    calendar = wide["close"].index
    labels, splits = {}, {}
    for h in sorted({c["horizon"] for c in cfg["candidates"]}):
        if h not in (5, 20):
            raise ValueError("仅支持既有 5/20 日标签")
        labels[h] = (wide["hclose"].shift(-h) / wide["hclose"] - 1).rank(axis=1, pct=True)
        splits[h] = {name: purged_dates(calendar[(calendar >= lo) & (calendar <= hi)],
                                       h, cfg["embargo"])[::h]
                     for name, lo, hi in [("train", train_start, train_end),
                                           ("valid", valid_start, valid_end)]}
        for name, dates in splits[h].items():
            print(f"h={h} {name}: {dates.min().date()}~{dates.max().date()}, {len(dates)} 截面", flush=True)
    rows, yearly_rows, series_rows = [], [], []
    ledger = ExperimentLedger()
    for candidate in cfg["candidates"]:
        expr, h = candidate["expr"], candidate["horizon"]
        print(f"重验 h={h} {expr}", flush=True)
        tree = entry.parse_expr(expr)
        with np.errstate(all="ignore"):
            factor = eval_node(tree, wide)
        with torch.no_grad():
            factor_gpu = gpu_eval.eval_gpu(tree, tensors)
        row = dict(expr=expr, horizon=h, prior_status=candidate["status"])
        all_ic, all_parity = [], True
        for split, dates in splits[h].items():
            positions = torch.tensor(calendar.get_indexer(dates), device=factor_gpu.device)
            gpu_sub = factor_gpu[positions]
            f, lab = factor.loc[dates], labels[h].loc[dates]
            ic = rank_ic_rows(f, lab, cfg["min_cross_section"])
            gpu_ic = pd.Series(rank_ic_rows_torch(gpu_sub, torch.tensor(lab.values,
                device=gpu_sub.device, dtype=gpu_sub.dtype), cfg["min_cross_section"]).cpu().numpy(), index=dates)
            old = old_ic_rows(f, lab, cfg["min_cross_section"])
            actual, expected = gpu_sub.cpu().numpy(), f.to_numpy()
            mask_diff = int(np.count_nonzero(np.isfinite(actual) != np.isfinite(expected)))
            finite = np.isfinite(actual) & np.isfinite(expected)
            numeric_diff = int(np.count_nonzero(~np.isclose(actual[finite], expected[finite],
                rtol=cfg["value_rtol"], atol=cfg["value_atol"])))
            ic_mask_diff = int(np.count_nonzero(ic.notna() != gpu_ic.notna()))
            max_delta = float((ic - gpu_ic).abs().max())
            ic_parity = ic_mask_diff == 0 and bool(np.allclose(ic, gpu_ic,
                rtol=0, atol=cfg["ic_atol"], equal_nan=True))
            all_parity &= mask_diff == 0 and numeric_diff == 0 and ic_parity
            row.update({f"{split}_ic": ic.mean(), f"{split}_n": int(ic.count()),
                        f"{split}_icir": ic.mean() / ic.std(),
                        f"{split}_old_formula_ic": old.mean(),
                        f"{split}_gpu_ic": gpu_ic.mean(),
                        f"{split}_ic_max_delta": max_delta,
                        f"{split}_mask_mismatch": mask_diff,
                        f"{split}_value_mismatch": numeric_diff,
                        f"{split}_ic_mask_mismatch": ic_mask_diff})
            lib_dates = dates[::cfg["library_sample_stride"]]
            correlations = {name: rank_ic_rows(f.loc[lib_dates], lib.loc[lib_dates],
                cfg["min_cross_section"]).mean() for name, lib in library.items()}
            row[f"{split}_lib_corr"] = max((abs(v) for v in correlations.values()
                if np.isfinite(v)), default=np.nan)
            for date in dates:
                series_rows.append(dict(expr=expr, horizon=h, split=split, date=date,
                    cpu_ic=ic.loc[date], gpu_ic=gpu_ic.loc[date], old_formula_ic=old.loc[date]))
            for year, values in ic.groupby(ic.index.year):
                yearly_rows.append(dict(expr=expr, horizon=h, year=year,
                                        ic=values.mean(), n=int(values.count())))
            all_ic.append(ic)
        row["decay"] = abs(row["train_ic"] / row["valid_ic"]) if abs(row["valid_ic"]) > 1e-6 else np.inf
        row["backend_parity"] = all_parity
        row["ic_screen_pass"] = bool(row["train_n"] >= cfg["min_train_days"]
            and row["valid_n"] >= cfg["min_valid_days"]
            and np.sign(row["train_ic"]) == np.sign(row["valid_ic"])
            and abs(row["train_ic"]) >= cfg["min_ic"]
            and abs(row["valid_ic"]) >= cfg["min_ic"] * cfg["valid_ic_ratio"]
            and row["decay"] <= cfg["valid_decay_max"]
            and row["valid_lib_corr"] < cfg["library_corr_max"])
        annual = pd.concat(all_ic).groupby(lambda d: d.year).mean()
        row["annual_sign_agreement"] = float((np.sign(annual) == np.sign(row["train_ic"])).mean())
        row["review_status"] = ("EXCLUDED_CONTROL" if candidate["status"] == "excluded_control"
            else "BACKEND_MISMATCH" if not all_parity
            else "IC_SCREEN_PASS" if row["ic_screen_pass"] else "IC_SCREEN_FAIL")
        rows.append(row)
        ledger.log(expr, f"fwd{h}_excess_rank", "TRAIN+VALID purged revalidation",
            row["train_ic"], row["train_icir"], np.nan, np.nan,
            params=dict(output=str(output), horizon=h, embargo=cfg["embargo"],
                        valid_ic=row["valid_ic"], prior_status=candidate["status"]),
            verdict=row["review_status"], note="既有候选口径重验；非新挖掘/非收益验证；TEST未访问")
        pd.DataFrame(rows).to_csv(output / "candidates.csv", index=False)
        pd.DataFrame(yearly_rows).to_csv(output / "yearly_ic.csv", index=False)
        pd.DataFrame(series_rows).to_csv(output / "ic_series.csv", index=False)
        print(f"  TRAIN {row['train_ic']:+.5f} / VALID {row['valid_ic']:+.5f}; {row['review_status']}", flush=True)
        del factor, factor_gpu
        gc.collect()
        torch.cuda.empty_cache()
    manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                    panel_rows_end=str(calendar.max()), shape=list(wide["close"].shape),
                    evaluated=len(rows), ledger_rows_after=len(ledger.df))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"完成：{output}，{manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
