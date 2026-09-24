"""IO 层: 分区数据集读取 (pyarrow 谓词下推 + 列裁剪) 与特征磁盘缓存。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from .config import HubConfig

def cn_dataset(cfg: HubConfig):
    # 对象创建很轻；每次发现文件，避免跨数据根串读或更新后沿用旧分区。
    return pads.dataset(str(cfg.cn_daily), format="parquet", partitioning="hive")


def _to_list(x):
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def _year_bounds(start=None, end=None):
    lo = hi = None
    if start is not None:
        lo = pd.Timestamp(start).year
    if end is not None:
        hi = pd.Timestamp(end).year
    return lo, hi


def read_cn_daily(cfg: HubConfig, codes=None, start=None, end=None, fields=None,
                  board=None, adjust: str | None = None, use_threads: bool = True) -> pd.DataFrame:
    """读 A股日线。codes=None 为全市场; adjust='hfq' 时 OHLC 换成后复权列。

    返回按 (ts_code, date) 升序的 DataFrame; 恒含 ts_code/date 列。
    """
    ds = cn_dataset(cfg)
    if adjust not in (None, "none", "raw", "hfq"):
        raise ValueError(f"adjust 只支持 None/'hfq', 收到: {adjust}")
    fl = []
    lo, hi = _year_bounds(start, end)
    if lo is not None:
        fl.append(pads.field("year") >= lo)
    if hi is not None:
        fl.append(pads.field("year") <= hi)
    codes_l = _to_list(codes)
    if codes_l is not None:
        fl.append(pads.field("ts_code").isin(pa.array(codes_l, type=pa.string())))
    if board is not None:
        fl.append(pads.field("board").isin(_to_list(board)))
    if start is not None:
        fl.append(pads.field("date") >= pd.Timestamp(start))
    if end is not None:
        fl.append(pads.field("date") <= pd.Timestamp(end))

    keep = ([c for c in ds.schema.names if c != "year"] if fields is None
            else list(dict.fromkeys(["ts_code", "date", *fields])))
    read_columns = list(keep)
    if adjust == "hfq":
        # 保留显式请求的 hfq_* 列，避免同时请求 close/hfq_close 出现重名列。
        read_columns = list(dict.fromkeys(
            "hfq_" + f if f in ("open", "high", "low", "close") else f for f in keep))
    expr = fl[0] if fl else None
    for f in fl[1:]:
        expr = expr & f
    table = ds.to_table(filter=expr, columns=read_columns, use_threads=use_threads)
    df = table.to_pandas()
    if adjust == "hfq":
        for f in ("open", "high", "low", "close"):
            if f in keep:
                df[f] = df["hfq_" + f]
        df = df[keep]
    return df.sort_values(["ts_code", "date"], kind="stable").reset_index(drop=True)


def read_us_daily(cfg: HubConfig, tickers=None, start=None, end=None, fields=None,
                  asset_type=None) -> pd.DataFrame:
    df = pd.read_parquet(cfg.us_daily)
    if tickers is not None:
        df = df[df["ticker"].isin(_to_list(tickers))]
    if asset_type is not None:
        df = df[df["asset_type"].isin(_to_list(asset_type))]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    keep = ["ticker", "date"] + [f for f in (fields or []) if f not in ("ticker", "date")]
    if fields:
        keep += [c for c in ("asset_type",) if c in df.columns and c not in keep]
        df = df[keep]
    return df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


def cn_max_date(cfg: HubConfig) -> pd.Timestamp:
    """数据集内最大交易日 (用于缓存键/新鲜度提示)。"""
    import pyarrow.compute as pc
    ds = cn_dataset(cfg)
    t = ds.to_table(columns=["date"])  # 单列扫描, ~100MB 内存峰值
    return pd.Timestamp(pc.max(t["date"]).as_py())


def cache_key(payload: dict, data_version: str) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "|research-v2|" + data_version
    return hashlib.sha1(blob.encode()).hexdigest()[:24]


def load_or_compute(cache_name: str, payload: dict, compute_fn, cfg: HubConfig,
                    data_version: str, refresh: bool = False):
    """特征磁盘缓存: key=payload+data_version, 命中读 parquet, 未命中算后落盘。"""
    key = cache_key(payload, data_version)
    path = cfg.cache_dir / f"{cache_name}_{key}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    out = compute_fn()
    out.to_parquet(path, index=False)
    return out


def clear_cache(cfg: HubConfig, older_than_days: float | None = None):
    import time
    now = time.time()
    for p in Path(cfg.cache_dir).glob("*.parquet"):
        if older_than_days is None or (now - p.stat().st_mtime) > older_than_days * 86400:
            p.unlink(missing_ok=True)


def parquet_dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6
