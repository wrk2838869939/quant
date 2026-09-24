"""并行层: 按代码分块的进程级并行特征计算 (joblib loky)。

每个 worker 独立从分区数据集读取自己负责的代码所需列, 避免大对象跨进程传输。
注意: 若 datahub 未 pip install 而是靠 sys.path 使用, 这里会把包路径注入
PYTHONPATH, 保证 loky 子进程也能 import datahub。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_env_pp = os.environ.get("PYTHONPATH", "")
if _SRC not in _env_pp:
    os.environ["PYTHONPATH"] = f"{_SRC}{os.pathsep}{_env_pp}" if _env_pp else _SRC


def chunk_list(xs: list, n_chunks: int) -> list[list]:
    n = max(1, min(n_chunks, len(xs)))
    k, rem = divmod(len(xs), n)
    out, i = [], 0
    for j in range(n):
        size = k + (1 if j < rem else 0)
        out.append(xs[i:i + size])
        i += size
    return [c for c in out if c]


def _worker_codes(chunk_codes, root, specs, price, start, end, board):
    """读取截至 end 的历史计算后截取输出，确保查询起点不影响数值。

    EWM 依赖全部前缀，停牌会令个股观测数少于日历天数，固定日历预热不可靠。
    """
    from .config import HubConfig
    from .io_utils import read_cn_daily
    from .indicators import compute_features, expand_required_fields

    cfg = HubConfig(root)
    df = read_cn_daily(cfg, codes=chunk_codes, start=None, end=end,
                       fields=expand_required_fields(specs, price), board=board)
    if df.empty:
        return compute_features(df, specs, price=price)
    parts = [compute_features(g, specs, price=price)
             for _, g in df.groupby("ts_code", sort=False)]
    out = pd.concat(parts, ignore_index=True).reset_index(drop=True)
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    return out


def parallel_cn_features(codes: list[str], specs: list[str], price: str,
                         start, end, root: str, n_jobs: int = 4,
                         board=None) -> pd.DataFrame:
    """按代码分块并行计算特征, 返回按 (ts_code, date) 排序的长表。"""
    if n_jobs <= 1 or len(codes) <= 1:
        return _worker_codes(codes, root, specs, price, start, end, board)
    chunks = chunk_list(codes, n_jobs * 2)
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_worker_codes)(c, root, specs, price, start, end, board)
        for c in chunks)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["ts_code", "date"], kind="stable").reset_index(drop=True)
