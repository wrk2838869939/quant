"""GPU 表达式求值: eval_node 的 torch 后端 (与 expr_engine CPU 版语义一致)。

用法:
    from gpu_eval import prep_wide, eval_gpu
    wide_t = prep_wide(wide_dict)          # pandas 宽表 → 常驻GPU float32
    factor_t = eval_gpu(tree, wide_t)      # → torch (n,m)
    factor_df = to_df(factor_t, ref_df)    # 需要时转回 pandas
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import gpu_ops as G
from expr_engine import Node, WINDOWS

_t = torch.Tensor


# ---------- 元素级 ----------
def _e_add(a, b): return a + b
def _e_sub(a, b): return a - b
def _e_mul(a, b): return a * b
def _e_div(a, b):
    out = a / b
    return torch.where(torch.isinf(out), float("nan"), out)
def _e_max(a, b): return torch.maximum(a, b)
def _e_min(a, b): return torch.minimum(a, b)
def _e_neg(a): return -a
def _e_abs(a): return torch.abs(a)
def _e_log(a): return torch.log(torch.where(a > 0, a, torch.full_like(a, float("nan"))))
def _e_sign(a): return torch.where(torch.isnan(a), a, torch.sign(a))
def _e_sqrt(a): return torch.sqrt(torch.where(a >= 0, a, torch.full_like(a, float("nan"))))
def _e_inv(a): return 1.0 / torch.where(a == 0, torch.full_like(a, float("nan")), a)


OPS_GPU = {}
for _n, _f in [("add", _e_add), ("sub", _e_sub), ("mul", _e_mul), ("div", _e_div),
               ("max", _e_max), ("min", _e_min)]:
    OPS_GPU[_n] = (_f, 2, False)
for _n, _f in [("neg", _e_neg), ("abs", _e_abs), ("log", _e_log), ("sign", _e_sign),
               ("sqrt", _e_sqrt), ("inv", _e_inv)]:
    OPS_GPU[_n] = (_f, 1, False)
OPS_GPU["rank"] = (G.x_rank, 1, False)
OPS_GPU["zscore"] = (G.x_zscore, 1, False)
for _n, _f in [("ts_mean", G.ts_mean), ("ts_sum", G.ts_sum), ("ts_std", G.ts_std),
               ("ts_max", G.ts_max), ("ts_min", G.ts_min), ("ts_delta", G.ts_delta),
               ("ts_delay", G.ts_delay), ("ts_zscore", G.ts_zscore),
               ("decay", G.decay_linear), ("tsrank", G.tsrank),
               ("argmax", G.ts_argmax), ("argmin", G.ts_argmin)]:
    OPS_GPU[_n] = (_f, 1, True)
OPS_GPU["corr"] = (G.ts_corr, 2, True)
OPS_GPU["cov"] = (G.ts_cov, 2, True)


def prep_wide(wide: dict) -> dict:
    out = {}
    for k, df in wide.items():
        v = torch.tensor(df.values, dtype=torch.float32, device=G.DEV)
        torch.cuda.empty_cache()
        out[k] = v
    return out


def to_df(t: _t, ref: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(t.detach().cpu().numpy(), index=ref.index, columns=ref.columns)


def eval_gpu(node: Node, wide_t: dict, cache: dict = None) -> _t:
    key = node.to_expr()
    if cache is not None and key in cache:
        return cache[key]
    if node.leaf:
        out = wide_t[node.leaf]
    else:
        fn, arity, need_w = OPS_GPU[node.op]
        args = [eval_gpu(c, wide_t, cache) for c in node.children]
        if need_w:
            out = fn(*args, node.window)
        else:
            out = fn(*args)
        del args
    if cache is not None:
        cache[key] = out
    return out
