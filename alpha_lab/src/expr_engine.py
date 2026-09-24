"""表达式引擎: 因子挖掘的算子库 + 向量化求值 (宽表法)。

数据组织: 每个基础序列是 DataFrame(index=date 升序, columns=code)。
  - 时序算子沿 axis=0 滚动 (全表一次 rolling, 比长表 groupby 快 10x+)
  - 截面算子沿 axis=1 (rank/zscore)
  - 元素级算子逐元素

算子库抄 101 Formulaic Alphas (Kakushadze 2016) + tsfresh 常用统计。
所有 ts 算子 right-aligned: 只用 T 及以前信息 (无未来函数)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import bottleneck as bn  # C 加速时序算子 (~20-50x)
    HAS_BN = True
except ImportError:
    HAS_BN = False


def _wrap(arr_fn, **kwargs):
    """bottleneck move_* (沿 axis=0) → DataFrame 包装。"""
    def f(a: pd.DataFrame, d: int):
        v = a.values.astype(np.float64)
        # bottleneck 不接受 window > 行数；补齐后截回，保留短历史的部分窗口。
        padded = (np.pad(v, ((0, d - len(v)), (0, 0)), constant_values=np.nan)
                  if len(v) < d else v)
        out = arr_fn(padded, window=d, min_count=max(2, d // 2), axis=0, **kwargs)[:len(v)]
        return pd.DataFrame(out, index=a.index, columns=a.columns)
    return f

# ---------- 元素级算子 (2元) ----------
def _add(a, b): return a + b
def _sub(a, b): return a - b
def _mul(a, b): return a * b
def _div(a, b): return a / b.replace(0, np.nan) if isinstance(b, pd.DataFrame) else a / (b + 0.0)


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan) if isinstance(out, pd.DataFrame) else out


def _max2(a, b): return np.maximum(a, b)
def _min2(a, b): return np.minimum(a, b)

# ---------- 元素级算子 (1元) ----------
def _neg(a): return -a
def _abs(a): return a.abs() if isinstance(a, pd.DataFrame) else np.abs(a)
def _log(a): return np.log(a.where(a > 0) if isinstance(a, pd.DataFrame) else np.where(a > 0, a, np.nan))
def _sign(a): return np.sign(a)
def _sqrt(a): return np.sqrt(a.where(a >= 0) if isinstance(a, pd.DataFrame) else np.where(a >= 0, a, np.nan))
def _inv(a): return 1.0 / a.replace(0, np.nan) if isinstance(a, pd.DataFrame) else 1.0 / a

# ---------- 截面算子 ----------
def x_rank(a: pd.DataFrame): return a.rank(axis=1, pct=True)
def x_zscore(a: pd.DataFrame):
    m, s = a.mean(axis=1), a.std(axis=1)
    return a.sub(m, axis=0).div(s.replace(0, np.nan), axis=0)

# ---------- 时序算子 ----------
WINDOWS = [5, 10, 20, 60]


def ts_mean(a, d): return a.rolling(d, min_periods=max(2, d // 2)).mean()
def ts_sum(a, d): return a.rolling(d, min_periods=max(2, d // 2)).sum()
def ts_std(a, d): return a.rolling(d, min_periods=max(2, d // 2)).std()
def ts_max(a, d): return a.rolling(d, min_periods=max(2, d // 2)).max()
def ts_min(a, d): return a.rolling(d, min_periods=max(2, d // 2)).min()
def ts_delta(a, d): return a - a.shift(d)
def ts_delay(a, d): return a.shift(d)
def ts_rank(a, d): return a.rolling(d, min_periods=max(2, d // 2)).apply(lambda x: (x.values > x.values[-1]).mean(), raw=False) if False else _tsrank_fast(a, d)


def _tsrank_fast(a: pd.DataFrame, d: int) -> pd.DataFrame:
    """当前值在过去 d 日的分位。bottleneck move_rank (axis=0, NaN感知)。"""
    if HAS_BN:
        return _wrap(bn.move_rank)(a, d)
    v = a.values
    out = np.full(v.shape, np.nan)
    lo = max(2, d // 2)
    for j in range(v.shape[1]):
        col = v[:, j]
        n = len(col)
        for i in range(lo - 1, n):
            if np.isnan(col[i]):
                continue
            w = col[max(0, i - d + 1):i + 1]
            w = w[~np.isnan(w)]
            if len(w) < lo:
                continue
            rank0 = (w < col[i]).sum() + ((w == col[i]).sum() - 1) / 2
            out[i, j] = 2 * rank0 / (len(w) - 1) - 1
    return pd.DataFrame(out, index=a.index, columns=a.columns)


def ts_zscore(a, d):
    m = a.rolling(d, min_periods=max(2, d // 2)).mean()
    s = a.rolling(d, min_periods=max(2, d // 2)).std()
    return (a - m) / s.replace(0, np.nan)


def ts_corr(a, b, d):
    return a.rolling(d, min_periods=max(3, int(d * 0.7))).corr(b)


def ts_cov(a, b, d):
    return a.rolling(d, min_periods=max(3, int(d * 0.7))).cov(b)


def decay_linear(a, d):
    """线性衰减加权均值 (滑窗矩阵乘, 向量化)。"""
    v = a.values.astype(np.float64)
    n = v.shape[0]
    lo = max(2, d // 2)
    if n < d:
        return a * np.nan
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(v, d, axis=0)          # (n-d+1, m, d)
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()
    out = np.full(v.shape, np.nan)
    ww = sw @ w                                     # (n-d+1, m)
    cnt = (~np.isnan(sw)).sum(axis=2)
    ww[cnt < lo] = np.nan
    out[d - 1:] = ww
    return pd.DataFrame(out, index=a.index, columns=a.columns)


def ts_argmax(a, d):
    if HAS_BN:
        return _wrap(bn.move_argmax)(a, d)
    return a.rolling(d, min_periods=max(2, d // 2)).apply(
        lambda w: np.nanargmax(w[::-1]), raw=True)


def ts_argmin(a, d):
    if HAS_BN:
        return _wrap(bn.move_argmin)(a, d)
    return a.rolling(d, min_periods=max(2, d // 2)).apply(
        lambda w: np.nanargmin(w[::-1]), raw=True)


# ---------- 算子注册表 ----------
# name -> (fn, arity, kind, need_window)  kind: e=元素级 x=截面 t=时序
OPS = {}
for _n, _f in [("add", _add), ("sub", _sub), ("mul", _mul), ("div", _safe_div),
               ("max", _max2), ("min", _min2)]:
    OPS[_n] = (_f, 2, "e", False)
for _n, _f in [("neg", _neg), ("abs", _abs), ("log", _log), ("sign", _sign),
               ("sqrt", _sqrt), ("inv", _inv)]:
    OPS[_n] = (_f, 1, "e", False)
OPS["rank"] = (x_rank, 1, "x", False)
OPS["zscore"] = (x_zscore, 1, "x", False)
if HAS_BN:
    OPS["ts_mean"] = (_wrap(bn.move_mean), 1, "t", True)
    OPS["ts_sum"] = (_wrap(bn.move_sum), 1, "t", True)
    OPS["ts_std"] = (_wrap(bn.move_std, ddof=1), 1, "t", True)
    OPS["ts_max"] = (_wrap(bn.move_max), 1, "t", True)
    OPS["ts_min"] = (_wrap(bn.move_min), 1, "t", True)
else:
    for _n, _f in [("ts_mean", ts_mean), ("ts_sum", ts_sum), ("ts_std", ts_std),
                   ("ts_max", ts_max), ("ts_min", ts_min)]:
        OPS[_n] = (_f, 1, "t", True)
for _n, _f in [("ts_delta", ts_delta), ("ts_delay", ts_delay), ("ts_zscore", ts_zscore)]:
    OPS[_n] = (_f, 1, "t", True)
OPS["decay"] = (decay_linear, 1, "t", True)
OPS["argmax"] = (ts_argmax, 1, "t", True)
OPS["argmin"] = (ts_argmin, 1, "t", True)
OPS["tsrank"] = (lambda a, d: _tsrank_fast(a, d), 1, "t", True)
OPS["corr"] = (ts_corr, 2, "t", True)
OPS["cov"] = (ts_cov, 2, "t", True)


# ---------- 表达式树 ----------
class Node:
    __slots__ = ("op", "children", "window", "leaf")

    def __init__(self, op: str, children=None, window: int = None, leaf: str = None):
        self.op = op
        self.children = children or []
        self.window = window
        self.leaf = leaf

    def to_expr(self) -> str:
        if self.leaf:
            return self.leaf
        args = [c.to_expr() for c in self.children]
        if self.window:
            args.append(str(self.window))
        return f"{self.op}({','.join(args)})"

    def depth(self):
        return 1 + max((c.depth() for c in self.children), default=0)

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def clone(self):
        return Node(self.op, [c.clone() for c in self.children], self.window, self.leaf)


def eval_node(node: Node, wide: dict, cache: dict = None) -> pd.DataFrame:
    """递归求值 (宽表)。cache: expr_str -> DataFrame 共享子树。"""
    key = node.to_expr()
    if cache is not None and key in cache:
        return cache[key]
    if node.leaf:
        out = wide[node.leaf]
    else:
        fn, arity, kind, need_w = OPS[node.op]
        args = [eval_node(c, wide, cache) for c in node.children]
        if need_w:
            out = fn(*args, node.window)
        else:
            out = fn(*args)
    if cache is not None:
        cache[key] = out
    return out
