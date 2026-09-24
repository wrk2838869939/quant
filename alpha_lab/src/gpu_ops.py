"""GPU (torch) 版时序/截面算子 —— ExprEngine 的加速后端。

数据常驻 GPU (float32): 9 序列 × 3084×5819×4B ≈ 0.65GB, 22GB 显存充裕。
算子与 expr_engine 对齐：右对齐、NaN 感知；普通滚动 min_periods=max(2,d//2)。
corr/cov 使用成对样本门槛；decay 使用完整无缺失窗口。

实现要点:
  ts_mean/sum/max/min/std : unfold 滑窗 (n,d,m) → 归约 (GPU 并行列×行)
  tsrank                  : unfold + (w < x).count (避免排序)
  截面 rank               : torch.sort(dim=1) 二分平均秩 (并列值同秩, 与 pandas pct 一致)
  corr/cov                : unfold 双序列 → E[xy]-E[x]E[y] 数值稳定式
  decay                   : unfold @ weights (batched matmul)
"""
from __future__ import annotations

import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_t = torch.Tensor


def _unfold(x: _t, d: int) -> _t:
    """左侧补 NaN，保留满窗之前满足 min_periods 的结果。"""
    pad = x.new_full((d - 1, x.shape[1]), float("nan"))
    return torch.cat((pad, x), dim=0).unfold(0, d, 1)


def _valid_cnt(w: _t) -> _t:
    """(k,m,d) 每窗非NaN数。"""
    return (~torch.isnan(w)).sum(-1)


def _apply_min_periods(out_tail: _t, w: _t, min_periods: int) -> _t:
    cnt = _valid_cnt(w)
    out_tail = torch.where(cnt >= min_periods, out_tail, torch.full_like(out_tail, float("nan")))
    return out_tail


def ts_mean(x: _t, d: int, mp=None) -> _t:
    mp = mp or max(2, d // 2)
    w = _unfold(x, d)
    return _apply_min_periods(torch.nanmean(w, dim=-1), w, mp)


def ts_sum(x: _t, d: int) -> _t:
    mp = max(2, d // 2)
    w = _unfold(x, d)
    r = torch.where(torch.isnan(w), 0.0, w).sum(-1)
    r = _apply_min_periods(r, w, mp)
    return r


def ts_max(x: _t, d: int) -> _t:
    mp = max(2, d // 2)
    w = _unfold(x, d)
    r = torch.where(torch.isnan(w), float("-inf"), w).max(-1).values
    r = _apply_min_periods(r, w, mp)
    return r


def ts_min(x: _t, d: int) -> _t:
    mp = max(2, d // 2)
    w = _unfold(x, d)
    nw = torch.where(torch.isnan(w), float("inf"), w)
    r = nw.min(-1).values
    r = torch.where(torch.isinf(r), float("nan"), r)
    r = _apply_min_periods(r, w, mp)
    return r


def ts_std(x: _t, d: int) -> _t:
    mp = max(2, d // 2)
    w = _unfold(x, d)
    m = torch.nanmean(w, -1, keepdim=True)
    diff2 = torch.where(torch.isnan(w), float("nan"), (w - m) ** 2)
    cnt = _valid_cnt(w).clamp(min=1)
    var = torch.where(cnt > 1, torch.nansum(diff2, -1) / (cnt - 1), float("nan"))
    r = torch.sqrt(torch.clamp(var, min=0))
    r = _apply_min_periods(r, w, mp)
    return r


def tsrank(x: _t, d: int) -> _t:
    """平均秩归一到 [-1, 1]，与 bottleneck move_rank 一致。"""
    mp = max(2, d // 2)
    w = _unfold(x, d)                       # (k,m,d) 最后一列是当前值
    cur = w[..., -1:]                        # (k,m,1)
    cnt = _valid_cnt(w)
    lt = (w < cur).sum(-1).to(x.dtype)       # 严格小于
    eq = (w == cur).sum(-1).to(x.dtype) - 1  # 并列(不含自身) — 平均秩: lt + eq/2
    r = 2.0 * (lt + eq / 2.0) / (cnt - 1).clamp(min=1) - 1.0
    r = torch.where(cnt >= mp, r, float("nan"))
    r = torch.where(torch.isnan(cur.squeeze(-1)), float("nan"), r)
    return r


def ts_delta(x: _t, d: int) -> _t:
    return x - ts_delay(x, d)


def ts_delay(x: _t, d: int) -> _t:
    out = torch.full_like(x, float("nan"))
    out[d:] = x[:-d]
    return out


def ts_zscore(x: _t, d: int) -> _t:
    m = ts_mean(x, d)
    s = ts_std(x, d)
    return (x - m) / torch.where(s == 0, float("nan"), s)


def _pair_moments(a: _t, b: _t, d: int):
    """用成对非缺失样本计算样本协方差及方差。"""
    mp = max(3, int(d * 0.7))
    wa, wb = _unfold(a, d), _unfold(b, d)
    ok = ~torch.isnan(wa) & ~torch.isnan(wb)
    wa2 = torch.where(ok, wa, float("nan"))
    wb2 = torch.where(ok, wb, float("nan"))
    n = ok.sum(-1).clamp(min=1).to(a.dtype)
    ma, mb = torch.nanmean(wa2, -1, keepdim=True), torch.nanmean(wb2, -1, keepdim=True)
    da, db = torch.where(ok, wa2 - ma, 0.0), torch.where(ok, wb2 - mb, 0.0)
    cov = (da * db).sum(-1) / (n - 1)
    va = (da ** 2).sum(-1) / (n - 1)
    vb = (db ** 2).sum(-1) / (n - 1)
    return cov, va, vb, n >= mp


def ts_corr(a: _t, b: _t, d: int) -> _t:
    cov, va, vb, valid = _pair_moments(a, b, d)
    denom = torch.sqrt(va * vb)
    return torch.where(valid & (denom > 0), cov / denom, float("nan"))


def ts_cov(a: _t, b: _t, d: int) -> _t:
    cov, _, _, valid = _pair_moments(a, b, d)
    return torch.where(valid, cov, float("nan"))


def decay_linear(x: _t, d: int) -> _t:
    w = _unfold(x, d)                        # (n,m,d)
    wt = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    wt = wt / wt.sum()
    # CPU 既有契约：完整窗口中任何 NaN 都传播，禁止隐式填零。
    r = (w * wt).sum(-1)
    return r


def x_rank(x: _t) -> _t:
    """截面 pct rank (并列平均秩, pandas rank(pct=True) 精确语义, NaN→NaN)。"""
    nan_mask = torch.isnan(x)
    xf = torch.where(nan_mask, torch.full_like(x, float("inf")), x)
    n, m = x.shape
    idx = torch.argsort(xf, dim=1)
    sx = torch.gather(xf, 1, idx)
    ranks = torch.arange(1, m + 1, device=x.device, dtype=x.dtype).unsqueeze(0).expand(n, -1)
    is_new = torch.ones_like(sx, dtype=torch.bool)
    is_new[:, 1:] = sx[:, 1:] != sx[:, :-1]
    first = torch.where(is_new, ranks, torch.zeros_like(ranks))
    is_end = torch.zeros_like(is_new)
    is_end[:, :-1] = is_new[:, 1:]
    is_end[:, -1] = True
    last = torch.where(is_end, ranks, torch.zeros_like(ranks))
    ff = torch.cummax(first, dim=1).values                                  # 段首秩前向填充 (first不减, cummax即最近非零)
    # 段尾秩后向填充: flip后取"左侧最近非零"用 iota-cummax-gather 技巧
    iota = torch.arange(m, device=x.device)
    lf = torch.flip(last, [1])
    posidx = torch.cummax(torch.where(lf > 0, iota, torch.full_like(iota, -1)), dim=1).values
    lb = torch.flip(torch.gather(lf, 1, posidx.clamp(min=0)), [1])
    avg = (ff + lb) / 2.0
    out = torch.empty_like(x)
    out.scatter_(1, idx, avg)
    denom = (~nan_mask).sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
    out = torch.where(nan_mask, torch.full_like(out, float("nan")), out / denom)
    return out


def x_zscore(x: _t) -> _t:
    m = torch.nanmean(x, dim=1, keepdim=True)
    cnt = (~torch.isnan(x)).sum(dim=1, keepdim=True)
    var = torch.nansum((x - m) ** 2, dim=1, keepdim=True) / (cnt - 1)
    s = torch.sqrt(torch.where(cnt > 1, var, float("nan")))
    return (x - m) / torch.where(s == 0, float("nan"), s)


def _ts_argextreme(x: _t, d: int, largest: bool) -> _t:
    """距当前日的位置；并列极值取最近者，与 bottleneck 一致。"""
    w = _unfold(x, d).flip(-1)
    fill = float("-inf") if largest else float("inf")
    values = torch.where(torch.isnan(w), fill, w)
    pos = values.argmax(-1) if largest else values.argmin(-1)
    return _apply_min_periods(pos.to(x.dtype), w, max(2, d // 2))


def ts_argmax(x: _t, d: int) -> _t:
    return _ts_argextreme(x, d, True)


def ts_argmin(x: _t, d: int) -> _t:
    return _ts_argextreme(x, d, False)
