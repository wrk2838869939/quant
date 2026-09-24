"""GPU 版挖掘任务: 覆盖 GPMiningTask 的求值/IC 路径 (表达式级 9~53x 实测)。

与 CPU 版共用: 进化循环 / 收割 / 台账 (gp_miner.GPMiningTask 继承)。
差异:
  - 宽表常驻 GPU float32 (prep_wide)
  - eval_gpu 求值; train/valid 日期预转为行号索引
  - 截面 IC 向量化在 GPU (x_rank + 逐行相关)
注意: GPU 单流评估 (线程并行改单进程顺序), 显存峰值 ~5GB/表达式。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import gpu_ops as G
import gpu_eval as GE
from expr_engine import Node
from gp_miner import GPMiningTask, GPConfig
from rank_ic import rank_ic_rows_torch


class GPGPMiningTask(GPMiningTask):
    """GPU 后端版。构造后调用 prep() 一次性搬运。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wide_t: dict = {}
        self.label_t: torch.Tensor | None = None
        self.train_rows: np.ndarray | None = None
        self.valid_rows: np.ndarray | None = None
        self.date_pos: pd.Series | None = None

    def prep(self, log=print):
        assert G.DEV.type == "cuda", "需要 GPU"
        ref = next(iter(self.wide.values()))
        self.date_pos = pd.Series(np.arange(len(ref.index)), index=ref.index)
        self.wide_t = GE.prep_wide(self.wide)
        self.label_t = torch.tensor(self.label.values, dtype=torch.float32, device=G.DEV)
        self.train_rows = self.date_pos.loc[self.train_idx].values
        self.valid_rows = self.date_pos.loc[self.valid_idx].values
        log(f"GPU prep: {len(self.wide_t)} 序列, train {len(self.train_rows)} / valid {len(self.valid_rows)} 截面, "
            f"显存 {torch.cuda.memory_allocated()/2**30:.2f}GB")

    # ---- GPU IC ----
    def _corr_rows(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """对成对有效样本重新排名，兼容原值或预排名输入。"""
        return rank_ic_rows_torch(f, y, self.cfg.min_cross_section)

    def ic_series_gpu(self, factor_t: torch.Tensor, rows: np.ndarray, dates: list) -> pd.Series:
        f = factor_t[torch.tensor(rows, device=factor_t.device)]
        y = self.label_t[torch.tensor(rows, device=factor_t.device)]
        ic = self._corr_rows(f, y)
        ic_np = ic.cpu().numpy()
        return pd.Series(ic_np, index=dates).dropna()

    def _eval_one(self, tree: Node, expr: str) -> dict:
        if tree.size() > self.cfg.max_nodes:
            return dict(ic=np.nan, fit=-9, expr=expr, reason="bloat")
        try:
            with np.errstate(all="ignore"):
                factor_t = GE.eval_gpu(tree, self.wide_t)
            valid = int((~torch.isnan(factor_t[self.train_rows_t])).sum().item())
            if valid < len(self.train_idx) * 50:
                return dict(ic=np.nan, fit=-9, expr=expr, reason="coverage")
            f = factor_t[self.train_rows_t]
            ic_t = self._corr_rows(f, self.label_t[self.train_rows_t])
            ic = pd.Series(ic_t.cpu().numpy(), index=self.train_idx).dropna()
            if len(ic) < 30:
                return dict(ic=np.nan, fit=-9, expr=expr, reason="ic_days")
            icm, icstd = ic.mean(), ic.std()
            corr = self._lib_corr_gpu(f)
            fit = (icm * min(icm / (icstd + 1e-9) / 10.0, 1.0)
                   - self.cfg.corr_lambda * max(0, corr - 0.5)
                   - self.cfg.parsimony_penalty * tree.size())
            del factor_t, f
            out = dict(ic=icm, icir=icm / (icstd + 1e-9), corr=corr, fit=fit,
                       expr=expr, n_ic_days=len(ic), reason="ok")
        except Exception as e:
            out = dict(ic=np.nan, fit=-9, expr=expr, reason=f"err:{type(e).__name__}")
        return out

    @property
    def train_rows_t(self) -> torch.Tensor:
        if not hasattr(self, "_trt"):
            self._trt = torch.tensor(self.train_rows, device=G.DEV, dtype=torch.long)
        return self._trt

    def _lib_corr_gpu(self, factor: torch.Tensor) -> float:
        """缓存库原值；每次与当前因子取共同样本后排名。"""
        if not hasattr(self, "_lib_values_t"):
            self._lib_values_t = {}
            self._sub_idx = torch.arange(0, len(self.train_rows), 4, device=G.DEV, dtype=torch.long)
            dates = self.train_idx[::4]
            for name, lib in self.library.items():
                self._lib_values_t[name] = torch.tensor(lib.loc[dates].values,
                                                       dtype=torch.float32, device=G.DEV)
        f = factor.index_select(0, self._sub_idx)
        worst = 0.0
        for name, lr in self._lib_values_t.items():
            c = self._corr_rows(f, lr).nanmean().item()
            if not np.isnan(c):
                worst = max(worst, abs(c))
        return worst

    def harvest(self, scored, top_n=20, valid_gate=True) -> pd.DataFrame:
        """收割: GPU 重执行 top 表达式 → VALID GPU IC 复核 (含衰减门+形态去族)。"""
        from gp_miner import _family_key
        rows = []
        seen_expr = set()
        seen_family = set()
        for s, t in scored[:top_n]:
            if s["expr"] in seen_expr:
                continue
            seen_expr.add(s["expr"])
            if s["reason"] != "ok" or abs(s["ic"]) < self.cfg.min_ic:
                continue
            try:
                factor_t = GE.eval_gpu(t, self.wide_t)
            except Exception:
                continue
            vic = self.ic_series_gpu(factor_t, self.valid_rows, self.valid_idx)
            # 与 CPU 收割路径使用相同的 VALID[::4] 采样。
            vrt = torch.tensor(self.valid_rows[::4], device=G.DEV, dtype=torch.long)
            # VALID 库相关: 直接用 valid 子集算 (库不大, 不缓存)
            fv = factor_t[vrt]
            worst_v = 0.0
            for name, lib in self.library.items():
                lv = torch.tensor(lib.loc[self.valid_idx[::4]].values,
                                  dtype=torch.float32, device=G.DEV)
                c = self._corr_rows(fv, lv).nanmean().item()
                if not np.isnan(c):
                    worst_v = max(worst_v, abs(c))
                del lv
            vcorr = worst_v
            sign_ok = len(vic) > 10 and np.sign(vic.mean()) == np.sign(s["ic"])
            decay = abs(s["ic"] / float(vic.mean())) if len(vic) and abs(vic.mean()) > 1e-6 else np.inf
            decay_ok = decay <= self.cfg.valid_decay_max
            passed = (sign_ok and decay_ok
                      and abs(float(vic.mean())) >= self.cfg.min_ic * 0.6
                      and vcorr < self.cfg.library_corr_max)
            fam = _family_key(s["expr"])
            if fam in seen_family and not passed:
                continue
            seen_family.add(fam)
            n = self.led.log(f"gp:{s['expr'][:60]}", "fwd5_excess_rank", "train+valid",
                             s["ic"], s.get("icir", np.nan), np.nan, np.nan,
                             params={"backend": "gpu", "valid_ic": round(float(vic.mean()), 4) if len(vic) else None},
                             verdict="VALID_PASS" if passed else "VALID_FAIL")
            rows.append(dict(expr=s["expr"], train_ic=round(float(s["ic"]), 4),
                             train_icir=round(float(s.get("icir", np.nan)), 2),
                             valid_ic=round(float(vic.mean()), 4) if len(vic) else np.nan,
                             decay=round(decay, 2) if np.isfinite(decay) else np.nan,
                             lib_corr=round(float(vcorr), 3), passed=passed, n_trials=n))
            del factor_t
        return pd.DataFrame(rows)

    def run(self, log=print):
        """GPU 版: 顺序评估 (GPU 单流), 复用 CPU 版进化循环参数。"""
        import time as _time
        cfg = self.cfg
        import random as _rd
        rng = _rd.Random(cfg.seed)
        from gp_miner import random_tree as _rt
        pop = [_rt(cfg.max_depth, self.terms, rng) for _ in range(cfg.n_population)]
        history = []
        t0 = _time.time()
        for gen in range(cfg.n_generations):
            todo = [t for t in pop if t.to_expr() not in self.seen]
            for t in todo:
                self.fitness(t)
            scored = [(self.seen[t.to_expr()], t) for t in pop]
            scored.sort(key=lambda x: x[0]["fit"], reverse=True)
            best = scored[0][0]
            n_ok = sum(1 for s, _ in scored if s["reason"] == "ok")
            history.append(dict(gen=gen, best_fit=round(best.get("fit", -9), 4),
                                best_ic=round(best.get("ic", np.nan), 4), n_valid=n_ok))
            log(f"  gen {gen:02d} | best IC {best.get('ic', float('nan')):+.4f} "
                f"fit {best.get('fit', float('nan')):+.4f} | valid {n_ok}/{cfg.n_population} "
                f"| {_time.time()-t0:.0f}s | VRAM {torch.cuda.memory_allocated()/2**30:.1f}GB")
            torch.cuda.empty_cache()
            elites = [t for _, t in scored[:cfg.elitism]]
            children = list(elites)
            from gp_miner import crossover as _cx, mutate as _mu
            while len(children) < cfg.n_population:
                k = min(cfg.tournament_k, len(scored))
                a = max(rng.sample(scored, k), key=lambda x: x[0]["fit"])[1]
                b = max(rng.sample(scored, k), key=lambda x: x[0]["fit"])[1]
                if rng.random() < cfg.crossover_rate:
                    c1, c2 = _cx(a, b, rng)
                    children.append(c1)
                    if len(children) < cfg.n_population:
                        children.append(c2)
                else:
                    children.append(a.clone())
            for i in range(cfg.elitism, len(children)):
                if rng.random() < cfg.mutate_rate:
                    children[i] = _mu(children[i], self.terms, rng)
            pop = children
        return scored, history
