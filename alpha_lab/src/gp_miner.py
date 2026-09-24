"""GP 因子挖掘器 (Phase 3a)。

设计 (依据调研报告 Phase 3):
  - 表达式树种群, 锦标赛选择 + 子树交叉/变异, elitism
  - 适应度 = 训练窗周采样截面 RankIC 均值 × ICIR稳定性 − λ·max|corr(因子库)|
    (AlphaGen 的"组合增量 IC"思想: 与已有库高相关直接惩罚)
  - 三段式防泄露: 挖掘只用 TRAIN 窗; VALID 窗只做入库筛选; TEST 窗最终一次性评估
  - 每个个体/每代都进实验台账 (尝试次数 → DSR)

输入: 宽表 dict {seq_name: DataFrame(date×code)} 由 build_wide() 预构建。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from expr_engine import OPS, WINDOWS, Node, eval_node
from ledger import ExperimentLedger, deflated_sharpe
from rank_ic import rank_ic_rows

TERMINALS = None  # 由 gp.run() 注入: 基础序列名列表


@dataclass
class GPConfig:
    n_population: int = 300
    n_generations: int = 40
    max_depth: int = 4
    max_nodes: int = 13            # 硬上限: 防表达式膨胀 (bloat)
    parsimony_penalty: float = 0.002  # 每节点惩罚 (Parsimony pressure)
    crossover_rate: float = 0.7
    mutate_rate: float = 0.15
    tournament_k: int = 5
    elitism: int = 8
    corr_lambda: float = 0.5        # 与因子库 max|corr| 的惩罚系数
    min_ic: float = 0.015           # 训练窗 IC 门槛
    valid_decay_max: float = 3.0    # |train|/|valid| 衰减率硬门 (防拟合噪声)
    library_corr_max: float = 0.7   # 入库相关性上限 (硬门)
    seed: int = 42
    min_cross_section: int = 100


def random_tree(depth: int, terms: list, rng: random.Random) -> Node:
    """随机生成表达式树 (ramped half-and-half, 尊重 max_nodes)。"""
    if depth <= 0 or rng.random() < 0.25:
        return Node(op=None, leaf=rng.choice(terms))
    # 深度小优先时序/截面算子构造形状; 排除易膨胀的复合型 (corr/cov 嵌套)
    if depth >= 3 and rng.random() < 0.35:
        op_names = [k for k in OPS if k not in ("corr", "cov")]
    else:
        op_names = list(OPS.keys())
    op = rng.choice(op_names)
    fn, arity, kind, need_w = OPS[op]
    children = [random_tree(depth - 1, terms, rng) for _ in range(arity)]
    window = rng.choice(WINDOWS) if need_w else None
    return Node(op, children, window, None)


def mutate(node: Node, terms: list, rng: random.Random) -> Node:
    """变异: 随机替换子树 / 换窗口 / 换算子。"""
    r = rng.random()
    if r < 0.5:
        return random_tree(rng.randint(1, 2), terms, rng)
    if node.op is not None and node.window and r < 0.75:
        node = node.clone()
        node.window = rng.choice(WINDOWS)
        return node
    node = node.clone()
    if node.op in OPS:
        fn, arity, kind, need_w = OPS[node.op]
        cand = [k for k, v in OPS.items() if v[1] == arity]
        node.op = rng.choice(cand)
    return node


def crossover(a: Node, b: Node, rng: random.Random):
    """子树交叉。"""
    ca, cb = a.clone(), b.clone()
    da = random_node(ca, rng)
    db = random_node(cb, rng)
    da.op, db.op = db.op, da.op
    da.children, db.children = db.children, da.children
    da.window, db.window = db.window, da.window
    da.leaf, db.leaf = db.leaf, da.leaf
    return ca, cb


def random_node(node: Node, rng: random.Random) -> Node:
    nodes = all_nodes(node)
    return rng.choice(nodes)


def all_nodes(node: Node) -> list:
    out = [node]
    for c in node.children:
        out += all_nodes(c)
    return out


class GPMiningTask:
    """一次挖掘任务: 给定宽表 + 标签 + 现有因子库。"""

    def __init__(self, wide: dict, label_wide: pd.DataFrame, library: dict,
                 train_dates: pd.DatetimeIndex, valid_dates: pd.DatetimeIndex,
                 sample_weeks: int = 1, cfg: GPConfig = None):
        self.wide = wide
        self.cfg = cfg or GPConfig()
        self.rng = random.Random(self.cfg.seed)
        self.terms = list(wide.keys())
        self.library = library  # {name: wide DataFrame}
        # 周采样截面 (无重叠标签窗)
        step = 5 * sample_weeks
        self.train_idx = [d for i, d in enumerate(train_dates) if i % step == 0]
        self.valid_idx = [d for i, d in enumerate(valid_dates) if i % step == 0]
        self.label = label_wide
        for name, frame in {**wide, **{f"lib:{k}": v for k, v in library.items()}}.items():
            if not frame.index.equals(self.label.index) or not frame.columns.equals(self.label.columns):
                raise ValueError(f"{name} 与标签的日期/股票轴不一致")
        self.led = ExperimentLedger()
        self.seen: dict[str, dict] = {}  # expr -> fitness (代内去重, 线程安全由 _lock 保证)
        self._lock = None

    def ic_series(self, factor: pd.DataFrame, dates: list) -> pd.Series:
        """先取因子与标签共同有效样本，再计算 Spearman。"""
        f, y = factor.loc[dates], self.label.loc[dates]
        return rank_ic_rows(f, y, self.cfg.min_cross_section).dropna()

    def max_lib_corr(self, factor: pd.DataFrame, dates: list) -> float:
        """与因子库在采样截面上的最大 |平均截面相关| (向量化)。"""
        f = factor.loc[dates]
        worst = 0.0
        for name, lib in self.library.items():
            l = lib.loc[dates]
            c = rank_ic_rows(f, l, self.cfg.min_cross_section).mean()
            if not np.isnan(c):
                worst = max(worst, abs(c))
        return worst

    def fitness(self, tree: Node) -> dict:
        expr = tree.to_expr()
        if expr in self.seen:
            return self.seen[expr]
        if tree.size() > self.cfg.max_nodes:  # bloat 硬拒
            out = dict(ic=np.nan, fit=-9, expr=expr, reason="bloat")
            if self._lock is not None:
                with self._lock:
                    self.seen[expr] = out
            else:
                self.seen[expr] = out
            return out
        out = self._eval_one(tree, expr)
        if self._lock is not None:
            with self._lock:
                self.seen[expr] = out
        else:
            self.seen[expr] = out
        return out

    def _eval_one(self, tree: Node, expr: str) -> dict:
        if tree.size() > self.cfg.max_nodes:
            return dict(ic=np.nan, fit=-9, expr=expr, reason="bloat")
        try:
            with np.errstate(all="ignore"):
                factor = eval_node(tree, self.wide, cache=None)
            if not isinstance(factor, pd.DataFrame):
                raise TypeError
            valid = factor.loc[self.train_idx].notna().sum().sum()
            if valid < len(self.train_idx) * 50:  # 覆盖太少
                return dict(ic=np.nan, fit=-9, expr=expr, reason="coverage")
            ic = self.ic_series(factor, self.train_idx)
            if len(ic) < 30:
                return dict(ic=np.nan, fit=-9, expr=expr, reason="ic_days")
            icm, icstd = ic.mean(), ic.std()
            corr = self.max_lib_corr(factor, self.train_idx[::4])
            fit = icm * min(icm / (icstd + 1e-9) / 10.0, 1.0) \
                - self.cfg.corr_lambda * max(0, corr - 0.5) \
                - self.cfg.parsimony_penalty * tree.size()   # 复杂度惩罚 (防 bloat)
            # ⚠️ 不缓存 factor/序列 —— 只存标量 (cgroup 48GB, 存 DataFrame 会 OOM)
            out = dict(ic=icm, icir=icm / (icstd + 1e-9), corr=corr, fit=fit,
                       expr=expr, n_ic_days=len(ic), reason="ok")
        except Exception as e:
            out = dict(ic=np.nan, fit=-9, expr=expr, reason=f"err:{type(e).__name__}")
        return out

    def run(self, log=print):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        cfg = self.cfg
        self._lock = threading.Lock()
        pop = [random_tree(cfg.max_depth, self.terms, self.rng) for _ in range(cfg.n_population)]
        history = []
        t0 = time.time()
        for gen in range(cfg.n_generations):
            todo = [t for t in pop if t.to_expr() not in self.seen]
            with ThreadPoolExecutor(max_workers=6) as ex:
                list(ex.map(self.fitness, todo))   # 并行评估 (rolling 为 C 路径, 释放 GIL)
            scored = [(self.seen[t.to_expr()], t) for t in pop]
            scored.sort(key=lambda x: x[0]["fit"], reverse=True)
            best = scored[0][0]
            n_ok = sum(1 for s, _ in scored if s["reason"] == "ok")
            history.append(dict(gen=gen, best_fit=round(best.get("fit", -9), 4),
                                best_ic=round(best.get("ic", np.nan), 4),
                                n_valid=n_ok))
            log(f"  gen {gen:02d} | best IC {best.get('ic', float('nan')):+.4f} "
                f"fit {best.get('fit', float('nan')):+.4f} | valid {n_ok}/{cfg.n_population} "
                f"| {time.time()-t0:.0f}s")

            elites = [t for _, t in scored[:cfg.elitism]]
            children = list(elites)
            while len(children) < cfg.n_population:
                a = self._select(scored)
                b = self._select(scored)
                if self.rng.random() < cfg.crossover_rate:
                    c1, c2 = crossover(a, b, self.rng)
                    children.append(c1)
                    if len(children) < cfg.n_population:
                        children.append(c2)
                else:
                    children.append(a.clone())
            # 变异
            for i in range(cfg.elitism, len(children)):
                if self.rng.random() < cfg.mutate_rate:
                    children[i] = mutate(children[i], self.terms, self.rng)
            pop = children
        return scored, history

    def _select(self, scored):
        k = min(self.cfg.tournament_k, len(scored))
        pick = self.rng.sample(scored, k)
        pick.sort(key=lambda x: x[0]["fit"], reverse=True)
        return pick[0][1]

    def harvest(self, scored, top_n=20, valid_gate=True) -> pd.DataFrame:
        """收割: 训练窗达标 → VALID 复核 (含衰减门) → 形态去族 → 台账登记。"""
        rows = []
        seen_expr = set()
        seen_family = set()   # 结构去族: 去掉窗口数字后的骨架
        for s, t in scored[:top_n]:
            if s["expr"] in seen_expr:
                continue
            seen_expr.add(s["expr"])
            if s["reason"] != "ok" or abs(s["ic"]) < self.cfg.min_ic:
                continue
            # 重新执行 (确定性) 拿 factor
            try:
                with np.errstate(all="ignore"):
                    factor = eval_node(t, self.wide, cache=None)
            except Exception:
                continue
            vic = self.ic_series(factor, self.valid_idx)
            vcorr = self.max_lib_corr(factor, self.valid_idx[::4])
            sign_ok = len(vic) > 10 and np.sign(vic.mean()) == np.sign(s["ic"])
            decay = abs(s["ic"] / vic.mean()) if len(vic) and abs(vic.mean()) > 1e-6 else np.inf
            decay_ok = decay <= self.cfg.valid_decay_max
            passed = (sign_ok and decay_ok
                      and abs(vic.mean()) >= self.cfg.min_ic * 0.6
                      and vcorr < self.cfg.library_corr_max)
            fam = _family_key(s["expr"])
            if fam in seen_family and not passed:
                continue  # 同族未过者不重复登记
            seen_family.add(fam)
            n = self.led.log(f"gp:{s['expr'][:60]}", "fwd5_excess_rank",
                             f"train+valid",
                             s["ic"], s.get("icir", np.nan), np.nan, np.nan,
                             params={"gen_harvest": True, "valid_ic": round(vic.mean(), 4) if len(vic) else None,
                                     "decay": round(decay, 2) if np.isfinite(decay) else None},
                             verdict="VALID_PASS" if passed else "VALID_FAIL")
            rows.append(dict(expr=s["expr"], train_ic=round(s["ic"], 4),
                             train_icir=round(s.get("icir", np.nan), 2),
                             valid_ic=round(vic.mean(), 4) if len(vic) else np.nan,
                             decay=round(decay, 2) if np.isfinite(decay) else np.nan,
                             lib_corr=round(vcorr, 3), passed=passed, n_trials=n))
        return pd.DataFrame(rows)


def _family_key(expr: str) -> str:
    """表达式形态骨架: 去窗口数字 + 截断。同族变体共享 key。"""
    import re
    sk = re.sub(r",\d+\)", ")", expr)
    return sk[:48]
