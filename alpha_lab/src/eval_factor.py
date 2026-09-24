"""因子评估件: 一条命令出全套诊断 (RankIC/ICIR/分层/换手/费后/子期稳定性)。

用法:
    from eval_factor import evaluate_factor
    rep = evaluate_factor(panel, "turnover_ma20", label="fwd5_excess_rank")
    rep.summary()          # 打印关键指标
    rep.to_csv(outputs/)   # 落盘明细 + 曲线图
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FactorReport:
    name: str
    label: str
    ic: pd.Series = None            # 日度 RankIC 序列
    quantile_ret: pd.DataFrame = None   # Q1..Q10 组均超额
    ls_ret: pd.Series = None        # 多空(Q10-Q1)日度序列
    turnover_top: float = np.nan    # top 10% 组合日换手
    fee_ret_ann: float = np.nan     # 费后年化 (top组)
    gross_ret_ann: float = np.nan
    n_days: int = 0

    @property
    def ic_mean(self): return self.ic.mean()

    @property
    def icir(self): return self.ic.mean() / (self.ic.std() + 1e-12)

    @property
    def ic_t(self): return self.ic.mean() / (self.ic.std() / np.sqrt(len(self.ic)) + 1e-12)

    def summary(self) -> str:
        by_year = self.ic.groupby(self.ic.index.year).mean()
        ys = "  ".join(f"{y}:{v:+.3f}" for y, v in by_year.items())
        mono = np.polyfit(range(1, len(self.quantile_ret) + 1),
                          self.quantile_ret.values, 0)[0] if self.quantile_ret is not None else np.nan
        return (f"[{self.name} vs {self.label}] n_days={self.n_days}\n"
                f"  RankIC 均值 {self.ic_mean:+.4f} | ICIR {self.icir:+.2f} | t {self.ic_t:+.1f}\n"
                f"  分年IC: {ys}\n"
                f"  多空年化 {(self.ls_ret.mean()*244 if self.ls_ret is not None else np.nan):+.2%} | "
                f"top组 毛收益 {self.gross_ret_ann:+.2%} / 费后 {self.fee_ret_ann:+.2%} | 日换手 {self.turnover_top:.1%}")


def evaluate_factor(df: pd.DataFrame, factor_col: str, label: str = "fwd5_excess_rank",
                    n_quantile: int = 10, cost_bps: float = 20.0,
                    neutralized: bool = True) -> FactorReport:
    """评估单因子。df 需含 date, ts_code, factor_col, label, 以及 (中性化时) fwd_ret_5。

    cost_bps: 双边成本 (基点)。默认 20bps ≈ 万2买+万2卖+冲击的保守个人口径。
    """
    d = df[df[factor_col].notna() & df[label].notna()].copy()
    rep = FactorReport(name=factor_col, label=label)

    # 1. 日度 RankIC
    ic = d.groupby("date")[[factor_col, label]].apply(
        lambda g: stats.spearmanr(g[factor_col], g[label])[0] if len(g) >= 50 else np.nan)
    ic = ic.dropna()
    rep.ic = ic
    rep.n_days = len(ic)

    # 2. 分位组合 (按因子值分组, Q10=最高值)
    d["q"] = d.groupby("date")[factor_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_quantile, labels=False) + 1)
    qr = d.groupby(["date", "q"])["fwd_ret_5"].mean().unstack()  # 超额近似: fwd_ret 减全市场均值已在label做, 这里用原始ret减截面均值
    mkt = d.groupby("date")["fwd_ret_5"].mean()
    qr = qr.sub(mkt, axis=0)
    rep.quantile_ret = qr.mean() * 244  # 年化

    # 3. 多空 & top 组费后
    rep.ls_ret = (qr[n_quantile] - qr[1])
    top = d[d["q"] == n_quantile].groupby("date")["ts_code"].apply(set)
    tos = []
    for a, b in zip(top.values[:-1], top.values[1:]):
        tos.append(1 - len(a & b) / max(len(b), 1))
    rep.turnover_top = float(np.mean(tos)) if tos else np.nan
    rep.gross_ret_ann = qr[n_quantile].mean() * 244
    rep.fee_ret_ann = rep.gross_ret_ann - rep.turnover_top * 244 * cost_bps / 1e4
    return rep
