"""实验台账 + Deflated Sharpe Ratio —— 把 audit 文化制度化 (调研报告 Phase 1)。

- ExperimentLedger: 每次因子/参数尝试必须记录 (Harvey t>3 与 DSR 都需要"尝试次数")
- deflated_sharpe: Bailey & López de Prado 2014; 需要 N 次尝试下的 SR 折减
"""
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

LEDGER = Path("/root/autodl-tmp/alpha_lab/outputs/experiment_ledger.csv")


class ExperimentLedger:
    def __init__(self, path: Path = LEDGER):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=[
                "ts", "factor", "params", "label", "period", "n_trials",
                "rank_ic", "icir", "ic_t", "ls_ann", "verdict", "note"])

    def log(self, factor: str, label: str, period: str,
            rank_ic: float, icir: float, ic_t: float, ls_ann: float,
            params: dict = None, verdict: str = "", note: str = "", *,
            experiment_id: str = None, parent_experiment: str = None,
            source_hashes: dict = None, input_hashes: dict = None,
            selection_note: str = "") -> int:
        """追加事件；行数仅为事件计数，不代表独立试验数。旧行不补写历史。"""
        row = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), factor=factor,
                   params=json.dumps(params or {}, ensure_ascii=False), label=label,
                   period=period, n_trials=len(self.df) + 1,
                   rank_ic=round(float(rank_ic), 5), icir=round(float(icir), 3),
                   ic_t=round(float(ic_t), 2), ls_ann=round(float(ls_ann), 4),
                   verdict=verdict, note=note,
                   event_id=uuid.uuid4().hex,
                   experiment_id=experiment_id or (params or {}).get("output") or uuid.uuid4().hex,
                   parent_experiment=parent_experiment,
                   source_hashes=json.dumps(source_hashes or {}, sort_keys=True),
                   input_hashes=json.dumps(input_hashes or {}, sort_keys=True),
                   selection_note=selection_note, count_semantics="logged_events_not_independent_trials")
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.path, index=False)
        return len(self.df)

    def n_of(self, factor_family: str = "") -> int:
        """某因子族的事件行数；不能直接作为DSR独立试验次数。"""
        if not factor_family:
            return len(self.df)
        return int(self.df["factor"].str.startswith(factor_family).sum())


def deflated_sharpe_report(sr_est: float, n_obs: int, n_trials: float = None,
                          sr_var: float = None, rets: np.ndarray = None, *,
                          assume_iid: bool = False) -> dict:
    """按 Bailey/López de Prado (2014) 式2计算条件DSR及其补概率。

    sr_est必须是rets的未年化均值/样本标准差；rets为同频超额收益。
    sr_var是同频、完整候选集合的Sharpe方差，不能用收益方差代替。
    n_trials由调用方提供独立试验数估计，不能传台账行数冒充它。
    assume_iid显式声明采用独立收益近似；不校正序列相关或遗漏选型。
    条件不足返回not_assessed及原因。单次试验退化为以0为门槛的PSR。
    """
    result = dict(status="not_assessed", reason="", dsr_probability=None,
                  p_value=None, method="DSR_2014_iid", independent_trials=n_trials)
    def unavailable(reason):
        return dict(result, reason=reason)
    if not assume_iid:
        return unavailable("iid_return_assumption_not_accepted")
    if rets is None:
        return unavailable("return_moments_missing")
    values = np.asarray(rets, dtype=float)
    if (values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all()
            or n_obs != len(values)):
        return unavailable("invalid_returns_or_observation_count")
    sd = values.std(ddof=1)
    if not np.isfinite(sr_est) or sd <= 0:
        return unavailable("invalid_sharpe_or_constant_returns")
    measured = values.mean() / sd
    if not np.isclose(sr_est, measured, rtol=1e-6, atol=1e-10):
        return unavailable("sharpe_frequency_or_return_mismatch")
    if (n_trials is None or not np.isfinite(n_trials)
            or n_trials < 1 or 1 < n_trials < 2):
        return unavailable("independent_trial_count_missing_or_invalid")
    if n_trials > 1 and (sr_var is None or not np.isfinite(sr_var) or sr_var < 0):
        return unavailable("cross_trial_sharpe_variance_missing_or_invalid")
    threshold = 0.
    if n_trials > 1:
        gamma = np.euler_gamma
        threshold = np.sqrt(sr_var) * ((1 - gamma) * norm.isf(1 / n_trials)
                    + gamma * norm.isf(1 / (n_trials * np.e)))
    # 普通峰度（正态为3），与论文式2一致；不使用excess kurtosis。
    skewness = float(skew(values, bias=True))
    kurt = float(kurtosis(values, fisher=False, bias=True))
    variance_term = 1 - skewness * sr_est + (kurt - 1) * sr_est ** 2 / 4
    if not np.isfinite(variance_term) or variance_term <= 0:
        return unavailable("invalid_sharpe_sampling_variance")
    se = np.sqrt(variance_term / (n_obs - 1))
    z = (sr_est - threshold) / se
    return dict(result, status="conditional_iid_estimate", reason="",
                dsr_probability=float(norm.cdf(z)), p_value=float(norm.sf(z)),
                benchmark_sharpe=float(threshold), sharpe_standard_error=float(se),
                skewness=skewness, kurtosis=kurt, n_obs=int(n_obs))


def deflated_sharpe(sr_est: float, n_obs: int, n_trials: float = None,
                    sr_var: float = None, rets: np.ndarray = None, *,
                    assume_iid: bool = False) -> float:
    """兼容旧调用的补概率接口：1-DSR，越小越显著；无法评估返回NaN。

    新报告优先调用deflated_sharpe_report，以保留无法评估的具体原因。
    """
    report = deflated_sharpe_report(sr_est, n_obs, n_trials, sr_var, rets,
                                    assume_iid=assume_iid)
    return np.nan if report["p_value"] is None else report["p_value"]
