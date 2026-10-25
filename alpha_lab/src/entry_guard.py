"""只约束月度新增仓位的相关门与集体退出暂停；不修改收益预测。"""
import numpy as np
import pandas as pd


class EntryGuard:
    def __init__(self, returns, variant, parameters):
        if variant not in ['baseline', 'correlation', 'pause', 'combined']:
            raise ValueError('未知已登记政策')
        self.returns = returns
        self.variant = variant
        self.parameters = parameters
        self.pair_cache = {}
        self.pair_audit = []
        self.review_audit = []
        self.candidate_audit = []

    def pause(self, date, candidates, trend_exit_codes):
        held = set(candidates.loc[candidates.shares.gt(0), 'ts_code'])
        failed = set(trend_exit_codes) & held
        trigger = (len(failed) >= self.parameters['minimum_collective_exits']
                   and len(failed) >= len(held) * self.parameters['collective_exit_fraction'])
        paused = trigger and self.variant in ['pause', 'combined']
        self.review_audit.append(dict(date=date, held_before=len(held), newly_confirmed_exits=len(failed),
            exit_codes='|'.join(sorted(failed)), collective_trigger=trigger, paused=paused))
        return paused

    def pair(self, date, left, right):
        key = (pd.Timestamp(date), *sorted([left, right]))
        if key not in self.pair_cache:
            sample = self.returns.loc[:date, [left, right]].tail(self.parameters['correlation_window']).dropna()
            valid = len(sample) >= self.parameters['minimum_observations']
            rho = float(sample.corr().iloc[0, 1]) if valid else np.nan
            if not np.isfinite(rho):
                rho = np.nan
            self.pair_cache[key] = rho
            self.pair_audit.append(dict(date=date, left=key[1], right=key[2], observations=len(sample),
                last_observation=sample.index.max() if len(sample) else pd.NaT, correlation=rho))
        return self.pair_cache[key]

    def reason(self, date, code, references, paused):
        reason, max_corr = None, np.nan
        if paused:
            reason = 'collective_exit_pause'
        elif self.variant in ['correlation', 'combined'] and references:
            correlations = [self.pair(date, code, other) for other in references]
            max_corr = float(np.nanmax(correlations)) if np.isfinite(correlations).any() else np.nan
            if not np.isfinite(correlations).all():
                reason = 'unknown_correlation'
            elif max_corr > self.parameters['maximum_correlation']:
                reason = 'correlation_limit'
        self.candidate_audit.append(dict(date=date, ts_code=code, references='|'.join(references),
                                         maximum_correlation=max_corr, guard_reason=reason or 'allowed'))
        return reason
