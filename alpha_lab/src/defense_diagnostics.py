"""仅作机制实验的收盘防守状态机，不修改月度选股和既有执行引擎。"""
import numpy as np
import pandas as pd

from patient_trend import review_dates


def risk_evidence(prices, spec):
    """从历史前缀计算趋势及持续转弱；缺失不作为安全或危险的证据。"""
    close = prices.where(prices > 0)
    medium, long = spec['medium_window'], spec['long_window']
    ma = close.rolling(long, min_periods=long).mean()
    r60, r120 = close / close.shift(medium) - 1, close / close.shift(long) - 1
    known = close.notna() & r60.notna() & r120.notna() & ma.notna()
    trend = r60.gt(0) & r120.gt(0) & close.gt(ma)
    broken = known & r120.le(0) & close.le(ma)
    weak = known & ~trend
    window = spec['persistence_window']
    count = weak.astype(float).rolling(window, min_periods=window).sum()
    n_known = known.astype(float).rolling(window, min_periods=window).sum()
    persistent = weak & n_known.eq(window) & count.ge(spec['persistent_weak_days'])
    return dict(known=known, trend=trend, broken=broken, weak=weak, persistent=persistent,
                alarm=broken & persistent, weak_count=count, ret60=r60, ret120=r120, ma_gap=close/ma-1)


class DefenseOverlay:
    """周初收盘检查；恢复仅在月度复核，冻结原风险篮子防止卖空后误判恢复。

    weights保存触发时的股票市值比例，不因卖出或随后涨跌重算。
    返回的退出请求由因果引擎在下一交易日按原限价规则处理。
    """
    def __init__(self, evidence, spec, variant):
        if variant not in ['weekly_exit', 'pause_only', 'weekly_pause', 'weekly_liquidate']:
            raise ValueError(variant)
        self.evidence, self.spec, self.variant = evidence, spec, variant
        self.weekly = set(review_dates(evidence['alarm'].index, spec['review_frequency']))
        self.weights = None
        self.records = []

    def on_close(self, date, monthly_review, values):
        if date not in self.evidence['alarm'].index:
            raise ValueError('风险证据缺当日')
        recovery = None
        if self.weights is not None and monthly_review:
            good = self.evidence['trend'].loc[date].reindex(self.weights.index).fillna(False)
            known = self.evidence['known'].loc[date].reindex(self.weights.index).fillna(False)
            recovery = float(self.weights.where(good & known, 0.).sum())
            if recovery > self.spec['recovery_fraction']:
                self.records.append(dict(date=date, kind='recovery', fraction=recovery,
                                         cohort='|'.join(self.weights.index)))
                self.weights = None
        exits = []
        if date in self.weekly and values:
            value = pd.Series(values, dtype=float)
            weights = value / value.sum()
            alarm = self.evidence['alarm'].loc[date].reindex(value.index).fillna(False)
            fraction = float(weights.where(alarm, 0.).sum())
            if self.variant != 'weekly_exit' and self.weights is None and fraction >= self.spec['affected_fraction']:
                self.weights = weights.copy()
                self.records.append(dict(date=date, kind='pause_trigger', fraction=fraction,
                                         cohort='|'.join(self.weights.index)))
            if self.variant in ['weekly_exit', 'weekly_pause', 'weekly_liquidate']:
                exits = value.index[alarm].tolist()
            if self.variant == 'weekly_liquidate' and self.weights is not None:
                exits = list(values)
            self.records.append(dict(date=date, kind='weekly_check', fraction=fraction,
                                     paused=self.weights is not None, n_exits=len(exits)))
        # 确认的全退意图每天保持，既有引擎负责未成交重试。
        if self.variant == 'weekly_liquidate' and self.weights is not None:
            exits = list(values)
        paused = self.weights is not None
        if monthly_review:
            self.records.append(dict(date=date, kind='monthly_entry_gate', paused=paused,
                                     recovery_fraction=recovery))
        return exits, paused
