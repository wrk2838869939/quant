"""原持仓篮子共同转弱的周频分级状态；卖出不会自动产生恢复信号。"""
import numpy as np
import pandas as pd


def basket_features(close):
    known = close.notna() & close.shift(20).notna() & close.rolling(60,min_periods=60).count().eq(60)
    return dict(known=known, weak=close.lt(close.rolling(20,min_periods=20).mean()),
        drawdown=1-close/close.rolling(60,min_periods=60).max(), ret20=close/close.shift(20)-1)


class CashState:
    def __init__(self, features, variant, lower=.05, upper=.10):
        if variant not in ['pause','half','full','staged','fixed_half']:
            raise ValueError('未知组合现金规则')
        if not 0 < lower < upper < 1: raise ValueError('阈值非法')
        self.features, self.variant, self.lower, self.upper = features, variant, lower, upper
        dates = features['known'].index
        self.weekly = set(dates.to_series().groupby(dates.to_period('W')).first())
        self.weights = None; self.cap = .5 if variant=='fixed_half' else 1.
        self.safe = 0; self.records = []; self.last_date = None

    def on_close(self, date, values, equity):
        if self.last_date is not None and date <= self.last_date: raise ValueError('状态日期必须递增')
        self.last_date = date
        if date not in self.features['known'].index: raise ValueError('缺组合风险证据日期')
        if self.variant=='fixed_half': return .5, False
        if date in self.weekly:
            weights = self.weights
            if weights is None and values:
                x = pd.Series(values,dtype=float); weights = x/x.sum()
            if weights is not None:
                known = self.features['known'].loc[date].reindex(weights.index).fillna(False)
                complete = bool(known.all())
                dd = float(self.features['drawdown'].loc[date].reindex(weights.index).mul(weights).sum()) if complete else np.nan
                weak = float(self.features['weak'].loc[date].reindex(weights.index).mul(weights).sum()) if complete else np.nan
                ret = float(self.features['ret20'].loc[date].reindex(weights.index).mul(weights).sum()) if complete else np.nan
                before = self.cap
                if complete and weak >= .6 and dd >= self.lower:
                    if self.weights is None: self.weights = weights.copy()
                    self.safe = 0
                    requested = 0. if self.variant=='full' or (self.variant=='staged' and dd>=self.upper) else .5
                    self.cap = min(self.cap, requested)
                elif self.weights is not None:
                    self.safe = self.safe+1 if complete and weak<=.4 and ret>0 else 0
                    if self.safe>=2:
                        self.weights = None; self.cap = 1.; self.safe = 0
                self.records.append(dict(date=date, known=complete, drawdown=dd, weak_weight=weak,
                    basket_ret20=ret, cap_before=before, cap_after=self.cap, safe_weeks=self.safe,
                    frozen=self.weights is not None, basket='|'.join(weights.index),
                    weights='|'.join(str(w) for w in weights), equity=equity))
        # pause只禁止新买，不减少旧仓；其它方案风险期间同样禁止补位。
        return (1. if self.variant=='pause' else self.cap), self.weights is not None
