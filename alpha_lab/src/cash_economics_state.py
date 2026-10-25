"""共同风险经济性消融：同步篮子回撤、长期趋势确认与恢复迟滞。"""
import numpy as np
import pandas as pd

from portfolio_cash_state import basket_features


class EconomicCashState:
    def __init__(self, close, mode):
        if mode not in ['quick_release', 'basket_dd', 'trend_break', 'trend_half']:
            raise ValueError('未知经济性消融')
        self.mode = mode
        self.variant = 'half' if mode == 'trend_half' else 'staged'
        self.close = close
        self.features = basket_features(close)
        self.weak60 = close.lt(close.rolling(60, min_periods=60).mean())
        self.ret60 = close / close.shift(60) - 1
        self.weekly = set(close.index.to_series().groupby(close.index.to_period('W')).first())
        self.weights = None
        self.cap = 1.
        self.safe = 0
        self.records = []
        self.last_date = None
        self.recovered_today = False

    def on_close(self, date, values, equity):
        if self.last_date is not None and date <= self.last_date:
            raise ValueError('状态日期必须递增')
        if date not in self.close.index:
            raise ValueError('缺风险证据日期')
        self.last_date = date
        self.recovered_today = False
        if date not in self.weekly:
            return self.cap, self.weights is not None
        weights = self.weights
        if weights is None and values:
            x = pd.Series(values, dtype=float)
            if not np.isfinite(x).all() or (x <= 0).any():
                raise ValueError('持仓市值非法')
            weights = x / x.sum()
        if weights is None:
            return self.cap, False
        known = self.features['known'].loc[date].reindex(weights.index).fillna(False)
        complete = bool(known.all())
        dd = basket_dd = weak = weak60 = ret = ret60 = np.nan
        if complete:
            dd = float(self.features['drawdown'].loc[date].reindex(weights.index).mul(weights).sum())
            weak = float(self.features['weak'].loc[date].reindex(weights.index).mul(weights).sum())
            ret = float(self.features['ret20'].loc[date].reindex(weights.index).mul(weights).sum())
            hist = self.close.loc[:date, weights.index].tail(60)
            basket = hist.div(hist.iloc[-1]).mul(weights).sum(axis=1, min_count=len(weights))
            basket_dd = float(1 - basket.iloc[-1] / basket.max())
            long_ret = self.ret60.loc[date].reindex(weights.index)
            if long_ret.notna().all() and np.isfinite(long_ret).all():
                ret60 = float(long_ret.mul(weights).sum())
                weak60 = float(self.weak60.loc[date].reindex(weights.index).mul(weights).sum())
        effective_dd = dd if self.mode == 'quick_release' else basket_dd
        trigger = complete and weak >= .6 and effective_dd >= .05
        if self.mode in ['trend_break', 'trend_half']:
            trigger = trigger and np.isfinite(ret60) and ret60 < 0 and weak60 >= .6
        before = self.cap
        if trigger:
            if self.weights is None:
                self.weights = weights.copy()
            requested = .5 if self.mode == 'trend_half' or effective_dd < .10 else 0.
            self.cap = min(self.cap, requested)
        elif self.weights is not None and complete and weak <= .4:
            # 一次周首收盘确认；仍是已知证据，恢复个股继续遵守引擎独立条件。
            self.weights = None
            self.cap = 1.
            self.recovered_today = True
        self.records.append(dict(date=date, known=complete, drawdown=dd, basket_drawdown=basket_dd,
            weak_weight=weak, weak60_weight=weak60, basket_ret20=ret, basket_ret60=ret60,
            cap_before=before, cap_after=self.cap, safe_weeks=0, frozen=self.weights is not None,
            basket='|'.join(weights.index), weights='|'.join(str(w) for w in weights), equity=equity))
        return self.cap, self.weights is not None
