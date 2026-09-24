"""新接口需复现旧账本，并验证反馈因果性与可突破初始策略的动作。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT / 'scripts')]
from test_portfolio import PortfolioTest
from portfolio import run_backtest as original
from policy_portfolio import run_backtest
from initialized_policy import PolicyAction, NeuralPolicy
from audit_monthly_portfolio import reconcile


class TestPolicyAccounting(PortfolioTest):
    def run_case(self, end=None, cfg=None):
        config = cfg or self.cfg
        signals = self.signals.copy()
        def callback(snapshot):
            date = snapshot['date']
            return PolicyAction(signals.loc[date], config['top_n']) if date in signals.index else None
        result = run_backtest(signals.iloc[:0].copy(), self.market, self.basics, self.st, self.limits,
                              self.actions, self.dates[0], end or self.dates[-1], config, policy=callback)
        expected = original(signals, self.market, self.basics, self.st, self.limits,
                            self.actions, self.dates[0], end or self.dates[-1], config)
        for name in ['daily', 'trades', 'events', 'holdings']:
            pd.testing.assert_frame_equal(getattr(result, name), getattr(expected, name))
        reconcile(result, config['capital'])
        return result


def fixture():
    f = PortfolioTest(); f.setUp()
    return f


def execute(f, callback, end=None):
    return run_backtest(f.signals.iloc[:0].copy(), f.market, f.basics, f.st, f.limits,
                        f.actions, f.dates[0], end or f.dates[-1], f.cfg, policy=callback)


def test_feedback_is_settled_copy_and_no_action_differs_from_exit():
    f, seen = fixture(), []
    def callback(s):
        seen.append(s)
        if s['date'] == f.dates[0]:
            return PolicyAction(f.signals.iloc[0], 1)
        if s['date'] == f.dates[3]:
            return PolicyAction(pd.Series(np.nan, index=f.signals.columns), 1)
        return None
    r = execute(f, callback)
    assert seen[0]['positions'] == {}
    assert seen[1]['positions']['000001.SZ']['available'] == 0
    assert seen[2]['positions']['000001.SZ']['available'] == 1000
    assert len(r.trades) == 2 and r.trades.date.iloc[1] == f.dates[4]
    assert seen[3]['positions']['000001.SZ']['shares'] == 1000
    assert all(t['date'] == s['date'] for s in seen for t in s['trades'])


def test_partial_reduction_retries_exact_quantity_after_limit_down():
    f = fixture()
    f.limits.loc[f.limits.date == f.dates[3], 'down_limit'] = 10.
    def callback(s):
        if s['date'] in [f.dates[0], f.dates[2]]:
            return PolicyAction(f.signals.iloc[0], 1, reduce={'000001.SZ': .4} if s['date'] == f.dates[2] else {})
    r = execute(f, callback)
    sells = r.trades[r.trades.side == 'sell']
    assert len(sells) == 1 and sells.shares.iloc[0] == 400 and sells.date.iloc[0] == f.dates[4]
    assert r.holdings.iloc[-1].shares == 600
    reconcile(r, f.cfg['capital'])


def test_cash_budget_and_unequal_new_position_allocation():
    f = fixture()
    def callback(s):
        if s['date'] == f.dates[0]:
            return PolicyAction(pd.Series([2., 1.], index=f.signals.columns), 2, .6,
                                weights={'000001.SZ': 2., '000002.SZ': 1.})
    r = execute(f, callback)
    assert list(r.trades.shares) == [400, 200]
    assert r.daily.cash.iloc[-1] == 4000.
    reconcile(r, f.cfg['capital'])


def test_observer_cannot_mutate_account_and_future_prices_do_not_change_prefix():
    f = fixture()
    def callback(s):
        if s['date'] == f.dates[0]:
            return PolicyAction(f.signals.iloc[0], 1)
        for p in s['positions'].values():
            p['shares'] = 0
        s['cash'] = -100
    a = execute(f, callback)
    f.edit(15, open=10.2, close=10.2)
    b = execute(f, callback, f.dates[12])
    pd.testing.assert_frame_equal(a.daily.loc[:f.dates[12]], b.daily)
    assert b.holdings.shares.eq(1000).all()


def package():
    f = fixture()
    dates, codes = f.dates, f.signals.columns
    feat = np.zeros((len(dates), 2, 8)); feat[:, :, :4] = np.array([.5, -.5])[:, None]
    feat[:, :, 4:] = 1
    return dict(dates=dates, codes=codes, features=feat, seed_score_start=0, seed_valid_start=4,
                schedules=np.tile([1, -1, -1], (len(dates), 1)), eligible=np.ones((len(dates), 2), bool),
                policies=[dict(frequency='daily', top_n=1)] * 4)


def snapshot(p):
    return dict(date=p['dates'][0], cash=10000., equity=10000., positions={}, events=[], trades=[], pending_sell={})


def test_trainable_heads_can_reverse_ranking_filter_schedule_and_allocation():
    p = package(); theta = NeuralPolicy.initialize(p, 4, 0, 8)
    a = NeuralPolicy(p, theta, 4, 10000)
    assert a(snapshot(p)).scores.idxmax() == p['codes'][0]
    b = NeuralPolicy(p, theta, 4, 10000)
    b.blocks['stock_linear'][0, 0] = -1
    b.blocks['global_weight'][4, 1:] = [2, .2]
    action = b(snapshot(p))
    assert action.scores.idxmax() == p['codes'][1] and action.top_n == 2 and action.cash_fraction == .2
    b.blocks['stock_linear'][4, 1] = -1
    assert b(snapshot(p)).scores.isna().all()
    b.blocks['global_weight'][:, 0] *= -1
    assert b(snapshot(p)) is None
    assert np.all(NeuralPolicy.scales(8, 4) > 0)


def test_invalid_action_rejected():
    p = package()
    with pytest.raises(ValueError):
        PolicyAction(pd.Series([1., 2.], index=p['codes']), 2, cash_fraction=1.1).validate(p['codes'])
