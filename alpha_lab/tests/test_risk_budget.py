"""风险折扣、整手配置、月度缩仓和前缀因果反例。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)]
from risk_budget_portfolio import run_backtest
from lifecycle_risk import risk_discount, lifecycle_reviews
import test_causal_portfolio_adversarial as fixture


class RiskBudgetTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.CausalPortfolioAdversarialTest()
        self.f.setUp()

    def run_case(self, scales=None, existing=False):
        f = self.f
        return run_backtest(f.signals, f.market, f.basics, f.st, f.limits, f.actions,
            f.dates[0], f.dates[-1], deepcopy(f.cfg), retail_forecasts=f.forecasts,
            retail_config=deepcopy(f.retail), hold_allowed=f.hold,
            risk_multipliers=scales, adjust_existing_risk=existing)

    def test_risk_discount_never_levers_and_unknown_rejected(self):
        np.testing.assert_allclose(risk_discount([-.2, -.1, -.05, 0., .1], .1), [.5, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            risk_discount([np.nan], .1)

    def test_entry_discount_leaves_real_cash_and_lots(self):
        f = self.f
        scales = f.signals * 0 + .5
        scales[:] = .5
        result = self.run_case(scales)
        self.assertEqual(result.trades.iloc[0].shares, 400)
        self.assertEqual(result.daily.iloc[-1]['cash'], 6000.)
        self.assertTrue(result.trades.shares.mod(100).eq(0).all())

    def test_existing_discount_only_applies_at_review(self):
        f = self.f
        f.add_review(5, preferred=0)
        scales = pd.DataFrame(1., index=f.signals.index, columns=f.codes)
        scales.iloc[1] = .5
        full = self.run_case(scales, True)
        entry = self.run_case(scales, False)
        sales = full.trades[full.trades.side.eq('sell')]
        self.assertEqual(sales.iloc[0].decision_date, f.dates[5])
        self.assertEqual(sales.iloc[0].date, f.dates[6])
        self.assertEqual(sales.iloc[0].shares, 400)
        self.assertFalse(entry.trades.side.eq('sell').any())

    def test_future_multiplier_does_not_change_earlier_trades(self):
        f = self.f
        f.add_review(5, preferred=0)
        scales = pd.DataFrame(1., index=f.signals.index, columns=f.codes)
        before = self.run_case(scales, True)
        scales.iloc[1] = .2
        after = self.run_case(scales, True)
        pd.testing.assert_frame_equal(before.trades[before.trades.date < f.dates[6]],
                                      after.trades[after.trades.date < f.dates[6]])

    def test_invalid_multiplier_fails_closed(self):
        f = self.f
        for value in [0., 1.1, np.nan]:
            scales = pd.DataFrame(value, index=f.signals.index, columns=f.codes)
            with self.assertRaises(ValueError):
                self.run_case(scales, True)

    def test_lifecycle_includes_entry_hold_unknown_and_exit(self):
        d = pd.date_range('2020-01-01', periods=4, freq='MS')
        s = pd.DataFrame({'a': [1., np.nan, np.nan, np.nan]}, index=d)
        h = pd.DataFrame({'a': [1., 1., np.nan, 0.]}, index=d)
        full = lifecycle_reviews(s, h, 1)
        self.assertEqual(full.phase.tolist(), ['entry', 'hold', 'hold', 'exit'])
        pd.testing.assert_frame_equal(full.iloc[:3], lifecycle_reviews(s.iloc[:3], h.iloc[:3], 1))


if __name__ == '__main__':
    unittest.main()
