"""真实减仓记忆、旧仓增持T+1、现金预约和整手缓冲反例。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)]
from risk_recovery_portfolio import run_backtest
import test_causal_portfolio_adversarial as fixture


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.CausalPortfolioAdversarialTest()
        self.f.setUp()
        self.f.add_review(3, preferred=0)
        self.f.add_review(7, preferred=0)
        self.scales = pd.DataFrame(1., index=self.f.signals.index, columns=self.f.codes)
        self.scales.iloc[1] = .5
        self.scales.iloc[2] = .8

    def run_case(self, restore=True, grace=False):
        f = self.f
        return run_backtest(f.signals, f.market, f.basics, f.st, f.limits, f.actions,
            f.dates[0], f.dates[-1], deepcopy(f.cfg), retail_forecasts=f.forecasts,
            retail_config=deepcopy(f.retail), hold_allowed=f.hold, risk_multipliers=self.scales,
            adjust_existing_risk=True, recovery_config=dict(restore=restore, lot_grace=grace))

    def test_restore_requires_actual_trim_and_preserves_old_shares_and_locks(self):
        r = self.run_case()
        trades = r.trades
        self.assertEqual(trades.side.tolist(), ['buy', 'sell', 'buy'])
        self.assertEqual(trades.shares.tolist(), [900, 400, 200])
        day = r.holdings[r.holdings.date.eq(self.f.dates[8])].iloc[0]
        self.assertEqual(day.shares, 700)
        self.assertEqual(day.available, 500)
        self.assertEqual(r.daily.iloc[-1]['cash'], 3000.)

    def test_failed_sale_never_creates_restore_permission(self):
        f = self.f
        f.market.loc[(f.market.ts_code == f.codes[0]) & (f.market.date >= f.dates[4]), ['open', 'high', 'low']] = np.nan
        r = self.run_case()
        self.assertFalse(r.events.kind.eq('risk_restore_planned').any())
        self.assertEqual(len(r.trades), 1)

    def test_no_risk_improvement_or_entry_eligibility_means_no_restore(self):
        self.scales.iloc[2] = .5
        r = self.run_case()
        self.assertFalse(r.events.kind.eq('risk_restore_planned').any())
        self.scales.iloc[2] = .8
        self.f.signals.iloc[2, 0] = np.nan
        r = self.run_case()
        self.assertFalse(r.events.kind.eq('risk_restore_planned').any())

    def test_grace_only_below_one_lot_and_original_cap_still_binding(self):
        self.scales.iloc[1:] = .85
        strict = self.run_case(False, False)
        grace = self.run_case(False, True)
        self.assertEqual(strict.trades.shares.tolist(), [900, 100])
        self.assertEqual(grace.trades.shares.tolist(), [900])
        self.assertTrue(grace.events.kind.eq('risk_cap_grace').any())
        self.scales.iloc[1:] = .8
        self.assertEqual(self.run_case(False, True).trades.shares.tolist(), [900, 100])
        self.scales[:] = 1.
        self.f.retail['max_stock_fraction'] = .85
        # 初始800股；涨价造成跨越原85%上限，不能因为不足一手免减。
        for i, d in enumerate(self.f.dates):
            value = 10. if i < 2 else 16.
            mask = self.f.market.date.eq(d) & self.f.market.ts_code.eq(self.f.codes[0])
            self.f.market.loc[mask, ['open','high','low','close']] = value
            self.f.market.loc[mask, 'pre_close'] = 10. if i <= 2 else 16.
        self.f.limits['up_limit'] = 100.
        self.f.limits['down_limit'] = 1.
        self.assertTrue(self.run_case(False, True).trades.side.eq('sell').any())

    def test_future_recovery_scale_does_not_change_prior_orders(self):
        before = self.run_case()
        self.scales.iloc[2] = 1.
        after = self.run_case()
        for name in ['trades', 'holdings']:
            a, b = getattr(before, name), getattr(after, name)
            pd.testing.assert_frame_equal(a[a.date < self.f.dates[8]], b[b.date < self.f.dates[8]])

    def test_restore_cash_is_reserved_before_any_next_day_sales(self):
        r = self.run_case()
        orders = r.events[r.events.kind.eq('order_planned') & r.events.side.eq('buy')]
        for date, group in orders.groupby('decision_date'):
            self.assertLessEqual(group.reserved_cash.sum(), r.daily.loc[pd.Timestamp(date), 'cash']+1e-8)
        plans = r.events[r.events.kind.eq('risk_restore_planned')]
        self.assertTrue((plans.frozen_position_notional <= plans.risk_cap+1e-8).all())


if __name__ == '__main__':
    unittest.main()
