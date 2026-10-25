"""合成路径独立验证持有许可、整手保留和不可撤销退出；不读取真实行情。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from portfolio import run_backtest, normalize_actions
from retail_rebalance import plan_rebalance


class PatientExecutionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / 'configs/monthly_portfolio_audit.yaml').read_text())['portfolio']
        self.cfg.update(capital=10000., top_n=1, buy_cost=0., sell_cost=0., min_fee=0.,
                        max_volume_participation=1., cash_dividend_haircut=0.)
        self.retail = dict(candidate_pool=2, max_stock_fraction=1., max_positions=1)
        self.dates = pd.bdate_range('2020-01-02', periods=8)
        self.codes = ['000001.SZ', '000002.SZ']
        self.market = pd.DataFrame([dict(date=d, ts_code=c, open=10., close=10.,
            pre_close=10., adj_factor=1., vol_share=1e7) for d in self.dates for c in self.codes])
        self.limits = self.market[['date', 'ts_code']].assign(up_limit=100., down_limit=1.)
        self.basics = pd.DataFrame(dict(ts_code=self.codes,
            list_date=[pd.Timestamp('2000-01-01')]*2, delist_date=[pd.NaT]*2))
        self.st = pd.DataFrame(dict(ts_code=pd.Series(dtype=str),
            start=pd.Series(dtype='datetime64[ns]'), end=pd.Series(dtype='datetime64[ns]')))
        self.actions = normalize_actions(pd.DataFrame())
        self.signals = pd.DataFrame([[2., 1.], [1., 2.]],
                                   index=self.dates[[0, 2]], columns=self.codes)
        self.forecasts = pd.DataFrame([[.1, -.1], [.01, .2]],
                                     index=self.signals.index, columns=self.codes)
        self.hold = pd.DataFrame(True, index=self.signals.index, columns=self.codes)

    def run_case(self, hold=True, risk=None):
        kwargs = dict(retail_forecasts=self.forecasts, retail_config=self.retail)
        if hold:
            kwargs['hold_allowed'] = self.hold
        if risk is not None:
            kwargs.update(risk_allowed=risk, risk_signal_max_age=10,
                          reentry_on_schedule_only=True)
        result = run_backtest(self.signals, self.market, self.basics, self.st,
            self.limits, self.actions, self.dates[0], self.dates[-1], self.cfg, **kwargs)
        # 直接重建现金流水，与每次成交后的现金及每日资产恒等式对账。
        cash = self.cfg['capital']
        for trade in result.trades.itertuples():
            cash += (-trade.gross if trade.side == 'buy' else trade.gross) - trade.fee
            self.assertAlmostEqual(cash, trade.cash_after)
        self.assertAlmostEqual(cash, result.daily.cash.iloc[-1])
        np.testing.assert_allclose(result.daily.equity,
            result.daily.cash + result.daily.stock_value + result.daily.receivables)
        self.assertLessEqual(result.daily.n_hold.max(), self.retail['max_positions'])
        if not result.events.empty and 'target_shares' in result.events:
            for row in result.events[result.events.kind.eq('allocation_decision')].itertuples():
                self.assertEqual(row.target_shares, row.old_shares + row.buy_shares - row.sell_shares)
                self.assertGreaterEqual(row.target_shares, row.min_target_shares)
        return result

    def test_rank_reversal_does_not_replace_old_position(self):
        result = self.run_case()
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(set(result.holdings.ts_code), {self.codes[0]})
        self.assertEqual(result.holdings.iloc[-1].shares, 1000)
        self.assertTrue(result.events.kind.eq('hold_retained').any())

    def test_false_exits_on_next_open(self):
        self.hold.loc[self.dates[2], self.codes[0]] = False
        result = self.run_case()
        sold = result.trades[result.trades.side.eq('sell')]
        self.assertEqual(list(sold.date), [self.dates[3]])
        self.assertEqual(list(sold.shares), [1000])
        self.assertEqual(result.holdings.iloc[-1].ts_code, self.codes[1])

    def test_unknown_and_missing_forecast_do_not_liquidate(self):
        self.hold = self.hold.astype(object)
        self.hold.loc[self.dates[2], self.codes[0]] = np.nan
        self.forecasts.loc[self.dates[2], self.codes[0]] = np.nan
        result = self.run_case()
        self.assertEqual(len(result.trades), 1)
        retained = result.events[result.events.kind.eq('hold_retained')]
        self.assertEqual(retained.iloc[-1].reason, 'unknown')

    def test_untradable_exit_retries_and_true_cannot_cancel(self):
        self.signals.loc[self.dates[3]] = [2., 1.]
        self.forecasts.loc[self.dates[3]] = [.9, -.1]
        self.hold = self.hold.reindex(self.signals.index, fill_value=True)
        self.hold.loc[self.dates[2], self.codes[0]] = False
        mask = self.limits.date.eq(self.dates[3]) & self.limits.ts_code.eq(self.codes[0])
        self.limits.loc[mask, 'down_limit'] = 10.
        result = self.run_case()
        sold = result.trades[result.trades.side.eq('sell')]
        self.assertEqual(list(sold.date), [self.dates[4]])
        self.assertEqual(list(sold.shares), [1000])
        self.assertTrue(result.events.kind.eq('sell_deferred').any())
        requested = result.events[result.events.kind.eq('hold_exit_requested')]
        self.assertEqual(requested.iloc[-1].reason, 'confirmed_pending')

    def test_untradable_exit_retries_without_scheduled_review(self):
        self.hold.loc[self.dates[2], self.codes[0]] = False
        mask = self.market.date.eq(self.dates[3]) & self.market.ts_code.eq(self.codes[0])
        self.market.loc[mask, 'vol_share'] = 0.
        result = self.run_case()
        sold = result.trades[result.trades.side.eq('sell')]
        self.assertEqual(list(sold.date), [self.dates[4]])

    def test_confirmed_exit_preserves_pending_order_for_locked_bonus(self):
        self.hold.loc[self.dates[2], self.codes[0]] = False
        self.actions = normalize_actions(pd.DataFrame([dict(ts_code=self.codes[0],
            div_proc='实施', record_date=self.dates[2], ex_date=self.dates[3],
            pay_date=self.dates[3], div_listdate=self.dates[5], cash_div_tax=0., stk_div=1.)]))
        mask = self.market.ts_code.eq(self.codes[0]) & self.market.date.ge(self.dates[3])
        self.market.loc[mask, ['open', 'close', 'pre_close']] = 5.
        self.market.loc[mask, 'adj_factor'] = 2.
        result = self.run_case()
        sold = result.trades[result.trades.side.eq('sell')]
        self.assertEqual(list(sold.date), [self.dates[3], self.dates[5]])
        self.assertEqual(list(sold.shares), [1000, 1000])
        self.assertEqual(result.daily.loc[self.dates[4], 'pending_sell'], 1)
        self.assertEqual(result.daily.loc[self.dates[5], 'pending_sell'], 0)

    def test_single_stock_cap_reduces_only_necessary_lots(self):
        self.retail['max_stock_fraction'] = .5
        mask = self.market.ts_code.eq(self.codes[0]) & self.market.date.ge(self.dates[2])
        self.market.loc[mask, ['open', 'close']] = 20.
        self.market.loc[mask & self.market.date.gt(self.dates[2]), 'pre_close'] = 20.
        result = self.run_case()
        sold = result.trades[result.trades.side.eq('sell')]
        # 500股涨至20元，资产15000、单股上限7500；必须卖200股，保留300股。
        self.assertEqual(list(sold.shares), [200])
        self.assertEqual(result.holdings.iloc[-1].shares, 300)
        self.assertTrue(result.events.kind.eq('hold_risk_cap_reduction').any())

    def test_retained_stock_cannot_receive_top_up(self):
        self.retail['max_stock_fraction'] = .5
        mask = self.market.ts_code.eq(self.codes[0]) & self.market.date.ge(self.dates[2])
        self.market.loc[mask, ['open', 'close']] = 5.
        self.market.loc[mask & self.market.date.gt(self.dates[2]), 'pre_close'] = 5.
        self.forecasts.loc[self.dates[2]] = [.9, -.1]
        result = self.run_case()
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.holdings.iloc[-1].shares, 500)

    def test_st_exit_overrides_hold(self):
        self.st = pd.DataFrame([dict(ts_code=self.codes[0], start=self.dates[2], end=pd.NaT)])
        result = self.run_case()
        sold = result.trades[result.trades.side.eq('sell')]
        self.assertEqual(list(sold.date), [self.dates[2]])

    def test_portfolio_risk_exit_overrides_hold(self):
        risk = pd.Series(1, index=self.dates)
        risk.loc[self.dates[2]:] = 0
        result = self.run_case(risk=risk)
        self.assertEqual(list(result.trades[result.trades.side.eq('sell')].date), [self.dates[3]])
        self.assertEqual(result.daily.n_hold.iloc[-1], 0)

    def test_default_semantics_still_replace_ranked_position(self):
        result = self.run_case(hold=False)
        self.assertEqual(list(result.trades.side), ['buy', 'sell', 'buy'])
        self.assertEqual(result.holdings.iloc[-1].ts_code, self.codes[1])

    def test_hold_calendar_and_position_limit_are_validated(self):
        original = self.hold
        self.hold = self.hold.iloc[:1]
        with self.assertRaises(ValueError):
            self.run_case()
        self.hold = original
        self.retail['max_positions'] = 6
        with self.assertRaises(ValueError):
            self.run_case()


class MinimumTargetTest(unittest.TestCase):
    def row(self, **kwargs):
        row = dict(ts_code='old', price=10., expected_return=-.5, shares=1050,
            available=1050, min_buy=100, buy_step=100, sell_step=100,
            max_notional=8200., can_buy=False, can_sell=True, min_target_shares=800)
        row.update(kwargs)
        return row

    def allocate(self, row):
        return plan_rebalance(pd.DataFrame([row]), cash=0., buy_rate=0., sell_rate=0.,
            min_fee=0., forecast_kind='expected_simple_return', max_positions=1)['plan'].iloc[0]

    def test_odd_lots_retain_largest_reachable_target(self):
        row = self.allocate(self.row())
        self.assertEqual(row.target_shares, 800)
        self.assertEqual(row.sell_shares, 250)
        self.assertEqual(row.min_target_shares, 800)

    def test_locked_shares_override_cap_without_imagined_sale(self):
        row = self.allocate(self.row(available=150, min_target_shares=900))
        self.assertEqual(row.target_shares, 900)
        self.assertEqual(row.sell_shares, 150)
        self.assertEqual(row.unavoidable_locked_excess, 800.)

    def test_minimum_validation_rejects_invalid_values(self):
        for value in [-1, 1.5, 1051, np.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.allocate(self.row(min_target_shares=value))

    def test_omitted_minimum_preserves_zero_default(self):
        row = self.row()
        del row['min_target_shares']
        plan = self.allocate(row)
        self.assertEqual(plan.target_shares, 0)
        self.assertEqual(plan.min_target_shares, 0)


if __name__ == '__main__':
    unittest.main()
