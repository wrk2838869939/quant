"""独立反事实检验收盘锁单、开盘兑现；不复述撮合器实现。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from causal_portfolio import run_backtest, AccountingError
from portfolio import normalize_actions


class CausalPortfolioAdversarialTest(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / 'configs/monthly_portfolio_audit.yaml').read_text())['portfolio']
        self.cfg.update(capital=10000., top_n=1, buy_cost=0., sell_cost=0., min_fee=0.,
                        max_volume_participation=1., cash_dividend_haircut=0.)
        self.retail = dict(candidate_pool=6, max_stock_fraction=1., max_positions=1,
                           selection_first=True, solver_relative_gap=.001)
        self.dates = pd.bdate_range('2022-01-04', periods=12)
        self.codes = ['000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ', '000007.SZ']
        self.market = pd.DataFrame([dict(date=d, ts_code=c, open=10., high=10., low=10., close=10.,
            pre_close=10., adj_factor=1., vol_share=1e7) for d in self.dates for c in self.codes])
        self.limits = self.market[['date', 'ts_code']].assign(up_limit=11., down_limit=9.)
        self.basics = pd.DataFrame(dict(ts_code=self.codes,
            list_date=[pd.Timestamp('2000-01-01')] * len(self.codes),
            delist_date=[pd.NaT] * len(self.codes)))
        self.st = pd.DataFrame(dict(ts_code=pd.Series(dtype=str),
            start=pd.Series(dtype='datetime64[ns]'), end=pd.Series(dtype='datetime64[ns]')))
        self.actions = normalize_actions(pd.DataFrame())
        self.signals = pd.DataFrame(np.nan, index=self.dates[[0]], columns=self.codes)
        self.signals.iloc[0, 0] = 1.
        self.forecasts = pd.DataFrame(.1, index=self.signals.index, columns=self.codes)
        self.hold = pd.DataFrame(True, index=self.signals.index, columns=self.codes)

    def run_case(self, market=None, end=None):
        result = run_backtest(self.signals, self.market if market is None else market,
            self.basics, self.st, self.limits, self.actions, self.dates[0],
            self.dates[-1] if end is None else end, deepcopy(self.cfg),
            retail_forecasts=self.forecasts, retail_config=deepcopy(self.retail),
            hold_allowed=self.hold, causal_config=dict(buy_limit_buffer=.02, max_entry_attempts=2,
                sell_limit_buffer=.10, st_sell_limit_buffer=.05, cancel_on_ex_date=True))
        np.testing.assert_allclose(result.daily.equity,
            result.daily.cash + result.daily.stock_value + result.daily.receivables, atol=1e-8)
        self.assertGreaterEqual(result.daily.cash.min(), -1e-8)
        self.assertLessEqual(result.daily.n_hold.max(), self.retail['max_positions'])
        if not result.trades.empty:
            self.assertTrue((result.trades.date > pd.to_datetime(result.trades.decision_date)).all())
            buys = result.trades[result.trades.side.eq('buy')]
            self.assertTrue((buys.shares % 100 == 0).all())
            self.assertTrue((buys.price <= buys.limit_price + 1e-9).all())
        return result

    def planned(self, result, decision=None, side=None):
        events = result.events[result.events.kind.eq('order_planned')].copy()
        if events.empty:
            return pd.DataFrame(columns=['order_id', 'decision_date', 'execution_date', 'ts_code', 'side',
                'shares', 'reference_price', 'limit_price', 'cash_at_decision', 'reserved_cash', 'budget_source'])
        if decision is not None:
            events = events[pd.to_datetime(events.decision_date).eq(decision)]
        if side is not None:
            events = events[events.side.eq(side)]
        fields = ['order_id', 'decision_date', 'execution_date', 'ts_code', 'side',
                  'shares', 'reference_price', 'limit_price', 'cash_at_decision',
                  'reserved_cash', 'budget_source']
        return events[fields].sort_values(['decision_date', 'side', 'ts_code']).reset_index(drop=True)

    def add_review(self, day, preferred=1, exit_code=None):
        self.signals.loc[self.dates[day]] = np.nan
        self.signals.loc[self.dates[day], self.codes[preferred]] = 1.
        self.signals = self.signals.sort_index()
        self.forecasts = self.forecasts.reindex(self.signals.index, fill_value=.1)
        self.hold = self.hold.reindex(self.signals.index, fill_value=True)
        if exit_code is not None:
            self.hold.loc[self.dates[day], self.codes[exit_code]] = False

    def mutate(self, day, code=0, **values):
        mask = self.market.date.eq(self.dates[day]) & self.market.ts_code.eq(self.codes[code])
        for field, value in values.items():
            self.market.loc[mask, field] = value

    def test_tomorrow_open_cannot_change_tonight_orders(self):
        baseline = self.run_case()
        orders = self.planned(baseline, self.dates[0], 'buy')
        self.assertGreater(len(orders), 0)
        for tomorrow_open in [9.5, 10.19, 10.5, 11., np.nan]:
            market = self.market.copy()
            market.loc[market.date.eq(self.dates[1]) & market.ts_code.eq(self.codes[0]), 'open'] = tomorrow_open
            altered = self.run_case(market=market)
            pd.testing.assert_frame_equal(orders, self.planned(altered, self.dates[0], 'buy'))

    def test_execution_day_full_volume_cannot_choose_orders_or_open_fills(self):
        baseline = self.run_case()
        initial_orders = self.planned(baseline, self.dates[0], 'buy')
        initial_fills = baseline.trades[baseline.trades.date.eq(self.dates[1])].reset_index(drop=True)
        self.assertGreater(len(initial_fills), 0)
        for volume in [0., np.nan, 1., 1e12]:
            market = self.market.copy()
            market.loc[market.date.eq(self.dates[1]), 'vol_share'] = volume
            changed = self.run_case(market=market)
            pd.testing.assert_frame_equal(initial_orders, self.planned(changed, self.dates[0], 'buy'))
            pd.testing.assert_frame_equal(initial_fills,
                changed.trades[changed.trades.date.eq(self.dates[1])].reset_index(drop=True))

    def test_only_known_prior_volume_can_limit_new_order(self):
        normal = self.run_case()
        self.market.loc[self.market.date.eq(self.dates[0]), 'vol_share'] = 50.
        tiny = self.run_case()
        self.assertGreater(len(self.planned(normal, self.dates[0], 'buy')), 0)
        self.assertEqual(len(self.planned(tiny, self.dates[0], 'buy')), 0)
        self.assertTrue(tiny.trades.empty)

    def test_gap_up_over_limit_cancels_without_repricing_or_lower_rank_chase(self):
        self.signals.iloc[0, 1] = .5
        self.mutate(1, open=10.5, high=10.6, low=10.4)
        result = self.run_case()
        orders = self.planned(result, self.dates[0], 'buy')
        self.assertEqual(list(orders.ts_code), [self.codes[0]])
        self.assertAlmostEqual(orders.iloc[0].limit_price, 10.2)
        self.assertTrue(result.trades.empty)
        self.assertTrue(result.events.kind.eq('order_cancelled').any())

    def test_day_limit_requires_strict_intraday_cross_and_never_replans(self):
        ordinary = self.run_case()
        self.mutate(1, open=10.5, high=10.6, low=10.2)
        touch = self.run_case()
        self.assertTrue(touch.trades.empty)
        self.mutate(1, low=10.19)
        crossed = self.run_case()
        self.assertEqual(len(crossed.trades), 1)
        self.assertAlmostEqual(crossed.trades.iloc[0].price, 10.2)
        pd.testing.assert_frame_equal(self.planned(ordinary, self.dates[0], 'buy'),
                                     self.planned(crossed, self.dates[0], 'buy'))
        self.assertEqual(crossed.trades.iloc[0].shares, ordinary.trades.iloc[0].shares)

    def test_price_improvement_does_not_increase_locked_quantity(self):
        ordinary = self.run_case()
        self.mutate(1, open=9.5)
        improved = self.run_case()
        plan = self.planned(ordinary, self.dates[0], 'buy').iloc[0]
        self.assertEqual(improved.trades.iloc[0].shares, plan.shares)
        self.assertEqual(ordinary.trades.iloc[0].shares, plan.shares)
        self.assertGreater(improved.daily.cash.iloc[1], ordinary.daily.cash.iloc[1])

    def test_order_reserves_worst_limit_price_and_minimum_fee(self):
        self.cfg.update(capital=1024., buy_cost=.001, min_fee=5.)
        empty = self.run_case()
        self.assertTrue(empty.trades.empty)  # 100 * 10.20 + 5 > 1024
        self.cfg['capital'] = 1025.
        result = self.run_case()
        order = self.planned(result, self.dates[0], 'buy').iloc[0]
        self.assertEqual(order.shares, 100)
        self.assertAlmostEqual(order.reserved_cash, 1025.)
        self.assertEqual(order.budget_source, 'closing_cash')
        self.assertEqual(result.trades.iloc[0].shares, 100)
        self.assertAlmostEqual(result.daily.cash.iloc[1], 20.)

    def test_new_buys_do_not_pre_spend_same_auction_sale_proceeds(self):
        self.add_review(2, preferred=1, exit_code=0)
        result = self.run_case()
        sale = result.trades[result.trades.side.eq('sell')].iloc[0]
        replacement = result.trades[result.trades.side.eq('buy') & result.trades.ts_code.eq(self.codes[1])]
        self.assertEqual(sale.date, self.dates[3])
        self.assertGreater(len(replacement), 0)
        self.assertTrue((replacement.date > sale.date).all())
        self.assertEqual(len(self.planned(result, self.dates[2], 'buy')), 0)
        self.assertTrue((pd.to_datetime(replacement.decision_date) >= sale.date).all())

    def test_blocked_exits_keep_position_slot_even_with_free_cash(self):
        self.cfg.update(capital=50000., top_n=5)
        self.retail.update(max_stock_fraction=.19, max_positions=5)
        self.signals.iloc[0] = [6., 5., 4., 3., 2., np.nan]
        self.add_review(2, preferred=5, exit_code=0)
        blocked = self.limits.date.isin(self.dates[3:7]) & self.limits.ts_code.eq(self.codes[0])
        self.limits.loc[blocked, 'down_limit'] = 10.
        result = self.run_case()
        self.assertEqual(result.daily.n_hold.iloc[1], 5)
        self.assertGreater(result.daily.cash.iloc[2], 4000.)
        held_before_release = result.holdings[result.holdings.date.le(self.dates[6])]
        self.assertFalse(held_before_release.ts_code.eq(self.codes[5]).any())
        self.assertLessEqual(result.daily.n_hold.max(), 5)

    def test_t_plus_one_when_exit_is_requested_on_purchase_day(self):
        self.add_review(1, preferred=1, exit_code=0)
        result = self.run_case()
        buy = result.trades[result.trades.side.eq('buy') & result.trades.ts_code.eq(self.codes[0])].iloc[0]
        sale = result.trades[result.trades.side.eq('sell') & result.trades.ts_code.eq(self.codes[0])].iloc[0]
        self.assertEqual(buy.date, self.dates[1])
        self.assertEqual(sale.date, self.dates[2])
        first = result.holdings[result.holdings.date.eq(buy.date) & result.holdings.ts_code.eq(self.codes[0])].iloc[0]
        self.assertEqual(first.available, 0)
        self.assertEqual(sale.shares, buy.shares)

    def test_limit_down_sale_retries_without_cancelling_confirmed_exit(self):
        self.add_review(2, preferred=1, exit_code=0)
        self.add_review(3, preferred=0)
        mask = self.limits.date.eq(self.dates[3]) & self.limits.ts_code.eq(self.codes[0])
        self.limits.loc[mask, 'down_limit'] = 10.
        result = self.run_case()
        sales = result.trades[result.trades.side.eq('sell') & result.trades.ts_code.eq(self.codes[0])]
        self.assertEqual(list(sales.date), [self.dates[4]])
        self.assertTrue(result.events.kind.eq('order_deferred').any())

    def test_official_limit_up_blocks_even_below_generous_entry_limit(self):
        mask = self.limits.date.eq(self.dates[1]) & self.limits.ts_code.eq(self.codes[0])
        self.limits.loc[mask, 'up_limit'] = 10.1
        self.mutate(1, open=10.1, high=10.1, low=10.1)
        result = self.run_case()
        self.assertGreater(len(self.planned(result, self.dates[0], 'buy')), 0)
        self.assertTrue(result.trades.empty)

    def test_missing_official_limits_cannot_fill_frozen_buy(self):
        self.limits = self.limits[~self.limits.date.eq(self.dates[1])]
        result = self.run_case()
        self.assertGreater(len(self.planned(result, self.dates[0], 'buy')), 0)
        self.assertTrue(result.trades.empty)

    def test_orders_and_accounts_at_history_cutoff_do_not_depend_on_later_rows(self):
        self.add_review(2, preferred=1, exit_code=0)
        full = self.run_case()
        short = self.run_case(end=self.dates[5])
        pd.testing.assert_frame_equal(full.daily.loc[:self.dates[5]], short.daily)
        pd.testing.assert_frame_equal(full.trades[full.trades.date.le(self.dates[5])].reset_index(drop=True),
                                     short.trades.reset_index(drop=True))
        pd.testing.assert_frame_equal(self.planned(full)[lambda x: pd.to_datetime(x.decision_date).le(self.dates[4])].reset_index(drop=True),
                                     self.planned(short)[lambda x: pd.to_datetime(x.decision_date).le(self.dates[4])].reset_index(drop=True))

    def test_pending_buy_crossing_ex_date_is_cancelled_not_repriced(self):
        self.actions = normalize_actions(pd.DataFrame([dict(ts_code=self.codes[0], div_proc='实施',
            record_date=self.dates[0], ex_date=self.dates[1], pay_date=self.dates[1],
            div_listdate=self.dates[1], cash_div_tax=1., stk_div=0.)]))
        self.actions['known_date'] = self.dates[0]
        mask = self.market.ts_code.eq(self.codes[0]) & self.market.date.ge(self.dates[1])
        self.market.loc[mask, ['open', 'high', 'low', 'close', 'pre_close']] = 9.
        self.market.loc[mask, 'adj_factor'] = 10 / 9
        self.limits.loc[self.limits.ts_code.eq(self.codes[0]) & self.limits.date.ge(self.dates[1]), 'down_limit'] = 8.1
        result = self.run_case()
        self.assertGreater(len(self.planned(result, self.dates[0], 'buy')), 0)
        self.assertTrue(result.trades.empty)
        self.assertTrue(result.events.kind.eq('order_cancelled').any())
        np.testing.assert_allclose(result.daily.equity, self.cfg['capital'])

    def test_bonus_new_shares_remain_locked_and_equity_is_conserved(self):
        self.add_review(1, preferred=1, exit_code=0)
        self.actions = normalize_actions(pd.DataFrame([dict(ts_code=self.codes[0], div_proc='实施',
            record_date=self.dates[1], ex_date=self.dates[2], pay_date=self.dates[2],
            div_listdate=self.dates[5], cash_div_tax=0., stk_div=1.)]))
        self.actions['known_date'] = self.dates[0]
        mask = self.market.ts_code.eq(self.codes[0]) & self.market.date.ge(self.dates[2])
        self.market.loc[mask, ['open', 'high', 'low', 'close', 'pre_close']] = 5.
        self.market.loc[mask, 'adj_factor'] = 2.
        lmask = self.limits.ts_code.eq(self.codes[0]) & self.limits.date.ge(self.dates[2])
        self.limits.loc[lmask, 'up_limit'] = 5.5
        self.limits.loc[lmask, 'down_limit'] = 4.5
        result = self.run_case()
        bought = int(result.trades[(result.trades.side=='buy') & (result.trades.ts_code==self.codes[0])].shares.sum())
        sold = result.trades[(result.trades.side=='sell') & (result.trades.ts_code==self.codes[0])]
        self.assertGreater(len(sold), 0)
        self.assertLessEqual(sold.loc[sold.date.lt(self.dates[5]), 'shares'].sum(), bought)
        self.assertEqual(sold.shares.sum(), bought * 2)
        np.testing.assert_allclose(result.daily.equity, self.cfg['capital'], atol=1e-8)

    def test_announced_next_day_st_blocks_order_before_effective_day(self):
        self.st = pd.DataFrame([dict(ts_code=self.codes[0], start=self.dates[1],
            end=pd.NaT, known_date=self.dates[0])])
        result = self.run_case()
        self.assertEqual(len(self.planned(result, self.dates[0], 'buy')), 0)
        self.assertTrue(result.trades.empty)

    def test_same_day_st_announcement_time_ambiguity_stops_path(self):
        self.st = pd.DataFrame([dict(ts_code=self.codes[0], start=self.dates[1],
            end=pd.NaT, known_date=self.dates[1])])
        with self.assertRaises(AccountingError):
            self.run_case()

    def test_future_delist_without_announcement_cannot_rewrite_prior_order(self):
        ordinary = self.run_case()
        self.basics.loc[self.basics.ts_code.eq(self.codes[0]), 'delist_date'] = self.dates[1]
        changed = self.run_case()
        pd.testing.assert_frame_equal(self.planned(ordinary, self.dates[0], 'buy'),
                                     self.planned(changed, self.dates[0], 'buy'))
        self.assertTrue(changed.trades.empty)

    def test_known_delist_blocks_order_for_effective_execution_day(self):
        mask = self.basics.ts_code.eq(self.codes[0])
        self.basics.loc[mask, 'delist_date'] = self.dates[1]
        self.basics['delist_known_date'] = pd.NaT
        self.basics.loc[mask, 'delist_known_date'] = self.dates[0]
        result = self.run_case()
        self.assertEqual(len(self.planned(result, self.dates[0], 'buy')), 0)
        self.assertTrue(result.trades.empty)

    def test_latest_of_ann_date_and_known_date_governs_st_knowledge(self):
        self.st = pd.DataFrame([dict(ts_code=self.codes[0], start=self.dates[2], end=pd.NaT,
            ann_date=self.dates[0], known_date=self.dates[5])])
        result = self.run_case()
        sales = result.trades[result.trades.side.eq('sell')]
        self.assertGreater(len(sales), 0)
        self.assertTrue((sales.date > self.dates[5]).all())

    def test_st_effective_date_cannot_override_later_known_date(self):
        self.st = pd.DataFrame([dict(ts_code=self.codes[0], start=self.dates[2],
            end=pd.NaT, known_date=self.dates[5])])
        result = self.run_case()
        sales = result.trades[result.trades.side.eq('sell')]
        self.assertGreater(len(sales), 0)
        self.assertTrue((sales.date > self.dates[5]).all())
        orders = self.planned(result, side='sell')
        self.assertTrue((pd.to_datetime(orders.decision_date) >= self.dates[5]).all())

    def test_missing_corporate_action_knowledge_cannot_produce_certified_path(self):
        self.actions = normalize_actions(pd.DataFrame([dict(ts_code=self.codes[0], div_proc='实施',
            record_date=self.dates[0], ex_date=self.dates[1], pay_date=self.dates[1],
            div_listdate=self.dates[1], cash_div_tax=1., stk_div=0.)]))
        with self.assertRaises(AccountingError):
            self.run_case()

    def test_action_first_known_on_execution_day_stops_ambiguous_cancellation(self):
        self.actions = normalize_actions(pd.DataFrame([dict(ts_code=self.codes[0], div_proc='实施',
            record_date=self.dates[1], ex_date=self.dates[1], pay_date=self.dates[1],
            div_listdate=self.dates[1], cash_div_tax=1., stk_div=0.)]))
        self.actions['known_date'] = self.dates[1]
        with self.assertRaises(AccountingError):
            self.run_case()

    def test_no_dividend_cash_is_spent_before_actual_payment(self):
        self.actions = normalize_actions(pd.DataFrame([dict(ts_code=self.codes[0], div_proc='实施',
            record_date=self.dates[1], ex_date=self.dates[2], pay_date=self.dates[6],
            div_listdate=self.dates[2], cash_div_tax=1., stk_div=0.)]))
        self.actions['known_date'] = self.dates[0]
        mask = self.market.ts_code.eq(self.codes[0]) & self.market.date.ge(self.dates[2])
        self.market.loc[mask, ['open', 'high', 'low', 'close', 'pre_close']] = 9.
        self.market.loc[mask, 'adj_factor'] = 10 / 9
        self.limits.loc[self.limits.ts_code.eq(self.codes[0]) & self.limits.date.ge(self.dates[2]), 'down_limit'] = 8.1
        self.add_review(2, preferred=1)
        self.retail['max_positions'] = self.cfg['top_n'] = 2
        result = self.run_case()
        initial = result.trades[(result.trades.side=='buy') & (result.trades.ts_code==self.codes[0])].iloc[0]
        self.assertEqual(result.daily.loc[self.dates[2], 'receivables'], initial.shares)
        self.assertEqual(len(self.planned(result, self.dates[2], 'buy')), 0)
        self.assertEqual(result.daily.loc[self.dates[5], 'cash'], self.cfg['capital'] - initial.gross)
        self.assertEqual(result.daily.loc[self.dates[6], 'receivables'], 0.)
        np.testing.assert_allclose(result.daily.equity, self.cfg['capital'], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
