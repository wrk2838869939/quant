"""用独立现金流和已知路径验证回测账本，而非复述实现。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))
from portfolio import run_backtest, daily_metrics, normalize_actions, AccountingError, Position
from audit_monthly_portfolio import reconcile


class PortfolioTest(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / "configs/monthly_portfolio_audit.yaml").read_text())["portfolio"]
        self.cfg.update(capital=10000., top_n=1, buy_cost=0., sell_cost=0., min_fee=0.,
                        cash_dividend_haircut=0., max_volume_participation=1.)
        self.dates = pd.bdate_range("2022-01-04", periods=25)
        self.codes = ["000001.SZ", "000002.SZ"]
        self.market = pd.DataFrame([dict(date=d, ts_code=c, open=10., close=10., pre_close=10.,
                                        adj_factor=1., vol_share=1e7)
                                    for d in self.dates for c in self.codes])
        self.limits = self.market[["date", "ts_code"]].assign(up_limit=11., down_limit=9.)
        self.basics = pd.DataFrame(dict(ts_code=self.codes,
            list_date=[pd.Timestamp("2000-01-01")] * 2, delist_date=[pd.NaT] * 2))
        self.st = pd.DataFrame({"ts_code": pd.Series(dtype=str), "start": pd.Series(dtype="datetime64[ns]"),
                                "end": pd.Series(dtype="datetime64[ns]")})
        self.actions = normalize_actions(pd.DataFrame())
        self.signals = pd.DataFrame([[1., np.nan]], index=self.dates[:1], columns=self.codes)

    def run_case(self, end=None, cfg=None):
        result = run_backtest(self.signals, self.market, self.basics, self.st, self.limits,
                             self.actions, self.dates[0], end or self.dates[-1], cfg or self.cfg)
        reconcile(result, (cfg or self.cfg)["capital"])
        return result

    def edit(self, day, code=None, **values):
        mask = (self.market.date == self.dates[day]) & (self.market.ts_code == (code or self.codes[0]))
        for key, value in values.items():
            self.market.loc[mask, key] = value
        if "close" in values:
            # 合成普通交易日的前收随前一有效收盘同步；显式除权价由用例覆盖。
            selected = self.market.ts_code == (code or self.codes[0])
            prior = self.market.loc[selected, "close"].ffill().shift(1).fillna(10.)
            self.market.loc[selected, "pre_close"] = prior
            if "pre_close" in values:
                self.market.loc[mask, "pre_close"] = values["pre_close"]

    def action(self, cash=1., bonus=0., pay=5, release=4):
        self.actions = normalize_actions(pd.DataFrame([dict(ts_code=self.codes[0], div_proc="实施",
            record_date=self.dates[1], ex_date=self.dates[2], pay_date=self.dates[pay],
            div_listdate=self.dates[release], cash_div_tax=cash, stk_div=bonus)]))
        reference = (10 - cash) / (1 + bonus)
        mask = (self.market.date >= self.dates[2]) & (self.market.ts_code == self.codes[0])
        self.market.loc[mask, ["open", "close", "pre_close"]] = reference
        self.market.loc[mask, "adj_factor"] = 10 / reference
        limmask = (self.limits.date >= self.dates[2]) & (self.limits.ts_code == self.codes[0])
        self.limits.loc[limmask, "up_limit"] = reference * 1.1
        self.limits.loc[limmask, "down_limit"] = reference * 0.9

    def test_intramonth_drawdown_and_end_mark(self):
        self.edit(10, open=5., close=5.)
        self.edit(24, close=12.)
        result = self.run_case()
        self.assertAlmostEqual(result.summary["mdd"], -0.5)
        self.assertAlmostEqual(result.summary["total"], 0.2)
        self.assertEqual(len(result.daily), 25)
        self.assertAlmostEqual(result.daily.iloc[-1].equity, 12000.)

    def test_differential_dividend_reference_preserves_actual_entitlement(self):
        raw = pd.DataFrame([dict(ts_code=self.codes[0], div_proc="实施",
            record_date=self.dates[1], ex_date=self.dates[2], pay_date=self.dates[3],
            div_listdate=self.dates[4], cash_div_tax=1., stk_div=.4)])
        rule = dict(ts_code=self.codes[0], ex_date=str(self.dates[2].date()),
                    ann_date=str(self.dates[0].date()), total_shares=10000, participating_shares=9000,
                    cash_div_tax=1., stk_div=.4, evidence="synthetic issuer announcement")
        self.actions = normalize_actions(raw, reference_events=[rule])
        reference = 9.1 / 1.36
        mask = self.market.ts_code.eq(self.codes[0]) & (self.market.date >= self.dates[2])
        self.market.loc[mask, ["open", "close", "pre_close"]] = reference
        self.market.loc[mask, "adj_factor"] = 10 / reference
        result = self.run_case()
        self.assertAlmostEqual(result.daily.iloc[-1].cash, 1000.)
        self.assertEqual(result.holdings.iloc[-1].shares, 1400)
        self.assertAlmostEqual(result.daily.iloc[-1].equity, 1000. + 1400 * reference)
        locked = result.holdings[result.holdings.date.eq(self.dates[2])].iloc[0]
        self.assertEqual(locked.available, 1000)
        with self.assertRaises(AccountingError):
            normalize_actions(raw, reference_events=[dict(rule, cash_div_tax=.9)])

    def test_002972_issuer_reference_preserves_actual_dividend(self):
        cfg = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())
        raw = pd.DataFrame([dict(ts_code="002972.SZ", div_proc="实施", record_date="20230524",
            ex_date="20230525", pay_date="20230525", div_listdate="20230525",
            cash_div_tax=.6, stk_div=.4)])
        normalized = normalize_actions(raw, reference_events=cfg["reference_adjustments"])
        action = normalized.iloc[0]
        reference = (18.60 - action.cash_div_tax * action.reference_share_fraction) / (
            1 + action.stk_div * action.reference_share_fraction)
        self.assertAlmostEqual(reference, 12.91, delta=.005)
        self.assertEqual(action.cash_div_tax, .6)
        self.assertEqual(action.stk_div, .4)

    def test_daily_metrics_include_starting_capital(self):
        equity = pd.Series([9000., 9900., 10890., 9801.])
        summary, returns = daily_metrics(equity, 10000., 244)
        np.testing.assert_allclose(returns, [-0.1, 0.1, 0.1, -0.1])
        self.assertAlmostEqual(summary["mdd"], -0.1)
        self.assertAlmostEqual(summary["sharpe"], 0., places=10)
        self.assertAlmostEqual(summary["ann"], (9801 / 10000) ** (244 / 4) - 1)

    def test_lots_minimum_fees_and_cash_conservation(self):
        self.cfg.update(capital=1000., buy_cost=0.001, min_fee=5.)
        result = self.run_case()
        self.assertTrue(result.trades.empty)  # 100 股还需手续费，不能透支
        self.cfg["capital"] = 1010.
        result = self.run_case()
        self.assertEqual(result.trades.iloc[0].shares, 100)
        self.assertEqual(result.trades.iloc[0].fee, 5)
        self.assertEqual(result.daily.iloc[-1].cash, 5)
        np.testing.assert_allclose(result.daily.equity,
            result.daily.cash + result.daily.stock_value + result.daily.receivables)

    def test_star_minimum_lot(self):
        for frame in (self.market, self.limits, self.basics):
            frame.loc[frame.ts_code == self.codes[0], "ts_code"] = "688001.SH"
        self.signals.columns = ["688001.SH", self.codes[1]]
        self.cfg["capital"] = 1999.
        self.assertTrue(self.run_case().trades.empty)
        self.cfg["capital"] = 2009.
        self.assertEqual(self.run_case().trades.iloc[0].shares, 200)
        self.cfg["capital"] = 2010.
        self.assertEqual(self.run_case().trades.iloc[0].shares, 201)

    def test_limit_down_sell_retried_next_day(self):
        self.signals.loc[self.dates[2]] = [np.nan, 1.]
        self.edit(3, open=9., close=9.)
        result = self.run_case()
        sale = result.trades[result.trades.side == "sell"].iloc[0]
        self.assertEqual(sale.date, self.dates[4])
        self.assertEqual(sale.signal_date, self.dates[2])

    def test_suspension_uses_last_mark_not_entry(self):
        self.edit(2, close=12.)
        self.edit(3, open=np.nan, close=np.nan, adj_factor=np.nan, vol_share=0.)
        result = self.run_case()
        self.assertEqual(result.daily.loc[self.dates[3], "equity"], 12000.)

    def test_delisting_does_not_invent_last_price_sale(self):
        self.basics.loc[0, "delist_date"] = self.dates[4]
        result = self.run_case()
        self.assertEqual(result.daily.loc[self.dates[4], "equity"], 0.)
        self.assertEqual(result.summary["mdd"], -1.)
        self.assertEqual(result.summary["delist_events"], 1)
        self.assertEqual(len(result.trades), 1)

    def test_dividend_receivable_survives_sale_until_payment(self):
        self.action(cash=1., pay=5)
        self.signals.loc[self.dates[1]] = [np.nan, np.nan]
        result = self.run_case()
        self.assertAlmostEqual(result.daily.loc[self.dates[2], "cash"], 9000.)
        self.assertAlmostEqual(result.daily.loc[self.dates[2], "receivables"], 1000.)
        self.assertAlmostEqual(result.daily.loc[self.dates[5], "cash"], 10000.)
        np.testing.assert_allclose(result.daily.equity, 10000.)

    def test_bonus_locked_until_listing(self):
        self.action(cash=0., bonus=1., release=4)
        self.signals.loc[self.dates[1]] = [np.nan, np.nan]
        result = self.run_case()
        sales = result.trades[result.trades.side == "sell"]
        self.assertEqual(list(sales.shares), [1000, 1000])
        self.assertEqual(list(sales.date), [self.dates[2], self.dates[4]])
        np.testing.assert_allclose(result.daily.equity, 10000.)

    def test_unexplained_adjustment_stops(self):
        self.edit(2, adj_factor=2., pre_close=5., open=5., close=5.)
        with self.assertRaises(AccountingError):
            self.run_case()

    def test_metadata_adjustment_does_not_create_shares_or_cash(self):
        self.edit(2, adj_factor=1.0002)
        result = self.run_case()
        np.testing.assert_allclose(result.daily.equity, 10000.)
        self.assertEqual(set(result.holdings.shares), {1000})
        self.assertGreater(result.summary["adjustment_metadata_events"], 0)

    def test_unexplained_reference_gap_without_factor_change_stops(self):
        self.edit(2, pre_close=5., open=5., close=5.)
        with self.assertRaises(AccountingError):
            self.run_case()

    def test_bad_dividend_units_stop(self):
        self.action()
        self.actions.loc[0, "cash_div_tax"] = 10.
        with self.assertRaises(AccountingError):
            self.run_case()

    def test_missing_limits_reject_trade(self):
        self.limits = self.limits.iloc[:0]
        result = self.run_case()
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.summary["missing_limit_events"], 1)

    def test_initial_fee_drawdown_and_no_same_day_signal_fill(self):
        self.cfg.update(buy_cost=0.01)
        result = self.run_case()
        self.assertEqual(result.trades.iloc[0].date, self.dates[1])
        self.assertEqual(result.daily.iloc[0].equity, 10000.)
        self.assertLess(result.summary["mdd"], 0.)
        buy_day = result.holdings[result.holdings.date == self.dates[1]].iloc[0]
        self.assertEqual(buy_day.available, 0)

    def test_future_market_rows_do_not_change_past_ledger(self):
        full = self.run_case()
        short = self.run_case(end=self.dates[12])
        pd.testing.assert_frame_equal(full.daily.iloc[:13], short.daily)

    def test_st_stock_exits_even_inside_keep_band(self):
        self.st = pd.DataFrame([dict(ts_code=self.codes[0], start=self.dates[3], end=pd.NaT)])
        sale = self.run_case().trades.query("side == 'sell'").iloc[0]
        self.assertEqual(sale.date, self.dates[3])

    def test_cash_dividend_not_reinvested_before_payment(self):
        self.action(cash=1., pay=5)
        self.signals.loc[self.dates[1]] = [np.nan, 1.]
        result = self.run_case()
        purchase = result.trades[(result.trades.ts_code == self.codes[1]) & (result.trades.side == "buy")].iloc[0]
        self.assertEqual(purchase.shares, 900)
        self.assertEqual(result.daily.loc[self.dates[2], "receivables"], 1000.)

    def test_price_limit_blocks_buy_without_chasing_lower_rank(self):
        self.edit(1, open=11., close=11.)
        self.signals.iloc[0] = [2., 1.]
        result = self.run_case()
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.daily.iloc[-1].cash, 10000.)

    def test_two_sided_cashflows_match_manual_ledger(self):
        self.cfg.update(buy_cost=0.001, sell_cost=0.002, min_fee=5.)
        self.signals.loc[self.dates[2]] = [np.nan, np.nan]
        result = self.run_case()
        # 可买900股：买9000+9，卖9000-18，净资产9973。
        self.assertEqual(list(result.trades.shares), [900, 900])
        self.assertAlmostEqual(result.daily.iloc[-1].equity, 9973.)
        self.assertAlmostEqual(result.summary["total_fees"], 27.)

    def test_only_verified_dividend_components_can_merge(self):
        self.action()
        raw = pd.concat([self.actions.assign(cash_div_tax=0.7),
                         self.actions.assign(cash_div_tax=0.3)], ignore_index=True)
        raw["div_proc"] = "实施"
        with self.assertRaises(AccountingError):
            normalize_actions(raw)
        rule = dict(ts_code=self.codes[0], ex_date=str(self.dates[2].date()),
                    component_count=2, cash_div_tax=1., stk_div=0.)
        self.actions = normalize_actions(raw, [rule])
        self.assertEqual(len(self.actions), 1)
        np.testing.assert_allclose(self.run_case().daily.equity, 10000.)
        raw.loc[0, "cash_div_tax"] = 0.8
        with self.assertRaises(AccountingError):
            normalize_actions(raw, [rule])

    def test_declining_rights_keeps_shares_and_bears_price_loss(self):
        # 每股配1股、配股价2元：10元除权到6元，不参与者仍持1000股。
        right = dict(ts_code=self.codes[0], ex_date=str(self.dates[2].date()),
                     ratio=1., subscription_price=2., policy="decline")
        mask = (self.market.ts_code == self.codes[0]) & (self.market.date >= self.dates[2])
        self.market.loc[mask, ["open", "close", "pre_close"]] = 6.
        self.market.loc[mask, "adj_factor"] = 10 / 6
        result = run_backtest(self.signals, self.market, self.basics, self.st, self.limits,
            self.actions, self.dates[0], self.dates[-1], self.cfg, [right])
        self.assertEqual(result.daily.iloc[-1].equity, 6000.)
        self.assertEqual(set(result.holdings.shares), {1000})
        self.assertEqual(result.daily.iloc[-1].cash, 0.)
        reconcile(result, self.cfg["capital"])


if __name__ == "__main__":
    unittest.main()
