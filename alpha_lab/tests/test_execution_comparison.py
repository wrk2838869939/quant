"""两种执行时段的资金、T+1、行情时间和前视隔离。"""
from copy import deepcopy
import unittest
import pandas as pd
import numpy as np
import execution_comparison_engine as engine
import predicted_risk_budget_engine as parent
from test_targeted_restore import RestoreEngineTests


class ExecutionComparisonTests(unittest.TestCase):
    def setUp(self):
        source=RestoreEngineTests();source.setUp();self.source=source;self.f=source.f

    def run_case(self,mode='legacy_day',quote_price=10.,decision=None):
        f=self.f
        quotes=f.market[['date','ts_code']].copy();quotes['price']=quote_price;quotes['volume']=1e9
        quotes['timestamp']=quotes.date+pd.Timedelta(hours=15 if mode=='close_auction' else 9,minutes=0 if mode=='close_auction' else 31)
        return engine.run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1],
            deepcopy(f.cfg),retail_forecasts=f.forecasts,retail_config=f.retail,hold_allowed=f.hold,
            cash_overlay=self.source.overlay(),restore_enabled=True,execution_mode=mode,
            decision_market=f.market if decision is None else decision,execution_quotes=quotes)

    def test_default_reproduces_parent_four_ledgers(self):
        f=self.f
        old=parent.run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1],
            deepcopy(f.cfg),retail_forecasts=f.forecasts,retail_config=f.retail,hold_allowed=f.hold,
            cash_overlay=self.source.overlay(),restore_enabled=True)
        new=self.run_case()
        for key in ['daily','trades','holdings','events']:pd.testing.assert_frame_equal(getattr(old,key),getattr(new,key))

    def test_same_day_close_fill_nav_and_t1(self):
        result=self.run_case('close_auction');buys=result.trades[result.trades.side.eq('buy')]
        self.assertTrue(len(buys)>0)
        self.assertTrue(buys.date.eq(buys.decision_date).all())
        first=result.holdings[result.holdings.date.eq(self.f.dates[0])]
        self.assertTrue(first.available.eq(0).all())
        self.assertLess(result.daily.iloc[0].cash,self.f.cfg['capital'])
        np.testing.assert_allclose(result.daily.equity,result.daily.cash+result.daily.stock_value+result.daily.receivables)

    def test_later_execution_price_cannot_change_prior_fixed_orders(self):
        first=self.run_case('open_fixed',10.)
        second=self.run_case('open_fixed',10.5)
        date=self.f.dates[0]
        def orders(x):return x.events[x.events.kind.eq('order_planned')&x.events.date.eq(date)].reset_index(drop=True)
        pd.testing.assert_frame_equal(orders(first),orders(second))
        self.assertTrue((second.daily.cash>=0).all())
        planned=orders(second).set_index('order_id').shares
        self.assertTrue(all(r.shares==planned[r.order_id] for r in second.trades[second.trades.decision_date.eq(date)].itertuples()))

    def test_close_fill_changes_do_not_change_1430_orders(self):
        first=self.run_case('close_auction',10.)
        second=self.run_case('close_auction',10.1)
        def orders(x):return x.events[x.events.kind.eq('order_planned')&x.events.date.eq(self.f.dates[0])].reset_index(drop=True)
        pd.testing.assert_frame_equal(orders(first),orders(second))


if __name__=='__main__':unittest.main()
