"""恢复仅增加一次收盘冻结机会，不变成每日追买。"""
import unittest
import pandas as pd
import portfolio_cash_reentry_engine as engine
from test_causal_portfolio_adversarial import CausalPortfolioAdversarialTest as Fixture


class RecoveryEntryTests(unittest.TestCase):
    def test_one_recovery_entry_and_no_future_open_in_plan(self):
        fixture=Fixture();fixture.setUp();dates=fixture.dates
        class Overlay:
            weekly=set(dates)
            def on_close(self,date,values,equity):
                self.recovered_today=date==dates[6]
                return (0.,True) if dates[3]<=date<dates[6] else (1.,False)
        def run(market):
            return engine.run_backtest(fixture.signals,market,fixture.basics,fixture.st,fixture.limits,
                fixture.actions,dates[0],dates[-1],fixture.cfg,retail_forecasts=fixture.forecasts,
                retail_config=fixture.retail,hold_allowed=fixture.hold,cash_overlay=Overlay())
        base=run(fixture.market)
        self.assertEqual(list(base.trades.loc[base.trades.side.eq('buy'),'date']),[dates[1],dates[7]])
        orders=base.events.loc[base.events.kind.eq('order_planned') & base.events.decision_date.eq(dates[6])]
        self.assertTrue(orders.reason.eq('recovery_entry').all())
        changed=fixture.market.copy();changed.loc[changed.date.eq(dates[7]),'open']=10.5
        altered=run(changed)
        other=altered.events.loc[altered.events.kind.eq('order_planned') & altered.events.decision_date.eq(dates[6])]
        pd.testing.assert_frame_equal(orders.reset_index(drop=True),other.reset_index(drop=True))


if __name__=='__main__':unittest.main()
