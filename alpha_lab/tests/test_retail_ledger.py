import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import test_portfolio
from portfolio import run_backtest
from audit_monthly_portfolio import reconcile


class RetailLedgerTest(unittest.TestCase):
    def setUp(self):
        self.f=test_portfolio.PortfolioTest();self.f.setUp()
        self.f.signals.iloc[0]=[2.,1.]
        self.mu=self.f.signals.copy();self.mu.iloc[0]=[.1,.09]

    def replay(self,delay=0):
        f=self.f
        r=run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1],f.cfg,
            retail_forecasts=self.mu,retail_config=dict(max_stock_fraction=.6,candidate_pool=2),extra_execution_delay=delay)
        reconcile(r,f.cfg['capital']);return r

    def test_joint_cash_and_two_positions_ignore_legacy_top_n(self):
        r=self.replay()
        self.assertEqual(r.daily.iloc[1].n_hold,2)
        self.assertEqual(r.daily.iloc[1].cash,0)
        self.assertEqual(r.trades.shares.tolist(),[600,400])

    def test_rebalance_uses_existing_position_without_round_trip(self):
        self.f.signals.loc[self.f.dates[3]]=[2.,1.]
        self.mu.loc[self.f.dates[3]]=[.1,.09]
        r=self.replay()
        self.assertEqual(len(r.trades),2)

    def test_negative_forecasts_allow_cash(self):
        self.mu.iloc[0]=[-.01,-.01]
        r=self.replay()
        self.assertTrue(r.trades.empty)
        self.assertEqual(r.daily.iloc[-1].cash,10000)

    def test_extra_day_delay_applies_to_order(self):
        r=self.replay(delay=1)
        self.assertTrue(r.trades.date.eq(self.f.dates[2]).all())
        self.assertEqual(r.daily.iloc[1].n_hold,0)

    def test_reentry_uses_latest_known_daily_forecasts(self):
        f=self.f
        gate=pd.Series(1,index=f.dates);gate.iloc[2:5]=0
        daily=pd.DataFrame([[1.,2.]]*len(f.dates),index=f.dates,columns=f.codes)
        forecasts=daily.copy();forecasts.iloc[:]=[-.1,.1]
        # 执行日收盘出现相反信号，不能反过来改变当天开盘订单。
        forecasts.iloc[6:]=[.1,-.1]
        r=run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1],f.cfg,
            retail_forecasts=self.mu,retail_config=dict(max_stock_fraction=.6,candidate_pool=2),
            risk_allowed=gate,risk_signal_max_age=21,reentry_signals=daily,reentry_forecasts=forecasts)
        buys=r.trades[r.trades.side.eq('buy') & r.trades.date.eq(f.dates[6])]
        self.assertEqual(buys.ts_code.tolist(),[f.codes[1]])
        self.assertEqual(buys.signal_date.tolist(),[f.dates[5]])
        reconcile(r,f.cfg['capital'])

    def test_explicit_position_limit_reaches_joint_allocator(self):
        f=self.f
        r=run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1],f.cfg,
            retail_forecasts=self.mu,retail_config=dict(max_stock_fraction=1.,candidate_pool=2,max_positions=1))
        self.assertEqual(r.daily.n_hold.max(),1)
        self.assertEqual(r.trades.ts_code.tolist(),[f.codes[0]])
        reconcile(r,f.cfg['capital'])

    def test_risk_exit_daily_but_reentry_waits_for_scheduled_review(self):
        f=self.f
        f.signals.loc[f.dates[10]]=[2.,1.];self.mu.loc[f.dates[10]]=[.1,.09]
        gate=pd.Series(1,index=f.dates);gate.iloc[2:5]=0
        r=run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1],f.cfg,
            retail_forecasts=self.mu,retail_config=dict(max_stock_fraction=.6,candidate_pool=2,max_positions=2),
            risk_allowed=gate,risk_signal_max_age=21,reentry_on_schedule_only=True)
        self.assertTrue(r.daily.n_hold.iloc[3:11].eq(0).all())
        self.assertTrue(r.trades.loc[r.trades.side.eq('sell'),'date'].eq(f.dates[3]).all())
        later=r.trades[r.trades.side.eq('buy')&r.trades.date.gt(f.dates[1])]
        self.assertEqual(later.date.unique().tolist(),[f.dates[11]])
        reconcile(r,f.cfg['capital'])


if __name__=='__main__':unittest.main()
