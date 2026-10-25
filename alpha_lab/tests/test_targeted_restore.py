"""定向恢复额度和五席位、真实现金、T+1对抗验证。"""
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
from restore_quota import RestoreQuota
import targeted_restore_engine as engine
import test_causal_portfolio_adversarial as fixtures


class QuotaTests(unittest.TestCase):
    def test_actual_credit_duplicate_restart_partial_consumption(self):
        q=RestoreQuota();q.start_episode()
        q.fill('s1','A','sell',250,'portfolio_cash_cap',1)
        q.fill('s1','A','sell',250,'portfolio_cash_cap',1)
        q=RestoreQuota(q.snapshot());q.fill('b1','A','buy',100,'targeted_restore',1)
        self.assertEqual(q.available('A'),150)
        with self.assertRaises(ValueError):q.fill('b2','A','buy',200,'targeted_restore',1)
        self.assertEqual(q.available('A'),150)

    def test_other_sale_and_old_episode_give_no_credit(self):
        q=RestoreQuota();q.start_episode();q.fill('s1','A','sell',100,'hold_false',1)
        self.assertEqual(q.available('A'),0)
        q.fill('s2','A','sell',100,'portfolio_cash_cap',1);q.start_episode()
        q.fill('s3','A','sell',100,'portfolio_cash_cap',1)
        self.assertEqual(q.available('A'),0)
        with self.assertRaises(ValueError):q.fill('b1','A','buy',100,'targeted_restore',1)

    def test_invalid_quota_and_conflicting_receipt_rejected(self):
        with self.assertRaises(ValueError):RestoreQuota(dict(epoch=0,shares={'A':-100},receipts={}))
        q=RestoreQuota();q.start_episode();q.fill('s1','A','sell',100,'portfolio_cash_cap',1)
        with self.assertRaises(ValueError):q.fill('s1','A','sell',200,'portfolio_cash_cap',1)


class RestoreEngineTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CausalPortfolioAdversarialTest();self.f.setUp()
        self.f.cfg.update(capital=100000,top_n=5)
        self.f.retail.update(max_positions=5,max_stock_fraction=.2)
        self.f.signals.iloc[0,:5]=[5,4,3,2,1]

    def overlay(self,weak=False):
        dates=self.f.dates;codes=self.f.codes
        class Overlay:
            variant='staged';weekly=set(dates);weights=None
            features={k:pd.DataFrame(v,index=dates,columns=codes) for k,v in [('known',True),('weak',weak),('ret20',.1)]}
            def on_close(self,date,values,equity):
                self.recovered_today=date==dates[6]
                self.weights={'risk':1} if dates[3]<=date<dates[6] else None
                return (.5,True) if self.weights is not None else (1.,False)
        return Overlay()

    def run_case(self,weak=False,market=None):
        f=self.f
        return engine.run_backtest(f.signals,f.market if market is None else market,f.basics,f.st,f.limits,f.actions,
            f.dates[0],f.dates[-1],deepcopy(f.cfg),retail_forecasts=f.forecasts,retail_config=f.retail,
            hold_allowed=f.hold,cash_overlay=self.overlay(weak),restore_enabled=True)

    def test_five_slots_restore_without_replacing_old_shares_and_lock_only_new(self):
        result=self.run_case();f=self.f
        plans=result.events.loc[result.events.kind.eq('targeted_restore_planned')]
        self.assertEqual(len(plans),5)
        self.assertTrue((plans.buy_shares<=plans.remaining_quota).all())
        self.assertLessEqual(plans.reserved_cash.sum(),result.daily.at[f.dates[6],'cash'])
        self.assertTrue((plans.frozen_position_notional<=plans.risk_cap+1e-8).all())
        self.assertEqual(result.daily.at[f.dates[7],'n_hold'],5)
        self.assertGreater(result.daily.at[f.dates[7],'stock_value']/result.daily.at[f.dates[7],'equity'],.9)
        day=result.holdings.loc[result.holdings.date.eq(f.dates[7])]
        self.assertTrue((day.available<day.shares).all())
        following=result.holdings.loc[result.holdings.date.eq(f.dates[8])]
        self.assertTrue((following.available==following.shares).all())

    def test_unrecovered_stock_does_not_get_restored(self):
        result=self.run_case(weak=True)
        self.assertFalse(result.events.kind.eq('targeted_restore_planned').any())
        self.assertLessEqual(result.daily.stock_value.iloc[-1]/result.daily.equity.iloc[-1],.5)

    def test_future_open_cannot_change_restoration_orders(self):
        base=self.run_case();market=self.f.market.copy()
        market.loc[market.date.eq(self.f.dates[7]),'open']=10.5
        changed=self.run_case(market=market)
        def plans(result):return result.events.loc[result.events.kind.eq('targeted_restore_planned')].reset_index(drop=True)
        pd.testing.assert_frame_equal(plans(base),plans(changed))


if __name__=='__main__':unittest.main()
