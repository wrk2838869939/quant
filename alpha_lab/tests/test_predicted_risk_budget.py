"""风险预算与禁止交易分离后，验证额度、恢复时点和缺证据行为。"""
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
import predicted_risk_budget_engine as engine
from predicted_risk_budget_state import PredictedRiskBudget
import test_causal_portfolio_adversarial as fixtures


class PredictedBudgetTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CausalPortfolioAdversarialTest();self.f.setUp()
        self.f.cfg.update(capital=100000,top_n=5)
        self.f.retail.update(max_positions=5,max_stock_fraction=.2)
        self.f.signals.iloc[0,:5]=[5,4,3,2,1]
        self.features={k:pd.DataFrame(v,index=self.f.dates,columns=self.f.codes) for k,v in [('known',True),('weak',False),('ret20',.1)]}

    def state(self,prob):
        state=PredictedRiskBudget(self.features,pd.Series(prob,index=self.f.dates),.7,.3,.8)
        state.weekly=set(self.f.dates)
        return state

    def run_case(self,state):
        f=self.f
        return engine.run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,
            f.dates[0],f.dates[-1],deepcopy(f.cfg),retail_forecasts=f.forecasts,retail_config=f.retail,
            hold_allowed=f.hold,cash_overlay=state,restore_enabled=True)

    def test_high_risk_can_enter_but_cannot_exceed_budget(self):
        result=self.run_case(self.state(.9))
        self.assertGreater(result.daily.n_hold.max(),0)
        self.assertLessEqual((result.daily.stock_value/result.daily.equity).max(),.801)

    def test_only_risk_release_restores_old_holdings(self):
        prob=pd.Series(.1,index=self.f.dates)
        prob.iloc[3:6]=.9
        result=self.run_case(self.state(prob))
        plans=result.events.loc[result.events.kind.eq('targeted_restore_planned')]
        self.assertGreater(len(plans),0)
        self.assertTrue(plans.date.ge(self.f.dates[6]).all())
        self.assertTrue((plans.buy_shares<=plans.remaining_quota).all())
        self.assertFalse(result.events.kind.eq('cash_risk_intent_cancelled').any())

    def test_hysteresis_and_unknown_probability(self):
        state=self.state(.1)
        state.probabilities.iloc[:4]=[.8,.5,.2,np.nan]
        self.assertEqual(state.on_close(self.f.dates[0],{},100000),(.8,False))
        self.assertEqual(state.on_close(self.f.dates[1],{},100000),(.8,False))
        self.assertEqual(state.on_close(self.f.dates[2],{},100000),(1.,False))
        self.assertTrue(state.recovered_today)
        with self.assertRaises(ValueError):state.on_close(self.f.dates[3],{},100000)


if __name__=='__main__':unittest.main()
