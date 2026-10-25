"""现金状态的未知数据、冻结篮子、次日执行和总预算约束。"""
import unittest
from copy import deepcopy
import numpy as np
import pandas as pd
import portfolio_cash_engine as engine
from portfolio_cash_state import CashState, basket_features
import test_causal_portfolio_adversarial as fixtures


class CashStateTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2024-01-01',periods=25)
        self.f={k:pd.DataFrame(v,index=self.dates,columns=['A','B']) for k,v in
                [('known',True),('weak',True),('drawdown',.06),('ret20',-.02)]}

    def test_freeze_and_two_week_recovery_even_when_empty(self):
        state=CashState(self.f,'staged')
        self.assertEqual(state.on_close(self.dates[0],{'A':60.,'B':40.},100.),(.5,True))
        self.f['drawdown'].loc[self.dates[5]:]=.12
        self.assertEqual(state.on_close(self.dates[5],{},100.),(0.,True))
        self.f['weak'].loc[self.dates[10]:]=False;self.f['ret20'].loc[self.dates[10]:]=.03
        self.assertEqual(state.on_close(self.dates[10],{},100.),(0.,True))
        self.assertEqual(state.on_close(self.dates[15],{},100.),(1.,False))

    def test_unknown_interrupts_recovery_and_cannot_count_as_safe(self):
        state=CashState(self.f,'full');state.on_close(self.dates[0],{'A':100.},100.)
        self.f['weak'].loc[:]=False;self.f['ret20'].loc[:]=.03
        state.on_close(self.dates[5],{},100.)
        self.f['known'].loc[self.dates[10],'A']=False
        self.assertEqual(state.on_close(self.dates[10],{},100.),(0.,True))
        self.assertEqual(state.on_close(self.dates[15],{},100.),(0.,True))
        self.assertEqual(state.on_close(self.dates[20],{},100.),(1.,False))

    def test_cash_or_new_basket_does_not_replace_original_cohort(self):
        state=CashState(self.f,'staged');state.on_close(self.dates[0],{'A':100.},100.)
        self.f['weak']['B']=False;self.f['ret20']['B']=.1
        state.on_close(self.dates[5],{'B':100.},100.)
        self.assertEqual(list(state.weights.index),['A'])
        self.assertEqual(state.cap,.5)

    def test_features_are_prefix_invariant(self):
        close=pd.DataFrame({'A':np.arange(100,220,dtype=float)},index=pd.bdate_range('2023-01-01',periods=120))
        full=basket_features(close);short=basket_features(close.iloc[:90])
        for k in full:pd.testing.assert_frame_equal(short[k],full[k].iloc[:90])


class CashEngineTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CausalPortfolioAdversarialTest();self.f.setUp()

    def run_case(self,overlay):
        f=self.f
        return engine.run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1],
            deepcopy(f.cfg),retail_forecasts=f.forecasts,retail_config=f.retail,hold_allowed=f.hold,cash_overlay=overlay)

    def test_fixed_half_buy_budget_is_real_total_budget(self):
        features={'known':pd.DataFrame(True,index=self.f.dates,columns=self.f.codes)}
        result=self.run_case(CashState(features,'fixed_half'))
        self.assertLessEqual(result.daily.stock_value.max(),5000.)
        self.assertGreaterEqual(result.daily.cash.min(),5000.)

    def test_liquidation_is_next_day_and_does_not_rebuy(self):
        dates=self.f.dates
        class Overlay:
            weekly=set(dates)
            def on_close(self,date,values,equity): return (0.,True) if date>=dates[3] else (1.,False)
        result=self.run_case(Overlay())
        sales=result.trades.loc[result.trades.side.eq('sell')]
        self.assertEqual(sales.date.iloc[0],dates[4])
        self.assertEqual(sales.decision_date.iloc[0],dates[3])
        self.assertTrue(result.daily.loc[dates[4]:].n_hold.eq(0).all())
        self.assertEqual(len(result.trades.loc[result.trades.side.eq('buy')]),1)


if __name__=='__main__':unittest.main()
