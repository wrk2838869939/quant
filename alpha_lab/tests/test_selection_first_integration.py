"""验证真实账本按显式排序选名，并保持旧仓与输入重排一致性。"""
import unittest
import pandas as pd
import test_patient_execution as fixtures


class SelectionFirstIntegrationTest(unittest.TestCase):
    def case(self, reverse=False, selection=True):
        f=fixtures.PatientExecutionTest();f.setUp()
        f.retail['selection_first']=selection
        f.forecasts.iloc[:]=[.02,.5]
        if reverse:
            columns=list(reversed(f.codes))
            f.signals=f.signals[columns];f.forecasts=f.forecasts[columns];f.hold=f.hold[columns]
            f.market=f.market.iloc[::-1].copy();f.basics=f.basics.iloc[::-1].copy()
        return f,f.run_case()

    def test_explicit_priority_defines_support_before_quantity_objective(self):
        _,r=self.case()
        self.assertEqual(set(r.trades.ts_code),{'000001.SZ'})
        choices=r.events[r.events.kind.eq('selection_decision')]
        self.assertTrue(choices.reason.eq('selected').any())
        self.assertEqual(len(r.trades),1)

    def test_reordered_signals_and_market_preserve_transactions(self):
        _,a=self.case();_,b=self.case(reverse=True)
        pd.testing.assert_frame_equal(a.trades,b.trades)
        pd.testing.assert_frame_equal(a.daily,b.daily)

    def test_default_objective_retains_previous_stock_choice(self):
        _,r=self.case(selection=False)
        self.assertEqual(set(r.trades.ts_code),{'000002.SZ'})
        self.assertFalse(r.events.kind.eq('selection_decision').any())

    def test_selection_requires_explicit_holding_calendar(self):
        f=fixtures.PatientExecutionTest();f.setUp();f.retail['selection_first']=True
        with self.assertRaises(ValueError):f.run_case(hold=False)


if __name__=='__main__':unittest.main()
