"""固定警报三动作的输入、资金和事件日历边界测试。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_policy_cash_execution import execution_cases, action_inputs, portfolio_config, common_alarms


class PolicyCashExecutionTest(unittest.TestCase):
    def test_sixteen_registered_cases(self):
        cfg = dict(capitals=[20000., 50000., 100000.], cash_actions=['none', 'entry_only', 'liquidate'],
            reference_capital=50000., cash_stress_cost_multiplier=2., cash_fixed_reserve_fraction=.5, cash_cases=16)
        cases = execution_cases(cfg)
        self.assertEqual(len(cases), 16)
        self.assertEqual(sum(c['reverse_input'] for c in cases), 3)
        self.assertEqual(sum(c['cost_multiplier'] == 2 for c in cases), 3)
        self.assertEqual(sum(c['reserve_fraction'] == .5 for c in cases), 1)
        self.assertEqual(cases[-1]['action'], 'none')

    def test_entry_veto_preserves_forecast_and_original_hold(self):
        dates = pd.to_datetime(['2022-01-04', '2022-02-07', '2022-03-01'])
        signals = pd.DataFrame({'000001.SZ': [1., 2., 3.]}, index=dates)
        points = signals / 100
        hold = pd.DataFrame({'000001.SZ': [True, np.nan, False]}, index=dates)
        risk = pd.Series([1, 0, 1], index=dates)
        chosen, after_hold, after_points = action_inputs(signals, hold, risk, 'entry_only', points)
        self.assertTrue(chosen.loc[dates[1]].isna().all())
        pd.testing.assert_frame_equal(chosen.drop(dates[1]), signals.drop(dates[1]))
        pd.testing.assert_frame_equal(after_hold, hold)
        pd.testing.assert_frame_equal(after_points, points)
        self.assertEqual(signals.loc[dates[1], '000001.SZ'], 2.)

    def test_original_confirmed_exit_added_to_entry_veto_is_redundant(self):
        hold = pd.Series([True, False, np.nan, False, True, np.nan])
        alarm = pd.Series([True, True, True, False, False, False])
        original_requested = hold.eq(False)
        original_plus_alarm_confirmation = original_requested | (alarm & original_requested)
        pd.testing.assert_series_equal(original_plus_alarm_confirmation, original_requested)
        self.assertFalse(original_plus_alarm_confirmation.loc[2])

    def test_cost_stress_and_fixed_reserve_affect_real_trading_capital(self):
        base = dict(capital=50000., buy_cost=.0002, sell_cost=.0012, min_fee=5.)
        case = dict(capital=50000., reserve_fraction=.5, cost_multiplier=2.)
        result = portfolio_config(base, case)
        self.assertEqual(result['capital'], 25000.)
        self.assertEqual(result['buy_cost'], .0004)
        self.assertEqual(result['sell_cost'], .0024)
        self.assertEqual(result['min_fee'], 10.)
        self.assertEqual(base['capital'], 50000.)

    def test_alarm_and_recovery_use_next_session_not_same_close(self):
        dates = pd.bdate_range('2022-01-03', periods=7)
        risk = pd.Series([1, 0, 0, 1, 1, 0, 1], index=dates)
        alarms = common_alarms(dates, risk)
        self.assertEqual(list(alarms.request_date), [dates[2], dates[6]])
        self.assertEqual(list(alarms.signal_date), [dates[1], dates[5]])
        self.assertEqual(alarms.iloc[0].recovery_execution_date, dates[4])
        self.assertTrue(pd.isna(alarms.iloc[1].recovery_execution_date))


if __name__ == '__main__':
    unittest.main()
