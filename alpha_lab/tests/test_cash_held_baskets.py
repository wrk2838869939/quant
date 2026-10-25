"""冻结篮子必须保留账户现金、缺价与摘牌，不重配后续组合。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('diagnose_cash_held_baskets', ROOT / 'scripts/diagnose_cash_held_baskets.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FrozenBasketTests(unittest.TestCase):
    def test_account_weights_keep_cash_and_mae_is_basket_path_minimum(self):
        prices = np.array([[100., 100.], [80., 120.], [120., 80.]])
        result = MODULE.frozen_price_path(prices, [np.inf, np.inf], 0, 2, [0.25, 0.5])
        # 25%现金零收益，不能将股票权重归一成1；两股票各自MAE不能先求和。
        np.testing.assert_allclose(result['total_return'], [0, 0.05, -0.05], atol=1e-15)
        np.testing.assert_allclose(result['mae'], [0, 0, -0.05], atol=1e-15)

    def test_missing_carries_and_delisting_never_revives(self):
        prices = np.array([[100.], [np.nan], [110.], [500.]])
        result = MODULE.frozen_price_path(prices, [2.], 0, 3, [0.4])
        np.testing.assert_allclose(result['total_return'], [0, 0, -0.4, -0.4])
        self.assertEqual(result['missing'].sum(), 1)
        np.testing.assert_array_equal(result['dead'][:, 0], [False, False, True, True])

    def test_baseline_uses_only_past_and_unknown_initial_mark_fails(self):
        prices = np.array([[100.], [np.nan], [120.]])
        result = MODULE.frozen_price_path(prices, [np.inf], 1, 2, [0.5])
        self.assertEqual(result['base'][0], 100)
        self.assertEqual(result['baseline_age'][0], 1)
        self.assertAlmostEqual(result['total_return'][-1], 0.1)
        with self.assertRaises(ValueError):
            MODULE.frozen_price_path(np.array([[np.nan], [120.]]), [np.inf], 0, 1, [0.5])

    def test_zero_holdings_preserve_zero_path(self):
        result = MODULE.frozen_price_path(np.empty((5, 0)), [], 1, 4, [])
        np.testing.assert_array_equal(result['total_return'], [0, 0, 0, 0])
        np.testing.assert_array_equal(result['mae'], [0, 0, 0, 0])

    def test_endpoint_conventions_and_unavailable_future_are_explicit(self):
        dates = pd.bdate_range('2023-01-02', periods=30)
        alarm = SimpleNamespace(signal_date=dates[0], recovery_execution_date=dates[10],
                                first_actual_buy_after_recovery=pd.NaT)
        endpoints = MODULE.endpoints(alarm, dates, [5, 10, 20])
        self.assertEqual(endpoints['alarm_next_5d'], dates[5])
        self.assertEqual(endpoints['recovery_signal_close'], dates[9])
        self.assertEqual(endpoints['recovery_next_5d'], dates[14])
        self.assertTrue(pd.isna(endpoints['first_actual_buy_after_recovery_close']))
        alarm.signal_date = dates[26]
        alarm.recovery_execution_date = pd.NaT
        endpoints = MODULE.endpoints(alarm, dates, [5, 10, 20])
        self.assertTrue(all(pd.isna(date) for date in endpoints.values()))


if __name__ == '__main__':
    unittest.main()
