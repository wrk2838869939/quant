"""缺月不能变成现金；有预测且门槛拒绝可以保留NaN分数。"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('coverage_guard', Path(__file__).resolve().parents[1] / 'scripts/validate_recent_input_coverage.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CoverageGuardTest(unittest.TestCase):
    def frames(self):
        dates = pd.to_datetime(['2026-01-05', '2026-02-02'])
        hold = pd.DataFrame(1., index=pd.DatetimeIndex(dates, name='date'), columns=pd.Index(['000001.SZ'], name='ts_code'))
        predictions = pd.DataFrame({'date': dates, 'ts_code': ['000001.SZ'] * 2,
            'low_turnover__score': [np.nan, np.nan], 'low_turnover__expected_return': [.03, .03]})
        features = predictions[['date', 'ts_code']].copy()
        signals = hold * np.nan
        forecasts = hold * .03
        return [features, predictions, hold, signals, forecasts]

    def test_valid_data_all_gates_reject_is_allowed(self):
        result = module.validate_coverage(*self.frames())
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(result['all_nan_signal_dates_with_observed_candidates']), 2)

    def test_missing_prediction_month_rejected_despite_reindex(self):
        args = self.frames()
        args[1] = args[1].iloc[:1]
        args[4].iloc[1] = np.nan
        with self.assertRaisesRegex(ValueError, 'predictions日期覆盖失败'):
            module.validate_coverage(*args)

    def test_missing_feature_month_rejected(self):
        args = self.frames()
        args[0] = args[0].iloc[:1]
        with self.assertRaisesRegex(ValueError, 'features日期覆盖失败'):
            module.validate_coverage(*args)

    def test_matrix_value_changed_rejected(self):
        args = self.frames()
        args[4].iloc[0, 0] = .04
        with self.assertRaises(AssertionError):
            module.validate_coverage(*args)


if __name__ == '__main__':
    unittest.main()
