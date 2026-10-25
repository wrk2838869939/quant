"""市场与原始个股共同证据仅改变已登记月度持有单元。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from run_policy_cash_confirm import confirmed_alarm_hold


class PolicyCashConfirmTest(unittest.TestCase):
    def test_alarm_and_known_failure_required_unknown_and_uncovered_preserved(self):
        dates = pd.date_range('2022-01-04', periods=3)
        original = pd.DataFrame({'A': [True, True, False], 'B': [np.nan, True, True], 'uncovered': [True, np.nan, False]}, index=dates)
        risk = pd.Series([0, 1, 0], index=dates)
        trend = np.array([[False, False], [False, False], [True, False]])
        known = np.array([[True, False], [True, True], [True, True]])
        result, evidence = confirmed_alarm_hold(original, risk, dates, ['A', 'B'], trend, known)
        self.assertFalse(result.loc[dates[0], 'A'])
        self.assertTrue(pd.isna(result.loc[dates[0], 'B']))
        self.assertTrue(result.loc[dates[1], 'A'])
        self.assertFalse(result.loc[dates[2], 'A'])
        self.assertFalse(result.loc[dates[2], 'B'])
        pd.testing.assert_series_equal(result.uncovered, original.uncovered, check_dtype=False)
        self.assertEqual(set(zip(evidence.date, evidence.ts_code)), {(dates[0], 'A'), (dates[2], 'B')})
        self.assertTrue(original.loc[dates[0], 'A'])

    def test_only_original_review_rows_can_change(self):
        dates = pd.bdate_range('2022-01-03', periods=5)
        original = pd.DataFrame({'A': [True, True]}, index=dates[[0, 4]])
        result, evidence = confirmed_alarm_hold(original, pd.Series(0, index=dates), dates, ['A'], np.zeros((5, 1), bool), np.ones((5, 1), bool))
        pd.testing.assert_index_equal(result.index, original.index)
        self.assertEqual(len(evidence), 2)

    def test_numeric_hold_parquet_roundtrip_after_boolean_override(self):
        from io import BytesIO
        dates = pd.date_range('2022-01-04', periods=3)
        original = pd.DataFrame({'A': [1., 0., np.nan]}, index=dates)
        result, _ = confirmed_alarm_hold(original, pd.Series([0, 0, 0], index=dates), dates,
            ['A'], np.zeros((3, 1), bool), np.array([[True], [True], [False]]))
        self.assertEqual(str(result.A.dtype), 'float64')
        buffer = BytesIO()
        result.to_parquet(buffer)
        pd.testing.assert_frame_equal(pd.read_parquet(buffer), result, check_freq=False)
        self.assertTrue(np.isnan(result.iloc[2, 0]))

    def test_test_boundary_rejected(self):
        dates = pd.to_datetime(['2024-01-02'])
        with self.assertRaises(AssertionError):
            confirmed_alarm_hold(pd.DataFrame({'A': [True]}, index=dates), pd.Series(0, index=dates), dates,
                                 ['A'], np.zeros((1,1), bool), np.ones((1,1), bool))


if __name__ == '__main__':
    unittest.main()
