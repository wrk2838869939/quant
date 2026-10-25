"""确认后收益的交易日、缺失和极值口径测试。"""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sector_confirmation_returns import window_metrics, cumulative_paths


class ReturnTests(unittest.TestCase):
    def test_extrema_and_endpoint(self):
        r = window_metrics([100, 110, 80, 105], 3)
        np.testing.assert_allclose([r['end_return'][0], r['max_return'][0], r['min_return'][0]], [.05, .1, -.2])
        self.assertEqual(r['max_day'][0], 1)
        self.assertEqual(r['min_day'][0], 2)

    def test_origin_is_not_extremum(self):
        self.assertAlmostEqual(window_metrics([100, 90, 80, 70], 3)['max_return'][0], -.1)
        self.assertAlmostEqual(window_metrics([100, 110, 120, 130], 3)['min_return'][0], .1)

    def test_gap_and_censor(self):
        self.assertFalse(window_metrics([100, 110, np.nan, 120], 3)['complete'][0])
        self.assertFalse(window_metrics([100, 110, 120], 3)['complete'][0])
        self.assertTrue(np.isnan(cumulative_paths([100, 110, np.nan, 120])[3, 0]))

    def test_horizons_and_assets_independent(self):
        values = np.array([[100, 100], [110, 90], [120, 80], [130, 70], [np.nan, 60]])
        self.assertTrue(window_metrics(values, 3)['complete'].all())
        np.testing.assert_array_equal(window_metrics(values, 4)['complete'], [False, True])


if __name__ == '__main__':
    unittest.main()
