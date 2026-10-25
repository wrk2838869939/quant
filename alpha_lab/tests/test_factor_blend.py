"""合成的极端值稳健性、轴对齐与缺失样本契约。"""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from factor_blend import blend_factors


class BlendTests(unittest.TestCase):
    def setUp(self):
        self.a = pd.DataFrame([[1., 2., 3.], [4., 2., 1.]], columns=list("abc"))

    def test_rank_invariant_to_outlier_scale_and_future_rows(self):
        factors = {"a": self.a, "b": self.a.iloc[:, ::-1].set_axis(list("abc"), axis=1)}
        expected = blend_factors(factors, {"a": 1., "b": -2.}, "rank_icir")
        changed = {k: v.copy() for k, v in factors.items()}
        changed["a"].iloc[0, 2] = 1e12
        changed["a"].iloc[1] = [300., 200., 100.]
        actual = blend_factors(changed, {"a": 1., "b": -2.}, "rank_icir")
        pd.testing.assert_series_equal(actual.iloc[0], expected.iloc[0])

    def test_duplicate_family_does_not_increase_weight(self):
        b = -self.a
        original = blend_factors({"a": self.a, "b": b}, {"a": 1., "b": 1.}, "family_rank", {"x": ["a"], "y": ["b"]})
        duplicate = blend_factors({"a": self.a, "aa": self.a, "b": b}, {"a": 1., "aa": 1., "b": 1.},
                                  "family_rank", {"x": ["a", "aa"], "y": ["b"]})
        pd.testing.assert_frame_equal(original, duplicate)

    def test_missing_and_invalid_axes_rejected(self):
        missing = self.a.copy()
        missing.iloc[0, 0] = np.nan
        result = blend_factors({"a": self.a, "b": missing}, {"a": 1., "b": 1.}, "rank_icir")
        self.assertTrue(np.isnan(result.iloc[0, 0]))
        with self.assertRaises(ValueError):
            blend_factors({"a": self.a, "b": self.a.iloc[::-1]}, {"a": 1., "b": 1.}, "rank_icir")
        with self.assertRaises(ValueError):
            blend_factors({"a": self.a}, {"a": 1.}, "family_rank", {"x": ["a", "a"]})


if __name__ == "__main__":
    unittest.main()
