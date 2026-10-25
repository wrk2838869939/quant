"""匹配对照不复用目标、不会重复买入对照、不能跨行业回退。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from style_matching import match_controls


class MatchingTest(unittest.TestCase):
    def test_exact_groups_and_no_replacement(self):
        u = pd.DataFrame(dict(board=["main"]*6, industry=["a"]*4+["b"]*2,
            log_mv=[1., 4., 1.1, 3.9, 10., 9.]), index=list("ABCDEF"))
        result = match_controls(u, ["A", "B", "E"], ["log_mv"], 1)
        self.assertEqual(result.control.tolist(), ["C", "D", "F"])
        self.assertEqual(result.control.nunique(), 3)
        pd.testing.assert_frame_equal(result, match_controls(u.iloc[::-1], ["A", "B", "E"], ["log_mv"], 1))

    def test_outcome_columns_cannot_change_matching(self):
        u = pd.DataFrame(dict(board=["main"]*4, industry=["a"]*4,
            log_mv=[1., 4., 1.1, 3.9], future_return=[1., 2., 3., 4.]), index=list("ABCD"))
        original = match_controls(u, ["A", "B"], ["log_mv"], 2)
        u["future_return"] = [-100., 1e6, 99., -99.]
        changed = match_controls(u, ["A", "B"], ["log_mv"], 2)
        pd.testing.assert_frame_equal(original, changed)

    def test_missing_metadata_and_empty_groups_are_explicit(self):
        u = pd.DataFrame(dict(board=["main"]*3, industry=[None, None, "b"],
                             log_mv=[1., 2., 3.]), index=list("ABC"))
        result = match_controls(u, ["A", "C"], ["log_mv"], 1)
        self.assertFalse(result.industry_known.iloc[0])
        self.assertEqual(result.status.iloc[1], "unmatched")
        self.assertTrue(pd.isna(result.control.iloc[1]))
        u.loc["A", "log_mv"] = np.nan
        self.assertEqual(match_controls(u, ["A"], ["log_mv"], 1).status.iloc[0], "unmatched")


if __name__ == "__main__":
    unittest.main()
