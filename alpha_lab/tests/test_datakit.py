"""datakit 构建产物与运行时 API 的关键守卫测试。"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from datakit import api  # noqa: E402


class DatakitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "data/datakit/meta.json").exists():
            raise unittest.SkipTest("datakit 未构建")

    def test_window_guard(self):
        with self.assertRaises(ValueError):
            api.load("2024-01-01", "2024-06-30")

    def test_missing_day_patched(self):
        m, *_ = api.load("2021-12-01", "2021-12-20")
        self.assertIn(pd.Timestamp("2021-12-13"), set(pd.DatetimeIndex(m.date.unique())))

    def test_excluded_codes_invisible(self):
        exc = api.exclusions()
        self.assertGreater(len(exc), 0)
        m, *_ = api.load("2019-01-01", "2019-12-31")
        self.assertFalse(set(m.ts_code) & exc)

    def test_shenhua_multi_component_merged(self):
        a = pd.read_parquet(ROOT / "data/datakit/actions.parquet")
        row = a[(a.ts_code == "601088.SH") & (a.ex_date == pd.Timestamp("2017-07-10"))]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.cash_div_tax.iloc[0]), 2.97, places=6)

    def test_attrs_and_month_ends(self):
        me = api.month_ends("2022-01-01", "2022-12-31")
        self.assertEqual(len(me), 12)
        snap = api.attrs(me[:2])
        self.assertIn("log_mv", snap.columns)
        self.assertIn("industry", snap.columns)

    def test_investable_mask_shape(self):
        me = api.month_ends("2022-06-01", "2022-06-30")
        mask = api.investable(me, capital=210000.0, top_n=50)
        self.assertEqual(mask.shape[0], len(me))
        self.assertGreater(int(mask.to_numpy().sum()), 0)


if __name__ == "__main__":
    unittest.main()
