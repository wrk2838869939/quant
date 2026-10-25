"""风格替换控制的时间边界与不充分匹配检查。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_style_controls import make_control


class StyleControlsTest(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/baseline_style_controls_v1.yaml').read_text())['matching']
        self.days = pd.date_range('2021-01-01', periods=3)
        self.signals = pd.DataFrame([[6., 5., 4., 3., 2., 1.]] * 3, index=self.days, columns=list('abcdef'))
        self.schedule = pd.Series(2, index=self.days)
        rank = pd.DataFrame([[.1, .3, .12, .32, .7, .9]] * 3, index=self.days, columns=self.signals.columns)
        self.ranks = {k: rank.copy() for k in ['market_cap', 'volatility', 'earnings_yield', 'turnover']}

    def test_disjoint_unique_and_matching(self):
        scores, rows = make_control(self.signals, self.schedule, self.ranks, list(self.ranks), 1, self.cfg)
        self.assertTrue(scores[['a', 'b']].isna().all().all())
        self.assertTrue(scores.notna().sum(axis=1).eq(2).all())
        self.assertTrue(rows.passed.all())
        self.assertEqual(set(scores.iloc[0].dropna().index), {'c', 'd'})

    def test_prefix_unchanged_by_future_extremes(self):
        a, ra = make_control(self.signals.iloc[:2], self.schedule.iloc[:2], {k: v.iloc[:2] for k, v in self.ranks.items()}, list(self.ranks), 3, self.cfg)
        for value in self.ranks.values():
            value.iloc[2] = np.arange(6)[::-1] / 6
        b, rb = make_control(self.signals, self.schedule, self.ranks, list(self.ranks), 3, self.cfg)
        pd.testing.assert_frame_equal(a, b.iloc[:2])
        pd.testing.assert_frame_equal(ra, rb[rb.date <= self.days[1]])

    def test_no_common_support_is_reported(self):
        for value in self.ranks.values():
            value.loc[:, ['c', 'd', 'e', 'f']] = .99
        scores, rows = make_control(self.signals, self.schedule, self.ranks, list(self.ranks), 1, self.cfg)
        self.assertEqual(rows.passed.sum(), 0)
        self.assertTrue(scores.notna().sum(axis=1).eq(2).all())

    def test_missing_target_never_passes(self):
        self.ranks['market_cap'].loc[:, 'a'] = np.nan
        _, rows = make_control(self.signals, self.schedule, self.ranks, list(self.ranks), 1, self.cfg)
        self.assertFalse(rows.loc[rows.target == 'a', 'passed'].any())

    def test_shortage_records_unmatched_target(self):
        self.signals.loc[:, ['d', 'e', 'f']] = np.nan
        scores, rows = make_control(self.signals, self.schedule, self.ranks, list(self.ranks), 1, self.cfg)
        self.assertTrue(scores.notna().sum(axis=1).eq(1).all())
        self.assertEqual(rows.missing.sum(), len(self.days))

    def test_random_priority_is_stable(self):
        a, _ = make_control(self.signals, self.schedule, self.ranks, [], 1, self.cfg)
        pd.testing.assert_series_equal(a.iloc[0], a.iloc[2], check_names=False)


if __name__ == '__main__':
    unittest.main()
