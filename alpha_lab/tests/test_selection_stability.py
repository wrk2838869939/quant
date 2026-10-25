"""入选持续性和排名缓冲的时序边界检验。"""
import unittest
import numpy as np
import pandas as pd
from selection_stability import smooth_scores, buffered_holds


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2020-01-01', periods=5)

    def test_future_changes_do_not_change_past(self):
        x = pd.DataFrame({'a': [.4, .3, .2, .1, .0]}, index=self.dates)
        y = x.copy(); y.iloc[3:] = -.5
        pd.testing.assert_frame_equal(smooth_scores(x, 2).iloc[:3], smooth_scores(y, 2).iloc[:3])

    def test_missing_day_counts_as_zero_and_current_gate_stays(self):
        x = pd.DataFrame({'a': [.5, np.nan, .5, .5, np.nan]}, index=self.dates)
        y = smooth_scores(x, 2)
        self.assertEqual(y.iloc[2, 0], .5)
        self.assertTrue(pd.isna(y.iloc[4, 0]))
        self.assertTrue(pd.isna(y.iloc[0, 0]))

    def test_window_one_keeps_rank_order(self):
        x = pd.DataFrame([[.2, .4, np.nan]], index=self.dates[:1])
        pd.testing.assert_frame_equal(smooth_scores(x, 1), x + .5)

    def test_buffer_needs_two_reviews_and_unknown_resets(self):
        x = pd.DataFrame({'a': [5, 5, 5, 5, 5], 'b': [4, 4, 4, 4, 4],
                          'c': [1, 1, np.nan, 1, 1]}, index=self.dates)
        hold = x.copy() * 0 + 1; hold.loc[:, :] = 1.
        y, _, _ = buffered_holds(x, hold, 2, 2, 2)
        self.assertEqual(y.c.tolist(), [1., 0., 1., 1., 0.])

    def test_original_exit_and_unknown_are_not_overridden(self):
        x = pd.DataFrame({'a': [5]*5, 'b': [4]*5, 'c': [1]*5}, index=self.dates)
        hold = x.astype(float) * 0 + 1
        hold.loc[self.dates[0], 'a'] = 0
        hold.loc[self.dates[2], 'c'] = np.nan
        y, _, _ = buffered_holds(x, hold, 2, 2, 2)
        self.assertEqual(y.iloc[0, 0], 0.)
        self.assertTrue(pd.isna(y.loc[self.dates[2], 'c']))

    def test_ties_do_not_force_exit(self):
        x = pd.DataFrame({'a': [5]*5, 'b': [4]*5, 'c': [4]*5}, index=self.dates)
        hold = x.astype(float) * 0 + 1
        y, _, _ = buffered_holds(x, hold, 2, 2, 2)
        self.assertTrue(y.eq(1).all().all())

    def test_hold_prefix_is_identical(self):
        x = pd.DataFrame({'a': [5]*5, 'b': [4]*5, 'c': [1]*5}, index=self.dates)
        hold = x.astype(float) * 0 + 1
        full = buffered_holds(x, hold, 2, 2, 2)[0]
        short = buffered_holds(x.iloc[:3], hold.iloc[:3], 2, 2, 2)[0]
        pd.testing.assert_frame_equal(full.iloc[:3], short)

    def test_bad_dates_and_windows_rejected(self):
        x = pd.DataFrame({'a': [.4]*5}, index=self.dates)
        with self.assertRaises(ValueError): smooth_scores(x, 0)
        with self.assertRaises(ValueError): smooth_scores(x.iloc[::-1], 2)


if __name__ == '__main__': unittest.main()
