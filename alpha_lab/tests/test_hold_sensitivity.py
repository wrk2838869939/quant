"""持有确认敏感性独立回归：仅变退出证据，固定复核日且禁止未来截断污染。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from hold_sensitivity import build_hold_variants
from run_patient_value_execution import confirmed_hold_states
import selection_first


class HoldSensitivityTest(unittest.TestCase):
    def setUp(self):
        self.spec = yaml.safe_load((ROOT / 'configs/hold_sensitivity_research.yaml').read_text())
        self.dates = pd.bdate_range('2022-01-03', periods=160)
        self.reviews = self.dates[[20, 45, 70, 95, 120, 145]]
        self.trend = pd.DataFrame(True, index=self.dates, columns=['A', 'B'])
        self.known = self.trend.copy()

    def build(self, trend=None, known=None, reviews=None):
        return build_hold_variants(self.trend if trend is None else trend,
                                   self.known if known is None else known,
                                   self.reviews if reviews is None else reviews, self.spec)

    def test_original_baseline_exact_and_unknown_resets_counter(self):
        self.trend.loc[self.reviews, 'A'] = False
        self.known.loc[self.reviews[1], 'A'] = False
        variants, evidence = self.build()
        expected = confirmed_hold_states(self.trend.loc[self.reviews], self.known.loc[self.reviews], 2)
        pd.testing.assert_frame_equal(variants['baseline'], expected)
        np.testing.assert_array_equal(evidence['monthly_failure_streak'].A, [1, 0, 1, 2, 3, 4])
        self.assertTrue(all(pd.isna(frame.loc[self.reviews[1], 'A']) for frame in variants.values()))

    def test_prefix_matches_full_and_future_mutation_cannot_change_past(self):
        self.trend.loc[self.dates[30:100], 'A'] = False
        full, _ = self.build()
        cutoff = self.reviews[2]
        prefix, _ = self.build(self.trend.loc[:cutoff], self.known.loc[:cutoff], self.reviews[:3])
        future = self.trend.copy()
        future.loc[future.index > cutoff] = ~future.loc[future.index > cutoff]
        changed, _ = self.build(future)
        for name in full:
            pd.testing.assert_frame_equal(full[name].loc[:cutoff], prefix[name])
            pd.testing.assert_frame_equal(full[name].loc[:cutoff], changed[name].loc[:cutoff])

    def test_fast_baseline_slow_order_and_recovery(self):
        self.trend.loc[self.reviews[:4], 'A'] = False
        variants, _ = self.build()
        np.testing.assert_array_equal(variants['fast'].A, [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(variants['baseline'].A, [1, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(variants['slow'].A, [1, 1, 0, 0, 1, 1])
        self.assertTrue((variants['fast'] <= variants['baseline']).all().all())
        self.assertTrue((variants['baseline'] <= variants['slow']).all().all())

    def test_acceleration_requires_twenty_known_trading_days(self):
        date = self.reviews[0]
        self.trend.loc[:date, 'A'] = False
        variants, evidence = self.build()
        self.assertEqual(evidence['daily_known_count'].loc[date, 'A'], 20)
        self.assertEqual(variants['persist10'].loc[date, 'A'], 0)
        self.known.loc[self.dates[10], 'A'] = False
        variants, evidence = self.build()
        self.assertEqual(evidence['daily_known_count'].loc[date, 'A'], 19)
        for name in ['persist5', 'persist10', 'persist15']:
            self.assertEqual(variants[name].loc[date, 'A'], 1)

    def test_partial_initial_window_cannot_accelerate(self):
        self.trend.loc[:, 'A'] = False
        variants, evidence = self.build(reviews=self.dates[[18]])
        self.assertTrue(pd.isna(evidence['daily_known_count'].iloc[0, 0]))
        self.assertEqual(variants['persist5'].iloc[0, 0], 1)

    def test_counts_are_occupancy_not_consecutive_days_and_include_review(self):
        date = self.reviews[0]
        # 最近20交易日中隔日10次失败，其中最后一日就是review。
        self.trend.loc[self.dates[2:21:2], 'A'] = False
        variants, evidence = self.build()
        self.assertEqual(evidence['daily_weak_count'].loc[date, 'A'], 10)
        self.assertEqual(variants['persist5'].loc[date, 'A'], 0)
        self.assertEqual(variants['persist10'].loc[date, 'A'], 0)
        self.assertEqual(variants['persist15'].loc[date, 'A'], 1)
        self.trend.loc[date, 'A'] = True
        recovered, _ = self.build()
        self.assertTrue(all(frame.loc[date, 'A'] == 1 for frame in recovered.values()))

    def test_original_two_failure_exit_survives_incomplete_daily_window(self):
        self.trend.loc[self.reviews[:2], 'A'] = False
        self.known.loc[self.dates[35], 'A'] = False
        variants, evidence = self.build()
        self.assertEqual(evidence['daily_known_count'].loc[self.reviews[1], 'A'], 19)
        for name in ['baseline', 'persist5', 'persist10', 'persist15']:
            self.assertEqual(variants[name].loc[self.reviews[1], 'A'], 0)

    def test_unknown_nan_value_is_ignored_only_with_explicit_unknown_flag(self):
        trend = self.trend.astype(float)
        trend.loc[self.reviews[0], 'A'] = np.nan
        known = self.known.copy()
        known.loc[self.reviews[0], 'A'] = False
        variants, _ = self.build(trend, known)
        self.assertTrue(all(pd.isna(frame.loc[self.reviews[0], 'A']) for frame in variants.values()))
        with self.assertRaises(ValueError):
            self.build(trend, self.known)
        bad_known = self.known.astype(float)
        bad_known.loc[self.reviews[0], 'A'] = np.nan
        with self.assertRaises(ValueError):
            self.build(self.trend, bad_known)

    def test_review_dates_are_preserved_and_bad_calendar_rejected(self):
        variants, _ = self.build()
        for frame in variants.values():
            self.assertTrue(frame.index.equals(self.reviews))
        with self.assertRaises(ValueError):
            self.build(reviews=self.reviews[::-1])
        with self.assertRaises(ValueError):
            self.build(reviews=self.reviews.append(self.reviews[:1]))

    def test_effective_hold_preserves_full_schema_unknown_and_original_bytes(self):
        from run_hold_sensitivity import effective_hold
        original = pd.DataFrame([[1., np.nan, 0.], [0., 1., np.nan]],
                                index=self.reviews[:2], columns=['B', 'A', 'C'])
        saved = original.copy(deep=True)
        subset = pd.DataFrame([[0.], [1.]], index=original.index, columns=['B'])
        actual = effective_hold(original, subset)
        self.assertTrue(actual.columns.equals(original.columns))
        self.assertTrue(actual.index.equals(original.index))
        pd.testing.assert_frame_equal(actual[['A', 'C']], original[['A', 'C']])
        pd.testing.assert_frame_equal(actual[['B']], subset)
        pd.testing.assert_frame_equal(original, saved)
        pd.testing.assert_frame_equal(effective_hold(original, original[['B']]), original)
        with self.assertRaises(ValueError):
            effective_hold(original, subset.rename(columns={'B': 'other'}))
        with self.assertRaises(ValueError):
            effective_hold(original, subset.iloc[::-1])

    def test_actual_reverse_receiver_and_exception_restoration(self):
        from run_hold_sensitivity import actual_reverse
        received, records = [], []

        def receiver(candidates, **kwargs):
            received.append(candidates.ts_code.tolist())
            return 37

        original_input = pd.DataFrame({'ts_code': ['B', 'A', 'C']})
        with patch.object(selection_first, 'plan_selection_first', receiver):
            with self.assertRaisesRegex(RuntimeError, 'test_stop'):
                with actual_reverse(True, records):
                    self.assertEqual(selection_first.plan_selection_first(original_input), 37)
                    raise RuntimeError('test_stop')
            self.assertIs(selection_first.plan_selection_first, receiver)
            with actual_reverse(False, records):
                selection_first.plan_selection_first(original_input)
        self.assertEqual(received, [['C', 'A', 'B'], ['B', 'A', 'C']])
        self.assertEqual(original_input.ts_code.tolist(), ['B', 'A', 'C'])
        self.assertEqual(records[0]['actual'], records[0]['before'][::-1])
        self.assertEqual(records[0]['function_module'], 'selection_first')
        self.assertEqual(len(records), 1)


if __name__ == '__main__':
    unittest.main()
