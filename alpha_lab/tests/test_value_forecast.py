"""条件均值模型的日期权重、标签隔离与重采样纪律。"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from value_forecast import fit_value_model, _month_block_counts


def sample_frame():
    """仅构造 2016—2019 年的虚拟月度股票截面。"""
    rng = np.random.default_rng(31)
    dates = pd.date_range("2016-01-01", periods=48, freq="MS")
    date = dates.repeat(4)
    x = rng.normal(size=len(date))
    month_effect = np.repeat(rng.normal(scale=0.025, size=len(dates)), 4)
    return pd.DataFrame({
        "date": date, "label_end_60": date + pd.Timedelta(days=90),
        "ret_60": 0.05 + 0.02 * x + month_effect,
        "mae_60": -0.1 + 0.01 * x,
        "entry_missing_60": False, "signal": x,
    })


def fit(frame, features=("signal",), **kwargs):
    """使用固定的合成实验配置。"""
    options = {"asof": "2020-05-01", "bootstrap_reps": 30, "block_dates": 6}
    options.update(kwargs)
    return fit_value_model(frame, features, **options)


class ValueForecastTests(unittest.TestCase):
    def test_future_labels_do_not_change_fit_or_input_hash(self):
        frame = sample_frame()
        baseline = fit(frame)
        future = frame.iloc[:8].copy()
        future["label_end_60"] = pd.Timestamp("2023-01-01")
        future["signal"] = np.inf
        future["ret_60"] = 1e10
        future["mae_60"] = -1e10
        actual = fit(pd.concat([frame, future], ignore_index=True))
        pd.testing.assert_frame_equal(actual.coefficients, baseline.coefficients)
        np.testing.assert_array_equal(actual.bootstrap_coefficients, baseline.bootstrap_coefficients)
        self.assertEqual(actual.training_input_hash, baseline.training_input_hash)
        self.assertEqual(actual.metadata["exclusion_counts"]["immature_label"], 8)
        self.assertLessEqual(actual.last_label, pd.Timestamp("2020-05-01"))

    def test_replicating_one_date_does_not_increase_its_weight(self):
        frame = sample_frame()
        baseline = fit(frame)
        one_date = frame[frame.date.eq(frame.date.iloc[0])]
        expanded = pd.concat([frame] + [one_date] * 13, ignore_index=True)
        actual = fit(expanded)
        np.testing.assert_allclose(actual.coefficients, baseline.coefficients, atol=1e-14)
        np.testing.assert_allclose(actual.bootstrap_coefficients, baseline.bootstrap_coefficients, atol=1e-14)
        pd.testing.assert_frame_equal(actual.predict(frame), baseline.predict(frame), atol=1e-13, rtol=1e-13)

    def test_constant_baseline_is_average_of_date_means(self):
        frame = sample_frame()
        first = frame.iloc[:1].copy()
        first["ret_60"] = 0.9
        frame = pd.concat([frame, first], ignore_index=True)
        model = fit(frame, features=[])
        expected = frame.groupby("date")["ret_60"].mean().mean()
        self.assertAlmostEqual(model.coefficients.loc["intercept", "expected_return"], expected)
        self.assertNotAlmostEqual(expected, frame.ret_60.mean(), places=7)
        self.assertEqual(model.predict(pd.DataFrame(index=["a", "b"])).shape, (2, 4))

    def test_seed_reproduces_models_and_quantile_intervals(self):
        frame = sample_frame()
        first, second = fit(frame, seed=5), fit(frame, seed=5)
        np.testing.assert_array_equal(first.bootstrap_coefficients, second.bootstrap_coefficients)
        other = fit(frame, seed=6)
        self.assertFalse(np.array_equal(first.bootstrap_coefficients, other.bootstrap_coefficients))
        x = np.column_stack([np.ones(len(frame)), frame.signal])
        expected = np.quantile(x @ first.bootstrap_coefficients[:, :, 0].T, [0.1, 0.9], axis=1)
        predictions = first.predict(frame)
        np.testing.assert_allclose(predictions[["mean_lower", "mean_upper"]].to_numpy().T, expected)
        self.assertIn("not_predictive", first.metadata["interval_semantics"])

    def test_rejects_insufficient_dates_and_degenerate_month_blocks(self):
        with self.assertRaisesRegex(ValueError, "日期不足"):
            fit(sample_frame().iloc[:35 * 4])
        with self.assertRaisesRegex(ValueError, "日历月数"):
            fit(sample_frame(), block_dates=48)

    def test_constant_targets_are_unpenalized_and_adverse_is_nonnegative(self):
        frame = sample_frame().assign(ret_60=0.07, mae_60=-0.03)
        model = fit(frame, alpha=1000)
        np.testing.assert_allclose(model.coefficients.loc["intercept"], [0.07, 0.03], atol=1e-14)
        np.testing.assert_allclose(model.coefficients.loc["signal"], 0, atol=1e-14)
        prediction = model.predict(frame)
        np.testing.assert_allclose(prediction.expected_return, 0.07, atol=1e-14)
        np.testing.assert_allclose(prediction.expected_adverse, 0.03, atol=1e-14)
        np.testing.assert_allclose(prediction[["mean_lower", "mean_upper"]], 0.07, atol=1e-14)
        extrapolated = fit(sample_frame()).predict(pd.DataFrame({"signal": [10000.0]}))
        self.assertEqual(extrapolated.expected_adverse.iloc[0], 0.0)

    def test_missing_entry_and_targets_are_excluded_with_counts(self):
        frame = sample_frame()
        extras = frame.iloc[:3].copy()
        extras.iloc[0, extras.columns.get_loc("entry_missing_60")] = True
        extras.iloc[0, extras.columns.get_loc("signal")] = np.inf
        extras.iloc[1, extras.columns.get_loc("ret_60")] = np.nan
        extras.iloc[2, extras.columns.get_loc("label_end_60")] = pd.NaT
        baseline = fit(frame)
        actual = fit(pd.concat([frame, extras], ignore_index=True))
        pd.testing.assert_frame_equal(actual.coefficients, baseline.coefficients)
        counts = actual.metadata["exclusion_counts"]
        self.assertEqual([counts[k] for k in ["entry_missing", "missing_target", "missing_label_end"]], [1, 1, 1])
        self.assertEqual(counts["fitted_rows"], len(frame))

    def test_rejects_nonfinite_features_targets_and_invalid_dates(self):
        frame = sample_frame()
        for column, value in [("signal", np.nan), ("signal", np.inf), ("ret_60", np.inf),
                              ("mae_60", -np.inf), ("mae_60", 0.1), ("date", pd.NaT),
                              ("entry_missing_60", "False")]:
            with self.subTest(column=column, value=value):
                bad = frame.copy()
                if column == "entry_missing_60":
                    bad[column] = bad[column].astype(object)
                bad.loc[0, column] = value
                with self.assertRaises(ValueError):
                    fit(bad)
        bad = frame.copy()
        bad.loc[0, "date"] = bad.loc[0, "label_end_60"] + pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "不得晚于"):
            fit(bad)
        model = fit(frame)
        with self.assertRaises(ValueError):
            model.predict(pd.DataFrame({"signal": [np.inf]}))

    def test_preserves_input_prediction_index_and_hash_order_invariance(self):
        frame = sample_frame()
        before = frame.copy(deep=True)
        model = fit(frame)
        query = frame.iloc[:5].copy()
        query.index = [5, 5, 3, 1, 8]
        before_query = query.copy(deep=True)
        prediction = model.predict(query)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(query, before_query)
        pd.testing.assert_index_equal(prediction.index, query.index)
        shuffled = fit(frame.sample(frac=1, random_state=7))
        self.assertEqual(shuffled.training_input_hash, model.training_input_hash)
        self.assertEqual(model.predict(query.iloc[:0]).shape, (0, 4))

    def test_sampling_preserves_contiguous_month_blocks(self):
        class FixedStarts:
            def integers(self, low, high, size):
                self.args = (low, high, size)
                return np.array([1, 5, 0])
        rng = FixedStarts()
        actual = _month_block_counts(8, 3, rng)
        # 抽到 [1,2,3]、[5,6,7]、[0,1]，最后一块按原月数截断。
        np.testing.assert_array_equal(actual, [1, 2, 1, 1, 0, 1, 1, 1])
        self.assertEqual(rng.args, (0, 6, 3))

    def test_rejects_invalid_configuration_and_label_features(self):
        frame = sample_frame()
        for options in [{"alpha": 0}, {"alpha": np.nan}, {"block_dates": 1.5},
                        {"bootstrap_reps": 0}, {"min_dates": True}, {"seed": -1}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit(frame, **options)
        with self.assertRaises(ValueError):
            fit(frame, features=["ret_60"])


if __name__ == "__main__":
    unittest.main()
