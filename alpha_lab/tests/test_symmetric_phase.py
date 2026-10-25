"""对称阶段正则化的独立对照及成熟标签隔离测试。"""

from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from divide_selection import phase_features
from symmetric_phase import DEFAULT_PHASES, fit_symmetric, phase_design
from value_forecast import fit_value_model


FEATURES = ("x_signal", "x_second")
OPTIONS = dict(asof="2020-05-01", bootstrap_reps=8, block_dates=6, seed=17)


def sample_frame():
    """含不等组规模、日期规模及阶段相关斜率的已知合成数据。"""
    rng = np.random.default_rng(481)
    rows = []
    for month, date in enumerate(pd.date_range("2016-01-01", periods=48, freq="MS")):
        for group, (mom60, previous) in enumerate([(0.12, -0.04), (0.16, 0.06), (0.07, 0.15)]):
            for _ in range(1 + group + month % 3):
                x, z = rng.uniform(-1, 1, size=2)
                rows.append(dict(date=date, label_end_60=date + pd.Timedelta(days=90),
                                 entry_missing_60=False, mom60=mom60,
                                 mom120=(1 + mom60) * (1 + previous) - 1,
                                 x_signal=x, x_second=z,
                                 ret_60=0.06 + 0.03 * x - 0.015 * z + [-0.04, 0.01, 0.03][group]
                                 + [-0.02, 0.03, -0.01][group] * x,
                                 mae_60=-0.12 + 0.008 * z - group * 0.01))
    return pd.DataFrame(rows)


def fit(frame, slopes=True, **kwargs):
    """冻结小规模重采样参数，仅允许测试明确改变分组顺序。"""
    options = dict(OPTIONS)
    options.update(kwargs)
    return fit_symmetric(frame, FEATURES, slopes=slopes, **options)


def orthonormal_prediction(frame, slopes, reference):
    """独立使用不同参考起点生成正交对比；罚项保持欧氏长度。"""
    phase, _ = phase_features(frame, [])
    onehot = np.column_stack([phase.phase.eq(group) for group in DEFAULT_PHASES]).astype(float)
    basis = np.eye(3)
    others = [i for i in range(3) if i != reference]
    contrasts, _ = np.linalg.qr(np.column_stack([basis[:, i] - basis[:, reference] for i in others]))
    coded = onehot @ contrasts
    values = frame[list(FEATURES)].to_numpy()
    blocks = [np.ones((len(frame), 1)), values, coded]
    if slopes:
        blocks.extend(coded[:, [j]] * values for j in range(2))
    x = np.column_stack(blocks)
    weights = 1 / frame.groupby("date").date.transform("size").to_numpy()
    y = frame[["ret_60", "mae_60"]].to_numpy() * [1, -1]
    # 直接解独立正交设计的加权正规方程，未调用模型求解器。
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(x.T @ (weights[:, None] * x) + penalty,
                           x.T @ (weights[:, None] * y))
    return x @ beta


class SymmetricPhaseTests(unittest.TestCase):
    def test_design_retains_all_groups_and_existing_mapping(self):
        frame = sample_frame()
        before = frame.copy(deep=True)
        expected, _ = phase_features(frame, [])
        for slopes in (False, True):
            design, columns = phase_design(frame, FEATURES, slopes=slopes)
            self.assertEqual(len(columns), len(FEATURES) + 3 + (3 * len(FEATURES) if slopes else 0))
            pd.testing.assert_series_equal(design.phase, expected.phase)
            centered = ["sym_phase_" + group for group in DEFAULT_PHASES]
            np.testing.assert_allclose(design[centered].sum(axis=1), 0, atol=1e-15)
            for group, column in zip(DEFAULT_PHASES, centered):
                np.testing.assert_allclose(design[column], expected.phase.eq(group) - 1 / 3)
        pd.testing.assert_frame_equal(frame, before)
        nine = tuple("x_" + str(i) for i in range(9))
        wide = frame.assign(**{name: frame.x_signal for name in nine})
        self.assertEqual(len(phase_design(wide, nine)[1]), 12)
        self.assertEqual(len(phase_design(wide, nine, slopes=True)[1]), 39)

    def test_orthonormal_reference_bases_and_group_order_give_same_predictions(self):
        frame = sample_frame()
        for slopes in (False, True):
            baseline = fit(frame, slopes)
            point = baseline.predict(frame)[["expected_return", "expected_adverse"]]
            for reference in range(3):
                with self.subTest(slopes=slopes, reference=reference):
                    np.testing.assert_allclose(point, orthonormal_prediction(frame, slopes, reference), atol=1e-13)
            for order in permutations(DEFAULT_PHASES):
                changed = fit(frame, slopes, phases=order)
                pd.testing.assert_frame_equal(baseline.predict(frame), changed.predict(frame), atol=1e-13, rtol=1e-13)

    def test_all_group_coefficients_sum_to_zero_including_bootstraps(self):
        model = fit(sample_frame()).underlying_model
        for suffix in ("",) + tuple("__" + f for f in FEATURES):
            names = ["sym_phase_" + group + suffix for group in DEFAULT_PHASES]
            np.testing.assert_allclose(model.coefficients.loc[names].sum(axis=0), 0, atol=1e-12)
            indices = model.coefficients.index.get_indexer(names)
            np.testing.assert_allclose(model.bootstrap_coefficients[:, indices].sum(axis=1), 0, atol=1e-12)

    def test_future_labels_and_invalid_future_phase_inputs_are_isolated(self):
        frame = sample_frame()
        baseline = fit(frame).underlying_model
        future = frame.iloc[:8].copy().assign(label_end_60=pd.Timestamp("2023-01-01"),
                                             ret_60=1e20, mae_60=-1e20, x_signal=np.inf,
                                             mom60=np.nan, mom120=np.inf)
        actual = fit(pd.concat([frame, future], ignore_index=True)).underlying_model
        pd.testing.assert_frame_equal(actual.coefficients, baseline.coefficients)
        np.testing.assert_array_equal(actual.bootstrap_coefficients, baseline.bootstrap_coefficients)
        self.assertEqual(actual.training_input_hash, baseline.training_input_hash)
        self.assertEqual(actual.metadata["exclusion_counts"]["immature_label"], 8)

    def test_unavailable_entry_and_missing_targets_do_not_require_phase_inputs(self):
        frame = sample_frame()
        baseline = fit(frame).underlying_model
        unavailable = frame.iloc[:2].copy().assign(entry_missing_60=True, mom60=np.nan)
        missing_target = frame.iloc[:2].copy().assign(ret_60=np.nan, mom120=np.inf)
        actual = fit(pd.concat([frame, unavailable, missing_target], ignore_index=True)).underlying_model
        pd.testing.assert_frame_equal(actual.coefficients, baseline.coefficients)
        self.assertEqual(actual.training_input_hash, baseline.training_input_hash)
        self.assertEqual(actual.metadata["exclusion_counts"]["entry_missing"], 2)
        self.assertEqual(actual.metadata["exclusion_counts"]["missing_target"], 2)

    def test_prediction_uses_no_labels_preserves_order_and_duplicate_index(self):
        frame = sample_frame()
        model = fit(frame)
        query = frame.iloc[:5][["mom60", "mom120"] + list(FEATURES)].copy()
        query.index = [4, 4, 2, 8, 1]
        before = query.copy(deep=True)
        predictions = model.predict(query)
        self.assertEqual(list(predictions), ["expected_return", "expected_adverse", "mean_lower", "mean_upper"])
        pd.testing.assert_index_equal(predictions.index, query.index)
        pd.testing.assert_frame_equal(query, before)
        shuffled = fit(frame.sample(frac=1, random_state=9))
        self.assertEqual(model.underlying_model.training_input_hash, shuffled.underlying_model.training_input_hash)
        pd.testing.assert_frame_equal(model.predict(query), shuffled.predict(query))
        self.assertEqual(model.predict(query.iloc[:0]).shape, (0, 4))

    def test_date_replication_keeps_date_equal_weight(self):
        frame = sample_frame()
        baseline = fit(frame)
        day = frame[frame.date.eq(frame.date.iloc[0])]
        expanded = fit(pd.concat([frame] + [day] * 7, ignore_index=True))
        np.testing.assert_allclose(baseline.underlying_model.coefficients,
                                   expanded.underlying_model.coefficients, atol=1e-13)

    def test_global_model_recovered_when_groups_have_identical_data(self):
        original = sample_frame()
        copies = []
        # 每个日期/特征向量在三组各复制一次，组偏差没有信息。
        for mom60, previous in [(0.12, -0.04), (0.16, 0.06), (0.07, 0.15)]:
            copies.append(original.assign(mom60=mom60, mom120=(1 + mom60) * (1 + previous) - 1))
        frame = pd.concat(copies, ignore_index=True)
        global_model = fit_value_model(frame, FEATURES, **OPTIONS)
        for slopes in (False, True):
            symmetric = fit(frame, slopes)
            pd.testing.assert_frame_equal(global_model.predict(frame), symmetric.predict(frame), atol=1e-13, rtol=1e-13)
            deviations = symmetric.underlying_model.coefficients.filter(like="sym_phase_", axis=0)
            np.testing.assert_allclose(deviations, 0, atol=1e-12)

    def test_fail_closed_invalid_phase_inputs_and_label_features(self):
        frame = sample_frame()
        for phase in [None, "unknown", "decelerating"]:
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                phase_design(frame.assign(phase=phase), FEATURES)
        for column, value in [("mom60", np.nan), ("mom120", np.inf), ("mom60", -1), ("x_signal", np.nan)]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                phase_design(frame.assign(**{column: value}), FEATURES)
        for feature in ["ret_20", "ret_60", "mae_60", "mfe_60", "label_end_60", "entry_missing_60", "phase"]:
            with self.subTest(feature=feature), self.assertRaisesRegex(ValueError, "标签或保留列"):
                phase_design(frame, [feature])
        with self.assertRaises(ValueError):
            phase_design(frame, FEATURES, phases=("recovery", "accelerating"))
        with self.assertRaises(ValueError):
            fit(frame, alpha=2)


if __name__ == "__main__":
    unittest.main()
