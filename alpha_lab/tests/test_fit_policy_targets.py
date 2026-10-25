"""固定目标对比的共同样本、独立正规方程和未来状态隔离。"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fit_policy_targets', ROOT / 'scripts/fit_policy_targets.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def config():
    """沿用登记参数，合成测试仅减少bootstrap次数。"""
    cfg = yaml.safe_load((ROOT / 'configs/policy_cash_research.yaml').read_text())
    cfg['forecast']['bootstrap_reps'] = 5
    return cfg


def synthetic():
    """每月三只股票，三收益目标共享同风险但存在不同收益斜率。"""
    cfg = config()
    rng = np.random.default_rng(341)
    dates = pd.date_range('2014-01-01', '2023-12-01', freq='MS').repeat(3)
    x = rng.uniform(-0.5, 0.5, (len(dates), 9))
    frame = pd.DataFrame(x, columns=cfg['features'])
    frame['date'] = dates
    frame['ts_code'] = np.tile(['000001.SZ', '600001.SH', '600002.SH'], len(dates) // 3)
    frame['monthly'] = True
    frame['trend'] = True
    frame['label_end_120'] = dates + pd.Timedelta(days=180)
    frame.loc[frame.label_end_120.ge('2024-01-01'), 'label_end_120'] = pd.NaT
    frame['entry_missing_120'] = False
    frame['execution_unknown_120'] = False
    frame['fixed60_return'] = 0.035 + 0.08 * x[:, 0]
    frame['fixed120_return'] = 0.04 + 0.12 * x[:, 1]
    frame['patient120_return'] = 0.045 + 0.08 * x[:, 2]
    frame['patient120_mae'] = -0.1 - 0.03 * x[:, 3]
    return frame


class PolicyTargetsTests(unittest.TestCase):
    def test_independent_date_equal_solver_and_shared_risk(self):
        frame, cfg = synthetic(), config()
        frame.loc[0, 'execution_unknown_120'] = True
        frame.loc[1, 'fixed60_return'] = np.nan
        frame.loc[2, 'entry_missing_120'] = True
        models, info = RUNNER.fit_models(frame, cfg, cfg['fit_end'])
        eligible, _ = RUNNER.common_sample(frame, cfg['fit_end'])
        cohort = frame.loc[eligible & frame.trend]
        self.assertEqual(info['common_training_rows'], len(cohort))
        self.assertEqual(len({m.metadata['common_cohort_hash'] for m in models.values()}), 1)
        for policy in RUNNER.POLICIES:
            model = models[policy]
            features = [] if policy == 'low_turnover' else cfg['features']
            x = np.column_stack([np.ones(len(cohort)), cohort[features].to_numpy()])
            ycol = 'patient120_return' if policy == 'low_turnover' else policy + '_return'
            y = np.column_stack([cohort[ycol], -cohort.patient120_mae])
            weight = 1 / cohort.groupby('date').date.transform('size').to_numpy()
            penalty = np.eye(x.shape[1])
            penalty[0, 0] = 0
            expected = np.linalg.solve(x.T @ (weight[:, None] * x) + penalty,
                                       x.T @ (weight[:, None] * y))
            np.testing.assert_allclose(model.coefficients, expected, atol=1e-13)
            self.assertLessEqual(model.last_label, pd.Timestamp(cfg['fit_end']))
        for policy in RUNNER.TARGETS[:-1]:
            np.testing.assert_array_equal(models[policy].coefficients.expected_adverse,
                                          models['patient120'].coefficients.expected_adverse)
            np.testing.assert_array_equal(models[policy].bootstrap_coefficients[:, :, 1],
                                          models['patient120'].bootstrap_coefficients[:, :, 1])

    def test_future_or_unknown_labels_cannot_affect_any_fit(self):
        frame, cfg = synthetic(), config()
        unknown = frame.index[:3]
        frame.loc[unknown, 'execution_unknown_120'] = True
        base, _ = RUNNER.fit_models(frame, cfg, cfg['fit_end'])
        changed = frame.copy()
        future = changed.label_end_120.gt(pd.Timestamp(cfg['fit_end'])) | changed.label_end_120.isna()
        mask = future | changed.execution_unknown_120
        changed.loc[mask, RUNNER.LABELS] = np.inf
        actual, _ = RUNNER.fit_models(changed, cfg, cfg['fit_end'])
        for policy in RUNNER.POLICIES:
            pd.testing.assert_frame_equal(base[policy].coefficients, actual[policy].coefficients)
            self.assertEqual(base[policy].training_input_hash, actual[policy].training_input_hash)

    def test_future_label_availability_never_filters_top_ranked_stock(self):
        frame, cfg = synthetic(), config()
        models, _ = RUNNER.fit_models(frame, cfg, cfg['fit_end'])
        query = frame[frame.date.eq(pd.Timestamp('2019-01-01'))].copy()
        before = RUNNER.predict_models(query, models, cfg)
        # 使未知股票明确成为低换手Top1，未来状态不能将其替换为第二名。
        first = query.x_turnover.idxmin()
        query.loc[first, 'execution_unknown_120'] = True
        after = RUNNER.predict_models(query, models, cfg)
        pd.testing.assert_frame_equal(before, after)
        tables = RUNNER.diagnostics(query, after, cfg, cfg['development'][1])
        selected = tables['selected_opportunities']
        one = selected[selected.policy.eq('low_turnover') & selected.k.eq(1)]
        self.assertEqual(one.ts_code.iloc[0], query.loc[first, 'ts_code'])
        self.assertFalse(one.common_label_available.iloc[0])
        self.assertTrue(pd.isna(one.patient120_return.iloc[0]))

    def test_common_maturity_does_not_grant_early_exit_extra_training_rows(self):
        frame, cfg = synthetic(), config()
        models, info = RUNNER.fit_models(frame, cfg, cfg['fit_end'])
        expected = frame[frame.label_end_120.le(pd.Timestamp(cfg['fit_end']))]
        self.assertEqual(info['common_training_rows'], len(expected))
        for model in models.values():
            self.assertEqual(len(model.fit_dates), expected.date.nunique())
        self.assertLess(expected.date.max(), pd.Timestamp('2018-08-01'))

    def test_known_blocked_top1_is_observed_cash_zero_but_not_training(self):
        frame, cfg = synthetic(), config()
        models, _ = RUNNER.fit_models(frame, cfg, cfg['fit_end'])
        query = frame[frame.date.eq(pd.Timestamp('2019-01-01'))].copy()
        first = query.x_turnover.idxmin()
        query.loc[first, 'entry_missing_120'] = True
        query.loc[first, RUNNER.LABELS] = 0.0
        train_available, flags = RUNNER.common_sample(query, cfg['development'][1])
        self.assertFalse(train_available.loc[first])
        self.assertTrue(flags.common_label_available.loc[first])
        prediction = RUNNER.predict_models(query, models, cfg)
        tables = RUNNER.diagnostics(query, prediction, cfg, cfg['development'][1])
        row = tables['top_k_dates'].query("policy == 'low_turnover' and k == 1 and target_kind == 'common_patient120'").iloc[0]
        self.assertEqual(row.selected_n, 1)
        self.assertEqual(row.observed_n, 1)
        self.assertEqual(row.no_entry_selected_n, 1)
        self.assertEqual(row.realized, 0.0)
        self.assertTrue(row.complete_selected_labels)
        self.assertTrue(row.complete_cash_slots)
        selected = tables['selected_opportunities'].query("policy == 'low_turnover' and k == 1").iloc[0]
        self.assertEqual(selected.ts_code, query.loc[first, 'ts_code'])
        self.assertEqual(selected.patient120_return, 0.0)
        errors = tables['errors_dates'].query("policy == 'low_turnover'")
        self.assertEqual(errors.query("target_kind == 'common_patient120'").n.iloc[0], 3)
        self.assertEqual(errors.query("target_kind == 'native_patient120'").n.iloc[0], 2)

    def test_rejections_have_explicit_cash_slots_and_targets_stay_separate(self):
        frame, cfg = synthetic(), config()
        models, _ = RUNNER.fit_models(frame, cfg, cfg['fit_end'])
        query = frame[frame.date.eq(pd.Timestamp('2019-01-01'))].copy()
        prediction = RUNNER.predict_models(query, models, cfg)
        prediction['fixed60__expected_return'] = -0.5
        prediction['fixed60__score'] = np.nan
        tables = RUNNER.diagnostics(query, prediction, cfg, cfg['development'][1])
        row = tables['top_k_dates'].query("policy == 'fixed60' and k == 5 and target_kind == 'common_patient120'").iloc[0]
        self.assertEqual(row.selected_n, 0)
        self.assertTrue(pd.isna(row.realized))
        self.assertEqual(row.cash_slot_return, 0)
        self.assertTrue(row.complete_cash_slots)
        self.assertFalse(row.complete_selected_labels)
        errors = tables['errors_dates'].query("policy == 'fixed60'")
        self.assertEqual(set(errors.target_kind), {'common_patient120', 'native_fixed60'})
        self.assertNotEqual(errors.mse.iloc[0], errors.mse.iloc[1])
        gates = tables['cash_rejection_dates'].query("policy == 'fixed60'").iloc[0]
        self.assertTrue(gates.no_admitted)
        self.assertEqual(gates.return_only_rejected, len(query))
        pairs = RUNNER.paired_comparisons(tables)
        self.assertEqual(pairs.query("comparison == 'selected_nonempty'").paired_complete_dates.max(), 0)
        self.assertEqual(pairs.query("comparison == 'fixed_k_slots_cash_zero'").paired_complete_dates.min(), 1)

    def test_rejects_undefined_execution_state_and_future_market_window(self):
        frame, cfg = synthetic(), config()
        for column, value in [('execution_unknown_120', np.nan), ('monthly', 'yes'),
                              ('date', pd.Timestamp('2024-01-01'))]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                RUNNER.check_inputs(frame.assign(**{column: value}), cfg)

    def test_complete_runner_freezes_all_candidates_without_execution(self):
        frame, cfg = synthetic(), config()
        frame.loc[frame.date.gt('2022-01-01'), 'execution_unknown_120'] = True
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'config.yaml').write_text(yaml.safe_dump(cfg))
            frame.to_parquet(run / 'policy_labels.parquet', index=False)
            report = RUNNER.run_fit(run)
            self.assertEqual(report['model_count'], 8)
            self.assertTrue(report['no_execution_replay'])
            pred = pd.read_parquet(run / 'execution_predictions.parquet')
            self.assertEqual(len(pred), int(frame.date.between(*cfg['execution']).sum()))
            self.assertEqual(pred.date.max(), pd.Timestamp('2023-12-01'))
            self.assertTrue((run / 'execution_diagnostics/paired_comparisons.csv').exists())
            with self.assertRaisesRegex(ValueError, '禁止覆盖'):
                RUNNER.run_fit(run)


if __name__ == '__main__':
    unittest.main()
