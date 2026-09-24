"""统计公式、尺度不变性、失败关闭及时间截面隔离。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ledger import deflated_sharpe, deflated_sharpe_report, ExperimentLedger
from neutralize import neutralize, standard_factor_pipeline


class SharpeTest(unittest.TestCase):
    def setUp(self):
        self.r = np.random.default_rng(7).normal(.001, .01, 1000)
        self.sr = self.r.mean() / self.r.std(ddof=1)

    def report(self, **kwargs):
        args = dict(sr_est=self.sr, n_obs=len(self.r), n_trials=20,
                    sr_var=.001, rets=self.r, assume_iid=True)
        args.update(kwargs)
        return deflated_sharpe_report(**args)

    def test_scale_invariance_and_probability_direction(self):
        base = self.report()
        for scale in [.01, 100.]:
            scaled = self.report(rets=self.r * scale)
            self.assertAlmostEqual(base["p_value"], scaled["p_value"], places=12)
        self.assertAlmostEqual(base["p_value"] + base["dsr_probability"], 1.)
        self.assertEqual(base["status"], "conditional_iid_estimate")

    def test_formula_from_central_moments(self):
        x = self.r - self.r.mean()
        skewness = np.mean(x ** 3) / np.mean(x ** 2) ** 1.5
        kurt = np.mean(x ** 4) / np.mean(x ** 2) ** 2
        threshold = np.sqrt(.001) * ((1 - np.euler_gamma) * norm.ppf(.95)
            + np.euler_gamma * norm.ppf(1 - 1 / (20 * np.e)))
        z = (self.sr - threshold) * np.sqrt(999) / np.sqrt(
            1 - skewness * self.sr + (kurt - 1) * self.sr ** 2 / 4)
        self.assertAlmostEqual(self.report()["dsr_probability"], norm.cdf(z), places=13)

    def test_more_trials_reduce_evidence(self):
        self.assertGreater(self.report(n_trials=2)["dsr_probability"],
                           self.report(n_trials=100)["dsr_probability"])
        self.assertEqual(self.report(n_trials=1, sr_var=None)["benchmark_sharpe"], 0.)

    def test_missing_assumptions_and_bad_frequency_are_not_assessed(self):
        for kwargs in [dict(assume_iid=False), dict(n_trials=None), dict(sr_var=None),
                       dict(n_obs=999), dict(sr_est=self.sr*np.sqrt(244)),
                       dict(rets=np.ones(1000)), dict(n_trials=0), dict(sr_var=-1),
                       dict(rets=np.r_[self.r[:-1], np.nan])]:
            with self.subTest(kwargs=list(kwargs)):
                report = self.report(**kwargs)
                self.assertEqual(report["status"], "not_assessed")
                self.assertIsNone(report["p_value"])
        self.assertTrue(np.isnan(deflated_sharpe(self.sr, 1000, 20, rets=self.r)))

    def test_sample_length_changes_sampling_error(self):
        r = np.tile(self.r, 4)
        longer = self.report(rets=r, sr_est=r.mean()/r.std(ddof=1), n_obs=len(r))
        self.assertLess(longer["sharpe_standard_error"], self.report()["sharpe_standard_error"])

    def test_ledger_preserves_legacy_rows_and_records_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "ledger.csv"
            pd.DataFrame([dict(factor="old", n_trials=1)]).to_csv(p, index=False)
            ledger = ExperimentLedger(p)
            ledger.log("new", "daily", "development", 0., 0., 0., 0.,
                       experiment_id="run1", parent_experiment="parent",
                       source_hashes={"code": "abc"}, selection_note="fixed controls")
            result = pd.read_csv(p)
            self.assertEqual(result.loc[0, "factor"], "old")
            self.assertTrue(pd.isna(result.loc[0, "event_id"]))
            self.assertEqual(result.loc[1, "parent_experiment"], "parent")
            self.assertEqual(result.loc[1, "count_semantics"], "logged_events_not_independent_trials")


class NeutralizationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        n = 120
        self.df = pd.DataFrame(dict(date=pd.Timestamp("2022-01-04"),
            log_mv=rng.normal(20, 1, n), sw_l1=np.repeat(["a", "b", "c"], 40)))
        self.df["signal"] = 3*self.df.log_mv + np.repeat([1, 2, 4], 40) + rng.normal(size=n)

    def test_residuals_remove_exposures(self):
        r, status = neutralize(self.df, "signal", return_diagnostics=True)
        self.assertEqual(status.status.iloc[0], "ok")
        self.assertAlmostEqual(float(r @ self.df.log_mv), 0., places=8)
        self.assertTrue(r.groupby(self.df.sw_l1).sum().abs().max() < 1e-10)

    def test_inadequate_or_singular_samples_never_return_raw_signal(self):
        for frame in [self.df.iloc[:49], self.df.assign(log_mv=20.)]:
            r, status = neutralize(frame, "signal", return_diagnostics=True)
            self.assertTrue(r.isna().all())
            self.assertNotEqual(status.status.iloc[0], "ok")

    def test_invalid_exposures_and_future_rows(self):
        self.df.loc[0, "log_mv"] = np.inf
        self.df.loc[1, "sw_l1"] = None
        r, status = neutralize(self.df, "signal", return_diagnostics=True)
        self.assertTrue(r.iloc[:2].isna().all())
        future = self.df.assign(date=pd.Timestamp("2022-02-01"), signal=-1e6)
        both = pd.concat([self.df, future], ignore_index=True)
        actual = neutralize(both, "signal")
        pd.testing.assert_series_equal(r, actual.iloc[:len(r)])
        self.assertEqual(status.n_output.iloc[0], 118)

    def test_pipeline_propagates_failure_status(self):
        r, status = standard_factor_pipeline(self.df.iloc[:20], "signal", return_diagnostics=True)
        self.assertTrue(r.isna().all())
        self.assertEqual(status.status.iloc[0], "insufficient_samples")


if __name__ == "__main__":
    unittest.main()
