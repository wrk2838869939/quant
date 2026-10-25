"""以 SciPy Spearman 为独立参照验证 IC、挖掘入口及隔离边界。"""
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from rank_ic import rank_ic_rows, rank_ic_rows_torch
from labels import purged_dates, purge_label_window
from gp_miner import GPMiningTask, GPConfig
from gp_miner_gpu import GPGPMiningTask
from ledger import ExperimentLedger
from expr_engine import Node
from synthesis_and_test import ic_series
import run_gp_mining


class RankICTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(614)
        self.f = rng.integers(-4, 5, (8, 150)).astype(float)
        self.y = rng.normal(size=(8, 150))
        self.f[0, :20] = np.nan
        self.y[0, 30:60] = np.nan
        self.f[1, :51] = np.nan
        self.f[2] = 1
        self.y[3] = np.nan
        self.f[4, :2] = [np.inf, -np.inf]
        self.y[4, 5:8] = np.inf
        self.f[5, :50] = np.nan
        self.y[6] = self.f[6] ** 3
        self.y[7] = -self.f[7]

    def oracle(self, min_obs=100):
        result = []
        for f, y in zip(self.f, self.y):
            mask = np.isfinite(f) & np.isfinite(y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result.append(spearmanr(f[mask], y[mask]).statistic
                              if mask.sum() >= min_obs else np.nan)
        return np.array(result)

    def test_cpu_and_torch_match_scipy(self):
        expected = self.oracle()
        np.testing.assert_allclose(rank_ic_rows(pd.DataFrame(self.f), pd.DataFrame(self.y)),
                                   expected, atol=1e-12, equal_nan=True)
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    actual = rank_ic_rows_torch(torch.tensor(self.f, device=device, dtype=dtype),
                                                torch.tensor(self.y, device=device, dtype=dtype))
                    np.testing.assert_allclose(actual.cpu(), expected, atol=2e-7, equal_nan=True)

    def test_exact_perfect_correlations_and_threshold(self):
        result = rank_ic_rows(pd.DataFrame(self.f), pd.DataFrame(self.y))
        self.assertTrue(np.isnan(result[1]))  # 99 个有效样本
        self.assertTrue(np.isfinite(result[5]))  # 100 个有效样本
        self.assertAlmostEqual(result[6], 1)
        self.assertAlmostEqual(result[7], -1)

    def test_axis_mismatch_rejected(self):
        f, y = pd.DataFrame(self.f), pd.DataFrame(self.y)
        with self.assertRaises(ValueError):
            rank_ic_rows(f, y.iloc[:, ::-1])

    def test_synthesis_uses_same_contract(self):
        f, y = pd.DataFrame(self.f), pd.DataFrame(self.y)
        pd.testing.assert_series_equal(ic_series(f, y, list(f.index)), rank_ic_rows(f, y).dropna())

    @unittest.skipUnless(torch.cuda.is_available(), "实际 CUDA 集成测试")
    def test_mining_fitness_library_and_harvest(self):
        rng = np.random.default_rng(981)
        idx = pd.bdate_range("2016-01-01", periods=270)
        f = pd.DataFrame(rng.integers(-5, 6, (270, 160)).astype(float), index=idx)
        # 两端使用完全相同的可表示输入，避免把 float32 合并近邻值误归为公式错误。
        y = (f ** 3 + rng.normal(size=f.shape)).astype(np.float32).astype(float)
        lib = pd.DataFrame(rng.normal(size=f.shape), index=idx).astype(np.float32).astype(float)
        f.iloc[::2, :20] = np.nan
        y.iloc[::3, 25:45] = np.nan
        lib.iloc[::4, 50:70] = np.nan
        y.iloc[::35, :80] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            with patch("gp_miner.ExperimentLedger", side_effect=lambda: ExperimentLedger(Path(tmp) / "ledger.csv")):
                args = ({"a": f}, y, {"reference": lib}, idx[:190], idx[190:])
                cpu = GPMiningTask(*args, cfg=GPConfig(min_ic=0))
                gpu = GPGPMiningTask(*args, cfg=GPConfig(min_ic=0))
            gpu.prep(log=lambda _: None)
            expected = cpu.ic_series(f, cpu.train_idx)
            actual = gpu.ic_series_gpu(gpu.wide_t["a"], gpu.train_rows, gpu.train_idx)
            np.testing.assert_allclose(actual, expected, atol=2e-7)
            self.assertEqual(list(actual.index), list(expected.index))
            self.assertAlmostEqual(cpu.max_lib_corr(f, cpu.train_idx[::4]),
                                   gpu._lib_corr_gpu(gpu.wide_t["a"][gpu.train_rows_t]), places=6)
            node = Node(None, leaf="a")
            scores = [task.fitness(node) for task in (cpu, gpu)]
            self.assertEqual(scores[0]["reason"], "ok")
            self.assertEqual(scores[1]["reason"], "ok")
            for key in ("ic", "icir", "corr", "fit"):
                np.testing.assert_allclose(scores[0][key], scores[1][key], rtol=1e-5, atol=2e-7)
            reports = [task.harvest([(score, node)]) for task, score in zip((cpu, gpu), scores)]
            for col in ("train_ic", "valid_ic", "lib_corr", "passed"):
                pd.testing.assert_series_equal(reports[0][col], reports[1][col])


class SplitIsolationTest(unittest.TestCase):
    def test_purge_and_embargo(self):
        dates = pd.bdate_range("2021-01-01", periods=50)
        for horizon in (5, 20):
            kept = purged_dates(dates, horizon, embargo=5)
            self.assertEqual(len(kept), 50 - horizon - 5)
            self.assertEqual(dates.get_loc(kept[-1]) + horizon, 44)

    def test_short_and_empty_purge(self):
        for n in (0, 3, 25):
            dates = pd.bdate_range("2021-01-01", periods=n)
            self.assertEqual(len(purged_dates(dates, 20, 5)), 0)
            self.assertTrue(purge_label_window(pd.DataFrame({"date": dates}), h=20).empty)

    def test_filtered_read_and_aligned_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = Path(tmp) / "panel.parquet"
            pd.DataFrame({"date": pd.to_datetime(["2023-12-28", "2023-12-28", "2024-01-02"]),
                          "ts_code": ["a", "b", "c"], "x": [1., 2., 999.],
                          "y": [np.nan, 3., 999.]}).to_parquet(panel)
            with patch.object(run_gp_mining, "PANEL", panel):
                wide = run_gp_mining.build_wide({"x": "x", "y": "y"}, end_date="2023-12-31")
            self.assertEqual(list(wide["x"].columns), ["a", "b"])
            self.assertEqual(wide["x"].shape, (1, 2))
            self.assertTrue(wide["x"].columns.equals(wide["y"].columns))
            self.assertTrue(np.isnan(wide["y"].iloc[0, 0]))


if __name__ == "__main__":
    unittest.main()
