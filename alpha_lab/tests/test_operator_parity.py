"""用合成面板核对表达式双后端，避免算子差异污染挖掘结果。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import expr_engine as cpu
import gpu_eval as gpu


class OperatorParityTest(unittest.TestCase):
    """覆盖缺失值、并列值、短历史和嵌套表达式。"""

    def setUp(self):
        rng = np.random.default_rng(731)
        self.a = rng.integers(-4, 5, size=(75, 6)).astype(float)
        self.b = rng.integers(-3, 7, size=(75, 6)).astype(float)
        self.a[10:14, 0] = np.nan
        self.b[12:17, 1] = np.nan
        self.a[:, 2] = np.nan
        self.a[:, 3] = 2
        self.b[:, 3] = 3
        self.a[:7, 4] = np.nan
        self.a[30] = np.nan
        self.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def check_node(self, node, a=None, b=None):
        wide = {"a": pd.DataFrame(self.a if a is None else a),
                "b": pd.DataFrame(self.b if b is None else b)}
        expected = cpu.eval_node(node, wide).to_numpy()
        for device in self.devices:
            for dtype in (torch.float64, torch.float32):
                with self.subTest(device=device, dtype=dtype):
                    tensors = {k: torch.tensor(v.values, dtype=dtype, device=device)
                               for k, v in wide.items()}
                    actual = gpu.eval_gpu(node, tensors).cpu().numpy()
                    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6,
                                               equal_nan=True)

    def test_registered_operators(self):
        self.assertEqual(set(cpu.OPS), set(gpu.OPS_GPU))

    def test_all_operators(self):
        for name, (_, arity, _, needs_window) in cpu.OPS.items():
            for window in (cpu.WINDOWS if needs_window else [None]):
                with self.subTest(operator=name, window=window):
                    children = [cpu.Node("leaf", leaf=k) for k in ("a", "b")[:arity]]
                    self.check_node(cpu.Node(name, children, window))

    def test_short_history(self):
        for name, (_, arity, _, needs_window) in cpu.OPS.items():
            if not needs_window:
                continue
            with self.subTest(operator=name):
                children = [cpu.Node("leaf", leaf=k) for k in ("a", "b")[:arity]]
                self.check_node(cpu.Node(name, children, 5), self.a[:3], self.b[:3])

    def test_nested_decay_missingness(self):
        node = cpu.Node("decay", [cpu.Node("sqrt", [cpu.Node("zscore", [
            cpu.Node("leaf", leaf="a")])])], 5)
        self.check_node(node)

    def test_time_operators_do_not_read_future(self):
        for name, (_, arity, _, needs_window) in cpu.OPS.items():
            if not needs_window:
                continue
            children = [cpu.Node("leaf", leaf=k) for k in ("a", "b")[:arity]]
            node = cpu.Node(name, children, 20)
            wide = {"a": pd.DataFrame(self.a), "b": pd.DataFrame(self.b)}
            prefix = {k: v.iloc[:35] for k, v in wide.items()}
            with self.subTest(operator=name):
                np.testing.assert_allclose(cpu.eval_node(node, prefix),
                                           cpu.eval_node(node, wide).iloc[:35],
                                           atol=1e-7, equal_nan=True)
                for device in self.devices:
                    tensors = {k: torch.tensor(v.values, dtype=torch.float32,
                                               device=device) for k, v in wide.items()}
                    actual = gpu.eval_gpu(node, {k: v[:35] for k, v in tensors.items()})
                    expected = gpu.eval_gpu(node, tensors)[:35]
                    torch.testing.assert_close(actual, expected, equal_nan=True)

    def test_cpu_fallbacks(self):
        a = pd.DataFrame(self.a)
        for name in ("tsrank", "argmax", "argmin"):
            fn = cpu.OPS[name][0]
            with self.subTest(operator=name):
                expected = fn(a, 5)
                with patch.object(cpu, "HAS_BN", False):
                    actual = fn(a, 5)
                np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_sample_standard_deviation(self):
        a = pd.DataFrame(self.a)
        actual = cpu.OPS["ts_std"][0](a, 5)
        expected = a.rolling(5, min_periods=2).std(ddof=1)
        np.testing.assert_allclose(actual, expected, atol=1e-7, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
