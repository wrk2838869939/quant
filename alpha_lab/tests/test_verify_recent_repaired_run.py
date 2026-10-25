"""修复后覆盖准入必须先于账户通过，且不能复用过期准入哈希。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_recent_repaired_run.py'
SPEC = importlib.util.spec_from_file_location('verify_recent_repaired_run', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CoverageAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        out = self.run / 'execution'
        out.mkdir()
        dates = pd.to_datetime(['2026-02-02', '2026-03-02'])
        matrix = pd.DataFrame({'600828.SH': [1., 1.]}, index=dates)
        for name in ['monthly_hold_allowed', 'low_turnover_signals', 'low_turnover_point_forecasts']:
            matrix.to_parquet(out / f'{name}.parquet')
        pd.DataFrame({'date': dates, 'ts_code': ['600828.SH'] * 2}).to_parquet(self.run / 'execution_predictions.parquet')
        gate_dir = self.run / 'input_coverage_audit'
        gate_dir.mkdir()
        script = gate_dir / 'validate_recent_input_coverage.py'
        script.write_text('# 本地测试覆盖报告来源\n')
        self.gate = gate_dir / 'coverage_gate.json'
        self.gate.write_text(json.dumps({'status': 'passed', 'input_hashes': {
            'execution_predictions.parquet': MODULE.sha(self.run / 'execution_predictions.parquet')},
            'script_sha256': MODULE.sha(script)}))

    def test_complete_dates_and_frozen_gate_pass(self):
        self.assertEqual(MODULE.verify_coverage(self.run)['review_count'], 2)

    def test_unselected_stock_may_keep_point_forecast(self):
        path = self.run / 'execution/low_turnover_signals.parquet'
        signals = pd.read_parquet(path)
        signals.iloc[0, 0] = float('nan')
        signals.to_parquet(path)
        self.assertEqual(MODULE.verify_coverage(self.run)['review_count'], 2)

    def test_selected_stock_requires_point_forecast(self):
        path = self.run / 'execution/low_turnover_point_forecasts.parquet'
        forecasts = pd.read_parquet(path)
        forecasts.iloc[0, 0] = float('nan')
        forecasts.to_parquet(path)
        with self.assertRaisesRegex(AssertionError, '有效候选缺少点预测'):
            MODULE.verify_coverage(self.run)

    def test_missing_month_is_rejected_even_with_previously_passed_gate(self):
        path = self.run / 'execution_predictions.parquet'
        frame = pd.read_parquet(path).iloc[:1]
        frame.to_parquet(path)
        with self.assertRaisesRegex(AssertionError, '预测仍完全缺少'):
            MODULE.verify_coverage(self.run)

    def test_gate_for_different_input_bytes_is_rejected(self):
        gate = json.loads(self.gate.read_text())
        gate['input_hashes']['execution_predictions.parquet'] = '0' * 64
        self.gate.write_text(json.dumps(gate))
        with self.assertRaisesRegex(AssertionError, '引用输入已漂移'):
            MODULE.verify_coverage(self.run)


if __name__ == '__main__':
    unittest.main()
