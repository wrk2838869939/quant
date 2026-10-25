"""真实执行矩阵、主板限制和完整倒序产物的边界验证。"""
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from run_symmetric_phase_execution import execution_cases, is_mainboard, compare_all_csv


class SymmetricExecutionTest(unittest.TestCase):
    def test_twenty_plus_preregistered_hold_variant(self):
        policies = ['low_turnover', 'global_return', 'symmetric_intercept', 'symmetric_slopes']
        cfg = dict(policies=policies, capitals=[20000., 50000., 100000.], reference_capital=50000.,
                   execution_transfer_policies=[f'transfer_{p}' for p in policies],
                   holding_variant=dict(name='keep_failure_memory', policy='low_turnover', capital=50000.))
        cases = execution_cases(cfg)
        self.assertEqual(len(cases), 21)
        self.assertEqual(sum(case['reverse_input'] for case in cases), 4)
        self.assertEqual(sum(case['training_universe'] == 'all_a' for case in cases), 4)
        self.assertEqual(cases[-1]['hold_variant'], 'keep_failure_memory')
        self.assertTrue(all(case['capital'] == 50000 for case in cases[12:]))
        self.assertTrue(all(case['execution_universe'] == 'mainboard' for case in cases))

    def test_mainboard_cannot_include_other_boards_or_malformed_codes(self):
        for code in ['000001.SZ', '001001.SZ', '002001.SZ', '003001.SZ', '600001.SH', '601001.SH', '603001.SH', '605001.SH']:
            self.assertTrue(is_mainboard(code))
        for code in ['688001.SH', '300001.SZ', '301001.SZ', '830001.BJ', '6001.SH', '600001.SZ', '000001.SH']:
            self.assertFalse(is_mainboard(code))

    def test_reverse_check_covers_all_csv_and_detects_auxiliary_drift(self):
        cfg = dict(policies=['low_turnover'], reference_capital=50000.)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for prefix in ['base', 'reverse']:
                folder = out / f'{prefix}_50000_low_turnover'
                folder.mkdir()
                for number in range(14):
                    pd.DataFrame({'value': [number]}).to_csv(folder / f'{number}.csv', index=False)
                pd.DataFrame({'name': [folder.name], 'year': [2022], 'value': [1.]}).to_csv(folder / 'yearly.csv', index=False)
            self.assertEqual(len(compare_all_csv(out, cfg)), 15)
            pd.DataFrame({'value': [100]}).to_csv(out / 'reverse_50000_low_turnover/13.csv', index=False)
            with self.assertRaises(AssertionError):
                compare_all_csv(out, cfg)


if __name__ == '__main__':
    unittest.main()
