"""研究运行器不得把整个候选代码缺数据悄悄当成单日不可成交。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_selection_stability as runner


class SelectionScopeTest(unittest.TestCase):
    def setUp(self):
        self.original = dict(runner.STATE)
        dates = pd.to_datetime(['2024-01-02', '2024-02-01'])
        scores = pd.DataFrame({'600000.SH': [.4, .4], '600001.SH': [.3, .3]}, index=dates)
        runner.STATE.clear()
        runner.STATE.update(schedules={1: {'baseline': dict(signals=scores, forecasts=scores, hold=scores)}},
                            risk=scores * 0 + 1,
                            frames={'market': pd.DataFrame({'ts_code': ['600000.SH']})})
        self.case = dict(start='2024-01-01', end='2024-03-01', offset=1, policy='baseline', kind='long')

    def tearDown(self):
        runner.STATE.clear(); runner.STATE.update(self.original)

    def test_whole_missing_candidate_rejected_even_if_low_priority(self):
        with self.assertRaisesRegex(ValueError, '完全缺执行行情'):
            runner.simulate(self.case)

    def test_missing_risk_rejected_before_execution(self):
        runner.STATE['frames']['market'] = pd.DataFrame({'ts_code': ['600000.SH', '600001.SH']})
        runner.STATE['risk'].loc[:, '600001.SH'] = np.nan
        with self.assertRaisesRegex(ValueError, '风险覆盖不全'):
            runner.simulate(self.case)


if __name__ == '__main__': unittest.main()
