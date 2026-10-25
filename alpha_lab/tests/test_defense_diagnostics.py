"""验证风险实验的历史前缀、冻结篮子与月度恢复纪律。"""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from defense_diagnostics import DefenseOverlay, risk_evidence


class DefenseTests(unittest.TestCase):
    def setUp(self):
        self.spec = dict(medium_window=3, long_window=6, persistence_window=4,
                         persistent_weak_days=2, affected_fraction=.5,
                         recovery_fraction=.5, review_frequency='weekly')
        self.dates = pd.bdate_range('2024-01-01', periods=40)

    def test_future_changes_cannot_change_prefix(self):
        close = pd.DataFrame({'A': np.arange(40, 0, -1.), 'B': np.arange(1., 41)}, index=self.dates)
        complete = risk_evidence(close, self.spec)
        prefix = risk_evidence(close.iloc[:25], self.spec)
        altered = close.copy()
        altered.iloc[25:] *= 100
        changed = risk_evidence(altered, self.spec)
        for key in complete:
            pd.testing.assert_frame_equal(prefix[key], complete[key].iloc[:25])
            pd.testing.assert_frame_equal(prefix[key], changed[key].iloc[:25])

    def test_unknown_prevents_alarm_and_recovery(self):
        close = pd.DataFrame({'A': np.arange(40, 0, -1.)}, index=self.dates)
        close.iloc[20] = np.nan
        e = risk_evidence(close, self.spec)
        self.assertFalse(e['alarm'].iloc[20:24].any().any())
        self.assertFalse(e['known'].iloc[20].any())

    def test_frozen_cohort_and_monthly_recovery(self):
        e = {key: pd.DataFrame(value, index=self.dates, columns=['A', 'B'])
             for key, value in [('alarm', False), ('trend', False), ('known', True)]}
        e['alarm'].loc[self.dates[0], 'A'] = True
        overlay = DefenseOverlay(e, self.spec, 'weekly_liquidate')
        exits, paused = overlay.on_close(self.dates[0], True, {'A': 60., 'B': 40.})
        self.assertEqual(set(exits), {'A', 'B'})
        self.assertTrue(paused)
        self.assertTrue(overlay.on_close(self.dates[1], False, {})[1])
        e['trend'].loc[self.dates[2], 'A'] = True
        self.assertTrue(overlay.on_close(self.dates[2], False, {})[1])
        e['trend'].loc[self.dates[3], 'A'] = True
        e['known'].loc[self.dates[3], 'A'] = False
        self.assertTrue(overlay.on_close(self.dates[3], True, {})[1])
        e['trend'].loc[self.dates[4], 'A'] = True
        self.assertFalse(overlay.on_close(self.dates[4], True, {})[1])

    def test_exit_does_not_imply_pause(self):
        e = {key: pd.DataFrame(value, index=self.dates, columns=['A', 'B'])
             for key, value in [('alarm', True), ('trend', False), ('known', True)]}
        overlay = DefenseOverlay(e, self.spec, 'weekly_exit')
        self.assertEqual(overlay.on_close(self.dates[0], True, {'A': 1.}), (['A'], False))
        pause = DefenseOverlay(e, self.spec, 'pause_only')
        self.assertEqual(pause.on_close(self.dates[0], True, {'A': 1.}), ([], True))


if __name__ == '__main__':
    unittest.main()
