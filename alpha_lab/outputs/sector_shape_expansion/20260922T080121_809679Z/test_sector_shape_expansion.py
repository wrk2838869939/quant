"""顺序确认、峰值重置及等待期的时间契约。"""
from pathlib import Path
import sys
import unittest
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sector_shape_expansion import stage_events


def frame():
    return pd.DataFrame(dict(event_id=['A'] * 8,
        date=pd.bdate_range('2020-01-01', periods=8),
        peak_date=pd.Timestamp('2019-12-20'), pullback_gate=True,
        fast_fall_seen=True, near_reference=True,
        support_broken=[True] + [False] * 7,
        slowed=[False, False, True, True, True, True, True, True]))


class StageTests(unittest.TestCase):
    def test_sequential_and_expiry(self):
        r = stage_events(frame(), 5)
        self.assertEqual(list(r.index[r.confirmed]), [2, 3, 4])
        self.assertEqual(list(r.index[r.first_confirmed]), [2])

    def test_new_peak_resets_evidence(self):
        f = frame()
        f.loc[2:, 'peak_date'] = f.date.iloc[1]
        self.assertFalse(stage_events(f, 5).confirmed.any())

    def test_prefix_and_no_trigger(self):
        f = frame()
        pd.testing.assert_frame_equal(stage_events(f.iloc[:4], 5), stage_events(f, 5).iloc[:4])
        f['pullback_gate'] = False
        self.assertFalse(stage_events(f, 5).confirmed.any())


if __name__ == '__main__':
    unittest.main()
