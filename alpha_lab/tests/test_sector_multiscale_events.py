"""多尺度计数不重复、不前视，独立回落和断段保留。"""
from pathlib import Path
import sys
import unittest
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sector_multiscale_events import merge_signals


class MergeTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(event_id=['a', 'b', 'c', 'd'], sector_code='S', segment_id=1,
            date=pd.to_datetime(['2020-02-10', '2020-02-11', '2020-02-20', '2020-02-21']),
            peak_date=pd.to_datetime(['2020-02-01', '2020-02-02', '2020-02-15', '2020-02-02']), priority=[0, 1, 0, 2]))

    def test_same_fall_not_new_event(self):
        r = merge_signals(self.frame())
        self.assertEqual(r.accepted.tolist(), [True, False, True, False])
        self.assertEqual(r.merged_event_id.iloc[0], r.merged_event_id.iloc[1])

    def test_prefix_exact(self):
        f = self.frame()
        pd.testing.assert_frame_equal(merge_signals(f.iloc[:2]), merge_signals(f).iloc[:2])

    def test_new_segment_and_same_day_priority(self):
        f = self.frame().iloc[:2].copy()
        f.loc[1, 'date'] = f.date.iloc[0]
        self.assertEqual(merge_signals(f).accepted.tolist(), [True, False])
        f.loc[1, 'segment_id'] = 2
        self.assertTrue(merge_signals(f).accepted.all())


if __name__ == '__main__':
    unittest.main()
