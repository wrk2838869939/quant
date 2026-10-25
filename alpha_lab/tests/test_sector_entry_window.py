"""五日资格的事件隔离、失效与个股趋势前缀验证。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sector_entry_window import entry_window_rows, trend_at_date
from left_side_trend import trend_pullback


class EntryWindowTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2024-01-04', periods=5)
        self.event = SimpleNamespace(event_id='source', merged_event_id='merged', sector_code='sector',
                                     window=15, segment_id=1, peak_date=pd.Timestamp('2023-12-29'), date=self.dates[0])
        self.states = pd.DataFrame(dict(event_id='source', segment_id=1, peak_date=self.event.peak_date,
                                        pullback_gate=True, near_reference=True,
                                        confirmed=[True, False, False, True, False]), index=self.dates)
        self.representative = {(d, 'sector'): True for d in self.dates}

    def test_qualification_survives_daily_slowdown_disappearing(self):
        rows = entry_window_rows(self.event, self.dates, self.states, self.representative)
        self.assertEqual([r['persistent_5d'] for r in rows], [True] * 5)
        self.assertEqual([r['same_event_today'] for r in rows], [True, False, False, True, False])
        self.assertEqual(rows[-1]['age_sessions'], 4)

    def test_invalidated_event_never_revives(self):
        self.states.loc[self.dates[1], 'near_reference'] = False
        rows = entry_window_rows(self.event, self.dates, self.states, self.representative)
        self.assertEqual([r['persistent_5d'] for r in rows], [True, False, False, False, False])
        self.assertTrue(rows[3]['same_event_today'])

    def test_new_peak_or_episode_cannot_borrow_old_score(self):
        for column, value in [('peak_date', self.dates[1]), ('event_id', 'new'), ('segment_id', 2)]:
            states = self.states.copy()
            states.loc[self.dates[1]:, column] = value
            rows = entry_window_rows(self.event, self.dates, states, self.representative)
            self.assertTrue(rows[1]['legacy_today'])
            self.assertFalse(any(r['persistent_5d'] or r['same_event_today'] for r in rows[1:]))

    def test_missing_state_cancels_qualification(self):
        states = self.states.drop(self.dates[1])
        rows = entry_window_rows(self.event, self.dates, states, self.representative)
        self.assertEqual(rows[1]['invalid_reason'], 'missing_source_state')
        self.assertFalse(rows[3]['persistent_5d'])

    def test_trend_original_parity_and_future_invariance(self):
        cfg = yaml.safe_load((ROOT / 'configs/left_side_trend_v2.yaml').read_text())['trend']
        rng = np.random.default_rng(20261013)
        close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.001, .02, (400, 12)), axis=0)),
                             index=pd.bdate_range('2022-01-01', periods=400))
        close.iloc[230:235, 1] = np.nan
        original = trend_pullback(close, cfg, cfg['stock_minimum_advance'], cfg['stock_minimum_pullback'])['passes'].eq(1)
        for i in [150, 250, 320, 399]:
            date = close.index[i]
            pd.testing.assert_series_equal(trend_at_date(close, date, cfg), original.iloc[i], check_names=False)
        date = close.index[320]
        altered = close.copy()
        altered.loc[close.index[321]:] *= 100
        pd.testing.assert_series_equal(trend_at_date(close, date, cfg), trend_at_date(altered, date, cfg))


if __name__ == '__main__':
    unittest.main()
