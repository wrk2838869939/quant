"""纯下跌反例、事件内锚点和时间契约。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sector_shape_events import identify_events,evaluate_outcomes


def panel(prices):
    return pd.DataFrame(dict(date=pd.bdate_range('2020-01-01',periods=len(prices)),sector_code='S',
        classification_version='SW2021',segment_id=1,index_valid=True,index_close=prices,
        turnover_pct=1.,total_mv_yuan=np.asarray(prices)*1e8,total_mv_yuan_members=10,added_members=0,removed_members=0))


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((ROOT/'configs/sector_shape_events_v1.yaml').read_text())
        self.cfg.update(trend_window=12,trend_average_window=8,trend_slope_days=3,minimum_rally_days=3,maximum_episode_days=90)
        self.prices=list(100+.1*np.sin(np.arange(25)))+[103,105,110,115,123,132,140,135,125,116,108,103,100,99.7,99.6,99.5]

    def test_sequence_prefix_and_future_shock(self):
        p=panel(self.prices)
        obs,_=identify_events(p,self.cfg)
        self.assertTrue(obs.candidate.any())
        hit=obs[obs.first_candidate].iloc[0]
        self.assertLess(hit.platform_end,hit.breakout_date)
        self.assertLess(hit.peak_date,hit.date)
        self.assertTrue(hit.trend_at_peak)
        self.assertGreater(hit.rally_return,.2)
        for length in [28,34,39]:
            short,_=identify_events(p.iloc[:length],self.cfg)
            pd.testing.assert_frame_equal(short,obs[obs.date<=p.date.iloc[length-1]].reset_index(drop=True))
        changed=pd.concat([p,panel([1000]*10).assign(date=pd.bdate_range(p.date.iloc[-1]+pd.Timedelta(days=1),periods=10))],ignore_index=True)
        future,_=identify_events(changed,self.cfg)
        pd.testing.assert_frame_equal(obs,future[future.date<=p.date.max()].reset_index(drop=True))

    def test_pure_decline(self):
        obs,events=identify_events(panel(np.linspace(200,80,200)),self.cfg)
        self.assertTrue(obs.empty and events.empty)

    def test_gap_breaks_episode_and_outcome(self):
        p=panel(self.prices+[99.4]*30)
        p.loc[42:,'segment_id']=2
        obs,events=identify_events(p,self.cfg)
        outcomes=evaluate_outcomes(p,obs,[20])
        self.assertTrue((events.end_reason=='segment_end').any())
        self.assertFalse(outcomes.complete.any())
        self.assertFalse((obs.segment_id==2).any())

    def test_reference_freezes_and_episode_expires(self):
        p=panel(self.prices+list(np.linspace(99.4,70,80)))
        obs,events=identify_events(p,self.cfg)
        first=obs[obs.event_id==obs.event_id.iloc[0]]
        self.assertEqual(first[first.reference_ready].reference_high.nunique(),1)
        self.assertTrue((events.end_reason=='pullback_timeout').any())
        self.assertFalse(obs[obs.date>p.date.iloc[-20]].candidate.any())


if __name__=='__main__':
    unittest.main()
