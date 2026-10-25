"""跟踪队列不按幸存形态筛选；未来结果不混入观察状态。"""
from pathlib import Path
import sys,unittest
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sector_state_evolution import build_paths,forward_labels,state_features


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2020-01-01',periods=40)
        self.cfg=dict(tracking_sessions=10,recent_days=2,preceding_days=3,slowdown_ratio=.65,
                      minimum_observation_coverage=.8,landmarks=[0,5,10],horizons=[5])
        self.panel=pd.DataFrame(dict(date=self.dates,sector_code='S',segment_id=1,index_valid=True,
            index_close=np.arange(40)+100.,up_share=.5,breadth_members=10,known_members=10))
        self.origin=pd.DataFrame(dict(event_id=['E'],sector_code=['S'],segment_id=[1],date=[self.dates[10]],index_close=[110.],peak_price=[130.]))
        self.obs=pd.DataFrame(dict(event_id=['E'],date=[self.dates[10]],pullback_gate=[True],candidate=[False]))

    def test_keep_event_after_episode_end(self):
        paths=build_paths(self.panel,self.origin,self.obs,self.dates,self.cfg)
        self.assertEqual(len(paths),11)
        self.assertEqual(paths.episode_active.sum(),1)
        self.assertTrue(paths.available.all())
        self.assertTrue(paths.iloc[5].index_close>0)

    def test_later_gap_does_not_skip_sessions_or_reconnect(self):
        p=self.panel.copy();p.loc[13:,'segment_id']=2
        paths=build_paths(p,self.origin,self.obs,self.dates,self.cfg)
        self.assertEqual(paths.available.sum(),3)
        labels=forward_labels(p,paths,self.dates,p.set_index('date').index_close,self.cfg)
        self.assertFalse(labels.complete.any())

    def test_prefix_and_no_prior_loss(self):
        full=build_paths(self.panel,self.origin,self.obs,self.dates,self.cfg)
        short=build_paths(self.panel.iloc[:16],self.origin,self.obs,self.dates[:16],self.cfg)
        pd.testing.assert_frame_equal(short,full[full.date<=self.dates[15]].reset_index(drop=True))
        self.assertTrue(full.state.eq('此前无下跌').all())
        features=state_features(self.panel,self.cfg)
        self.assertTrue(features.slowdown_ratio.isna().all())

    def test_labels_start_at_landmark(self):
        paths=build_paths(self.panel,self.origin,self.obs,self.dates,self.cfg)
        labels=forward_labels(self.panel,paths,self.dates,pd.Series(100.,index=self.dates),self.cfg)
        row=labels[labels.offset.eq(5)].iloc[0]
        self.assertAlmostEqual(row.close_return,120/115-1)
        self.assertEqual(row.end_date,self.dates[20])


if __name__=='__main__':unittest.main()
