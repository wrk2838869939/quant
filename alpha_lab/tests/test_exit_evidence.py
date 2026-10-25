"""特征前缀、未来标记边界、重叠事件剔除与缺失保留。"""
from pathlib import Path
import sys,unittest
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from exit_evidence import state_features,forward_marks,nonoverlapping_events

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.d=pd.bdate_range('2020-01-01',periods=200);self.p=pd.DataFrame({'a':100+np.arange(200)*.1},index=self.d)
        self.t=pd.DataFrame(True,index=self.d,columns=['a']);self.o=self.t.copy()
    def test_features_future_prefix(self):
        full=state_features(self.p,self.t,self.o);short=state_features(self.p.iloc[:175],self.t.iloc[:175],self.o.iloc[:175])
        for name in full:pd.testing.assert_frame_equal(full[name].iloc[:175],short[name],check_exact=True)
    def test_labels_begin_next_day_and_require_full_horizon(self):
        r=forward_marks(self.p,self.d[150],'a',20)
        self.assertEqual(r['label_start'],self.d[151]);self.assertEqual(r['label_end'],self.d[171])
        self.assertAlmostEqual(r['ret'],self.p.iloc[171,0]/self.p.iloc[151,0]-1)
        self.assertTrue(forward_marks(self.p,self.d[-2],'a',20)['right_censored'])
    def test_missing_future_never_forward_filled(self):
        self.p.iloc[160]=np.nan;r=forward_marks(self.p,self.d[150],'a',20)
        self.assertTrue(r['missing']);self.assertTrue(np.isnan(r['ret']))
    def test_current_day_not_part_of_future_label(self):
        before=forward_marks(self.p,self.d[150],'a',20);self.p.iloc[150]=1.;after=forward_marks(self.p,self.d[150],'a',20);self.assertEqual(before,after)
    def test_nonoverlap_requires_strictly_more_than_horizon(self):
        f=pd.DataFrame(dict(date=self.d[[0,1,60,61,90,122]],ts_code=['a']*6))
        r=nonoverlapping_events(f,self.d,60)
        self.assertEqual(r.nonoverlap_primary.tolist(),[True,False,False,True,False,True])
        short=nonoverlapping_events(f.iloc[:4],self.d[:100],60)
        pd.testing.assert_frame_equal(r.iloc[:4],short,check_exact=True)
    def test_unknown_recent_history_does_not_become_known_weakness(self):
        self.o.iloc[-5]=False
        self.assertTrue(np.isnan(state_features(self.p,self.t,self.o)['weak_fraction20'].iloc[-1,0]))

if __name__=='__main__':unittest.main()
