import unittest,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trend_entry import marked_paths,entry_features,patient_memberships


class TrendEntryTest(unittest.TestCase):
    spec=dict(max_adverse=.15,peak_for_roundtrip=.1,positive_checkpoints=2)

    def test_missing_quotes_are_carried_not_deleted(self):
        x=marked_paths([10.],np.array([[11.],[np.nan],[9.]]),[np.nan],[np.inf],self.spec).iloc[0]
        self.assertAlmostEqual(x.ret,-.1);self.assertTrue(x.exit_marked)
        self.assertEqual(x.missing_marks,2);self.assertAlmostEqual(x.giveback,.2)

    def test_delisting_cannot_be_undone_by_later_spurious_quote(self):
        x=marked_paths([10.],np.array([[11.],[12.],[15.]]),[20.],[2],self.spec).iloc[0]
        self.assertEqual(x.ret,-1.);self.assertEqual(x.mdd,-1.);self.assertTrue(x.delisted)

    def test_missing_entry_is_cash_even_with_later_large_rise(self):
        x=marked_paths([np.nan],np.array([[11.],[12.],[15.]]),[20.],[np.inf],self.spec).iloc[0]
        self.assertEqual(x.ret,0.);self.assertEqual(x.state,'no_entry')

    def test_feature_prefix(self):
        dates=pd.bdate_range('2015-01-01',periods=180)
        close=pd.DataFrame({'a':np.exp(np.arange(180)*.01),'b':np.exp(np.arange(180)*-.001)},index=dates)
        full=entry_features(close,close*1e9,close*0+2)
        short=entry_features(close.iloc[:150],close.iloc[:150]*1e9,close.iloc[:150]*0+2)
        for field in full:pd.testing.assert_frame_equal(full[field].iloc[:150],short[field])

    def test_keep_old_name_despite_new_rank_and_confirm_trend_failure(self):
        dates=pd.bdate_range('2020-01-01',periods=4)
        frame=pd.DataFrame([dict(date=d,ts_code=c,trend=(c=='b' or i<2),score=(2 if i==0 and c=='a' else 3 if c=='b' and i else 1))
            for i,d in enumerate(dates) for c in ['a','b']])
        result=patient_memberships(frame,'score',1,2)
        self.assertEqual(result.ts_code.tolist(),['a','a','a','b'])

    def test_no_entry_when_score_is_insufficient(self):
        frame=pd.DataFrame([dict(date=pd.Timestamp('2020-01-01'),ts_code='a',trend=True,score=-.1)])
        self.assertTrue(patient_memberships(frame,'score',5,2,0.).empty)


if __name__=='__main__':unittest.main()
