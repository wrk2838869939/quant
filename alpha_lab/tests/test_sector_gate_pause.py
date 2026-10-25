"""暂停板块门槛不放宽事件去重、日期及个股筛选。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'alpha_lab/experiments/sector_live_v1'))
sys.path.insert(0,str(ROOT/'datahub/src'))
from scan_opportunities import current_events,sector_stock_eligibility
from stock_selection_asof import select_positive_top


class SectorGatePauseTests(unittest.TestCase):
    def test_disabled_admits_negative_and_missing_diagnostics(self):
        events=pd.DataFrame({'score':[-.04,.02,np.nan],'threshold':[.003,.003,np.nan]},index=[4,7,9])
        self.assertEqual(sector_stock_eligibility(events,False).to_dict(),{4:True,7:True,9:True})

    def test_enabled_restores_existing_gate_and_rejects_unknown(self):
        events=pd.DataFrame({'score':[-.04,.003,np.nan],'threshold':[.003,.003,.003]})
        self.assertEqual(sector_stock_eligibility(events,True).tolist(),[False,True,False])

    def test_old_confirmations_and_unaccepted_windows_do_not_reopen(self):
        events=pd.DataFrame({'date':pd.to_datetime(['2026-09-17','2026-10-21','2026-10-21']),
            'event_id':['hotel-old','fresh','duplicate-window'],
            'merged_event_id':['hotel','fresh','fresh'],'accepted':[True,True,False]})
        current=current_events(events,'2026-10-21')
        self.assertEqual(current.event_id.tolist(),['fresh'])
        self.assertEqual(current.source_event_id.tolist(),['fresh'])

    def test_duplicate_accepted_event_is_rejected(self):
        events=pd.DataFrame({'date':['2026-10-21']*2,'event_id':['a','b'],'merged_event_id':['same']*2})
        with self.assertRaises(ValueError):current_events(events,'2026-10-21')

    def test_no_first_confirmation_remains_empty(self):
        events=pd.DataFrame({'date':['2026-09-17'],'event_id':['old'],'merged_event_id':['old']})
        today=current_events(events,'2026-10-21')
        self.assertTrue(today.empty)
        self.assertTrue(sector_stock_eligibility(today,False).empty)

    def test_stock_shape_positive_score_and_top_n_remain_enforced(self):
        frame=pd.DataFrame({'event_id':['same']*7,'ts_code':[str(i) for i in range(7)],
            'price':[False,True,True,True,True,True,True],'model_score':[100,-1,0,.5,.4,.3,.2]})
        picked=select_positive_top(frame,'model_score',{'top_n':3,'model':{'minimum_predicted_excess':0.0}})
        self.assertEqual(picked.tolist(),[3,4,5])

    def test_string_false_is_not_silently_truthy(self):
        with self.assertRaises(TypeError):sector_stock_eligibility(pd.DataFrame(),'false')


if __name__=='__main__':unittest.main()
