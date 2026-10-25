"""独立输入构建的时间边界和数据完整性测试。"""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import yaml

from shadow_inputs import build_bundle
from shadow_planner import ShadowError


class ShadowInputsTests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/shadow_execution.yaml').read_text())
        self.date='2024-04-01'
        dates=pd.bdate_range('2022-01-03','2024-04-03')
        rows=[]
        for i,day in enumerate(dates):
            for j,code in enumerate(['600000.SH','601988.SH','300001.SZ']):
                price=10+i*.02+j*.5+np.sin(i)*.01
                rows.append(dict(date=day,ts_code=code,hfq_close=price,close=price,
                    vol_share=1e6,total_mv_yuan=1e9,turnover_rate=1.+j))
        self.raw=pd.DataFrame(rows)
        self.basics=pd.DataFrame([dict(ts_code=c,list_date=pd.Timestamp('2000-01-01'),delist_date=pd.NaT)
            for c in ['600000.SH','601988.SH','300001.SZ']])
        self.st=pd.DataFrame(columns=['ts_code','ann_date','start','end'])
        self.actions=pd.DataFrame(columns=['ts_code','known_date','ex_date'])
        self.cal=dict(source='synthetic calendar',sessions=[str(d.date()) for d in dates])
        self.model=dict(year=2024,train_label_end='2023-12-01',expected_return=.03,
            intercept=-.1,coefficients=[-.01,.01],center=[0.,0.],scale=[1.,1.],reference_risk=.07,uniform_discount=.8)
        self.receipt=dict(asof=self.date,complete=True,source='synthetic',downloaded_at=self.date+'T18:00:00+08:00',expected_rows=3)

    def run_bundle(self,raw=None):
        x=self.raw if raw is None else raw
        return build_bundle(x,x,self.basics,self.st,self.actions,self.date,self.cal,self.cfg,self.model,['600000.SH'],self.receipt)

    def test_future_prices_do_not_change_today(self):
        old=self.run_bundle();changed=self.raw.copy()
        changed.loc[changed.date.gt(self.date),['hfq_close','close','turnover_rate']]*=100
        self.assertEqual(old,self.run_bundle(changed))
        self.assertNotIn('300001.SZ',old['signals'])
        self.assertEqual(old['provenance']['full_universe'],3)

    def test_missing_eod_or_wrong_row_count_blocks(self):
        missing=self.raw[~self.raw.date.eq(self.date)]
        with self.assertRaises(ShadowError):self.run_bundle(missing)
        self.receipt['expected_rows']=4
        with self.assertRaises(ShadowError):self.run_bundle()

    def test_premarket_receipt_and_unknown_model_block(self):
        self.receipt['downloaded_at']=self.date+'T14:59:00+08:00'
        with self.assertRaises(ShadowError):self.run_bundle()
        self.receipt['downloaded_at']=self.date+'T18:00:00+08:00';self.model['year']=2023
        with self.assertRaises(ShadowError):self.run_bundle()

    def test_announced_next_day_action_blocked_without_future_prices(self):
        self.actions=pd.DataFrame([dict(ts_code='600000.SH',known_date=pd.Timestamp('2024-03-28'),ex_date=pd.Timestamp('2024-04-02'))])
        self.assertIn('600000.SH',self.run_bundle()['blocked_next_day'])
        self.actions['known_date']=pd.Timestamp('2024-04-02')
        self.assertNotIn('600000.SH',self.run_bundle()['blocked_next_day'])

    def test_known_date_st_metadata_supported(self):
        self.st=pd.DataFrame([dict(ts_code='601988.SH',known_date=pd.Timestamp('2024-03-20'),
            start=pd.Timestamp('2024-03-21'),end=pd.NaT)])
        self.assertNotIn('601988.SH',self.run_bundle()['signals'])


if __name__=='__main__':unittest.main()
