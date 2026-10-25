"""新执行行情的单位、主键、日期和元数据覆盖失败测试。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from execution_market_feed import normalize_day, require_current_metadata, file_hash
from shadow_planner import ShadowError


def frames():
    key=dict(ts_code=['000001.SZ'],trade_date=['20260911'])
    return (pd.DataFrame(dict(key,open=[10.],high=[11.],low=[9.],close=[10.],pre_close=[10.],vol=[100.],amount=[100.])),
        pd.DataFrame(dict(key,close=[10.],turnover_rate=[.2],total_mv=[1000.])),
        pd.DataFrame(dict(key,adj_factor=[2.])))


class ExecutionMarketTests(unittest.TestCase):
    def test_units(self):
        f,m,check=normalize_day(*frames(),'2026-09-11')
        self.assertEqual(f.hfq_close.iloc[0],20.)
        self.assertEqual(f.total_mv_yuan.iloc[0],10000000.)
        self.assertEqual(m.vol_share.iloc[0],10000.)
        self.assertEqual(m.amount_yuan.iloc[0],100000.)

    def test_bad_keys_dates_prices_and_coverage(self):
        for mode in ['duplicate','date','missing_basic','missing_adj','close','ohlc','nan','negative']:
            d,b,a=frames()
            if mode=='duplicate':d=pd.concat([d,d])
            if mode=='date':b['trade_date']='20260910'
            if mode=='missing_basic':b['ts_code']='000002.SZ'
            if mode=='missing_adj':a['ts_code']='000002.SZ'
            if mode=='close':b['close']=10.01
            if mode=='ohlc':d['high']=9
            if mode=='nan':d['vol']=float('nan')
            if mode=='negative':d['amount']=-1
            with self.subTest(mode=mode),self.assertRaises(ShadowError):normalize_day(d,b,a,'2026-09-11')

    def test_adj_only_codes_retained_as_diagnostic(self):
        d,b,a=frames();extra=a.copy();extra['ts_code']='000002.SZ'
        _,_,check=normalize_day(d,b,pd.concat([a,extra]),'2026-09-11')
        self.assertEqual(check['adj_only_codes'],['000002.SZ'])

    def test_metadata_missing_stale_and_tampered(self):
        package=dict(asof='2026-09-11',price_checks_passed=True,metadata={})
        with self.assertRaises(ShadowError):require_current_metadata(package,'2026-09-11')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metadata.json';path.write_text('{}')
            for key in ['security_master','st_history','corporate_actions','trading_calendar']:
                package['metadata'][key]=dict(source='test',path=str(path),sha256=file_hash(path),verified_through='2026-09-11')
            self.assertTrue(require_current_metadata(package,'2026-09-11'))
            stale=deepcopy(package);stale['metadata']['st_history']['verified_through']='2026-09-10'
            with self.assertRaises(ShadowError):require_current_metadata(stale,'2026-09-11')
            path.write_text('{"changed":true}')
            with self.assertRaises(ShadowError):require_current_metadata(package,'2026-09-11')


if __name__=='__main__':unittest.main()
