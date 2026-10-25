"""决策日历的休市、跨年、前缀和确认时间尺度测试。"""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rolling_review import decision_dates


class RollingReviewTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2022-01-04','2024-03-15')

    def test_all_calendar_prefixes(self):
        for parameter in [1,8,15,22]:
            full=decision_dates(self.dates,'calendar',parameter)
            for cut in [1,7,21,54,270,len(self.dates)-1]:
                short=self.dates[:cut]
                pd.testing.assert_index_equal(decision_dates(short,'calendar',parameter),full[full<=short[-1]])

    def test_rolling_prefix_and_spacing(self):
        for offset in [0,5,10,15]:
            full=decision_dates(self.dates,'rolling',offset,anchor='2022-01-01')
            positions=self.dates.get_indexer(full)
            self.assertTrue(((positions[1:]-positions[:-1])==20).all())
            short=self.dates[:270]
            pd.testing.assert_index_equal(decision_dates(short,'rolling',offset,anchor='2022-01-01'),
                                          full[full<=short[-1]])

    def test_holiday_waits_for_known_trading_day(self):
        dates=pd.to_datetime(['2024-02-01','2024-02-02','2024-02-19','2024-02-20','2024-03-01'])
        actual=decision_dates(dates,'calendar',8)
        pd.testing.assert_index_equal(actual,pd.DatetimeIndex(['2024-02-19']))

    def test_no_end_of_month_backfill(self):
        short=pd.to_datetime(['2024-02-01','2024-02-02'])
        self.assertEqual(len(decision_dates(short,'calendar',22)),0)

    def test_anchor_required_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            decision_dates(self.dates,'rolling',0)
        with self.assertRaises(ValueError):
            decision_dates(pd.to_datetime(['2024-01-02','2024-01-02']),'calendar',1)


if __name__=='__main__':
    unittest.main()
