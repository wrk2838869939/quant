"""字段修订、同日冲突及累计口径混用的失败模式。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from financial_events import append_events, asof_events, normalize_events, quarter_difference


def event(value=-10., date='2019-04-30', field='cfo_yuan', role='disclosed_value', basis='original'):
    return dict(ts_code='A', period_end='2018-12-31', field=field, scope='consolidated',
                published_date=date, value=value, unit='CNY', basis_id=basis,
                source_path='/not/opened/in/selection.pdf', source_sha256='a'*64,
                pdf_page=1, source_token=str(value), source_url='https://issuer.example/report.pdf', event_role=role)


class FinancialEventsTest(unittest.TestCase):
    def query(self, events, dates, field='cfo_yuan'):
        req=pd.DataFrame([dict(date=d, ts_code='A', period_end='2018-12-31', field=field, scope='consolidated') for d in dates])
        return asof_events(pd.DataFrame(events), req)

    def test_revision_not_backdated_and_same_day_excluded(self):
        got=self.query([event(),event(3., '2020-06-18', basis='restated')],
                       ['2019-04-30','2019-05-01','2020-06-18','2020-06-19'])
        self.assertEqual(list(got.status),['unavailable','available','available','available'])
        self.assertEqual(list(got.value.iloc[1:]),[-10.,-10.,3.])
        self.assertFalse(got.history_complete.any())

    def test_historical_before_column_not_republished(self):
        got=self.query([event(),event(-10.,'2020-06-18',role='historical_comparator'),event(3.,'2020-06-18',basis='restated')], ['2020-06-19'])
        self.assertEqual(got.iloc[0].value,3.)

    def test_field_update_does_not_update_unmentioned_fields(self):
        records=[event(),event(.18,field='eps'),event(3.,'2020-06-18',basis='restated')]
        self.assertEqual(self.query(records,['2020-06-19'],'eps').iloc[0].value,.18)

    def test_same_day_conflict_not_silently_tiebroken(self):
        got=self.query([event(),event(99.)],['2019-05-01'])
        self.assertEqual(got.iloc[0].status,'same_day_conflict')
        self.assertTrue(pd.isna(got.iloc[0].value))

    def test_disclosed_missing_not_filled_with_old_value(self):
        got=self.query([event(),event(np.nan,'2020-06-18')],['2020-06-19'])
        self.assertEqual(got.iloc[0].status,'disclosed_missing')

    def test_immutable_id_and_append(self):
        first=normalize_events(pd.DataFrame([event()]))
        full=append_events(first,pd.DataFrame([event(3.,'2020-06-18')]))
        self.assertEqual(len(full),2)
        self.assertIn(first.event_id.iloc[0],full.event_id.tolist())
        first.loc[0,'value']=9.
        with self.assertRaises(ValueError):normalize_events(first)

    def test_row_order_and_future_events_preserve_past(self):
        a=self.query([event()],['2019-05-01'])
        b=self.query([event(3.,'2020-06-18'),event()],['2019-05-01'])
        pd.testing.assert_frame_equal(a,b)

    def test_quarter_difference_requires_compatible_basis(self):
        curr=dict(status='available',ts_code='A',field='cfo_ytd',scope='consolidated',basis_id='first_release',unit='CNY',period_end='2021-09-30',value=20.,published_date=pd.Timestamp('2021-10-30'),lineage='c')
        prev=dict(curr,period_end='2021-06-30',value=5.,published_date=pd.Timestamp('2021-08-20'),lineage='p')
        self.assertEqual(quarter_difference(curr,prev)['value'],15.)
        with self.assertRaises(ValueError):quarter_difference(curr,dict(prev,basis_id='restated'))
        with self.assertRaises(ValueError):quarter_difference(curr,dict(prev,scope='parent_only'))
        with self.assertRaises(ValueError):quarter_difference(curr,dict(prev,period_end='2020-12-31'))


if __name__=='__main__':unittest.main()
