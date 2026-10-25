"""财务公告边界、晚到修订与未知来源的真实失败模式。"""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from financial_candidates import join_financial_candidates, require_verified_financial_history


class FinancialCandidatesTest(unittest.TestCase):
    def reports(self):
        return pd.DataFrame([
            ['A', '20210420', '20210331', 10., 20.],
            ['A', '20220420', '20220331', 12., 25.],
            ['A', '20220510', '20210331', 8., 19.],
        ], columns=['ts_code', 'ann_date', 'end_date', 'q_sales_yoy', 'q_ocf_to_sales'])

    def join(self, dates, reports=None):
        return join_financial_candidates(pd.DataFrame({'date': dates, 'ts_code': 'A'}),
                                         self.reports() if reports is None else reports)

    def test_same_day_excluded_late_revision_is_not_backdated(self):
        got = self.join(['2022-04-20', '2022-04-21', '2022-05-10', '2022-05-11'])
        self.assertFalse(got.iloc[0].numeric_available)
        self.assertEqual(list(got.sales_growth_change_pp.iloc[1:]), [2., 2., 4.])
        self.assertEqual(list(got.cash_conversion_change_pp.iloc[1:]), [5., 5., 6.])
        self.assertTrue(got.report_end.iloc[1:].eq(pd.Timestamp('2022-03-31')).all())

    def test_new_missing_report_does_not_fall_back(self):
        data = pd.concat([self.reports(), pd.DataFrame([
            ['A', '20220801', '20220630', np.nan, 1.]], columns=self.reports().columns)])
        got = self.join(['2022-08-02'], data).iloc[0]
        self.assertEqual(got.report_end, pd.Timestamp('2022-06-30'))
        self.assertFalse(got.numeric_available)
        self.assertTrue(pd.isna(got.q_sales_yoy))

    def test_conflicting_same_day_revision_rejected(self):
        data = self.reports()
        conflict = data.iloc[[0]].copy()
        conflict['q_sales_yoy'] = 100.
        with self.assertRaises(ValueError):
            self.join(['2022-04-21'], pd.concat([data, conflict]))

    def test_future_and_row_order_do_not_change_past(self):
        data = self.reports()
        future = pd.DataFrame([['A', '20231001', '20230930', 999., 999.]], columns=data.columns)
        left = self.join(['2022-04-21'])
        right = self.join(['2022-04-21'], pd.concat([data, future]).iloc[::-1])
        pd.testing.assert_frame_equal(left, right)

    def test_missing_code_remains_and_unverified_blocks_training(self):
        got = join_financial_candidates(pd.DataFrame({'date': ['2022-04-21'], 'ts_code': ['B']}), self.reports())
        self.assertEqual(len(got), 1)
        self.assertFalse(got.iloc[0].numeric_available)
        with self.assertRaises(ValueError):
            require_verified_financial_history(self.join(['2022-04-21']))

    def test_same_day_annual_and_quarter_uses_latest_period(self):
        data = self.reports()
        annual = pd.DataFrame([['A', '20220420', '20211231', 800., 800.]], columns=data.columns)
        got = self.join(['2022-04-21'], pd.concat([data, annual])).iloc[0]
        self.assertEqual(got.report_end, pd.Timestamp('2022-03-31'))
        self.assertEqual(got.sales_growth_change_pp, 2.)

    def test_invalid_disclosure_date_rejected(self):
        data = self.reports()
        data.loc[0, 'ann_date'] = '20210301'
        with self.assertRaises(ValueError):
            self.join(['2022-04-21'], data)

    def test_nonstandard_report_preserves_identity_without_quarter_features(self):
        data = self.reports()
        unusual = pd.DataFrame([['A', '20220601', '20220430', 5., 6.]], columns=data.columns)
        got = self.join(['2022-06-02'], pd.concat([data, unusual])).iloc[0]
        self.assertEqual(got.report_end, pd.Timestamp('2022-04-30'))
        self.assertFalse(got.standard_quarter)
        self.assertFalse(got.numeric_available)


if __name__ == '__main__':
    unittest.main()
