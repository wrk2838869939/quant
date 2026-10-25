"""个股排序的时点边界与拒绝规则测试。"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'alpha_lab/src'))
from stock_selection_asof import cross_section_features, fit_stock_model, mature_history, select_positive_top


class StockSelectionAsOfTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range('2015-01-01', '2023-12-31')
        rows = []
        rng = np.random.default_rng(71)
        for i, date in enumerate(self.calendar[::20]):
            for j in range(5):
                rows.append(dict(event_id=f'e{i}', date=date, ts_code=str(j), exit_date=date + pd.offsets.BDay(21),
                                 label_complete=True, target=(j - 2) * .01 + rng.normal(0, .02), feature=(j - 2) / 5))
        self.data = pd.DataFrame(rows)
        self.cfg = dict(embargo_sessions=20, training_years=8, validation_fraction=.25,
                        minimum_fit_events=20, minimum_validation_events=8, minimum_validation_dates=4,
                        alphas=[1., 10., 100.])

    def test_future_poison_and_prefix_produce_identical_model(self):
        start = pd.Timestamp('2021-01-01')
        model, audit = fit_stock_model(self.data, start, self.calendar, self.cfg, ['feature'])
        _, cutoff = mature_history(self.data, start, self.calendar, self.cfg)
        poisoned = self.data.copy()
        poisoned.loc[poisoned.exit_date.ge(cutoff), ['target', 'feature']] = 1e9
        changed, caudit = fit_stock_model(poisoned, start, self.calendar, self.cfg, ['feature'])
        prefix, paudit = fit_stock_model(self.data[self.data.date.lt(start)], start, self.calendar, self.cfg, ['feature'])
        self.assertEqual(audit, caudit)
        self.assertEqual(audit, paudit)
        np.testing.assert_array_equal(model.coef_, changed.coef_)
        np.testing.assert_array_equal(model.coef_, prefix.coef_)
        fit = self.data[self.data.event_id.isin(audit['fit_events'])]
        valid = self.data[self.data.event_id.isin(audit['valid_events'])]
        self.assertTrue(set(fit.date).isdisjoint(valid.date))
        self.assertTrue(fit.exit_date.lt(pd.Timestamp(audit['inner_cutoff'])).all())

    def test_cross_section_does_not_mix_future_events(self):
        frame = pd.DataFrame(dict(event_id=['a', 'a', 'b', 'b'], date=pd.to_datetime(['2020-01-01'] * 2 + ['2021-01-01'] * 2),
                                  ts_code=['1', '2', '1', '2'], strength=[1., 2., 1e9, np.nan], trend=[True, False, True, True], price=[True] * 4))
        cfg = dict(rank_features={'strength': True}, binary_features=['trend', 'price'], include_interaction=True)
        full, columns = cross_section_features(frame, cfg)
        prefix, _ = cross_section_features(frame[frame.event_id.eq('a')], cfg)
        pd.testing.assert_frame_equal(full[full.event_id.eq('a')], prefix)
        self.assertTrue(full[columns].notna().all().all())

    def test_no_forced_selection_and_insufficient_history(self):
        frame = pd.DataFrame(dict(event_id=['a'] * 3, ts_code=['1', '2', '3'], price=[True, True, False], score=[-.1, 0., .3]))
        config = dict(top_n=5, model={'minimum_predicted_excess': 0.})
        self.assertEqual(len(select_positive_top(frame, 'score', config)), 0)
        with self.assertRaises(ValueError):
            fit_stock_model(self.data, '2015-02-01', self.calendar, self.cfg, ['feature'])


if __name__ == '__main__':
    unittest.main()
