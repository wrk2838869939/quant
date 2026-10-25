"""生命周期、未来截断与标签成熟约束的反例测试。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from exit_history import candidate_spells, annual_fit, predict_frame


class HistoryTests(unittest.TestCase):
    def test_unknown_is_not_exit_and_rank_loss_does_not_sell(self):
        dates = pd.date_range('2020-01-01', periods=5, freq='MS')
        signal = pd.DataFrame({'a': [1., np.nan, np.nan, np.nan, 1.]}, index=dates)
        hold = pd.DataFrame({'a': [1., 1., np.nan, 0., 1.]}, index=dates)
        exits, entries, active = candidate_spells(signal, hold, 1)
        self.assertEqual(exits.date.tolist(), [dates[3]])
        self.assertEqual(entries.date.tolist(), [dates[0], dates[4]])
        self.assertEqual(active.entry_date.tolist(), [dates[4]])
        prefix = candidate_spells(signal.iloc[:4], hold.iloc[:4], 1)
        pd.testing.assert_frame_equal(exits, prefix[0])

    def test_top_n_is_per_review_not_cap_on_shadow_population(self):
        d = pd.date_range('2020-01-01', periods=2, freq='MS')
        s = pd.DataFrame({'a': [2., 1.], 'b': [1., 2.]}, index=d)
        _, entries, active = candidate_spells(s, s * 0 + 1, 1)
        self.assertEqual(len(entries), 2)
        self.assertEqual(len(active), 2)

    def test_future_and_incomplete_labels_cannot_train(self):
        f = pd.DataFrame(dict(date=pd.to_datetime(['2018-01-01', '2018-02-01', '2018-12-01']),
            label_end_60=pd.to_datetime(['2018-04-01', '2018-05-01', '2019-03-01']),
            ret_60=[.1, -.1, 1000.], x=[1., 2., 999.], features_complete=True))
        cfg = dict(features=['x'], model=dict(alpha=10., min_train_events=2, min_train_dates=2))
        train, model, mean, _ = annual_fit(f, 2019, 60, 'ret', cfg)
        self.assertEqual(len(train), 2)
        expected = predict_frame(f, model, mean, ['x'])
        f.loc[2, 'ret_60'] = -1000.
        _, altered, mean2, _ = annual_fit(f, 2019, 60, 'ret', cfg)
        np.testing.assert_array_equal(expected, predict_frame(f, altered, mean2, ['x']))
        f.loc[0, 'features_complete'] = False
        _, fallback, mean3, _ = annual_fit(f, 2019, 60, 'ret', cfg)
        self.assertIsNone(fallback)
        self.assertAlmostEqual(mean3, -.1)


if __name__ == '__main__':
    unittest.main()
