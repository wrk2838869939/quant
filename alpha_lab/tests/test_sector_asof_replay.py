"""时间边界与未来扰动测试。"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'alpha_lab/src'), str(ROOT / 'alpha_lab/experiments/sector_fuzzy_v1')]
from sector_asof_replay import announced_normal_names, history_before, training_plan, fit_asof, predict_asof


class AsOfTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range('2018-01-01', '2022-12-31')
        dates = self.calendar[::10]
        rng = np.random.default_rng(73)
        self.data = pd.DataFrame(dict(event_id=[str(i) for i in range(len(dates))], date=dates,
                                     end_date=dates + pd.offsets.BDay(20), complete=True,
                                     target=rng.normal(0, .1, len(dates)), x=rng.normal(size=len(dates))))
        self.cfg = dict(embargo_sessions=20, training_years=8, minimum_train_events=8,
                        minimum_validation_events=4, minimum_validation_dates=4, validation_fraction=.25,
                        maximum_epochs=20, rules=[2], learning_rates=[.003], tuning_seeds=[17],
                        ensemble_seeds=[17], half_life_days=1095, retention=.5)
        self.mcfg = dict(features=['x'], target_scale=.1, scaler_floor=.001, feature_clip=5,
                         fnn=dict(width_initial=1.5, width_floor=.1, weight_decay=.01,
                                  minimum_epochs=10, check_interval=10, patience_epochs=10,
                                  improvement_tolerance=1e-5, gradient_clip=5))

    def test_mature_labels_and_whole_dates(self):
        train, fit, valid, cutoff, inner_cutoff = training_plan(self.data, '2021-01-01', self.calendar, self.cfg)
        self.assertTrue(train.end_date.lt(cutoff).all())
        self.assertTrue(fit.end_date.lt(inner_cutoff).all())
        self.assertTrue(set(fit.date).isdisjoint(valid.date))
        self.assertTrue(train.date.lt(pd.Timestamp('2021-01-01')).all())

    def test_future_label_and_feature_poison_leaves_model_identical(self):
        start = pd.Timestamp('2021-01-01')
        original = fit_asof(self.data, start, self.calendar, self.cfg, self.mcfg)
        poisoned = self.data.copy()
        _, cutoff = history_before(poisoned, start, self.calendar, self.cfg)
        future = poisoned.end_date.ge(cutoff)
        poisoned.loc[future, ['target', 'x']] = 1e9
        changed = fit_asof(poisoned, start, self.calendar, self.cfg, self.mcfg)
        self.assertEqual(original['audit'], changed['audit'])
        probe = pd.DataFrame({'x': [-1., 0., 1.]})
        np.testing.assert_array_equal(predict_asof(original, probe, self.mcfg)[0], predict_asof(changed, probe, self.mcfg)[0])
        prefix = self.data[self.data.date.lt(start)]
        cut = fit_asof(prefix, start, self.calendar, self.cfg, self.mcfg)
        np.testing.assert_array_equal(predict_asof(original, probe, self.mcfg)[0], predict_asof(cut, probe, self.mcfg)[0])

    def test_name_announcement_unknown_and_conflict_rejected(self):
        names = pd.DataFrame(dict(ts_code=['known', 'future', 'same', 'unknown', 'st', 'dup', 'dup'],
                                  name=['正常', '正常', '正常', '正常', '*ST某', '正常', '正常'],
                                  start_date=pd.Timestamp('2020-01-01'), end_date=pd.NaT,
                                  ann_date=pd.to_datetime(['2019-12-31', '2021-01-02', '2021-01-01', None,
                                                          '2019-12-31', '2019-12-31', '2019-12-31'])))
        codes = ['known', 'future', 'same', 'unknown', 'st', 'dup', 'missing']
        result = announced_normal_names(names, pd.Timestamp('2021-01-01'), codes)
        self.assertEqual(result.tolist(), [True, False, False, False, False, False, False])

    def test_insufficient_past_is_rejected(self):
        with self.assertRaises(ValueError):
            training_plan(self.data, '2018-02-01', self.calendar, self.cfg)


if __name__ == '__main__':
    unittest.main()
