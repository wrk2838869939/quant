"""联合选择的时间边界、非单调比较与切换状态契约。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from policy_balance import RollingPolicy, select_balance, stable_update, quarter_excess


class BalanceTest(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/policy_balance_v1.yaml').read_text())

    def row(self, k=8, score=.1, **extra):
        row = dict(id=f'k{k}', window_months=6, generation=k, retrain_months=3,
                   evaluation_end='2021-12-31', quarters=4, score=score, log_excess=.1,
                   metrics=dict(feasible=True, total=.1))
        row.update(extra)
        return row

    def test_future_and_same_day_rejected(self):
        for end in ['2022-01-01', '2022-02-01']:
            with self.assertRaises(ValueError):
                select_balance([self.row(evaluation_end=end)], '2022-01-01', self.cfg['selection'])

    def test_nonmonotonic_interior_wins(self):
        rows = [self.row(8, .02), self.row(32, .10), self.row(64, .03)]
        self.assertEqual(select_balance(rows, '2022-01-01', self.cfg['selection'])['generation'], 32)

    def test_cost_tie_considers_retraining(self):
        rows = [self.row(8, .10, retrain_months=3), self.row(16, .098, retrain_months=12)]
        self.assertEqual(select_balance(rows, '2022-01-01', self.cfg['selection'])['generation'], 16)

    def test_infeasible_and_negative_excess_fallback(self):
        rows = [self.row(metrics=dict(feasible=False, total=10)), self.row(log_excess=-.01)]
        self.assertIsNone(select_balance(rows, '2022-01-01', self.cfg['selection']))

    def test_incomplete_history_cannot_win(self):
        self.assertIsNone(select_balance([self.row(quarters=3)], '2022-01-01', self.cfg['selection']))

    def test_training_equal_fitness_keeps_incumbent(self):
        row = dict(metrics=dict(feasible=True, total=.2, mdd=-.1, worst_block_mdd=-.1, turnover=1))
        old = np.array([1., 2.])
        evaluated = [(np.array([3., 4.]), dict(row)) for _ in range(12)]
        center, theta, champion, *_ = stable_update(evaluated, old, row, 1., self.cfg['search'])
        np.testing.assert_array_equal(theta, old)
        self.assertIs(champion, row)
        self.assertFalse(np.array_equal(center, old))

    def test_better_candidate_still_replaces(self):
        row = dict(metrics=dict(feasible=True, total=.2, mdd=-.1, worst_block_mdd=-.1, turnover=1))
        better = dict(metrics=dict(row['metrics'], total=.3))
        candidate = np.array([3., 4.])
        _, theta, champion, *_ = stable_update([(candidate, better)] * 12, np.array([1., 2.]), row, 1., self.cfg['search'])
        np.testing.assert_array_equal(theta, candidate)
        self.assertIs(champion, better)

    def test_state_age_preserved_across_different_axes(self):
        class Dummy:
            def __init__(self, dates):
                self.lookup = {d: i for i, d in enumerate(dates)}
                self.held_since, self.entry_price = {}, {}
                self.peak = self.previous_equity = 100000
            def __call__(self, snapshot):
                return self.lookup[snapshot['date']] - self.held_since.get('x', self.lookup[snapshot['date']])
        dates = pd.bdate_range('2022-01-03', periods=8)
        old, new = Dummy(dates), Dummy(dates[2:])
        old.held_since, old.entry_price = {'x': 1}, {'x': 23.}
        old.peak, old.previous_equity = 120000, 110000
        policy = RollingPolicy({dates[0]: old, dates[4]: new})
        self.assertEqual(policy({'date': dates[3]}), 2)
        self.assertEqual(policy({'date': dates[4]}), 3)
        self.assertEqual(new.peak, 120000)
        self.assertEqual(new.previous_equity, 110000)
        self.assertEqual(new.entry_price, {'x': 23.})
        self.assertEqual(policy({'date': dates[5]}), 4)

    def test_calendar_matched_excess(self):
        idx = pd.to_datetime(['2021-03-31', '2021-04-01'])
        a = pd.DataFrame({'ret': [.1, -.1]}, index=idx)
        b = pd.DataFrame({'ret': [.05, -.05]}, index=idx)
        q = quarter_excess(a, b)
        self.assertEqual(len(q), 2)
        np.testing.assert_allclose(q.sum(), np.log(.99 / .9975))
        with self.assertRaises(AssertionError):
            quarter_excess(a, b.iloc[:1])


if __name__ == '__main__':
    unittest.main()
