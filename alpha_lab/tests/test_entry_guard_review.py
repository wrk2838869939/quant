"""独立审查新仓门：截断、受阻持仓、整手预约和基线机械等价。"""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.append(str(ROOT / 'scripts'))
import guarded_selection as guarded
from entry_guard import EntryGuard
from selection_first import plan_selection_first as original


def row(code, **updates):
    result = dict(ts_code=code, price=10., expected_return=.1, shares=0,
                  available=0, min_buy=100, buy_step=100, sell_step=100,
                  max_notional=10000., can_buy=True, can_sell=True,
                  min_target_shares=0)
    result.update(updates)
    return result


class EntryGuardIndependentReview(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(correlation_window=120, minimum_observations=120,
                               maximum_correlation=.7, minimum_collective_exits=2,
                               collective_exit_fraction=.5)
        self.dates = pd.bdate_range('2022-01-03', periods=160)
        rng = np.random.default_rng(146)
        a, b = rng.normal(size=(2, 160))
        self.returns = pd.DataFrame(dict(A=a, B=a, C=b, D=-a), index=self.dates)
        self.date = self.dates[119]
        self.saved = guarded.ACTIVE_GUARD

    def tearDown(self):
        guarded.ACTIVE_GUARD = self.saved

    def allocate(self, rows, variant='correlation', exits=(), **kwargs):
        guarded.ACTIVE_GUARD = EntryGuard(self.returns, variant, self.parameters)
        args = dict(priorities={r['ts_code']: 10-i for i, r in enumerate(rows)},
                    cash=5000., buy_rate=.001, sell_rate=.002, min_fee=5.,
                    forecast_kind='expected_simple_return', max_positions=5)
        args.update(kwargs)
        return guarded.plan_selection_first(pd.DataFrame(rows), decision_date=self.date,
                                             trend_exit_codes=exits, **args)

    def test_future_returns_cannot_change_pair_or_admission(self):
        first = EntryGuard(self.returns, 'correlation', self.parameters)
        poisoned = self.returns.copy()
        poisoned.loc[poisoned.index > self.date, 'B'] = -poisoned.loc[poisoned.index > self.date, 'A'] * 1e6
        second = EntryGuard(poisoned, 'correlation', self.parameters)
        self.assertEqual(first.reason(self.date, 'B', ['A'], False), 'correlation_limit')
        self.assertEqual(first.pair(self.date, 'A', 'B'), second.pair(self.date, 'A', 'B'))
        self.assertEqual(second.reason(self.date, 'B', ['A'], False), 'correlation_limit')
        self.assertLessEqual(second.pair_audit[0]['last_observation'], self.date)
        self.assertEqual(second.pair_audit[0]['observations'], 120)

    def test_missing_and_constant_returns_are_unknown_not_zero(self):
        for mode in ['missing', 'constant']:
            data = self.returns.copy()
            if mode == 'missing':
                data.loc[self.date, 'B'] = np.nan
            else:
                data['B'] = 0.
            guard = EntryGuard(data, 'correlation', self.parameters)
            self.assertEqual(guard.reason(self.date, 'B', ['A'], False), 'unknown_correlation')

    def test_blocked_exit_remains_correlation_reference_and_not_cash(self):
        result = self.allocate([row('A', shares=200, available=200,
                                   max_notional=0., can_sell=False), row('B')])
        self.assertEqual(result['retained_codes'], ['A'])
        self.assertEqual(result['selected_codes'], [])
        self.assertEqual(result['cash_after_required_sales'], 5000.)
        plan = result['plan'].set_index('ts_code')
        self.assertEqual(plan.loc['A', 'target_shares'], 200)
        self.assertEqual(result['selection_audit'].iloc[0].reason, 'correlation_limit')

    def test_uneconomic_candidate_does_not_occupy_correlation_reference(self):
        result = self.allocate([row('A', expected_return=.004, max_notional=1000.), row('B')])
        self.assertEqual(result['selected_codes'], ['B'])
        audit = result['selection_audit'].set_index('ts_code')
        self.assertEqual(audit.loc['A', 'reason'], 'no_positive_net_profit_within_cap_and_cash')
        self.assertEqual(guarded.ACTIVE_GUARD.candidate_audit[-1]['references'], '')

    def test_successful_new_reservations_constrain_later_new_names(self):
        result = self.allocate([row('A'), row('B'), row('D')])
        self.assertEqual(result['selected_codes'], ['A', 'D'])
        self.assertEqual(result['selection_audit'].set_index('ts_code').loc['B', 'reason'], 'correlation_limit')
        self.assertGreater(result['reserved_shares']['A'], 0)

    def test_collective_pause_uses_requests_even_when_exits_blocked_and_resets(self):
        rows = [row('A', shares=200, available=200, can_sell=False, max_notional=0.),
                row('B', shares=200, available=200, can_sell=False, max_notional=0.), row('C')]
        result = self.allocate(rows, variant='pause', exits=['A', 'B'])
        self.assertEqual(result['selected_codes'], [])
        guard = guarded.ACTIVE_GUARD
        self.assertTrue(guard.review_audit[-1]['paused'])
        self.assertEqual(result['cash_after'], 5000.)
        self.assertFalse(guard.pause(self.dates[140], pd.DataFrame(rows), []))
        self.assertFalse(guard.pause(self.dates[140], pd.DataFrame(rows), ['A']))

    def test_baseline_selector_matches_original_all_outputs(self):
        rows = [row('A', shares=200, available=200, max_notional=1000.),
                row('B', expected_return=.004, max_notional=1000.), row('C'), row('D')]
        args = dict(priorities=dict(A=4, B=3, C=2, D=1), cash=4000., buy_rate=.001,
                    sell_rate=.002, min_fee=5., forecast_kind='expected_simple_return', max_positions=3)
        expected = original(pd.DataFrame(rows), **args)
        actual = self.allocate(rows, variant='baseline', exits=['A', 'B'], **args)
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            if isinstance(expected[key], pd.DataFrame):
                pd.testing.assert_frame_equal(expected[key], actual[key])
            elif isinstance(expected[key], float):
                self.assertAlmostEqual(expected[key], actual[key])
            else:
                self.assertEqual(expected[key], actual[key])

    def test_reverse_helper_changes_actual_receiver_and_restores_after_exception(self):
        from run_entry_guard_research import reversed_guard_input
        received, evidence = [], []

        def receiver(candidates, **kwargs):
            received.append(candidates.ts_code.tolist())
            return 'actual_receiver_called'

        with patch.object(guarded, 'plan_selection_first', receiver):
            with self.assertRaisesRegex(RuntimeError, 'synthetic_stop'):
                with reversed_guard_input(True, evidence):
                    result = guarded.plan_selection_first(pd.DataFrame([row('A'), row('C'), row('B')]),
                                                          decision_date=self.date)
                    self.assertEqual(result, 'actual_receiver_called')
                    raise RuntimeError('synthetic_stop')
            self.assertIs(guarded.plan_selection_first, receiver)
            with reversed_guard_input(False, evidence):
                guarded.plan_selection_first(pd.DataFrame([row('A'), row('B')]), decision_date=self.date)
        self.assertEqual(received, [['B', 'C', 'A'], ['A', 'B']])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]['actual'], evidence[0]['before'][::-1])
        self.assertEqual(evidence[0]['function_module'], 'guarded_selection')

    def test_portfolio_copy_only_has_review_context_plumbing(self):
        original_source = (ROOT / 'src/portfolio.py').read_text()
        changed = (ROOT / 'src/portfolio_entry_guard.py').read_text()
        edits = [
            ('                new_trend_exit_codes = []\n', ''),
            ("                            if reason == 'hold_false':\n                                new_trend_exit_codes.append(code)\n", ''),
            ('from guarded_selection import plan_selection_first', 'from selection_first import plan_selection_first'),
            ("                        selection_kwargs['decision_date'] = signal_date\n", ''),
            ("                        selection_kwargs['trend_exit_codes'] = new_trend_exit_codes\n", ''),
        ]
        for before, after in edits:
            self.assertEqual(changed.count(before), 1)
            changed = changed.replace(before, after)
        self.assertEqual(ast.dump(ast.parse(changed)), ast.dump(ast.parse(original_source)))


if __name__ == '__main__':
    unittest.main()
