"""名单优先适配器的整手、现金、保留旧仓与顺序不变性合成检验。"""
from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from retail_rebalance import plan_rebalance
from selection_first import plan_selection_first


def row(code, **kwargs):
    values = dict(ts_code=code, price=10., expected_return=.1, shares=0,
                  available=0, min_buy=100, buy_step=100, sell_step=100,
                  max_notional=10000., can_buy=True, can_sell=True,
                  min_target_shares=0)
    values.update(kwargs)
    return values


class SelectionFirstTest(unittest.TestCase):
    def allocate(self, rows, priorities=None, **kwargs):
        args = dict(cash=5000., buy_rate=.001, sell_rate=.002, min_fee=5.,
                    forecast_kind='expected_simple_return', max_positions=2)
        args.update(kwargs)
        priorities = priorities if priorities is not None else {r['ts_code']: 1. for r in rows}
        result = plan_selection_first(pd.DataFrame(rows), priorities=priorities, **args)
        plan = result['plan']
        cash = args['cash'] + (plan.sell_shares * plan.price - plan.sell_fee
            - plan.buy_shares * plan.price - plan.buy_fee - plan.slippage_cost).sum()
        self.assertAlmostEqual(cash, result['cash_after'])
        self.assertGreaterEqual(cash, -1e-6)
        self.assertLessEqual(int(plan.target_shares.gt(0).sum()), args['max_positions'])
        np.testing.assert_array_equal(plan.target_shares,
                                      plan.shares + plan.buy_shares - plan.sell_shares)
        return result

    def test_permutation_invariant_selection_and_quantities(self):
        rows = [row('C'), row('A'), row('B')]
        reference = None
        for perm in permutations(rows):
            result = self.allocate(list(perm))
            self.assertEqual(result['selected_codes'], ['A', 'B'])
            plan = result['plan'].set_index('ts_code')
            if reference is None:
                reference = plan
            pd.testing.assert_frame_equal(reference, plan)

    def test_priority_selects_names_even_when_other_has_larger_return(self):
        result = self.allocate([row('ranked', expected_return=.03), row('other', expected_return=.8)],
                               priorities={'ranked': 2, 'other': 1}, max_positions=1)
        self.assertEqual(result['selected_codes'], ['ranked'])
        self.assertEqual(result['plan'].set_index('ts_code').loc['other', 'buy_shares'], 0)

    def test_expensive_minimum_lot_not_rejected_by_average_seat_budget(self):
        result = self.allocate([row('expensive', price=80.), row('cheap')],
                               priorities={'expensive': 2, 'cheap': 1}, cash=10000., max_positions=5)
        self.assertIn('expensive', result['selected_codes'])
        self.assertEqual(result['plan'].set_index('ts_code').loc['expensive', 'buy_shares'], 100)

    def test_cash_shortfall_leaves_fewer_positions(self):
        result = self.allocate([row('A'), row('B')], cash=1500., max_positions=5)
        self.assertEqual(result['selected_codes'], ['A'])
        self.assertIn('insufficient_cash_for_minimum_lot', result['selection_audit'].reason.tolist())

    def test_net_profit_gate_uses_actual_return_not_priority(self):
        result = self.allocate([row('A', expected_return=.001)], priorities={'A': 100000.})
        self.assertEqual(result['selected_codes'], [])
        self.assertEqual(result['cash_after'], 5000.)
        self.assertEqual(result['selection_audit'].reason.iloc[0], 'nonpositive_net_margin')

    def test_multiple_lots_can_amortize_minimum_commission(self):
        result = self.allocate([row('cheap', price=1., expected_return=.03)],
                               cash=405., max_positions=1)
        self.assertEqual(result['selected_codes'], ['cheap'])
        self.assertEqual(result['reserved_shares'], {'cheap': 400})
        self.assertEqual(result['plan'].buy_shares.iloc[0], 400)
        audit = result['selection_audit'].iloc[0]
        self.assertLess(audit.minimum_lot_net_profit, 0)
        self.assertGreater(audit.reserved_net_profit, 0)
        self.assertEqual(audit.reserved_shares, 400)

    def test_cash_for_only_three_lots_cannot_fund_four_lot_economic_block(self):
        result = self.allocate([row('cheap', price=1., expected_return=.03)],
                               cash=305., max_positions=1)
        self.assertEqual(result['selected_codes'], [])
        self.assertEqual(result['plan'].buy_shares.iloc[0], 0)
        self.assertEqual(result['selection_audit'].reason.iloc[0],
                         'no_positive_net_profit_within_cap_and_cash')

    def test_stock_cap_can_prevent_economic_minimum_block(self):
        result = self.allocate([row('cheap', price=1., expected_return=.03, max_notional=300.)],
                               cash=5000., max_positions=1)
        self.assertEqual(result['selected_codes'], [])

    def test_old_position_is_not_replaced_or_topped_up(self):
        old = row('old', shares=200, available=200, min_target_shares=200,
                  expected_return=-.5, can_buy=False)
        result = self.allocate([old, row('new', expected_return=.8)],
                               priorities={'new': 10, 'old': -10}, max_positions=1)
        plan = result['plan'].set_index('ts_code')
        self.assertEqual(plan.loc['old', 'target_shares'], 200)
        self.assertEqual(plan.loc['old', 'sell_shares'], 0)
        self.assertEqual(result['selected_codes'], [])

    def test_explicit_exit_releases_cash_before_new_reservation(self):
        old = row('old', shares=200, available=200, max_notional=0., can_buy=False)
        result = self.allocate([old, row('new')], priorities={'new': 1}, cash=0., max_positions=1)
        plan = result['plan'].set_index('ts_code')
        self.assertEqual(plan.loc['old', 'sell_shares'], 200)
        self.assertEqual(plan.loc['new', 'buy_shares'], 100)
        self.assertEqual(result['cash_after_required_sales'], 1995.)

    def test_unavailable_exit_does_not_release_cash_or_position_slot(self):
        old = row('old', shares=200, available=200, max_notional=0., can_buy=False, can_sell=False)
        result = self.allocate([old, row('new')], priorities={'new': 1}, cash=0., max_positions=1)
        plan = result['plan'].set_index('ts_code')
        self.assertEqual(plan.loc['old', 'target_shares'], 200)
        self.assertEqual(plan.loc['old', 'unavoidable_locked_excess'], 2000.)
        self.assertEqual(result['selected_codes'], [])

    def test_cap_trim_odd_lot_preserves_largest_reachable_holding(self):
        old = row('old', shares=1050, available=1050, max_notional=8200.,
                  min_target_shares=800, can_buy=False, expected_return=-.9)
        result = self.allocate([old], cash=0., max_positions=1)
        self.assertEqual(result['plan'].target_shares.iloc[0], 800)
        self.assertEqual(result['plan'].sell_shares.iloc[0], 250)

    def test_minimum_reservation_fees_are_not_charged_twice(self):
        result = self.allocate([row('A')], cash=5005., buy_rate=.0001, sell_rate=.0001,
                               min_fee=5., max_positions=1)
        self.assertEqual(result['plan'].buy_shares.iloc[0], 500)
        self.assertEqual(result['plan'].buy_fee.iloc[0], 5.)
        self.assertEqual(result['cash_after'], 0.)

    def test_non_main_board_minimum_and_unit_increment(self):
        result = self.allocate([row('star', min_buy=200, buy_step=1, sell_step=1)],
                               cash=2205., max_positions=1)
        self.assertEqual(result['plan'].buy_shares.iloc[0], 220)

    def test_slippage_enters_profit_gate_and_cash_reservation(self):
        result = self.allocate([row('A', expected_return=.01)], cash=1006., slippage_rate=.001)
        self.assertEqual(result['selected_codes'], [])
        result = self.allocate([row('A')], cash=1006., slippage_rate=.001)
        self.assertEqual(result['plan'].buy_shares.iloc[0], 100)
        self.assertAlmostEqual(result['cash_after'], 0.)


class RequiredNewCodesTest(unittest.TestCase):
    def allocate(self, rows, **kwargs):
        args = dict(cash=10000., buy_rate=.001, sell_rate=.002, min_fee=5.,
                    forecast_kind='expected_simple_return', max_positions=2)
        args.update(kwargs)
        return plan_rebalance(pd.DataFrame(rows), **args)

    def test_required_new_is_hard_constraint_but_default_unchanged(self):
        rows = [row('A', expected_return=-.1)]
        self.assertEqual(self.allocate(rows)['plan'].buy_shares.iloc[0], 0)
        self.assertEqual(self.allocate(rows, required_new_codes=['A'])['plan'].buy_shares.iloc[0], 100)

    def test_unknown_old_and_forbidden_codes_rejected(self):
        for rows, codes in [([row('A')], ['missing']),
                            ([row('A', shares=100, available=100)], ['A']),
                            ([row('A', can_buy=False)], ['A'])]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.allocate(rows, required_new_codes=codes)

    def test_impossible_cash_or_position_limit_fails_closed(self):
        for args in [dict(cash=100.), dict(max_positions=0)]:
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                self.allocate([row('A')], required_new_codes=['A'], **args)

    def test_required_new_shares_is_hard_quantity_constraint(self):
        rows = [row('A', expected_return=-.1)]
        plan = self.allocate(rows, required_new_shares={'A': 400})['plan']
        self.assertEqual(plan.buy_shares.iloc[0], 400)

    def test_required_new_shares_must_match_legal_units(self):
        for quantity in [0, 99, 150, 300.5, np.nan]:
            with self.subTest(quantity=quantity), self.assertRaises(ValueError):
                self.allocate([row('A')], required_new_shares={'A': quantity})


if __name__ == '__main__':
    unittest.main()
