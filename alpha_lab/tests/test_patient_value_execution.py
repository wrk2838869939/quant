"""执行驱动的三态确认、准入/点预测分离与空仓归因测试。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from portfolio import BacktestResult
from run_patient_value_execution import confirmed_hold_states, make_policy_matrices, process_report, case_matrix


class KnownPredictions:
    def predict(self, frame):
        return frame[['expected_return', 'expected_adverse', 'mean_lower', 'mean_upper']].copy()


class NegativeConstant:
    def predict(self, frame):
        return pd.DataFrame({'expected_return': [-.01]*len(frame)}, index=frame.index)


class PatientValueExecutionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT/'configs/autonomous_value_research.yaml').read_text())

    def test_two_failures_unknown_break_and_prefix_invariance(self):
        dates = pd.bdate_range('2022-01-01', periods=6)
        trend = pd.DataFrame({'stock': [False, False, False, True, False, False]}, index=dates)
        known = pd.DataFrame({'stock': [True, True, False, True, True, True]}, index=dates)
        full = confirmed_hold_states(trend, known, 2)
        np.testing.assert_allclose(full.stock, [1., 0., np.nan, 1., 1., 0.], equal_nan=True)
        pd.testing.assert_frame_equal(confirmed_hold_states(trend.iloc[:4], known.iloc[:4], 2), full.iloc[:4])

    def test_fixed_twenty_one_cases(self):
        cases = case_matrix(self.cfg)
        self.assertEqual(len(cases), 21)
        self.assertEqual(len({c['name'] for c in cases}), 21)
        stress = [c for c in cases if c['name'].startswith('stress')]
        self.assertEqual(len(stress), 3)
        self.assertTrue(all(c['capital']==50000 and c['frequency']=='monthly'
                            and c['delay']==1 and c['slippage_rate']==.001 for c in stress))

    def test_confidence_only_gates_new_entry_and_never_replaces_point_mu(self):
        date = pd.Timestamp('2022-01-04')
        table = pd.DataFrame(dict(date=[date]*4, ts_code=list('ABCD'), monthly=True,
            trend=True, x_turnover=[-.2, -.1, .1, .2],
            expected_return=[.1, .08, .06, .07], expected_adverse=[.05, .2, .1, .1],
            mean_lower=[-.1, .02, .015, .02], mean_upper=[.2]*4))
        holds = {'monthly': pd.DataFrame(True, index=[date], columns=list('ABCD'))}
        matrices, _, _ = make_policy_matrices(table, holds,
            {'full': KnownPredictions(), 'constant': NegativeConstant()}, self.cfg)
        signal, mu = matrices[('monthly', 'full_confidence')]
        self.assertTrue(signal[['A', 'B']].isna().all().all())
        self.assertEqual(signal.loc[date].idxmax(), 'D')
        self.assertEqual(mu.loc[date, 'A'], .1)
        self.assertEqual(mu.loc[date, 'C'], .06)
        low_signal, low_mu = matrices[('monthly', 'low_turnover')]
        self.assertTrue(low_signal.isna().all().all())
        self.assertTrue(low_mu.eq(-.01).all().all())

    def test_empty_cash_case_has_zero_pnl_and_readable_reports(self):
        dates = pd.bdate_range('2022-01-01', periods=3)
        daily = pd.DataFrame(dict(cash=1000., stock_value=0., receivables=0., equity=1000., n_hold=0), index=dates)
        result = BacktestResult(daily, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
        process, pnl, episodes, exits, stats = process_report(result, 1000.)
        self.assertEqual(stats['asset_pnl_max_error'], 0.)
        self.assertEqual(stats['n_cash_days'], 3)
        self.assertEqual(stats['max_holdings'], 0)
        self.assertTrue(process.two_sided_turnover.eq(0).all())
        self.assertEqual(list(pnl.columns), ['ts_code', 'pnl_yuan'])
        self.assertTrue(episodes.empty and exits.empty)

    def test_dividend_payment_not_double_counted_in_asset_pnl(self):
        dates = pd.bdate_range('2022-01-01', periods=3)
        daily = pd.DataFrame(dict(cash=[0., 0., 10.], stock_value=1000.,
            receivables=[0., 10., 0.], equity=[1000., 1010., 1010.], n_hold=1), index=dates)
        trades = pd.DataFrame([dict(date=dates[0], ts_code='A', side='buy', shares=100,
            price=10., gross=1000., fee=0., cash_after=0.)])
        holdings = pd.DataFrame([dict(date=d, ts_code='A', shares=100, value=1000.) for d in dates])
        events = pd.DataFrame([dict(date=dates[1], ts_code='A', kind='dividend_receivable', amount=10.),
                              dict(date=dates[2], ts_code='A', kind='dividend_paid', amount=10.)])
        _, pnl, _, _, stats = process_report(BacktestResult(daily, trades, events, holdings, {}), 1000.)
        self.assertEqual(pnl.pnl_yuan.iloc[0], 10.)
        self.assertEqual(stats['asset_pnl_max_error'], 0.)


if __name__ == '__main__':
    unittest.main()
