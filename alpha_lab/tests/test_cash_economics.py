"""新风险状态的同步峰值、因果前缀和恢复证据验证。"""
import unittest
import numpy as np
import pandas as pd
from cash_economics_state import EconomicCashState
from robust_cash_state import RobustCashState


class EconomicStateTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2023-01-02', periods=160)
        self.close = pd.DataFrame({'A': np.linspace(20,10,160), 'B':np.linspace(30,15,160)}, index=self.dates)
        self.week = self.dates[self.dates.dayofweek == 0][-3]

    def test_synchronous_basket_does_not_mix_individual_peaks(self):
        close = self.close.copy()
        close['A'] = 10 + np.sin(np.arange(160))
        close['B'] = 20 - 2*np.sin(np.arange(160))
        state = EconomicCashState(close, 'basket_dd')
        p = close.loc[self.week]
        state.on_close(self.week, {'A':p.A*100,'B':p.B*50}, 2000)
        self.assertAlmostEqual(state.records[-1]['basket_drawdown'], 0., places=12)
        self.assertGreater(state.records[-1]['drawdown'], .05)

    def test_severe_long_break_full_vs_half(self):
        for mode, expected in [('trend_break',0.),('trend_half',.5)]:
            state = EconomicCashState(self.close, mode)
            cap, paused = state.on_close(self.week, {'A':1000,'B':1000}, 2000)
            self.assertEqual(cap, expected)
            self.assertTrue(paused)

    def test_one_confirm_can_release_before_ret20_positive(self):
        state = EconomicCashState(self.close, 'quick_release')
        state.on_close(self.week, {'A':1000,'B':1000}, 2000)
        nxt = self.week + pd.Timedelta(days=7)
        state.features['weak'].loc[nxt] = False
        state.features['ret20'].loc[nxt] = -.1
        cap, paused = state.on_close(nxt, {}, 2000)
        self.assertEqual(cap, 1.)
        self.assertFalse(paused)
        self.assertTrue(state.recovered_today)

    def test_missing_evidence_cannot_release_frozen_basket(self):
        state = EconomicCashState(self.close, 'quick_release')
        state.on_close(self.week, {'A':1000,'B':1000}, 2000)
        old = state.weights.copy()
        nxt = self.week + pd.Timedelta(days=7)
        state.features['known'].loc[nxt,'A'] = False
        state.features['weak'].loc[nxt] = False
        cap, paused = state.on_close(nxt, {}, 2000)
        self.assertTrue(paused)
        self.assertFalse(state.recovered_today)
        pd.testing.assert_series_equal(old, state.weights)

    def test_future_prices_cannot_change_past_decisions(self):
        cutoff = self.dates[120]
        changed = self.close.copy()
        changed.loc[changed.index>cutoff] *= 3
        for mode in ['quick_release','basket_dd','trend_break','trend_half']:
            results = []
            for frame in [self.close, changed, self.close.loc[:cutoff]]:
                state = EconomicCashState(frame, mode)
                for date in self.dates[:121]:
                    state.on_close(date, {'A':1000,'B':1000},2000)
                results.append(pd.DataFrame(state.records))
            pd.testing.assert_frame_equal(results[0], results[1],check_exact=True)
            pd.testing.assert_frame_equal(results[0], results[2],check_exact=True)

    def test_unknown_mode_and_repeated_date_rejected(self):
        with self.assertRaises(ValueError):
            EconomicCashState(self.close, 'best')
        state = EconomicCashState(self.close, 'trend_break')
        state.on_close(self.week, {}, 2000)
        with self.assertRaises(ValueError):
            state.on_close(self.week, {}, 2000)

    def test_neighbor_center_reproduces_original_state(self):
        a=EconomicCashState(self.close,'trend_break')
        b=RobustCashState(self.close,60,.05)
        for date in self.dates:
            self.assertEqual(a.on_close(date,{'A':1000,'B':1000},2000),
                             b.on_close(date,{'A':1000,'B':1000},2000))
            self.assertEqual(a.recovered_today,b.recovered_today)

    def test_all_neighbor_prefixes_are_causal(self):
        cut=self.dates[120]
        for window,dd in [(40,.05),(80,.05),(60,.04),(60,.06)]:
            a=RobustCashState(self.close,window,dd)
            b=RobustCashState(self.close.loc[:cut],window,dd)
            for date in self.dates[:121]:
                self.assertEqual(a.on_close(date,{'A':1000,'B':1000},2000),
                                 b.on_close(date,{'A':1000,'B':1000},2000))
            pd.testing.assert_frame_equal(pd.DataFrame(a.records),pd.DataFrame(b.records),check_exact=True)

    def test_missing_long_history_not_interpreted_as_complete_trend(self):
        frame=self.close.copy()
        index=frame.index.get_loc(self.week)
        frame.iloc[index-70,:]=np.nan
        state=RobustCashState(frame,80,.05)
        cap,paused=state.on_close(self.week,{'A':1000,'B':1000},2000)
        self.assertEqual(cap,1.)
        self.assertFalse(paused)
        self.assertTrue(np.isnan(state.records[-1]['basket_ret_trend']))


if __name__ == '__main__':
    unittest.main()
