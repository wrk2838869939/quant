import unittest,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from patient_trend import review_dates,trend_state
from retail_rebalance import plan_rebalance


class PatientTrendTest(unittest.TestCase):
    def test_calendar_prefix_invariance_at_midweek_and_midmonth(self):
        calendar=pd.bdate_range('2020-01-01',periods=150)
        for freq in ['weekly','monthly']:
            full=review_dates(calendar,freq)
            for stop in [3,17,45,79]:
                pd.testing.assert_index_equal(review_dates(calendar[:stop],freq),full[full<=calendar[stop-1]])

    def test_future_prices_cannot_change_past_trend(self):
        dates=pd.bdate_range('2020-01-01',periods=200)
        close=pd.DataFrame({'rise':np.exp(np.arange(200)*.001),'fall':np.exp(-np.arange(200)*.001)},index=dates)
        spec=dict(medium_window=60,long_window=120,min_return=0.,require_above_long_ma=True)
        full,state=trend_state(close,spec)
        self.assertTrue(state.rise.iloc[120:].all());self.assertFalse(state.fall.any())
        short,_=trend_state(close.iloc[:150],spec)
        pd.testing.assert_frame_equal(short,full.iloc[:150])

    def test_six_affordable_candidates_do_not_force_six_holdings(self):
        rows=[dict(ts_code=str(i),price=10,expected_return=.1+i*.01,shares=0,available=0,min_buy=100,
            buy_step=100,sell_step=100,max_notional=2000,can_buy=True,can_sell=True) for i in range(6)]
        result=plan_rebalance(pd.DataFrame(rows),cash=20000,buy_rate=0.,sell_rate=0.,min_fee=5.,
            forecast_kind='expected_simple_return',max_positions=5)
        self.assertEqual(result['plan'].target_shares.gt(0).sum(),5)
        self.assertGreater(result['cash_after'],0)


if __name__=='__main__':unittest.main()
