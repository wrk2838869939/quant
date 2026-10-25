import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from risk_states import fixed_risk_plan,market_risk_features
from portfolio import run_backtest,AccountingError
import test_portfolio


class RiskLedgerTest(unittest.TestCase):
    def setUp(self):
        self.f=test_portfolio.PortfolioTest();self.f.setUp()
        self.gate=pd.Series(1,index=self.f.dates)

    def replay(self,age=21):
        f=self.f
        return run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,
            f.dates[0],f.dates[-1],f.cfg,risk_allowed=self.gate,risk_signal_max_age=age)

    def test_all_allowed_matches_existing_ledger(self):
        a=self.f.run_case();b=self.replay()
        pd.testing.assert_frame_equal(a.daily,b.daily[a.daily.columns])
        pd.testing.assert_frame_equal(a.trades,b.trades)

    def test_close_signal_exits_next_open_and_reenters_between_rebalances(self):
        self.gate.iloc[2:5]=0
        r=self.replay()
        self.assertEqual(r.trades.side.tolist(),['buy','sell','buy'])
        self.assertEqual(r.trades.date.tolist(),[self.f.dates[1],self.f.dates[3],self.f.dates[6]])
        self.assertEqual(r.daily.loc[self.f.dates[2],'n_hold'],1)
        self.assertEqual(r.daily.loc[self.f.dates[3],'risk_state'],'cash')

    def test_recovery_does_not_cancel_unfilled_exit(self):
        self.gate.iloc[2:4]=0
        self.f.edit(3,open=np.nan,close=np.nan,vol_share=0.)
        self.f.edit(4,open=np.nan,close=np.nan,vol_share=0.)
        r=self.replay()
        self.assertEqual(r.daily.loc[self.f.dates[4],'risk_state'],'exit_pending')
        self.assertEqual(r.trades.date.tolist(),[self.f.dates[1],self.f.dates[5],self.f.dates[6]])

    def test_stale_reentry_waits_for_new_signal(self):
        self.gate.iloc[2:8]=0
        r=self.replay(age=5)
        self.assertEqual(r.trades.side.tolist(),['buy','sell'])
        self.assertEqual(r.daily.iloc[-1].risk_state,'reentry_wait')

    def test_missing_risk_date_is_fatal(self):
        self.gate=self.gate.iloc[:-1]
        with self.assertRaises(AccountingError):self.replay()

    def test_bonus_locks_survive_risk_exit(self):
        self.f.action(cash=0.,bonus=.5,release=5)
        self.gate.iloc[1:]=0
        r=self.replay()
        sells=r.trades[r.trades.side=='sell']
        self.assertEqual(sells.shares.tolist(),[1000,500])
        self.assertEqual(sells.date.tolist(),[self.f.dates[2],self.f.dates[5]])
        self.assertEqual(r.daily.loc[self.f.dates[4],'risk_state'],'exit_pending')


class RiskFeatureTest(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((ROOT/'configs/active_cash_audit.yaml').read_text())

    def test_recovery_confirmation_and_prefix_invariance(self):
        dates=pd.bdate_range('2020-01-01',periods=8)
        f=pd.DataFrame({'trend_return':[.1,.1,.1,-.1,.1,.1,.1,.1],
                        'above_ma_ratio':[.6,.6,.6,.2,.6,.6,.6,.6]},index=dates)
        p=fixed_risk_plan(f,'breadth_trend',self.cfg)
        self.assertEqual(p.allowed.tolist(),[0,0,1,0,0,0,1,1])
        pd.testing.assert_frame_equal(p.iloc[:5],fixed_risk_plan(f.iloc[:5],'breadth_trend',self.cfg))

    def test_future_shock_cannot_change_past_features(self):
        cfg=dict(self.cfg,min_market_names=1,trend_days=2,median_days=2,
                 volatility_days=2,quantile_days=3,short_return_days=2)
        x=pd.DataFrame(dict(ts_code=['a']*12,date=pd.bdate_range('2020-01-01',periods=12),
                            pct_chg=[1.,-1.,2.,-.5]*3))
        full=market_risk_features(x,cfg)
        x.loc[10:,'pct_chg']=90.
        shocked=market_risk_features(x,cfg)
        pd.testing.assert_frame_equal(full.iloc[:10],shocked.iloc[:10])
        past=full.volatility.iloc[1:4].quantile(.8)
        self.assertAlmostEqual(full.past_vol_quantile.iloc[4],past)
