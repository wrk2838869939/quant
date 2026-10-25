"""验证等待、恢复与整手缓冲的经济行为；复用原撮合对抗集。"""
from copy import deepcopy
from pathlib import Path
import sys,unittest
import numpy as np
import pandas as pd
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
from path_mechanism_portfolio import run_backtest
from duration_hold import duration_holds
import test_causal_portfolio_adversarial as fixtures

class PathMechanisms(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.CausalPortfolioAdversarialTest();self.f.setUp()
    def simulate(self,mode='none',grace=0):
        f=self.f
        return run_backtest(f.signals,f.market,f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1],deepcopy(f.cfg),retail_forecasts=f.forecasts,retail_config=deepcopy(f.retail),hold_allowed=f.hold,path_config=dict(wait_mode=mode,cap_grace_lots=grace,cap_hard_fraction=.35))
    def exit_fixture(self):
        f=self.f;reviews=f.dates[[0,3,7,10]]
        f.signals=pd.DataFrame(np.nan,index=reviews,columns=f.codes)
        f.signals.iloc[:,0]=.1;f.signals.iloc[1:,1]=.4
        f.forecasts=pd.DataFrame(.1,index=reviews,columns=f.codes)
        f.hold=pd.DataFrame(True,index=reviews,columns=f.codes);f.hold.iloc[1,0]=False
        f.retail['candidate_pool']=1
    def test_wait_requires_filled_exit_and_next_review(self):
        self.exit_fixture();base=self.simulate();wait=self.simulate('wait')
        b=base.trades[base.trades.side.eq('buy')];w=wait.trades[wait.trades.side.eq('buy')]
        self.assertEqual(b.iloc[1].date,self.f.dates[5]);self.assertEqual(w.iloc[1].date,self.f.dates[8])
        self.assertEqual(w.iloc[1].ts_code,self.f.codes[1]);self.assertTrue(wait.daily.loc[self.f.dates[4:8],'n_hold'].eq(0).all())
    def test_recovery_can_cross_pool_cutoff_but_not_entry_eligibility(self):
        self.exit_fixture();recovered=self.simulate('recover')
        self.assertEqual(recovered.trades[recovered.trades.side.eq('buy')].iloc[1].ts_code,self.f.codes[0])
        self.f.signals.iloc[2:,0]=np.nan
        missing=self.simulate('recover');self.assertEqual(missing.trades[missing.trades.side.eq('buy')].iloc[1].ts_code,self.f.codes[1])
    def test_no_slot_wait_on_failed_exit(self):
        self.exit_fixture();f=self.f
        f.market.loc[(f.market.ts_code==f.codes[0])&(f.market.date>=f.dates[4]),['open','high','low']]=np.nan
        f.market.loc[(f.market.ts_code==f.codes[0])&(f.market.date>=f.dates[4]),'pre_close']=10.
        result=self.simulate('recover')
        self.assertFalse(result.events.kind.eq('recovery_slot_wait').any())
        self.assertEqual(len(result.trades),1)
    def test_lot_buffer_keeps_small_excess_and_hard_limit_still_trims(self):
        f=self.f;f.retail['max_stock_fraction']=.25
        f.signals=f.signals.reindex(f.dates[[0,3]]).ffill();f.forecasts=f.forecasts.reindex(f.signals.index).ffill();f.hold=f.hold.reindex(f.signals.index).ffill()
        # 初始200股，升至14元后轻微越过25%；跨日参考保持连续。
        for j,d in enumerate(f.dates):
            value=10. if j<2 else 14.
            f.market.loc[(f.market.date==d)&(f.market.ts_code==f.codes[0]),['open','high','low','close']]=value
            f.market.loc[(f.market.date==d)&(f.market.ts_code==f.codes[0]),'pre_close']=10. if j<=2 else 14.
        f.limits['up_limit']=100.;f.limits['down_limit']=1.
        strict=self.simulate();buffer=self.simulate(grace=1)
        self.assertEqual(len(strict.trades[strict.trades.side.eq('sell')]),1)
        self.assertEqual(len(buffer.trades[buffer.trades.side.eq('sell')]),0)
        self.assertTrue(buffer.events.kind.eq('lot_cap_grace').any())
        f.market.loc[(f.market.date>=f.dates[2])&(f.market.ts_code==f.codes[0]),['open','high','low','close']]=30.
        f.market.loc[(f.market.date>=f.dates[3])&(f.market.ts_code==f.codes[0]),'pre_close']=30.
        hard=self.simulate(grace=1);self.assertEqual(len(hard.trades[hard.trades.side.eq('sell')]),1)

class DurationTests(unittest.TestCase):
    def fixture(self):
        dates=pd.bdate_range('2020-01-01',periods=180)
        p=pd.DataFrame(100.,index=dates,columns=['a']);t=pd.DataFrame(True,index=dates,columns=['a']);o=t.copy();t.iloc[-12:]=False
        return p,t,o,dict(window=20,usual_weak_days=15,broken_weak_days=10,long_window=120)
    def test_same_current_day_does_not_depend_on_prior_review_dates(self):
        p,t,o,c=self.fixture()
        a=duration_holds(p,t,o,p.index[[120,140,179]],c);b=duration_holds(p,t,o,p.index[[130,150,179]],c)
        pd.testing.assert_series_equal(a.iloc[-1],b.iloc[-1]);self.assertEqual(a.iloc[-1,0],0.)
    def test_unknown_history_cannot_confirm_exit(self):
        p,t,o,c=self.fixture();o.iloc[-4]=False
        a=duration_holds(p,t,o,p.index[[-1]],c);self.assertEqual(a.iloc[-1,0],1.)
        o.iloc[-1]=False;self.assertTrue(np.isnan(duration_holds(p,t,o,p.index[[-1]],c).iloc[0,0]))
    def test_future_truncation(self):
        p,t,o,c=self.fixture();cut=p.index[-6];reviews=p.index[::5]
        a=duration_holds(p,t,o,reviews,c);b=duration_holds(p.loc[:cut],t.loc[:cut],o.loc[:cut],reviews[reviews<=cut],c)
        pd.testing.assert_frame_equal(a.loc[:cut],b,check_exact=True)

if __name__=='__main__':
    unittest.main()
