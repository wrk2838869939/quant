"""把新增防守开关接入真实引擎，检验次日成交及暂停后的现金隔离。"""
from copy import deepcopy
import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import test_causal_portfolio_adversarial as original
import defense_research_portfolio as research
from defense_diagnostics import DefenseOverlay


class DefenseExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = original.CausalPortfolioAdversarialTest()
        self.fixture.setUp()
        f = self.fixture
        self.spec = dict(affected_fraction=.5,recovery_fraction=.5,review_frequency='weekly')
        self.evidence = {k:pd.DataFrame(v,index=f.dates,columns=f.codes)
                         for k,v in [('alarm',False),('trend',False),('known',True)]}
        # 第一个入场之后的周一收盘触发。
        self.trigger = f.dates[f.dates.weekday==0][0]
        self.evidence['alarm'].loc[self.trigger,f.codes[0]] = True

    def run_case(self, market=None, end=None):
        f = self.fixture
        overlay = DefenseOverlay(self.evidence,self.spec,'weekly_liquidate')
        return research.run_backtest(f.signals,f.market if market is None else market,
            f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1] if end is None else end,
            deepcopy(f.cfg),retail_forecasts=f.forecasts,retail_config=f.retail,
            hold_allowed=f.hold,defense_overlay=overlay)

    def test_close_alarm_only_sells_next_day_and_blocks_monthly_buy(self):
        f = self.fixture
        number=f.dates.get_loc(self.trigger)
        f.add_review(number+1)
        result=self.run_case()
        sells=result.trades[result.trades.side.eq('sell')]
        self.assertEqual(sells.date.tolist(),[f.dates[number+1]])
        self.assertEqual(sells.decision_date.tolist(),[self.trigger])
        self.assertEqual(result.trades.side.eq('buy').sum(),1)
        self.assertEqual(result.daily.iloc[-1].n_hold,0)

    def test_tomorrow_open_cannot_change_risk_exit_order(self):
        f=self.fixture;number=f.dates.get_loc(self.trigger)
        baseline=self.run_case()
        market=f.market.copy()
        market.loc[market.date.eq(f.dates[number+1]),'open']=9.5
        changed=self.run_case(market)
        pd.testing.assert_frame_equal(f.planned(baseline,self.trigger,'sell'),
                                      f.planned(changed,self.trigger,'sell'))

    def test_failed_sale_remains_pending_after_recovery(self):
        f=self.fixture;number=f.dates.get_loc(self.trigger)
        f.add_review(number+1)
        self.evidence['trend'].loc[f.dates[number+1],f.codes[0]]=True
        # 停牌/缺报价使最初卖单无法成交，恢复信号不能撤销已确认退出。
        market=f.market.copy()
        mask=market.date.eq(f.dates[number+1]) & market.ts_code.eq(f.codes[0])
        market.loc[mask,['open','high','low']]=float('nan')
        result=self.run_case(market)
        sell=result.trades[result.trades.side.eq('sell')]
        self.assertEqual(sell.date.tolist(),[f.dates[number+2]])
        self.assertEqual(sell.shares.sum(),result.trades.loc[result.trades.side.eq('buy'),'shares'].iloc[0])


if __name__=='__main__':
    unittest.main()
