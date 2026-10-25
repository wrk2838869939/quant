"""用可手算的组合核对现金选择、高价股与整手约束。"""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from retail_allocation import allocate_new_positions


class RetailAllocationTest(unittest.TestCase):
    def run_case(self, rows, cash=10000, max_positions=10, **kwargs):
        frame=pd.DataFrame(rows,columns=['ts_code','price','expected_return','min_buy','buy_step','max_notional'])
        return allocate_new_positions(frame,cash=cash,buy_rate=0.,exit_rate=0.,minimum_fee=5.,
            max_positions=max_positions,forecast_kind=kwargs.get('forecast_kind','expected_simple_return'))

    def test_high_price_not_removed_by_average_budget(self):
        # 平均每席1000元买不起6000元的一手，但账户完全可承担。
        out=self.run_case([['high',60,.10,100,100,6000],['low',10,.01,100,100,9000]])
        self.assertEqual(out['orders'].shares.tolist(),[100,300])
        self.assertEqual(out['cash_after'],990.)
        self.assertAlmostEqual(out['expected_net_profit'],610.)

    def test_cash_beats_nonpositive_net_edge(self):
        out=self.run_case([['weak',10,.0001,100,100,9000],['negative',20,-.1,100,100,9000]])
        self.assertEqual(out['orders'].shares.sum(),0)
        self.assertEqual(out['cash_after'],10000.)

    def test_joint_allocation_beats_greedy_rank(self):
        out=self.run_case([['a',60,.10,100,100,6000],['b',50,.09,100,100,5000],['c',49,.09,100,100,4900]])
        self.assertEqual(out['orders'].shares.tolist(),[0,100,100])
        self.assertAlmostEqual(out['expected_net_profit'],871.)

    def test_max_positions_is_only_upper_bound(self):
        out=self.run_case([['a',10,.1,100,100,1000]],max_positions=8)
        self.assertEqual(out['orders'].shares.tolist(),[100])
        self.assertEqual(out['cash_after'],8995.)

    def test_star_minimum_and_increment(self):
        out=self.run_case([['star',10,.1,200,1,2500]],cash=2510)
        self.assertEqual(out['orders'].shares.tolist(),[250])
        self.assertEqual(out['cash_after'],5.)

    def test_rank_scores_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case([['a',10,.1,100,100,1000]],forecast_kind='rank')

    def test_fee_makes_whole_account_infeasible(self):
        out=self.run_case([['a',100,.2,100,100,10000]])
        self.assertEqual(out['orders'].shares.sum(),0)
        self.assertEqual(out['orders'].decision.iloc[0],'whole_account_cash_below_one_lot')

    def test_proportional_fees_above_minimum(self):
        frame=pd.DataFrame([dict(ts_code='a',price=10.,expected_return=.1,
            min_buy=100,buy_step=100,max_notional=10000.)])
        out=allocate_new_positions(frame,cash=10000.,buy_rate=.001,exit_rate=.002,
            minimum_fee=5.,max_positions=1,forecast_kind='expected_simple_return')
        self.assertEqual(out['orders'].shares.iloc[0],900)
        self.assertEqual(out['cash_after'],991.)
        self.assertAlmostEqual(out['expected_net_profit'],873.)

    def test_missing_explicit_risk_limit_rejected(self):
        frame=pd.DataFrame([dict(ts_code='a',price=10.,expected_return=.1,
            min_buy=100,buy_step=100)])
        with self.assertRaises(ValueError):
            allocate_new_positions(frame,cash=10000.,buy_rate=.001,exit_rate=.002,
                minimum_fee=5.,max_positions=1,forecast_kind='expected_simple_return')


if __name__=='__main__':
    unittest.main()
