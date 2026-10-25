import unittest,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from retail_rebalance import plan_rebalance


def row(code,price,mu,shares=0,available=0,cap=10000,**kw):
    return dict(ts_code=code,price=price,expected_return=mu,shares=shares,available=available,
                max_notional=cap,min_buy=100,buy_step=100,sell_step=100,can_buy=True,can_sell=True,**kw)


class RebalanceTest(unittest.TestCase):
    def run_case(self,rows,cash=10000):
        return plan_rebalance(pd.DataFrame(rows),cash=cash,buy_rate=0.,sell_rate=0.,min_fee=5.,forecast_kind='expected_simple_return')

    def test_old_position_released_to_better_candidate(self):
        r=self.run_case([row('old',10,.001,1000,1000),row('new',50,.1,cap=10000)],cash=0)
        self.assertEqual(r['plan'].sell_shares.tolist(),[1000,0])
        self.assertEqual(r['plan'].buy_shares.tolist(),[0,100])
        self.assertEqual(r['cash_after'],4990.)

    def test_locked_position_cannot_fund_purchase(self):
        r=self.run_case([row('old',10,-.1,1000,0),row('new',10,.2)],cash=0)
        self.assertEqual(r['plan'].target_shares.tolist(),[1000,0])
        self.assertEqual(r['cash_after'],0.)

    def test_partial_reduction_and_odd_lot_exit(self):
        r=self.run_case([row('odd',10,-.1,305,205,cap=500)],cash=0)
        self.assertEqual(r['plan'].sell_shares.tolist(),[205])
        self.assertEqual(r['plan'].target_shares.tolist(),[100])
        self.assertEqual(r['plan'].unavoidable_locked_excess.tolist(),[500])

    def test_preserve_profitable_holding_without_round_trip(self):
        r=self.run_case([row('old',10,.1,1000,1000)],cash=0)
        self.assertEqual(r['plan'].sell_shares.sum()+r['plan'].buy_shares.sum(),0)

    def test_high_price_retained_and_cash_possible(self):
        r=self.run_case([row('high',60,.1,cap=6000),row('low',10,-.1)])
        self.assertEqual(r['plan'].target_shares.tolist(),[100,0])
        self.assertEqual(r['cash_after'],3995.)

    def test_suspended_shares_remain(self):
        old=row('old',10,-.1,1000,1000);old['can_sell']=False
        r=self.run_case([old],cash=0)
        self.assertEqual(r['plan'].target_shares.tolist(),[1000])

    def test_suspended_overweight_is_reported_not_forced_sold(self):
        old=row('old',10,-.1,1000,1000,cap=2000);old['can_sell']=False
        r=self.run_case([old],cash=0)
        self.assertEqual(r['plan'].target_shares.tolist(),[1000])
        self.assertEqual(r['plan'].unavoidable_locked_excess.tolist(),[8000])

    def test_slippage_is_additional_to_minimum_fee(self):
        r=plan_rebalance(pd.DataFrame([row('a',10,.1,cap=1000)]),cash=1010,
            buy_rate=0.,sell_rate=0.,min_fee=5.,forecast_kind='expected_simple_return',slippage_rate=.001)
        self.assertEqual(r['plan'].buy_shares.tolist(),[100])
        self.assertEqual(r['cash_after'],4.)

    def test_matches_exhaustive_joint_cash_and_old_holdings(self):
        from itertools import product
        rows=[row('a',5,.035,100,100,cap=2000),row('b',7,.05,200,200,cap=2100)]
        result=self.run_case(rows,cash=1000)
        feasible=[]
        for qa,qb in product(range(0,401,100),range(0,301,100)):
            value=0.;cash=1000.
            for target,record in zip([qa,qb],rows):
                delta=target-record['shares'];fee=5. if delta else 0.
                cash-=delta*record['price']+fee
                value+=target*record['price']*record['expected_return']-fee-(5. if target else 0.)
            if cash>=0:feasible.append(value)
        self.assertAlmostEqual(result['expected_net_profit'],max(feasible))
        self.assertLessEqual(result['solver_gap'],.001)


if __name__=='__main__':unittest.main()
