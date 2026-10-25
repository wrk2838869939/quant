"""应计分红不可重复计入，交易与末日估值应精确对账。"""
from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_attribution import attribute_account


class AttributionTest(unittest.TestCase):
    def test_cash_dividend_and_final_liquidation(self):
        dates = pd.date_range('2022-01-01', periods=3)
        daily = pd.DataFrame(dict(stock_value=[100., 110., 0.], equity=[999., 1019., 1023.]), index=dates)
        holdings = pd.DataFrame([dict(date=dates[0], ts_code='x', value=100.), dict(date=dates[1], ts_code='x', value=110.)])
        trades = pd.DataFrame([dict(date=dates[0], ts_code='x', side='buy', gross=100., fee=1.),
                               dict(date=dates[2], ts_code='x', side='sell', gross=115., fee=1.)])
        events = pd.DataFrame([dict(date=dates[1], ts_code='x', kind='dividend_receivable', amount=10.),
                               dict(date=dates[2], ts_code='x', kind='dividend_paid', amount=10.)])
        stock, pnl, _, error = attribute_account(daily, trades, events, holdings, 1000.)
        self.assertEqual(stock.loc['x', 'net_pnl'], 23.)
        self.assertEqual(stock.loc['x', 'dividends'], 10.)
        self.assertEqual(stock.loc['x', 'fees'], 2.)
        self.assertEqual(error, 0.)
        self.assertEqual(pnl.iloc[1, 0], 20.)

    def test_delisting_cash_recovery(self):
        date = pd.Timestamp('2022-01-01')
        daily = pd.DataFrame(dict(stock_value=[0.], equity=[919.]), index=[date])
        trades = pd.DataFrame([dict(date=date, ts_code='x', side='buy', gross=100., fee=1.)])
        events = pd.DataFrame([dict(date=date, ts_code='x', kind='delist_writeoff', cash_recovery=20.)])
        stock, _, _, _ = attribute_account(daily, trades, events, pd.DataFrame(), 1000.)
        self.assertEqual(stock.loc['x', 'net_pnl'], -81.)


if __name__ == '__main__':
    unittest.main()
