"""独立现金不是每日重新平衡：用可手算路径验证总账户收益与对账。"""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from portfolio import BacktestResult
from check_cash_reserve import add_cash_reserve


class CashReserveTest(unittest.TestCase):
    def test_reserve_remains_cash_while_stock_weight_drifts(self):
        dates = pd.bdate_range("2022-01-04", periods=3)
        daily = pd.DataFrame(dict(cash=[0., 0., 0.], stock_value=[1000., 2000., 4000.],
                                 equity=[1000., 2000., 4000.]), index=dates)
        trades = pd.DataFrame([dict(date=dates[0], ts_code="000001.SZ", side="buy", shares=100, gross=1000., fee=0.)])
        holdings = pd.DataFrame(dict(date=dates, ts_code="000001.SZ", shares=100))
        result = BacktestResult(daily, trades, pd.DataFrame(), holdings, dict(capital=1000.))
        result = add_cash_reserve(result, 1000., 244)
        np.testing.assert_allclose(result.daily.equity, [2000., 3000., 5000.])
        np.testing.assert_allclose(result.daily.ret, [0., .5, 2/3])
        self.assertAlmostEqual(result.summary["total"], 1.5)
        self.assertAlmostEqual(result.daily.cash.iloc[-1], 1000.)
        self.assertAlmostEqual(result.summary["active_capital"], 1000.)


if __name__ == "__main__":
    unittest.main()
