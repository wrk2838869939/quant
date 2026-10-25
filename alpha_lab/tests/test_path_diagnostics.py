"""独立合成持仓验证利润归属、股息不重复和持仓段归属。"""
from pathlib import Path
import sys
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from path_diagnostics import asset_daily_states, holding_episodes


class PathTest(unittest.TestCase):
    def test_dividend_and_sale_reconcile_without_double_count(self):
        dates = pd.date_range("2022-01-01", periods=4)
        daily = pd.DataFrame(dict(equity=[999., 1009., 1019., 1038.]), index=dates)
        daily.index.name = "date"
        trades = pd.DataFrame(dict(date=[dates[0], dates[3]], ts_code=["A", "A"],
            side=["buy", "sell"], gross=[100., 140.], fee=[1., 1.]))
        holdings = pd.DataFrame(dict(date=dates[:3], ts_code=["A"]*3, value=[100., 110., 120.]))
        # 第二日确认10元股息，但股价除息后净市值为100。
        holdings.loc[1, "value"] = 100.
        holdings.loc[2, "value"] = 110.
        trades.loc[1, "gross"] = 130.
        events = pd.DataFrame(dict(date=[dates[1], dates[2]], ts_code=["A", "A"],
            kind=["dividend_receivable", "dividend_paid"], amount=[10., 10.]))
        states = asset_daily_states(daily, trades, holdings, events, 1000.)
        self.assertEqual(states.pnl.tolist(), [-1., 10., 10., 19.])
        _, episodes = holding_episodes(states, trades, dates)
        self.assertTrue(episodes.closed.iloc[0])
        self.assertEqual(episodes.pnl.iloc[0], 38.)
        self.assertEqual(episodes.trading_days.iloc[0], 4)

    def test_accounting_mismatch_fails(self):
        dates = pd.date_range("2022-01-01", periods=1)
        daily = pd.DataFrame(dict(equity=[1200.]), index=dates)
        daily.index.name = "date"
        trades = pd.DataFrame(dict(date=dates, ts_code=["A"], side=["buy"], gross=[100.], fee=[0.]))
        holdings = pd.DataFrame(dict(date=dates, ts_code=["A"], value=[100.]))
        events = pd.DataFrame(columns=["date", "ts_code", "kind", "amount"])
        with self.assertRaises(ValueError):
            asset_daily_states(daily, trades, holdings, events, 1000.)


if __name__ == "__main__":
    unittest.main()
