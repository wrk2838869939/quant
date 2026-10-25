"""实际撮合中验证排名退出后不会同月又把同一股票补回。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)]
from risk_budget_portfolio import run_backtest
import test_causal_portfolio_adversarial as fixture


class RankEntryAlignmentTest(unittest.TestCase):
    def test_exiting_stock_cannot_be_supplemented_in_same_review(self):
        f = fixture.CausalPortfolioAdversarialTest(); f.setUp(); f.add_review(5, preferred=0)
        f.hold = f.hold.astype(float)
        f.hold.loc[f.dates[5], f.codes[0]] = 0.
        def execute(signals):
            return run_backtest(signals, f.market, f.basics, f.st, f.limits, f.actions,
                f.dates[0], f.dates[-1], deepcopy(f.cfg), retail_forecasts=f.forecasts,
                retail_config=deepcopy(f.retail), hold_allowed=f.hold,
                risk_multipliers=pd.DataFrame(1., index=signals.index, columns=signals.columns), adjust_existing_risk=True)
        unaligned = execute(f.signals)
        signals = f.signals.copy(); signals.loc[f.dates[5], f.codes[0]] = np.nan
        aligned = execute(signals)
        old = unaligned.trades
        self.assertTrue(((old.ts_code == f.codes[0]) & old.side.eq('buy') & old.date.gt(f.dates[5])).any())
        new = aligned.trades
        self.assertFalse(((new.ts_code == f.codes[0]) & new.side.eq('buy') & new.date.gt(f.dates[5])).any())
        self.assertTrue(new.side.eq('sell').any())
        pd.testing.assert_frame_equal(old[old.date < f.dates[5]], new[new.date < f.dates[5]])


if __name__ == '__main__': unittest.main()
