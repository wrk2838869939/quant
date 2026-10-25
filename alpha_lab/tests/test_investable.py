"""验证事前预算、板块手数与时点过滤，防止借下一日行情替换股票。"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from investable import investable_signals


class InvestableTest(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / "configs/monthly_portfolio_audit.yaml").read_text())["portfolio"]
        self.cfg.update(capital=1000., top_n=1, buy_cost=0., min_fee=5.)
        self.dates = pd.bdate_range("2022-01-04", periods=2)
        self.codes = ["000001.SZ", "688001.SH"]
        self.signals = pd.DataFrame(1., index=self.dates, columns=self.codes)
        self.close = self.signals * 5.
        self.volume = self.signals * 1e6
        self.basics = pd.DataFrame(dict(ts_code=self.codes, list_date=pd.Timestamp("2000-01-01"), delist_date=pd.NaT))
        self.st = pd.DataFrame(dict(ts_code=[self.codes[0]], start=[self.dates[1]], end=[pd.NaT]))

    def test_known_constraints_and_future_isolation(self):
        scores, _ = investable_signals(self.signals, self.close, self.volume, self.basics, self.st, self.cfg, 1.)
        self.assertTrue(pd.notna(scores.iloc[0, 0]))
        self.assertTrue(pd.isna(scores.iloc[1, 0]))  # 次日开始ST不能提前一天排除
        self.assertTrue(scores.iloc[:, 1].isna().all())  # 科创200股加最低费用超过预算
        self.close.iloc[1] = 99999.
        changed, _ = investable_signals(self.signals, self.close, self.volume, self.basics, self.st, self.cfg, 1.)
        pd.testing.assert_series_equal(changed.iloc[0], scores.iloc[0])

    def test_fee_price_buffer_and_prior_volume(self):
        self.close.iloc[0, 0] = 9.9
        unbuffered, _ = investable_signals(self.signals, self.close, self.volume, self.basics, self.st, self.cfg, 1.)
        buffered, _ = investable_signals(self.signals, self.close, self.volume, self.basics, self.st, self.cfg, 1.05)
        self.assertTrue(pd.notna(unbuffered.iloc[0, 0]))
        self.assertTrue(pd.isna(buffered.iloc[0, 0]))
        self.volume.iloc[0, 0] = 9900.  # 1%不足100股
        limited, _ = investable_signals(self.signals, self.close, self.volume, self.basics, self.st, self.cfg, 1.)
        self.assertTrue(pd.isna(limited.iloc[0, 0]))


if __name__ == "__main__":
    unittest.main()
