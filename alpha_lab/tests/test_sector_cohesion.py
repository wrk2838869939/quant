"""滚动残差的时间契约与缺失样本口径验证。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sector_cohesion import residualize


class CohesionTests(unittest.TestCase):
    def test_lagged_ols_matches_independent_fit_with_missing(self):
        rng = np.random.default_rng(4)
        idx = pd.date_range('2020-01-01', periods=100)
        f = pd.DataFrame(rng.normal(0, .02, (100, 2)), index=idx, columns=['market', 'size'])
        y = pd.DataFrame({'A': .003 + f.market*1.3-f['size']*.4+rng.normal(0, .005, 100)}, index=idx)
        y.iloc[40:45] = np.nan
        cfg = dict(beta_window=60, beta_minimum_days=30, covariance_epsilon=1e-14)
        for columns in [['market'], ['market', 'size']]:
            res, model = residualize(y, f[columns], cfg)
            part = pd.concat([y, f[columns]], axis=1).iloc[39:99].dropna()
            coef = np.linalg.lstsq(np.column_stack([np.ones(len(part)), part[columns]]), part.A, rcond=None)[0]
            expected = y.iloc[99, 0] - np.dot(np.r_[1, f[columns].iloc[99]], coef)
            self.assertAlmostEqual(res.iloc[99, 0], expected, places=12)
            changed = y.copy()
            changed.iloc[99] = 100
            _, changed_model = residualize(changed, f[columns], cfg)
            pd.testing.assert_frame_equal(model['intercept'], changed_model['intercept'])
            for a, b in zip(model['beta'], changed_model['beta']):
                pd.testing.assert_frame_equal(a, b)
            short, _ = residualize(y.iloc[:80], f[columns].iloc[:80], cfg)
            pd.testing.assert_frame_equal(short, res.iloc[:80])


if __name__ == '__main__':
    unittest.main()
