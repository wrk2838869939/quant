"""授权延长日期不能改变全A排名或原未知复核语义。"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_recent_inputs', ROOT / 'scripts/prepare_recent_low_turnover_inputs.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecentLowTurnoverInputsTests(unittest.TestCase):
    def test_unknown_month_breaks_confirmation_instead_of_retaining_failure(self):
        dates = pd.date_range('2023-01-01', periods=5, freq='MS')
        trend = pd.DataFrame({'A': [False, False, False, False, True]}, index=dates)
        observable = pd.DataFrame({'A': [True, False, True, True, True]}, index=dates)
        actual = MODULE.monthly_hold_states(trend, observable, 2)
        np.testing.assert_allclose(actual.A, [1., np.nan, 1., 0., 1.], equal_nan=True)

    def test_all_A_ranks_and_history_prefix_survive_extension(self):
        dates = pd.bdate_range('2021-01-01', '2024-03-29')
        rows = []
        for code, turnover, speed in [('000001.SZ', 1., 0.0006), ('300001.SZ', 2., 0.0008)]:
            for i, date in enumerate(dates):
                rows.append(dict(date=date, ts_code=code, hfq_close=10 * np.exp(speed * i),
                                 total_mv_yuan=1e9 * (1 + 0.0001 * i), turnover_rate=turnover))
        raw = pd.DataFrame(rows)
        basics = pd.DataFrame({'ts_code': ['000001.SZ', '300001.SZ'],
            'list_date': ['2010-01-01'] * 2, 'delist_date': [None, None]})
        names = pd.DataFrame(columns=['ts_code', 'name', 'start_date', 'end_date'])
        cfg = yaml.safe_load((MODULE.FEATURE_PARENT / 'config.yaml').read_text())
        table, hold = MODULE.build_inputs(raw, basics, names, cfg, '2022-01-01', '2024-03-29')
        # 主板最小换手在两股全A截面rank为0，若先删创业板会错误变成0.5。
        np.testing.assert_allclose(table.loc[table.ts_code.eq('000001.SZ'), 'x_turnover'], 0.)
        np.testing.assert_allclose(table.loc[table.ts_code.eq('300001.SZ'), 'x_turnover'], 0.5)
        before, old_hold = MODULE.build_inputs(raw[raw.date.le('2023-12-31')], basics, names,
                                               cfg, '2022-01-01', '2023-12-31')
        pd.testing.assert_frame_equal(table[table.date.le('2023-12-31')].reset_index(drop=True), before, check_exact=True)
        pd.testing.assert_frame_equal(hold.loc[:'2023-12-31'], old_hold, check_exact=True)


if __name__ == '__main__':
    unittest.main()
