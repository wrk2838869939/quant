"""板块聚合的时序、计量单位、成员变化与缺失行为。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sector_dataset import FIELDS, aggregate, build_membership, capped_weights


class SectorDatasetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/sector_dataset_v1.yaml').read_text())
        self.cfg.update(minimum_members=2, maximum_member_weight=.7, minimum_quote_coverage=1.)
        self.dates = pd.date_range('2020-01-01', periods=5)
        self.mapping = np.zeros((5, 3), np.int16)
        self.active = np.ones((5, 1), bool)
        self.wide = {k: np.full((5, 3), 10.) for k in FIELDS}
        self.wide['total_mv_yuan'][:] = [100., 200., 300.]
        self.wide['vol_share'][:] = 1000.
        self.wide['float_share'][:] = 10.

    def run_panel(self):
        return aggregate(self.wide, self.dates, self.mapping, ['S'], self.active, self.cfg)

    def test_previous_weights_and_future_prefix(self):
        self.wide['hfq_close'][2, 0] = 12.
        self.wide['total_mv_yuan'][2, 0] = 1000000.
        full = self.run_panel()
        self.assertAlmostEqual(full.index_return.iloc[2], .2 / 6)
        prefix = aggregate({k: v[:3] for k, v in self.wide.items()}, self.dates[:3],
                           self.mapping[:3], ['S'], self.active[:3], self.cfg)
        pd.testing.assert_frame_equal(full.iloc[:3], prefix)

    def test_membership_addition_does_not_jump_index(self):
        self.mapping[:2, 2] = -1
        out = self.run_panel()
        self.assertAlmostEqual(out.index_return.iloc[2], 0.)
        self.assertEqual(out.member_add_cap_yuan.iloc[2], 300.)
        self.assertEqual(out.matched_cap_change_yuan.iloc[2], 0.)
        self.assertEqual(out.observed_cap_change_yuan.iloc[2], 300.)

    def test_turnover_units_and_all_missing(self):
        out = self.run_panel()
        self.assertAlmostEqual(out.turnover_pct.iloc[1], 1.)
        self.wide['amount_yuan'][2] = np.nan
        self.wide['total_mv_yuan'][2] = np.nan
        out = self.run_panel()
        self.assertTrue(np.isnan(out.amount_yuan.iloc[2]))
        self.assertTrue(np.isnan(out.total_mv_yuan.iloc[2]))

    def test_gap_starts_new_segment(self):
        self.wide['hfq_close'][2] = np.nan
        out = self.run_panel()
        self.assertEqual(out.segment_id.tolist(), [0, 1, 0, 0, 2])
        self.assertEqual(out.index_close.iloc[4], 1000.)
        self.assertTrue(np.isnan(out.chain_return.iloc[4]))

    def test_membership_exclusive_end_conflict_and_listing(self):
        catalog = pd.DataFrame({'index_code': ['S', 'T'], 'src': ['SW2014', 'SW2014']})
        members = pd.DataFrame({'index_code': ['S', 'T'], 'con_code': ['A', 'A'],
                                'in_date': ['20200101', '20200102'], 'out_date': ['20200103', None]})
        basic = pd.DataFrame({'ts_code': ['A'], 'list_date': ['20200102'], 'delist_date': [None]})
        matrix, _, _, _ = build_membership(members, catalog, self.dates, ['A'], basic, self.cfg)
        self.assertEqual(matrix[:, 0].tolist(), [-1, -2, 1, 1, 1])

    def test_weight_cap(self):
        w = capped_weights([1, 1, 100], .4)
        np.testing.assert_allclose(w, [.3, .3, .4])
        with self.assertRaises(ValueError):
            capped_weights([1, 100], .4)


if __name__ == '__main__':
    unittest.main()
