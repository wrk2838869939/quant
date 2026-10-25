"""厂商同事件修订重复的显式合并: 全键一致仅现金微差 → 保留最后修订值; 其他差异仍报错。"""

import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from portfolio import AccountingError, normalize_actions  # noqa: E402


def revision_rows(cash_a=0.149971, cash_b=0.150000, stk_b=None, cash_gap=False):
    """合成厂商修订重复: 两行除现金分红外全部字段一致 (002846.SZ 2021-03-30 实例)。"""
    common = dict(ts_code="002846.SZ", end_date="20201231", ann_date="20210209",
                  div_proc="实施", stk_div=0.6, record_date="20210329",
                  ex_date="20210330", pay_date="20210330", div_listdate="20210330",
                  imp_ann_date="20210324")
    rows = [dict(common, cash_div=cash_a, cash_div_tax=cash_a),
            dict(common, cash_div=cash_b, cash_div_tax=cash_b)]
    if stk_b is not None:
        rows[1]["stk_div"] = stk_b
    if cash_gap:
        rows[1]["cash_div"] = rows[1]["cash_div_tax"] = cash_b * 2
    return pd.DataFrame(rows)


class VendorRevisionDuplicateTests(unittest.TestCase):
    def normalize(self, frame):
        return normalize_actions(frame)

    def test_merge_keeps_last_revision_with_audit(self):
        result = self.normalize(revision_rows())
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.cash_div_tax.iloc[0], 0.150000)
        self.assertAlmostEqual(result.stk_div.iloc[0], 0.6)
        audit = result.attrs["source_corrections"]
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["kind"], "vendor_revision_duplicate")
        self.assertEqual(audit[0]["n_rows"], 2)
        self.assertAlmostEqual(audit[0]["kept"], 0.150000)

    def test_material_cash_difference_is_not_merged(self):
        with self.assertRaises(AccountingError):
            self.normalize(revision_rows(cash_gap=True))

    def test_differing_stk_div_is_not_merged(self):
        with self.assertRaises(AccountingError):
            self.normalize(revision_rows(stk_b=1.2))

    def test_exact_duplicates_still_dedup_silently(self):
        frame = revision_rows(cash_a=0.15, cash_b=0.15)
        result = self.normalize(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result.attrs["source_corrections"]), 0)


if __name__ == "__main__":
    unittest.main()
