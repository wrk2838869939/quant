"""公告支持的公司行动源修正：完整键、旧值、重复与时间边界失败关闭。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from portfolio import AccountingError, normalize_actions
from audit_monthly_portfolio import load_inputs


def source_rows():
    """合成原接口两条原始记录；不读取市场或接口缓存。"""
    common = dict(ts_code="600600.SH", ann_date="20230323", div_proc="实施",
                  stk_div=0.0, record_date="20230713", ex_date="20230714",
                  pay_date="20230714", div_listdate=None, imp_ann_date="20230707")
    return pd.DataFrame([
        dict(common, end_date="20230323", cash_div=0.5, cash_div_tax=0.5),
        dict(common, end_date="20221231", cash_div=1.8, cash_div_tax=0.0),
    ])


class SourceCorrectionTests(unittest.TestCase):
    def setUp(self):
        config = yaml.safe_load((ROOT / "configs/corporate_action_components.yaml").read_text())
        self.rule = deepcopy(config["source_corrections"][0])
        self.components = [r for r in config["components"] if r["ts_code"] == "600600.SH"]
        self.frame = source_rows()

    def normalize(self, frame=None, rule=None):
        return normalize_actions(self.frame if frame is None else frame,
                                 self.components, source_corrections=[self.rule if rule is None else rule])

    def test_correct_ordinary_component_then_aggregate_without_mutating_raw(self):
        original = self.frame.copy(deep=True)
        result = self.normalize()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.cash_div_tax.iloc[0], 1.8)
        self.assertEqual(result.stk_div.iloc[0], 0.0)
        self.assertEqual(result.ex_date.iloc[0], pd.Timestamp("2023-07-14"))
        audit = result.attrs["source_corrections"]
        self.assertEqual(audit[0]["matched_rows"], 1)
        self.assertEqual(audit[0]["replacement_values"], {"cash_div": 1.3, "cash_div_tax": 1.3})
        self.assertTrue(audit[0]["evidence"].endswith("10809161-0.PDF"))
        pd.testing.assert_frame_equal(self.frame, original)

    def test_changed_raw_expected_values_are_rejected(self):
        for column, value in [("cash_div", 1.81), ("cash_div_tax", 1.3), ("stk_div", 0.1)]:
            with self.subTest(column=column):
                frame = self.frame.copy()
                frame.loc[1, column] = value
                with self.assertRaisesRegex(AccountingError, "原始值改变"):
                    self.normalize(frame)

    def test_exact_duplicate_is_rejected_before_normalization_deduplication(self):
        frame = pd.concat([self.frame, self.frame.iloc[[1]]], ignore_index=True)
        with self.assertRaisesRegex(AccountingError, "匹配数应为1，实际2"):
            self.normalize(frame)

    def test_full_key_change_or_missing_target_with_stock_present_is_rejected(self):
        for key in ["end_date", "ann_date", "imp_ann_date", "record_date", "pay_date", "ex_date"]:
            with self.subTest(key=key):
                frame = self.frame.copy()
                frame.loc[1, key] = "20230712"
                with self.assertRaisesRegex(AccountingError, "匹配数应为1，实际0"):
                    self.normalize(frame)
        with self.assertRaisesRegex(AccountingError, "匹配数应为1，实际0"):
            self.normalize(self.frame.iloc[:1])

    def test_no_rule_leaves_original_amount_and_does_not_silently_correct(self):
        original = normalize_actions(self.frame.iloc[[1]])
        self.assertEqual(original.cash_div_tax.iloc[0], 0.0)
        self.assertEqual(original.attrs["source_corrections"], [])
        with self.assertRaisesRegex(AccountingError, "组成与核验配置不符"):
            normalize_actions(self.frame, self.components)

    def test_unrelated_stock_is_unchanged_and_absent_target_is_skipped(self):
        other = self.frame.iloc[[0]].copy().assign(ts_code="000001.SZ")
        expected = normalize_actions(other)
        result = self.normalize(other)
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(result.attrs["source_corrections"], [])
        combined = self.normalize(pd.concat([self.frame, other], ignore_index=True))
        self.assertEqual(combined.loc[combined.ts_code.eq("000001.SZ"), "cash_div_tax"].iloc[0], 0.5)

    def test_future_announcement_and_incomplete_or_duplicate_rule_are_rejected(self):
        for change in ["future", "key", "evidence"]:
            rule = deepcopy(self.rule)
            if change == "future":
                rule["source_ann_date"] = "2023-07-15"
            elif change == "key":
                del rule["match"]["imp_ann_date"]
            else:
                rule["evidence"] = ""
            with self.subTest(change=change), self.assertRaises(AccountingError):
                self.normalize(rule=rule)
        with self.assertRaisesRegex(AccountingError, "重复修正"):
            normalize_actions(self.frame, self.components, source_corrections=[self.rule, self.rule])

    def test_loader_passes_correction_and_persists_audit_in_experiment_directory(self):
        def read_parquet(path, **kwargs):
            path = Path(path)
            if path.name == "daily_panel.parquet":
                self.assertEqual(kwargs["filters"][1], ("date", "<=", pd.Timestamp("2023-12-31")))
                return pd.DataFrame()
            if path.name == "stock_basic_all.parquet":
                return pd.DataFrame([dict(ts_code="600600.SH", list_date="19930827", delist_date=None)])
            if path.name == "namechange_all.parquet":
                return pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"])
            if path.parent.name == "dividend":
                return self.frame.copy(deep=True)
            return pd.DataFrame([dict(trade_date="20230714", up_limit=114.96, down_limit=94.06)])
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            (prepared / "execution_codes.csv").write_text("ts_code\n600600.SH\n")
            (prepared / "execution_cache.txt").write_text(str(prepared / "cache"))
            with patch("audit_monthly_portfolio.pd.read_parquet", side_effect=read_parquet):
                *_, actions = load_inputs(prepared, {"valid": ["2022-01-01", "2023-12-31"]})
            self.assertAlmostEqual(actions.cash_div_tax.iloc[0], 1.8)
            audit = json.loads((prepared / "corporate_action_source_corrections.json").read_text())
            self.assertFalse(audit["raw_cache_modified"])
            self.assertEqual(len(audit["applied"]), 1)


if __name__ == "__main__":
    unittest.main()
