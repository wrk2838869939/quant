"""实施日期、输入时刻、空响应和禁止过窗补发的回归测试。"""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile
import unittest

import pandas as pd

from execution_metadata import normalize_dividends,inspect_metadata,guard_plan
from shadow_planner import ShadowError,fingerprint


def dividend():
    return pd.DataFrame([dict(ts_code='601728.SH',div_proc='实施',ann_date='20260821',imp_ann_date='20260903',
        record_date='20260909',ex_date='20260910',pay_date='20260910',div_listdate=None,cash_div_tax=.1606,stk_div=0.)])


def fixture():
    bundle=dict(asof='2026-09-11',signals={'601728.SH':.4})
    snapshot=dict(asof='2026-09-11',positions={})
    calendar=dict(source='test',sessions=['2026-09-11','2026-09-14'])
    metadata=dict(price_date='2026-09-11',next_session='2026-09-14',first_available_at='2026-09-11T13:00:00+00:00',
        blockers=[],account_scope_supplied=True,action_scope=['601728.SH'],all_corporate_actions_covered=True)
    provenance=dict(historical_backfill=False,bundle_sha256=fingerprint(bundle),snapshot_sha256=fingerprint(snapshot))
    return bundle,snapshot,calendar,metadata,'2026-09-11T21:01:00+08:00',provenance


class MetadataTests(unittest.TestCase):
    def test_implementation_date_never_proposal(self):
        accepted,bad=normalize_dividends(dividend())
        self.assertEqual(accepted.known_date.iloc[0],pd.Timestamp('2026-09-03'))
        self.assertNotIn('ann_date',accepted);self.assertTrue(bad.empty)

    def test_missing_implementation_is_quarantined(self):
        x=dividend();x['imp_ann_date']=None
        accepted,bad=normalize_dividends(x)
        self.assertTrue(accepted.empty);self.assertEqual(len(bad),1)

    def test_exact_duplicates_and_proposal_variants_recorded(self):
        x=dividend();y=x.copy();y['ann_date']='20260822'
        accepted,_=normalize_dividends(pd.concat([x,x,y]))
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted.attrs['exact_duplicate_rows_removed'],1)
        self.assertEqual(accepted.attrs['proposal_only_variants_removed'],1)

    def test_conflicting_terms_not_deduplicated(self):
        x=dividend();y=x.copy();y['cash_div_tax']=.2
        with self.assertRaises(ShadowError):normalize_dividends(pd.concat([x,y]))

    def test_settled_historical_conflicts_are_quarantined_not_summed(self):
        x=dividend();y=x.copy();y['cash_div_tax']=.2
        accepted,bad=normalize_dividends(pd.concat([x,y]),active_from='2026-09-14')
        self.assertTrue(accepted.empty)
        self.assertEqual(len(bad),2)
        self.assertEqual(set(bad.cash_div_tax),{.1606,.2})
        self.assertTrue(bad.quarantine_reason.eq('historical_multiple_implemented_records').all())

    def test_current_or_unpaid_conflicts_still_block(self):
        x=dividend();y=x.copy();y['cash_div_tax']=.2
        with self.assertRaises(ShadowError):normalize_dividends(pd.concat([x,y]),active_from='2026-09-10')
        y['pay_date']='20260915'
        with self.assertRaises(ShadowError):normalize_dividends(pd.concat([x,y]),active_from='2026-09-14')
        y['pay_date']='20260910';y['stk_div']=.1;y['div_listdate']=None
        with self.assertRaises(ShadowError):normalize_dividends(pd.concat([x,y]),active_from='2026-09-14')

    def test_invalid_date_order_and_missing_money(self):
        x=dividend();x['imp_ann_date']='20260911'
        accepted,bad=normalize_dividends(x);self.assertTrue(accepted.empty)
        x=dividend();x['cash_div_tax']=None
        with self.assertRaises(ShadowError):normalize_dividends(x)

    def test_guard_accepts_complete_synthetic_contract(self):
        self.assertTrue(guard_plan(*fixture()))

    def test_guard_rejects_late_execution_and_fake_future_calendar(self):
        values=list(fixture());values[4]='2026-09-14T09:25:00+08:00'
        with self.assertRaisesRegex(ShadowError,'错过'):guard_plan(*values)
        values=list(fixture());values[2]['sessions']=['2026-09-11','2026-09-15']
        with self.assertRaisesRegex(ShadowError,'日历'):guard_plan(*values)

    def test_guard_no_future_metadata_or_premature_eod(self):
        values=list(fixture());values[3]['first_available_at']='2026-09-14T06:00:00+00:00'
        with self.assertRaisesRegex(ShadowError,'之后'):guard_plan(*values)
        values=list(fixture());values[4]='2026-09-11T20:59:00+08:00'
        with self.assertRaisesRegex(ShadowError,'21点'):guard_plan(*values)

    def test_guard_rejects_empty_current_responses(self):
        values=list(fixture());values[3]['blockers']=['stock_st_unavailable']
        with self.assertRaisesRegex(ShadowError,'不可用'):guard_plan(*values)

    def test_guard_rejects_missing_account_or_new_candidate(self):
        values=list(fixture());values[3]['account_scope_supplied']=False
        with self.assertRaisesRegex(ShadowError,'账户'):guard_plan(*values)
        values=list(fixture());values[0]['signals']['600000.SH']=.3
        with self.assertRaisesRegex(ShadowError,'候选'):guard_plan(*values)

    def test_guard_rejects_backfill_changed_bundle_and_incomplete_actions(self):
        for key in ['backfill','bundle','actions']:
            values=list(fixture())
            if key=='backfill':values[5]['historical_backfill']=True
            if key=='bundle':values[5]['bundle_sha256']='changed'
            if key=='actions':values[3]['all_corporate_actions_covered']=False
            with self.subTest(key=key),self.assertRaises(ShadowError):guard_plan(*values)


if __name__=='__main__':unittest.main()
