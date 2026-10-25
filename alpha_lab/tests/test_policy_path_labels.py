"""统一成熟时点、三时钟与不可成交延期的独立合成路径检验。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from policy_path_labels import policy_path_labels
from trend_entry import marked_paths


class PolicyPathLabelsTest(unittest.TestCase):
    def inputs(self,horizon=6,n=1):
        return dict(adjusted_open=np.full((horizon+1,n),10.),
            adjusted_close=np.full((horizon,n),10.),review_mask=np.ones((horizon,n),bool),
            trend=np.ones((horizon,n)),observable=np.ones((horizon,n),bool),
            entry_allowed=np.ones(n),sell_allowed=np.ones((horizon+1,n)),
            delist_step=np.full(n,np.inf),observation_horizon=horizon,
            policy='monthly_two_failures',confirmations=2)

    def test_fixed_120_matches_original_marked_paths_if_tradeable(self):
        data = self.inputs(120,2)
        data['adjusted_open'][:,0] = np.linspace(10,12,121)
        data['adjusted_close'][:,0] = np.linspace(9,12,120)
        data['adjusted_open'][:,1] = np.linspace(10,8,121)
        data['adjusted_close'][:,1] = np.linspace(9.9,7.5,120)
        data.update(policy='fixed',fixed_exit_step=120)
        result = policy_path_labels(**data)
        original = marked_paths(data['adjusted_open'][0],data['adjusted_close'],
            data['adjusted_open'][-1],data['delist_step'],
            dict(max_adverse=.15,peak_for_roundtrip=.1,positive_checkpoints=2))
        np.testing.assert_allclose(result.labels.label_return,original.ret)
        np.testing.assert_allclose(result.labels.label_mae,original.mae)
        np.testing.assert_array_equal(result.labels.actual_sell_open_step,[120,120])

    def test_fixed_60_cash_is_observed_to_120_not_reinvested(self):
        data = self.inputs(120)
        data.update(policy='fixed',fixed_exit_step=60)
        data['adjusted_open'][60,0] = 12.
        data['adjusted_open'][61:,0] = 1.
        data['adjusted_close'][60:,0] = 1.
        result = policy_path_labels(**data)
        row = result.labels.iloc[0]
        self.assertEqual(row.exit_request_close_step,59)
        self.assertEqual(row.actual_sell_open_step,60)
        self.assertAlmostEqual(row.label_return,.2)
        np.testing.assert_allclose(result.marked_nav[61:],1.2)

    def test_unknown_resets_original_counter_without_C1_memory(self):
        data = self.inputs(6)
        data['trend'][:,0] = [0,0,0,0,1,1]
        data['observable'][1,0] = False
        result = policy_path_labels(**data)
        row = result.labels.iloc[0]
        np.testing.assert_array_equal(result.failure_counts[:4,0],[1,0,1,2])
        self.assertEqual(row.first_failure_review_step,0)
        self.assertEqual(row.exit_request_close_step,3)
        self.assertEqual(row.earliest_sell_open_step,4)
        self.assertEqual(row.actual_sell_open_step,4)

    def test_pending_sell_survives_recovery_unknown_and_blocked_open(self):
        data = self.inputs(6)
        data['trend'][:2,0] = 0
        data['observable'][3,0] = False
        data['sell_allowed'][2:4,0] = 0.
        data['sell_allowed'][4,0] = np.nan
        data['adjusted_open'][5,0] = 8.
        result = policy_path_labels(**data)
        row = result.labels.iloc[0]
        self.assertEqual(row.exit_request_close_step,1)
        self.assertEqual(row.actual_sell_open_step,5)
        self.assertEqual(row.blocked_sell_days,3)
        self.assertEqual(row.execution_unknown_sell_days,1)
        self.assertEqual(row.request_to_fill_intervals,3)
        self.assertAlmostEqual(row.label_return,-.2)

    def test_unfilled_terminal_quote_is_mark_not_sale(self):
        data = self.inputs(6)
        data.update(policy='fixed',fixed_exit_step=3)
        data['sell_allowed'][3:,0] = 0.
        data['adjusted_open'][-1,0] = 7.
        result = policy_path_labels(**data)
        row = result.labels.iloc[0]
        self.assertTrue(row.pending_exit)
        self.assertTrue(row.terminal_is_unsettled_mark)
        self.assertTrue(np.isnan(row.actual_sell_open_step))
        self.assertTrue(np.isnan(row.settled_label_return))
        self.assertAlmostEqual(row.label_return,-.3)

    def test_missing_quote_cannot_fill_even_with_allowed_mask(self):
        data = self.inputs(6)
        data.update(policy='fixed',fixed_exit_step=3)
        data['adjusted_open'][3:5,0] = np.nan
        data['adjusted_open'][5,0] = 9.
        result = policy_path_labels(**data)
        self.assertEqual(result.labels.actual_sell_open_step.iloc[0],5)
        self.assertEqual(result.labels.execution_unknown_sell_days.iloc[0],2)

    def test_known_suspension_is_blocked_not_unknown_execution(self):
        data = self.inputs(6)
        data.update(policy='fixed',fixed_exit_step=3)
        data['adjusted_open'][3:5,0] = np.nan
        data['adjusted_close'][3:5,0] = np.nan
        data['sell_allowed'][3:5,0] = 0.
        result = policy_path_labels(**data)
        self.assertEqual(result.labels.actual_sell_open_step.iloc[0],5)
        self.assertEqual(result.labels.blocked_sell_days.iloc[0],2)
        self.assertEqual(result.labels.execution_unknown_sell_days.iloc[0],0)
        self.assertEqual(result.labels.missing_hold_marks.iloc[0],2)

    def test_no_entry_is_not_deleted_and_unknown_does_not_allow_buy(self):
        data = self.inputs(6,3)
        data['entry_allowed'][:] = [0,np.nan,1]
        data['adjusted_open'][0,2] = np.nan
        result = policy_path_labels(**data)
        self.assertEqual(len(result.labels),3)
        self.assertTrue(result.labels.no_entry.all())
        np.testing.assert_array_equal(result.labels.label_return,[0,0,0])
        np.testing.assert_array_equal(result.labels.entry_execution_unknown,[False,True,False])

    def test_delisting_writes_off_active_stock_but_not_cash_after_sale(self):
        data = self.inputs(6,2)
        data.update(policy='fixed',fixed_exit_step=2)
        data['sell_allowed'][2:,0] = 0
        data['delist_step'][:] = 4
        result = policy_path_labels(**data)
        np.testing.assert_allclose(result.labels.label_return,[-1,0])
        self.assertEqual(result.labels.delist_writeoff_step.iloc[0],4)
        self.assertTrue(np.isnan(result.labels.delist_writeoff_step.iloc[1]))
        self.assertTrue(np.isnan(result.labels.actual_sell_open_step.iloc[0]))

    def test_early_sale_never_matures_before_common_endpoint(self):
        full = self.inputs(8)
        full.update(policy='fixed',fixed_exit_step=2)
        result = policy_path_labels(**full)
        short = dict(full)
        for key in ['adjusted_open','sell_allowed']:
            short[key] = short[key][:5]
        for key in ['adjusted_close','review_mask','trend','observable']:
            short[key] = short[key][:4]
        prefix = policy_path_labels(**short)
        self.assertEqual(prefix.labels.actual_sell_open_step.iloc[0],2)
        self.assertFalse(prefix.labels.mature.iloc[0])
        self.assertTrue(prefix.labels[['label_return','label_mae','settled_label_return']].isna().all().all())
        np.testing.assert_allclose(prefix.marked_nav[:-1],result.marked_nav[:5])
        np.testing.assert_array_equal(prefix.failure_counts,result.failure_counts[:4])
        np.testing.assert_array_equal(prefix.pending_after_open,result.pending_after_open[:5])

    def test_healthy_monthly_policy_is_censored_not_forced_liquidated(self):
        data = self.inputs()
        data['adjusted_open'][-1,0] = 11.
        result = policy_path_labels(**data)
        row = result.labels.iloc[0]
        self.assertTrue(row.no_request_administrative_censor)
        self.assertFalse(row.pending_exit)
        self.assertTrue(np.isnan(row.exit_request_close_step))
        self.assertTrue(np.isnan(row.settled_label_return))
        self.assertAlmostEqual(row.label_return,.1)

    def test_no_review_no_failure_and_no_same_day_sell(self):
        data = self.inputs()
        data['review_mask'][1:,0] = False
        data['trend'][:,0] = 0
        data['confirmations'] = 1
        result = policy_path_labels(**data)
        self.assertEqual(result.labels.exit_request_close_step.iloc[0],0)
        self.assertEqual(result.labels.actual_sell_open_step.iloc[0],1)

    def test_infinite_permission_is_rejected(self):
        data = self.inputs()
        data['sell_allowed'][2,0] = np.inf
        with self.assertRaises(ValueError):
            policy_path_labels(**data)

    def test_st_review_is_common_risk_and_primary_simultaneous_reason(self):
        for policy in ['fixed','monthly_two_failures']:
            data = self.inputs()
            data.update(policy=policy,fixed_exit_step=6 if policy=='fixed' else None)
            data['trend'][:2,0] = 0
            data['risk_exit_review'] = np.zeros_like(data['review_mask'])
            data['risk_exit_review'][1,0] = True
            result = policy_path_labels(**data)
            self.assertEqual(result.labels.exit_request_reason.iloc[0],'st')
            self.assertEqual(result.labels.exit_request_close_step.iloc[0],1)
            self.assertEqual(result.labels.actual_sell_open_step.iloc[0],2)

    def test_st_off_review_is_not_backfilled_and_delist_preempts_fill(self):
        data = self.inputs()
        data['risk_exit_review'] = np.zeros_like(data['review_mask'])
        data['risk_exit_review'][0,0] = True
        data['review_mask'][0,0] = False
        result = policy_path_labels(**data)
        self.assertTrue(np.isnan(result.labels.exit_request_close_step.iloc[0]))
        data.update(policy='fixed',fixed_exit_step=2)
        data['delist_step'][0] = 2
        result = policy_path_labels(**data)
        self.assertEqual(result.labels.delist_writeoff_step.iloc[0],2)
        self.assertTrue(np.isnan(result.labels.actual_sell_open_step.iloc[0]))
        self.assertEqual(result.labels.label_return.iloc[0],-1)

    def test_st_known_at_execution_open_does_not_invent_previous_close_request(self):
        for policy in ['fixed','monthly_two_failures']:
            data = self.inputs()
            data.update(policy=policy,fixed_exit_step=6 if policy=='fixed' else None)
            data['risk_exit_open'] = np.zeros_like(data['sell_allowed'],dtype=bool)
            data['risk_exit_open'][2,0] = True
            data['adjusted_open'][2,0] = 8.
            result = policy_path_labels(**data)
            row = result.labels.iloc[0]
            self.assertTrue(np.isnan(row.exit_request_close_step))
            self.assertEqual(row.exit_request_open_step,2)
            self.assertEqual(row.earliest_sell_open_step,2)
            self.assertEqual(row.actual_sell_open_step,2)
            self.assertEqual(row.exit_request_reason,'st')
            self.assertAlmostEqual(row.request_mark_return,-.2)
            self.assertAlmostEqual(row.label_return,-.2)

    def test_st_open_request_remains_pending_then_delist_preempts_sale(self):
        data = self.inputs()
        data['risk_exit_open'] = np.zeros_like(data['sell_allowed'],dtype=bool)
        data['risk_exit_open'][2,0] = True
        data['sell_allowed'][2,0] = 0
        data['delist_step'][0] = 3
        result = policy_path_labels(**data)
        row = result.labels.iloc[0]
        self.assertEqual(row.exit_request_open_step,2)
        self.assertTrue(np.isnan(row.exit_request_close_step))
        self.assertTrue(np.isnan(row.actual_sell_open_step))
        self.assertEqual(row.delist_writeoff_step,3)
        self.assertEqual(row.label_return,-1)

    def test_later_st_open_does_not_rewrite_prior_trend_request(self):
        data = self.inputs()
        data['trend'][:2,0] = 0
        data['sell_allowed'][2:4,0] = 0
        data['risk_exit_open'] = np.zeros_like(data['sell_allowed'],dtype=bool)
        data['risk_exit_open'][3,0] = True
        result = policy_path_labels(**data)
        row = result.labels.iloc[0]
        self.assertEqual(row.exit_request_close_step,1)
        self.assertTrue(np.isnan(row.exit_request_open_step))
        self.assertEqual(row.exit_request_reason,'trend_failure')
        self.assertEqual(row.actual_sell_open_step,4)


if __name__ == '__main__':
    unittest.main()
