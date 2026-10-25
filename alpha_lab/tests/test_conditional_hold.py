"""长期状态条件退出的独立单元与原执行边界测试。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from conditional_hold import build_conditional_holds
from hold_sensitivity import build_hold_variants


class ConditionalHoldTest(unittest.TestCase):
    def setUp(self):
        self.spec = yaml.safe_load((ROOT/'configs/conditional_hold_research.yaml').read_text())
        self.dates = pd.bdate_range('2022-01-03', periods=220)
        self.reviews = self.dates[[130, 150, 170, 190, 210]]
        self.prices = pd.DataFrame(100., index=self.dates, columns=['I', 'B', 'M'])
        self.prices['M'] = 90.
        self.prices.loc[self.dates[[10, 30, 50, 70, 90]], 'M'] = 100.
        self.prices.loc[self.reviews] = [120., 80., 100.]
        # 外生趋势/可观测输入用于隔离分类逻辑；真实价格一致性由run矩阵核验负责。
        self.trend = pd.DataFrame(False, index=self.dates, columns=self.prices.columns)
        self.known = pd.DataFrame(True, index=self.dates, columns=self.prices.columns)

    def build(self, prices=None, trend=None, known=None, reviews=None):
        return build_conditional_holds(self.prices if prices is None else prices,
            self.trend if trend is None else trend, self.known if known is None else known,
            self.reviews if reviews is None else reviews, self.spec)

    def test_classification_and_two_by_two_actions(self):
        variants, evidence = self.build()
        self.assertEqual(evidence['weak_regime'].iloc[0].tolist(), ['long_intact_weak', 'long_broken_weak', 'mixed'])
        np.testing.assert_array_equal(variants['patient'].I.iloc[:3], [1, 1, 0])
        np.testing.assert_array_equal(variants['hybrid'].I.iloc[:3], [1, 1, 0])
        np.testing.assert_array_equal(variants['accelerated'].B.iloc[:3], [0, 0, 0])
        np.testing.assert_array_equal(variants['patient'].B.iloc[:3], [1, 0, 0])
        np.testing.assert_array_equal(variants['hybrid'].M.iloc[:3], [1, 0, 0])
        np.testing.assert_array_equal(variants['persist10'].M.iloc[:3], [0, 0, 0])

    def test_original_two_references_exact(self):
        variants, _ = self.build()
        reference, _ = build_hold_variants(self.trend, self.known, self.reviews, dict(persistence_window=20,
            variants=dict(baseline=dict(monthly_confirmations=2), persist10=dict(monthly_confirmations=2,persistent_weak_days=10))))
        for name in reference:
            pd.testing.assert_frame_equal(variants[name], reference[name], check_exact=True)

    def test_state_changes_do_not_reset_counter_or_extend_beyond_three(self):
        self.prices.loc[self.reviews[:3], 'B'] = [80., 120., 120.]
        variants, evidence = self.build()
        np.testing.assert_array_equal(evidence['monthly_failure_streak'].B.iloc[:3], [1, 2, 3])
        np.testing.assert_array_equal(variants['hybrid'].B.iloc[:3], [0, 1, 0])
        np.testing.assert_array_equal(variants['patient'].B.iloc[:3], [1, 1, 0])
        # 此处矩阵许可可恢复，但原引擎已提交的退出订单不可撤销，另有执行测试。

    def test_unknown_resets_counter_and_cannot_accelerate(self):
        self.known.loc[self.reviews[1], 'B'] = False
        variants, evidence = self.build()
        self.assertTrue(all(pd.isna(v.loc[self.reviews[1], 'B']) for v in variants.values()))
        self.assertEqual(evidence['monthly_failure_streak'].loc[self.reviews[1], 'B'], 0)
        self.assertEqual(evidence['weak_regime'].loc[self.reviews[1], 'B'], 'unknown')
        self.known.loc[self.dates[120], 'B'] = False
        variants, evidence = self.build()
        self.assertEqual(evidence['daily_known_count'].loc[self.reviews[0], 'B'], 19)
        self.assertEqual(variants['accelerated'].loc[self.reviews[0], 'B'], 1)

    def test_strong_state_resets_and_disables_both_interventions(self):
        self.trend.loc[self.reviews[2]] = True
        variants, evidence = self.build()
        self.assertTrue(all(v.loc[self.reviews[2]].eq(1).all() for v in variants.values()))
        self.assertTrue(evidence['monthly_failure_streak'].loc[self.reviews[2]].eq(0).all())
        self.assertTrue(evidence['weak_regime'].loc[self.reviews[2]].eq('strong').all())

    def test_zero_long_return_and_ma_equality_are_not_positive(self):
        self.prices.loc[:, 'B'] = 100.
        _, evidence = self.build()
        self.assertTrue(evidence['long_return'].B.eq(0).all())
        self.assertTrue(evidence['long_ma'].B.eq(100).all())
        self.assertTrue(evidence['long_broken'].B.all())
        self.assertFalse(evidence['long_intact'].B.any())

    def test_allowed_state_partial_order(self):
        v, _ = self.build()
        for left, right in [('persist10','accelerated'),('accelerated','baseline'),('baseline','patient'),
                            ('accelerated','hybrid'),('hybrid','patient')]:
            self.assertTrue((v[left] <= v[right]).all().all(), (left,right))
        self.assertGreater(v['hybrid'].I.iloc[1], v['baseline'].I.iloc[1])
        self.assertLess(v['hybrid'].B.iloc[0], v['baseline'].B.iloc[0])

    def test_history_prefix_and_future_price_poison(self):
        full, _ = self.build()
        cutoff = self.reviews[2]
        prefix, _ = self.build(self.prices.loc[:cutoff], self.trend.loc[:cutoff], self.known.loc[:cutoff], self.reviews[:3])
        poisoned = self.prices.copy(); poisoned.loc[poisoned.index > cutoff] *= 1e6
        changed, _ = self.build(prices=poisoned)
        for name in full:
            pd.testing.assert_frame_equal(prefix[name], full[name].loc[:cutoff], check_exact=True)
            pd.testing.assert_frame_equal(changed[name].loc[:cutoff], full[name].loc[:cutoff], check_exact=True)

    def test_invalid_alignment_or_known_price_evidence_rejected(self):
        with self.assertRaises(ValueError):
            self.build(prices=self.prices.iloc[::-1])
        damaged = self.prices.copy(); damaged.loc[self.reviews[0], 'I'] = np.nan
        with self.assertRaises(ValueError):
            self.build(prices=damaged)
        unknown = self.known.copy(); unknown.loc[self.reviews[0], 'I'] = False
        variants, _ = self.build(prices=damaged, known=unknown, reviews=self.reviews[:1])
        self.assertTrue(pd.isna(variants['hybrid'].loc[self.reviews[0], 'I']))

    def test_pending_exit_not_cancelled_by_later_patient_permission(self):
        import test_patient_execution as fixture_module
        fixture = fixture_module.PatientExecutionTest(); fixture.setUp()
        fixture.signals = fixture.signals.reindex(fixture.dates[[0,1,2,3]])
        fixture.signals.loc[:] = [2.,1.]
        fixture.forecasts = fixture.forecasts.reindex(fixture.signals.index); fixture.forecasts.loc[:] = [.1,-.1]
        fixture.hold = pd.DataFrame([[1.,1.],[0.,1.],[1.,1.],[1.,1.]], index=fixture.signals.index, columns=fixture.codes)
        mask = fixture.limits.date.eq(fixture.dates[2]) & fixture.limits.ts_code.eq(fixture.codes[0])
        fixture.limits.loc[mask,'down_limit'] = 10.
        result = fixture.run_case()
        sold = result.trades[result.trades.side.eq('sell')]
        self.assertEqual(list(sold.date), [fixture.dates[3]])
        requests = result.events[result.events.kind.eq('hold_exit_requested')]
        self.assertEqual(requests.iloc[-1].reason, 'confirmed_pending')
        self.assertEqual(requests.iloc[-1].signal_date, fixture.dates[2])


if __name__ == '__main__':
    unittest.main()
