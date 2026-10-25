"""9:30开盘后连续竞价执行的窗口契约、计划生成与账本边界。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from execution_clock import prepare_cutoff, timing
from manual_execution import ManualAccount, bulletin
from shadow_planner import ShadowError, fingerprint, make_plan
from test_shadow_execution import fixture
from test_monthly_risk_manual import fixture as monthly_fixture

ROOT = Path(__file__).resolve().parents[1]


def review(order, at):
    return dict(timestamp=at, source='manual broker screen', evidence='synthetic review',
        account_id='test', positions={}, unresolved_external_orders=False, ts_code=order['ts_code'],
        status_checked=True, corporate_actions_checked=True, broker_accepts_limit=True,
        suspended=False, corporate_action=False, buy_allowed=True, is_st=False,
        down_limit=9., up_limit=11., sellable_shares=0, available_cash=5000.)


class OpeningContinuousTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.a = ManualAccount(Path(self.temp.name)/'account.sqlite')
        self.s, self.b, self.cal, self.cfg = fixture()
        self.cfg['execution_session'] = 'opening_continuous'
        self.b['provenance']['receipt'] = dict(complete=True, asof='2024-01-02',
            downloaded_at='2024-01-02T21:00:00+08:00')
        self.a.import_snapshot(self.s)
        self.plan = self.a.prepare(self.b, self.cal, self.cfg, '2024-01-02T21:01:00+08:00')
        self.config = yaml.safe_load((ROOT/'configs/manual_execution.yaml').read_text())

    def tearDown(self):
        self.a.close(); self.temp.cleanup()

    def test_window_contract(self):
        self.assertEqual(timing('opening_continuous'),
            dict(prepare_after='21:00:00', execution_start='09:30:00', execution_end='09:35:00'))
        self.assertEqual(prepare_cutoff('opening_continuous'), '09:30:00')
        self.assertEqual(prepare_cutoff('opening_auction'), '09:25:00')
        with self.assertRaises(ShadowError): timing('noon')
        with self.assertRaises(ShadowError): prepare_cutoff('noon')

    def test_plan_is_next_day_continuous_with_prev_close_limits(self):
        self.assertEqual(self.plan['session'], 'opening_continuous')
        self.assertEqual(self.plan['decision_date'], '2024-01-02')
        self.assertEqual(self.plan['execution_date'], '2024-01-03')
        self.assertTrue(self.plan['orders'])
        for order in self.plan['orders']:
            self.assertEqual(order['side'], 'buy')
            self.assertEqual(order['limit_price'], 10.20)  # 前收盘10元×(1+2%缓冲)
            self.assertEqual(order['shares'] % 100, 0)

    def test_orders_match_opening_auction_except_session(self):
        opening = make_plan(self.s, self.b, self.cal, {**self.cfg, 'execution_session': 'opening_auction'})
        keys = lambda plan: [(o['ts_code'], o['side'], o['shares'], o['limit_price']) for o in plan['orders']]
        self.assertEqual(keys(self.plan), keys(opening))
        self.assertNotEqual(self.plan['session'], opening['session'])

    def test_intraday_snapshot_rejected_for_open_execution(self):
        s = deepcopy(self.s)
        s.update(phase='intraday', timestamp='2024-01-02T14:40:00+08:00')
        with self.assertRaises(ShadowError): make_plan(s, self.b, self.cal, self.cfg)

    def test_prepare_cutoff_enforced(self):
        path = Path(self.temp.name)/'cutoff.sqlite'
        account = ManualAccount(path); account.import_snapshot(self.s)
        account.prepare(self.b, self.cal, self.cfg, '2024-01-03T09:29:00+08:00')
        account.close()
        account = ManualAccount(path); account.import_snapshot(self.s)
        with self.assertRaises(ShadowError):
            account.prepare(self.b, self.cal, self.cfg, '2024-01-03T09:30:00+08:00')
        account.close()
        account = ManualAccount(Path(self.temp.name)/'early.sqlite'); account.import_snapshot(self.s)
        with self.assertRaises(ShadowError):
            account.prepare(self.b, self.cal, self.cfg, '2024-01-02T20:59:00+08:00')
        account.close()

    def test_review_window_and_idempotence(self):
        first = self.plan['orders'][0]
        with self.assertRaises(ShadowError):
            self.a.review_order(first['order_id'], review(first, '2024-01-03T09:29:59+08:00'))
        checked = review(first, '2024-01-03T09:30:00+08:00')
        self.assertEqual(self.a.review_order(first['order_id'], checked)['status'], 'ready')
        # 完全相同的核对内容重复提交才可幂等；内容改变必须报错。
        self.assertEqual(self.a.review_order(first['order_id'], checked)['status'], 'duplicate')
        with self.assertRaises(ShadowError):
            self.a.review_order(first['order_id'], review(first, '2024-01-03T09:31:00+08:00'))
        second = self.plan['orders'][1]
        with self.assertRaises(ShadowError):
            self.a.review_order(second['order_id'], review(second, '2024-01-03T09:35:00+08:00'))

    def test_fill_within_limit_and_partial_terminal(self):
        first, second = self.plan['orders'][:2]
        # 先逐单核对再提交；核对时的持仓与账本一致（空仓）。
        self.a.review_order(first['order_id'], review(first, '2024-01-03T09:30:30+08:00'))
        self.a.review_order(second['order_id'], review(second, '2024-01-03T09:30:45+08:00'))
        fill = dict(execution_id='open-cont-fill-1', order_id=first['order_id'], date='2024-01-03',
            shares=first['shares'], price=10.10, fee=5., evidence='synthetic actual fill',
            timestamp='2024-01-03T09:31:00+08:00')
        self.assertEqual(self.a.record_fill(fill)['status'], 'filled')
        over = dict(fill, execution_id='open-cont-over', order_id=second['order_id'], shares=100, price=10.21)
        with self.assertRaises(ShadowError): self.a.record_fill(over)
        partial = dict(fill, execution_id='open-cont-fill-2', order_id=second['order_id'], shares=50)
        self.assertEqual(self.a.record_fill(partial)['status'], 'partial')
        self.a.terminal(second['order_id'], 'cancelled', 'synthetic remainder cancelled after window')
        with self.assertRaises(ShadowError): self.a.skip_order(second['order_id'], '不能隐去部分成交')

    def test_bulletin_text_and_expiry(self):
        result = bulletin(self.plan, {}, '2024-01-02T21:05:00+08:00', self.config)
        self.assertFalse(result['historical'])
        self.assertIn('09:30开盘后连续竞价', result['body'])
        self.assertIn('09:30—09:35', result['body'])
        self.assertIn('可以撤单', result['body'])
        self.assertNotIn('收盘集合竞价', result['body'])
        late = bulletin(self.plan, {}, '2024-01-03T09:35:00+08:00', self.config, backfill=False)
        self.assertTrue(late['historical'])

    def test_monthly_config_session_accepted(self):
        on_disk = yaml.safe_load((ROOT/'configs/monthly_risk_open_day15.yaml').read_text())
        self.assertEqual(on_disk['execution_session'], 'opening_continuous')
        s, b, cal, cfg = monthly_fixture()
        cfg['execution_session'] = 'opening_continuous'
        b['strategy_config'] = fingerprint(cfg)
        account = ManualAccount(Path(self.temp.name)/'monthly.sqlite')
        account.import_snapshot(s)
        plan = account.prepare(b, cal, cfg, '2024-01-02T21:01:00+08:00')
        self.assertEqual(plan['session'], 'opening_continuous')
        self.assertEqual(plan['execution_date'], '2024-01-03')
        account.close()


if __name__ == '__main__':
    unittest.main()
