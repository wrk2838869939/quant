"""同日盘中决策、收盘实际成交、T+1和不可改写的双阶段账本。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import yaml

from execution_clock import check_proxy
from manual_execution import ManualAccount, bulletin
from shadow_planner import ShadowError, make_plan
from test_shadow_execution import fixture


def receipt(date='2024-01-02'):
    return dict(asof=date, kind='intraday_proxy', complete=True, source='synthetic only',
        sha256='fixture', downloaded_at=date+'T14:31:00+08:00',
        quote_min_at=date+'T14:30:00+08:00', quote_max_at=date+'T14:31:00+08:00',
        market_index=dict(code='000300.SH', sha256='fixture', timestamp=date+'T14:30:50+08:00'))


class ClosingExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'synthetic.sqlite'
        self.a = ManualAccount(self.path)
        self.s, self.b, self.cal, self.cfg = fixture()
        self.s.update(phase='intraday', timestamp='2024-01-02T14:32:00+08:00')
        self.cfg['execution_session'] = 'closing_auction'
        self.b.update(eod_complete=False, input_kind='intraday_proxy', provenance=dict(receipt=receipt()))
        self.a.import_snapshot(self.s)

    def tearDown(self):
        self.a.close(); self.temp.cleanup()

    def plan(self):
        return self.a.prepare(self.b, self.cal, self.cfg, '2024-01-02T14:33:00+08:00')

    def review(self, order, timestamp='2024-01-02T14:57:01+08:00'):
        return dict(timestamp=timestamp, source='synthetic broker', evidence='fixture',
            account_id='test', positions={}, unresolved_external_orders=False, ts_code=order['ts_code'],
            status_checked=True, corporate_actions_checked=True, broker_accepts_limit=True,
            suspended=False, corporate_action=False, buy_allowed=True, is_st=False,
            down_limit=9., up_limit=11., sellable_shares=0, available_cash=5000.)

    def test_same_day_freeze_restart_and_actual_cash_only(self):
        plan = self.plan()
        self.assertEqual(plan['execution_date'], '2024-01-02')
        self.assertTrue(plan['orders'])
        self.assertTrue(all(o['shares'] % 100 == 0 for o in plan['orders']))
        self.assertLessEqual(sum(o['reserved_cash'] for o in plan['orders']), self.s['cash'])
        self.a.close(); self.a = ManualAccount(self.path)
        self.assertEqual(self.plan(), plan)
        self.b['quotes']['600000.SH']['close'] += .1
        with self.assertRaises(ShadowError): self.plan()

    def test_same_day_partial_fill_eod_reconciliation_preserves_both_snapshots(self):
        plan = self.plan(); order = plan['orders'][0]
        self.a.review_order(order['order_id'], self.review(order))
        fill = dict(execution_id='C1', order_id=order['order_id'], date='2024-01-02',
            timestamp='2024-01-02T15:00:00+08:00', shares=50, price=10., fee=5., evidence='synthetic broker fill')
        self.assertEqual(self.a.record_fill(fill)['status'], 'partial')
        self.assertEqual(self.a.record_fill(fill)['status'], 'duplicate')
        eod = deepcopy(self.s)
        eod.update(phase='eod', timestamp='2024-01-02T15:10:00+08:00', cash=4495., equity=4995.,
            positions={order['ts_code']:dict(shares=50, sellable_next_day=50, close=10.)})
        with self.assertRaises(ShadowError): self.a.import_snapshot(eod)
        self.a.terminal(order['order_id'], 'expired', 'broker remaining expired')
        for o in plan['orders'][1:]: self.a.skip_order(o['order_id'], 'never submitted')
        self.a.import_snapshot(eod)
        self.assertEqual(self.a.db.execute('SELECT COUNT(*) FROM snapshot_history').fetchone()[0], 2)
        self.assertEqual(self.a.import_snapshot(eod)['status'], 'duplicate')
        fill['execution_id'] = 'C2'
        with self.assertRaises(ShadowError): self.a.record_fill(fill)
        # 下一日盘中快照与当日可卖股数可继续，旧收盘快照不覆盖。
        tomorrow = deepcopy(eod)
        tomorrow.update(asof='2024-01-03', phase='intraday', timestamp='2024-01-03T14:32:00+08:00')
        tomorrow['positions'][order['ts_code']]['sellable_today'] = 50
        self.a.import_snapshot(tomorrow)

    def test_today_locked_shares_cannot_use_next_day_sellable(self):
        s = deepcopy(self.s); s.update(cash=0., equity=5000.)
        s['positions'] = {'600000.SH':dict(shares=500, sellable_today=0, sellable_next_day=500, close=10.)}
        b = deepcopy(self.b); b['holds'] = {'600000.SH':False}
        self.assertEqual(make_plan(s,b,self.cal,self.cfg)['orders'], [])
        s['positions']['600000.SH']['sellable_today'] = 200
        plan = make_plan(s,b,self.cal,self.cfg)
        self.assertEqual(sum(o['shares'] for o in plan['orders']), 200)
        self.assertTrue(all(o['side']=='sell' for o in plan['orders']))

    def test_wrong_sessions_stale_future_and_eod_masquerade_block(self):
        for at in ['14:29:59','14:42:00','14:55:00','15:00:00']:
            with self.subTest(at=at), self.assertRaises(ShadowError):
                self.a.prepare(self.b,self.cal,self.cfg,'2024-01-02T'+at+'+08:00')
        plan = self.plan(); order = plan['orders'][0]
        for at in ['09:20:00','14:56:59','15:00:00']:
            with self.subTest(at=at), self.assertRaises(ShadowError):
                self.a.review_order(order['order_id'], self.review(order,'2024-01-02T'+at+'+08:00'))
        self.b['eod_complete'] = True
        with self.assertRaises(ShadowError): make_plan(self.s,self.b,self.cal,self.cfg)

    def test_bulletin_selects_closing_window_even_with_old_mail_config(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/manual_execution.yaml').read_text())
        plan = self.plan()
        draft = bulletin(plan,{},'2024-01-02T14:34:00+08:00',cfg)
        self.assertIn('14:57:00—15:00:00',draft['body'])
        self.assertIn('不是最终收盘数据',draft['body'])
        self.assertNotIn('09:15',draft['body'])
        self.assertFalse(draft['historical'])
        self.assertTrue(bulletin(plan,{},'2024-01-02T15:00:00+08:00',cfg)['historical'])

    def test_unknown_fill_time_and_same_day_snapshot_overwrite_block(self):
        plan = self.plan(); order = plan['orders'][0]
        self.a.review_order(order['order_id'],self.review(order))
        for at in [None,'2024-01-02T14:32:00+08:00','2024-01-02T14:59:00+08:00']:
            fill = dict(execution_id='bad',order_id=order['order_id'],date='2024-01-02',
                shares=10,price=10.,fee=0.,evidence='synthetic',timestamp=at)
            before=self.a.status()
            with self.assertRaises(ShadowError): self.a.record_fill(fill)
            self.assertEqual(before,self.a.status())
        changed=deepcopy(self.s);changed['timestamp']='2024-01-02T14:34:00+08:00'
        with self.assertRaises(ShadowError): self.a.import_snapshot(changed)

    def test_sale_cash_supplement_uses_previous_close_receipt_once(self):
        from shadow_planner import empty_memory
        s=deepcopy(self.s);s.update(asof='2024-01-03',timestamp='2024-01-03T14:32:00+08:00')
        b=deepcopy(self.b);b.update(asof=s['asof'],is_review=False)
        b['provenance']['receipt']=receipt(s['asof'])
        for q in b['quotes'].values():q['date']=s['asof']
        memory=empty_memory();memory.update(signal_date='2024-01-02',last_sale_date='2024-01-02',last_sale_signal='2024-01-02')
        first=make_plan(s,b,self.cal,self.cfg,memory)
        self.assertTrue(first['orders'])
        self.assertEqual(make_plan(s,b,self.cal,self.cfg,first['next_memory'])['orders'],[])

    def test_mail_deadline_is_same_day_close_and_does_not_send(self):
        from mail_outbox import MailOutbox
        box=MailOutbox(Path(self.temp.name)/'mail.sqlite')
        try:
            cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/manual_execution.yaml').read_text())
            settings=dict(sender='test@example.com',recipients=['user@example.com'],host='example.com',port=465,
                username='fixture',password='fixture',tls='ssl')
            ids=box.enqueue(self.plan(),{},settings,'2024-01-02T14:34:00+08:00',cfg)
            row=box.db.execute('SELECT deadline,status,attempts FROM mail WHERE id=?',(ids[0],)).fetchone()
            self.assertEqual(row,('2024-01-02T15:00:00+08:00','pending',0))
        finally:box.close()

    def test_simulated_observation_cannot_become_trade_plan(self):
        self.b['provenance']['receipt']['synthetic']=True
        with self.assertRaises(ShadowError): self.plan()


if __name__ == '__main__': unittest.main()
