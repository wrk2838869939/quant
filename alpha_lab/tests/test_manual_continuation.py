"""人工缺席、部分卖出及补税后的跨日账本接续。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from manual_execution import ManualAccount
from shadow_planner import ShadowError
from test_shadow_execution import fixture


class ManualContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'manual.sqlite'
        self.a = ManualAccount(self.path)
        self.snapshot, self.bundle, self.calendar, self.cfg = fixture()

    def tearDown(self):
        self.a.close()
        self.temp.cleanup()

    def prepare(self, date, review):
        b = deepcopy(self.bundle)
        b.update(asof=date, is_review=review)
        for q in b['quotes'].values(): q['date'] = date
        b['provenance']['receipt'] = dict(complete=True, asof=date, downloaded_at=date+'T21:00:00+08:00')
        return self.a.prepare(b, self.calendar, self.cfg, date+'T21:01:00+08:00')

    def restart(self):
        self.a.close()
        self.a = ManualAccount(self.path)

    def test_missed_buy_does_not_become_a_fill_or_unlimited_daily_retry(self):
        self.a.import_snapshot(self.snapshot)
        plan = self.prepare('2024-01-02', True)
        for order in plan['orders']: self.a.skip_order(order['order_id'], '人工未提交')
        self.restart()
        snap = dict(self.snapshot, asof='2024-01-03')
        self.a.import_snapshot(snap)
        next_plan = self.prepare('2024-01-03', False)
        self.assertEqual(next_plan['orders'], [])
        self.assertEqual(self.a.state()['book_cash'], 5000.)
        self.assertEqual(self.a.state()['book_positions'], {})

    def sale_plan(self):
        self.snapshot.update(cash=0., positions={'600000.SH':dict(shares=500, sellable_next_day=500, close=10.)})
        self.bundle['holds'] = {'600000.SH': False}
        self.bundle['signals'] = {'601988.SH': .3}
        self.a.import_snapshot(self.snapshot)
        plan = self.prepare('2024-01-02', True)
        self.assertEqual(len(plan['orders']), 1)
        return plan['orders'][0]

    def test_missed_sale_survives_restart_without_releasing_cash_or_slot(self):
        order = self.sale_plan()
        self.a.skip_order(order['order_id'], '人工未提交')
        self.restart()
        self.a.import_snapshot(dict(self.snapshot, asof='2024-01-03'))
        plan = self.prepare('2024-01-03', False)
        self.assertEqual(len(plan['orders']), 1)
        self.assertEqual(plan['orders'][0]['side'], 'sell')
        self.assertEqual(plan['orders'][0]['shares'], 500)
        self.assertEqual(plan['cash'], 0.)

    def test_partial_sale_only_releases_actual_proceeds(self):
        order = self.sale_plan()
        review = dict(timestamp='2024-01-03T09:16:00+08:00', source='synthetic client', evidence='synthetic',
            account_id='test', positions={'600000.SH':500}, unresolved_external_orders=False,
            ts_code='600000.SH', status_checked=True, corporate_actions_checked=True,
            broker_accepts_limit=True, suspended=False, corporate_action=False, buy_allowed=True,
            is_st=False, down_limit=9., up_limit=11., sellable_shares=500, available_cash=0.)
        self.a.review_order(order['order_id'], review)
        self.a.record_fill(dict(execution_id='partial-sale', order_id=order['order_id'], date='2024-01-03',
            shares=200, price=10., fee=5., evidence='synthetic receipt'))
        self.a.terminal(order['order_id'], 'cancelled', 'synthetic cancellation')
        self.restart()
        snap = dict(self.snapshot, asof='2024-01-03', cash=1995., equity=4995.,
            positions={'600000.SH':dict(shares=300, sellable_next_day=300, close=10.)})
        self.a.import_snapshot(snap)
        plan = self.prepare('2024-01-03', False)
        sale = next(o for o in plan['orders'] if o['side']=='sell')
        buys = [o for o in plan['orders'] if o['side']=='buy']
        self.assertEqual(sale['shares'], 300)
        self.assertTrue(buys)
        self.assertLessEqual(sum(o['reserved_cash'] for o in buys), 1995.)
        self.assertTrue(plan['next_memory']['supplementary_used'])

    def test_unresolved_missed_order_blocks_next_day_snapshot(self):
        self.a.import_snapshot(self.snapshot)
        self.prepare('2024-01-02', True)
        with self.assertRaises(ShadowError):
            self.a.import_snapshot(dict(self.snapshot, asof='2024-01-03'))

    def test_actual_dividend_tax_requires_separate_evidence_and_reconciles(self):
        self.a.import_snapshot(self.snapshot)
        snap = dict(self.snapshot, asof='2024-01-03', cash=4940., equity=4940.)
        with self.assertRaises(ShadowError): self.a.import_snapshot(snap)
        receipt = dict(event_id='tax-1', kind='tax', date='2024-01-03', cash_delta=-60., evidence='synthetic broker tax debit')
        self.a.adjustment(receipt)
        self.assertEqual(self.a.adjustment(receipt)['status'], 'duplicate')
        self.restart()
        self.a.import_snapshot(snap)
        self.assertEqual(self.a.state()['book_cash'], 4940.)


if __name__ == '__main__': unittest.main()
