"""真实状态接续和订单冻结的对抗测试，不用模拟涨跌证明成交。"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import yaml

from shadow_account import ShadowAccount
from shadow_planner import ShadowError, make_plan, empty_memory

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    cfg = yaml.safe_load((ROOT/'configs/shadow_execution.yaml').read_text())
    snapshot = dict(account_id='test', asof='2024-01-02', source='test-broker', cash=5000.,
                    receivables=0., equity=5000., frozen_cash=0., open_orders=[], positions={})
    calendar = dict(source='test-calendar',sessions=['2024-01-02','2024-01-03','2024-01-04','2024-01-05'])
    bundle = dict(asof='2024-01-02', signal_date='2024-01-02', is_review=True,
        eod_complete=True, provenance={'test':True}, model_year=2024, model_cutoff='2023-09-01',
        expected_return=.031, signals={'600000.SH':.4, '601988.SH':.3}, holds={},
        risk_discounts={'600000.SH':1., '601988.SH':1.}, st_codes=[], blocked_next_day=[],
        quotes={c:dict(date='2024-01-02', close=10., volume=1e6) for c in ['600000.SH','601988.SH']},
        eligible={'600000.SH':True,'601988.SH':True})
    return snapshot, bundle, calendar, cfg


def preflight(plan):
    return dict(timestamp=plan['execution_date']+'T09:16:00+08:00', source='test-broker',
        account_id=plan['account_id'],positions=plan['snapshot_positions'],open_orders=[],
        session='opening_auction', corporate_actions_checked=True, available_cash=plan['cash'],
        securities={o['ts_code']:dict(down_limit=1.,up_limit=100.,suspended=False,corporate_action=False,
                     broker_accepts_limit=True,buy_allowed=True,account_sellable=10000) for o in plan['orders']})


class ShadowExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'account.sqlite'
        self.account = ShadowAccount(self.path)
        self.s, self.b, self.cal, self.cfg = fixture()
        self.account.import_snapshot(self.s)

    def tearDown(self):
        self.account.close(); self.temp.cleanup()

    def plan(self):
        return self.account.plan(self.b,self.cal,self.cfg)

    def test_last_day_does_not_need_tomorrow_quote(self):
        plan = self.plan()
        self.assertEqual(plan['execution_date'],'2024-01-03')
        self.assertTrue(plan['orders'])
        self.assertLessEqual(sum(o['reserved_cash'] for o in plan['orders']),self.s['cash'])
        self.assertTrue(all(o['shares']%100==0 for o in plan['orders']))

    def test_restart_duplicate_plan_has_same_ids_no_new_orders(self):
        old = self.plan(); self.account.close(); self.account=ShadowAccount(self.path)
        self.assertEqual(old,self.plan())
        self.assertEqual(len(self.account.status()['orders']),len(old['orders']))

    def test_concurrent_duplicate_planning_commits_once(self):
        def work(_):
            connection=ShadowAccount(self.path)
            try:return connection.plan(self.b,self.cal,self.cfg)
            finally:connection.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(work,range(2)))
        self.assertEqual(results[0],results[1])
        self.assertEqual(len(self.account.status()['orders']),len(results[0]['orders']))

    def test_changed_same_day_input_refused(self):
        self.plan(); self.b['quotes']['600000.SH']['close']+=.1
        with self.assertRaises(ShadowError): self.plan()

    def test_stale_missing_or_future_inputs_block(self):
        for key,value in [('asof','2024-01-01'),('eod_complete',False),('model_cutoff','2024-01-01')]:
            b=deepcopy(self.b);b[key]=value
            with self.assertRaises(ShadowError): make_plan(self.s,b,self.cal,self.cfg)
        b=deepcopy(self.b);b['risk_discounts'].pop('600000.SH')
        with self.assertRaises(ShadowError): make_plan(self.s,b,self.cal,self.cfg)
        with self.assertRaises(ShadowError): make_plan(self.s,self.b,dict(source='test',sessions=['2024-01-02']),self.cfg)

    def test_no_buy_with_cash_from_pending_sale(self):
        s=deepcopy(self.s);s.update(cash=0.,equity=5000.)
        s['positions']={'600000.SH':dict(shares=500,sellable_next_day=500,close=10.)}
        b=deepcopy(self.b);b['holds']={'600000.SH':False}
        plan=make_plan(s,b,self.cal,self.cfg)
        self.assertTrue(plan['orders']);self.assertTrue(all(o['side']=='sell' for o in plan['orders']))

    def test_five_slots_and_locked_old_shares(self):
        codes=['600000.SH','601988.SH','601398.SH','601288.SH','601998.SH']
        s=deepcopy(self.s);s['positions']={c:dict(shares=100,sellable_next_day=0,close=10.) for c in codes};s['equity']=10000.
        b=deepcopy(self.b);b['holds']={c:False for c in codes}
        for c in codes:b['quotes'][c]=dict(date=s['asof'],close=10.,volume=1e6)
        plan=make_plan(s,b,self.cal,self.cfg)
        self.assertEqual(plan['orders'],[])

    def test_preflight_time_and_limits_and_action_rollback(self):
        plan=self.plan();m=preflight(plan);m['timestamp']='2024-01-03T09:31:00+08:00'
        with self.assertRaises(ShadowError):self.account.preflight(plan['plan_id'],m)
        m=preflight(plan);m['securities'][plan['orders'][-1]['ts_code']]['up_limit']=5.
        with self.assertRaises(ShadowError):self.account.preflight(plan['plan_id'],m)
        self.assertTrue(all(o['status']=='planned' for o in self.account.status()['orders']))
        m=preflight(plan);m['securities'][plan['orders'][0]['ts_code']]['corporate_action']=True
        with self.assertRaises(ShadowError):self.account.preflight(plan['plan_id'],m)

    def test_partial_fill_duplicate_and_cancel_restart_reconcile(self):
        plan=self.plan();self.account.preflight(plan['plan_id'],preflight(plan));order=plan['orders'][0]
        fill=dict(execution_id='F1',order_id=order['order_id'],date='2024-01-03',shares=40,price=10.,fee=5.,evidence='broker/F1')
        self.assertEqual(self.account.record_fill(fill)['status'],'partial')
        self.assertEqual(self.account.record_fill(fill)['status'],'duplicate')
        self.account.terminal(order['order_id'],'cancelled','cancel1')
        for other in plan['orders'][1:]:self.account.terminal(other['order_id'],'rejected','reject')
        self.account.close();self.account=ShadowAccount(self.path)
        s=deepcopy(self.s);s.update(asof='2024-01-03',cash=4595.,equity=4995.)
        s['positions']={order['ts_code']:dict(shares=40,sellable_next_day=40,close=10.)}
        self.assertEqual(self.account.import_snapshot(s)['status'],'reconciled')

    def test_conflicting_fill_and_overfill_rollback(self):
        plan=self.plan();self.account.preflight(plan['plan_id'],preflight(plan));order=plan['orders'][0]
        fill=dict(execution_id='F1',order_id=order['order_id'],date='2024-01-03',shares=100000,price=10.,fee=5.,evidence='broker')
        before=self.account.status()
        with self.assertRaises(ShadowError):self.account.record_fill(fill)
        self.assertEqual(before,self.account.status())
        fill['shares']=20;self.account.record_fill(fill);fill['fee']=6.
        with self.assertRaises(ShadowError):self.account.record_fill(fill)

    def test_unresolved_or_mismatched_snapshot_block(self):
        self.plan();s=deepcopy(self.s);s['asof']='2024-01-03'
        with self.assertRaises(ShadowError):self.account.import_snapshot(s)
        for o in self.account.status()['orders']:self.account.terminal(o['order_id'],'expired','broker-expiry')
        s['cash']-=10;s['equity']-=10
        with self.assertRaises(ShadowError):self.account.import_snapshot(s)

    def test_verified_cash_adjustment_is_idempotent(self):
        item=dict(event_id='D1',kind='dividend',date='2024-01-03',cash_delta=50.,evidence='broker-dividend')
        self.account.adjustment(item);self.assertEqual(self.account.adjustment(item)['status'],'duplicate')
        s=deepcopy(self.s);s.update(asof='2024-01-03',cash=5050.,equity=5050.)
        self.account.import_snapshot(s)
        self.assertEqual(self.account.state()['book_cash'],5050.)

    def test_future_cash_event_cannot_reconcile_earlier_snapshot(self):
        self.account.adjustment(dict(event_id='future',kind='deposit',date='2024-01-05',cash_delta=50.,evidence='test'))
        s=deepcopy(self.s);s.update(asof='2024-01-03',cash=5050.,equity=5050.)
        with self.assertRaises(ShadowError):self.account.import_snapshot(s)

    def test_premarket_actual_position_change_blocks(self):
        plan=self.plan();market=preflight(plan);market['positions']={'601988.SH':100}
        with self.assertRaises(ShadowError):self.account.preflight(plan['plan_id'],market)

    def test_unquoted_candidate_is_explicitly_skipped_not_held(self):
        self.b['quotes'].pop('600000.SH')
        plan=self.plan()
        self.assertIn(dict(ts_code='600000.SH',reason='no_known_closing_reference'),plan['decisions'])
        self.assertFalse(any(o['ts_code']=='600000.SH' for o in plan['orders']))

    def test_unselected_audit_null_persists_and_invalid_config_rejected(self):
        self.b['quotes']['601988.SH']['close']=500.
        plan=self.plan()
        skipped=next(x for x in plan['decisions'] if x['ts_code']=='601988.SH')
        self.assertIsNone(skipped['reserved_entry_cash'])
        self.assertEqual(self.plan(),plan)
        cfg=deepcopy(self.cfg);cfg['max_stock_fraction']=2.
        with self.assertRaises(ShadowError):make_plan(self.s,self.b,self.cal,cfg)

    def test_pending_reduction_survives_partial_cancel(self):
        s=deepcopy(self.s);s.update(cash=100.,equity=10100.)
        s['positions']={'600000.SH':dict(shares=1000,sellable_next_day=1000,close=10.)}
        b=deepcopy(self.b);b['holds']={'600000.SH':True}
        plan=make_plan(s,b,self.cal,self.cfg)
        target=plan['next_memory']['pending_sells']['600000.SH']['target_shares']
        s.update(asof='2024-01-03',cash=3095.,equity=10095.);s['positions']['600000.SH']['shares']=700;s['positions']['600000.SH']['sellable_next_day']=700
        b['asof']=s['asof'];b['is_review']=False
        for q in b['quotes'].values():q['date']=s['asof']
        second=make_plan(s,b,self.cal,self.cfg,plan['next_memory'])
        sale=next(o for o in second['orders'] if o['side']=='sell')
        self.assertEqual(sale['shares'],700-target)

    def test_supplement_only_after_actual_sale_once(self):
        b=deepcopy(self.b);b.update(asof='2024-01-03',is_review=False)
        s=deepcopy(self.s);s['asof']='2024-01-03'
        for q in b['quotes'].values():q['date']=s['asof']
        state=empty_memory();state['signal_date']='2024-01-02'
        self.assertEqual(make_plan(s,b,self.cal,self.cfg,state)['orders'],[])
        state.update(last_sale_date='2024-01-03',last_sale_signal='2024-01-02')
        plan=make_plan(s,b,self.cal,self.cfg,state)
        self.assertTrue(plan['orders']);self.assertTrue(plan['next_memory']['supplementary_used'])
        self.assertEqual(make_plan(s,b,self.cal,self.cfg,plan['next_memory'])['orders'],[])


if __name__=='__main__':unittest.main()
