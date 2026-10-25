"""人工播报、逐单核对、未执行与真实成交分离。"""
from copy import deepcopy
from email import policy
from email.parser import BytesParser
from pathlib import Path
import tempfile
import unittest

import yaml

from manual_execution import ManualAccount,bulletin
from shadow_planner import ShadowError
from test_shadow_execution import fixture

ROOT=Path(__file__).resolve().parents[1]


class ManualExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.a=ManualAccount(Path(self.temp.name)/'account.sqlite')
        s,self.b,self.cal,self.cfg=fixture();self.a.import_snapshot(s)
        self.b['provenance']['receipt']=dict(complete=True,asof='2024-01-02',downloaded_at='2024-01-02T21:00:00+08:00')
        self.plan=self.a.prepare(self.b,self.cal,self.cfg,'2024-01-02T21:01:00+08:00')
        self.config=yaml.safe_load((ROOT/'configs/manual_execution.yaml').read_text())

    def tearDown(self):
        self.a.close();self.temp.cleanup()

    def review(self,order):
        return dict(timestamp='2024-01-03T09:16:00+08:00',source='manual broker screen',evidence='synthetic review',
            account_id='test',positions={},unresolved_external_orders=False,ts_code=order['ts_code'],
            status_checked=True,corporate_actions_checked=True,broker_accepts_limit=True,
            suspended=False,corporate_action=False,buy_allowed=True,is_st=False,
            down_limit=9.,up_limit=11.,sellable_shares=0,available_cash=5000.)

    def test_email_preparation_not_fills_or_delivery(self):
        before=self.a.status()
        result=bulletin(self.plan,dict(blockers=['stock_st_unavailable'],all_corporate_actions_covered=False),
            '2024-01-02T21:05:00+08:00',self.config)
        self.assertFalse(result['mail_sent']);self.assertFalse(result['historical'])
        self.assertIn('需要在客户端核对',result['body']);self.assertIn('逐股人工例外处理',result['body'])
        parsed=BytesParser(policy=policy.default).parsebytes(result['eml'])
        self.assertIn('手动执行',str(parsed['Subject']))
        self.assertEqual(self.a.status(),before)

    def test_expired_draft_visibly_historical(self):
        result=bulletin(self.plan,{},'2024-01-03T09:25:00+08:00',self.config)
        self.assertTrue(result['historical']);self.assertIn('勿执行',result['subject'])
        with self.assertRaises(ShadowError):self.a.prepare(self.b,self.cal,self.cfg,'2024-01-03T09:25:00+08:00')

    def test_per_order_exception_does_not_block_other_stock(self):
        first,second=self.plan['orders'][:2]
        bad=self.review(first);bad['corporate_action']=True
        with self.assertRaises(ShadowError):self.a.review_order(first['order_id'],bad)
        self.a.review_order(second['order_id'],self.review(second))
        self.assertEqual(self.a.order(first['order_id'])['status'],'planned')
        self.assertEqual(self.a.order(second['order_id'])['status'],'ready')
        self.assertEqual(self.a.state()['book_positions'],{})

    def test_manual_fill_skip_and_next_snapshot(self):
        first,second=self.plan['orders'][:2]
        self.a.skip_order(first['order_id'],'人工核对后决定不提交')
        self.a.review_order(second['order_id'],self.review(second))
        fill=dict(execution_id='manual-receipt-1',order_id=second['order_id'],date='2024-01-03',
            shares=50,price=10.,fee=5.,evidence='synthetic actual fill')
        self.a.record_fill(fill)
        self.a.record_fill(fill)
        with self.assertRaises(ShadowError):self.a.skip_order(second['order_id'],'不能隐去部分成交')
        self.a.terminal(second['order_id'],'cancelled','synthetic broker cancellation')
        snap=dict(account_id='test',source='manual closing reconciliation',asof='2024-01-03',cash=4495.,
            equity=4995.,receivables=0.,frozen_cash=0.,open_orders=[],
            positions={second['ts_code']:dict(shares=50,sellable_next_day=50,close=10.)})
        self.a.import_snapshot(snap)
        self.assertEqual(self.a.state()['book_positions'],{second['ts_code']:50})
        self.assertTrue(self.a.order(first['order_id'])['manual_not_submitted'])

    def test_repeat_review_and_skip_are_idempotent(self):
        first,second=self.plan['orders'][:2]
        review=self.review(first);self.a.review_order(first['order_id'],review)
        self.assertEqual(self.a.review_order(first['order_id'],review)['status'],'duplicate')
        self.a.skip_order(second['order_id'],'manual')
        self.assertEqual(self.a.skip_order(second['order_id'],'manual')['status'],'duplicate')

    def test_client_checks_and_position_mismatch_rejected(self):
        first=self.plan['orders'][0]
        for key,value in [('status_checked',False),('is_st',True),('positions',{'600000.SH':100}),('available_cash',1.),('down_limit',10.9)]:
            item=self.review(first);item[key]=value
            with self.subTest(key=key),self.assertRaises(ShadowError):self.a.review_order(first['order_id'],item)

    def test_future_and_backfill_input_rejected(self):
        for field,value in [('downloaded_at','2024-01-04T21:00:00+08:00'),('downloaded_at','2024-01-02T14:00:00+08:00'),('historical_backfill',True)]:
            b=deepcopy(self.b);b['provenance']['receipt'][field]=value
            with self.subTest(field=field),self.assertRaises(ShadowError):
                self.a.prepare(b,self.cal,self.cfg,'2024-01-02T21:05:00+08:00')


if __name__=='__main__':unittest.main()
