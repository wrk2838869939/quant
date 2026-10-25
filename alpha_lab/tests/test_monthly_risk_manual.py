"""新候选的实际成交、事务重启、风险回合与输入边界。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from manual_execution import ManualAccount, bulletin
from monthly_risk_live import load_model, probability, risk_history
from shadow_planner import ShadowError, fingerprint, make_plan
from restore_quota import RestoreQuota
from test_shadow_execution import preflight

ROOT = Path(__file__).resolve().parents[1]


def fixture(date='2024-01-02', high=False):
    cfg=yaml.safe_load((ROOT/'configs/monthly_risk_manual.yaml').read_text())
    cfg['schedule']['anchor']='2024-01-01'
    dates=pd.bdate_range('2023-07-03','2024-05-06').difference(pd.DatetimeIndex(['2024-01-01']))
    cal=dict(source='synthetic calendar',sessions=[str(d.date()) for d in dates])
    codes=['600000.SH','601988.SH','600001.SH','600002.SH','600003.SH']
    s=dict(account_id='test',asof=date,source='synthetic broker fixture',cash=80000.,receivables=0.,equity=80000.,frozen_cash=0.,open_orders=[],positions={})
    history=[]
    for d in ['2024-01-02','2024-02-01','2024-03-01','2024-04-01','2024-05-01']:
        if d>date:continue
        x=dict(ret20=0.,drawdown60=0.,vol20=.04 if (high and d==date) or (d=='2024-02-01' and date>'2024-02-01') else .001,vol_shock=1.)
        history.append(dict(date=d,features=x,probability=probability(x,load_model(cfg))))
    b=dict(asof=date,signal_date=history[-1]['date'],is_review=date==history[-1]['date'],eod_complete=True,
        provenance=dict(receipt=dict(complete=True,asof=date,downloaded_at=date+'T21:00:00+08:00')),
        model_year=2024,model_cutoff='2023-12-01',expected_return=.031,
        signals={c:.4-i*.1 for i,c in enumerate(codes)},holds={c:True for c in codes},
        risk_discounts={c:1. for c in codes},st_codes=[],blocked_next_day=[],
        quotes={c:dict(date=date,close=10.,volume=1e6) for c in codes},eligible={c:True for c in codes},
        strategy=cfg['strategy'],strategy_config=fingerprint(cfg),market_risk_history=history,
        stock_recovery={c:dict(known=True,weak=False,ret20=.1) for c in codes})
    b['provenance']['receipt']['market_index']=dict(code='000300.SH',asof=date,complete=True,source='synthetic index',sha256='synthetic',downloaded_at=date+'T21:00:00+08:00')
    return s,b,cal,cfg


class MonthlyManualTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'account.sqlite'
        self.a=ManualAccount(self.path)
        s,b,self.cal,self.cfg=fixture();self.a.import_snapshot(s)
        self.initial=self.a.prepare(b,self.cal,self.cfg,'2024-01-02T21:01:00+08:00')
        self.fill_plan(self.initial)

    def tearDown(self):self.a.close();self.tmp.cleanup()

    def fill_plan(self,plan,partial=False):
        self.a.preflight(plan['plan_id'],preflight(plan))
        fills=[]
        for o in plan['orders']:
            q=min(50,o['shares']) if partial else o['shares']
            f=dict(execution_id=o['order_id']+'-fill',order_id=o['order_id'],date=o['execution_date'],shares=q,price=10.,fee=5.,evidence='synthetic fill fixture')
            self.a.record_fill(f);fills.append(f)
            if partial:self.a.terminal(o['order_id'],'cancelled','synthetic unfilled remainder')
        return fills

    def next_plan(self,date,high=False,mutate=None):
        _,b,_,_=fixture(date,high)
        s=deepcopy(self.a.state()['snapshot']);state=self.a.state()
        s.update(asof=date,cash=state['book_cash'],positions={c:dict(shares=q,sellable_next_day=q,close=10.) for c,q in state['book_positions'].items()})
        s['equity']=s['cash']+sum(p['shares']*10 for p in s['positions'].values())
        self.a.import_snapshot(s)
        # 已冻结的历史概率不能被后续fixture重写。
        old=self.a.db.execute('SELECT payload FROM plans ORDER BY date DESC LIMIT 1').fetchone()
        import json
        prior=json.loads(old[0]);date0=prior.get('risk_review_date')
        for r in b['market_risk_history']:
            if r['date']==date0 and prior.get('risk_active'):
                r['features']['vol20']=.04;r['probability']=probability(r['features'],load_model(self.cfg))
        if mutate:mutate(b)
        return self.a.prepare(b,self.cal,self.cfg,date+'T21:01:00+08:00')

    def test_partial_sell_credit_restart_and_no_daily_restore(self):
        p=self.next_plan('2024-02-01',True)
        self.assertTrue(p['risk_active']);self.assertTrue(p['orders'])
        self.assertEqual(RestoreQuota(self.a.state()['memory']['monthly_risk']['quota']).data['shares'],{})
        fills=self.fill_plan(p,partial=True)
        before=self.a.state();self.a.close();self.a=ManualAccount(self.path)
        self.assertEqual(before,self.a.state())
        for f in fills:self.assertEqual(self.a.record_fill(f)['status'],'duplicate')
        quota=RestoreQuota(self.a.state()['memory']['monthly_risk']['quota'])
        self.assertTrue(all(v==50 for v in quota.data['shares'].values()))
        p=self.next_plan('2024-02-02')
        self.assertFalse(any(o['reason']=='targeted_restore' for o in p['orders']))

    def test_recovery_five_slots_and_actual_credit_consumption(self):
        p=self.next_plan('2024-02-01',True);self.fill_plan(p)
        credit=deepcopy(self.a.state()['memory']['monthly_risk']['quota']['shares'])
        self.assertEqual(len(self.a.state()['book_positions']),5)
        p=self.next_plan('2024-03-01')
        restore=[o for o in p['orders'] if o['reason']=='targeted_restore']
        self.assertTrue(restore);self.assertFalse(p['risk_active'])
        self.assertLessEqual(sum(o['reserved_cash'] for o in p['orders']),p['cash'])
        self.assertEqual(self.a.state()['memory']['monthly_risk']['quota']['shares'],credit)
        self.fill_plan(p,partial=True)
        for o in restore:self.assertEqual(self.a.state()['memory']['monthly_risk']['quota']['shares'][o['ts_code']],credit[o['ts_code']]-50)
        nextp=self.next_plan('2024-03-04')
        self.assertFalse(any(o['reason']=='targeted_restore' for o in nextp['orders']))

    def test_formal_exit_overrides_unfilled_risk_sale(self):
        p=self.next_plan('2024-02-01',True)
        for o in p['orders']:self.a.skip_order(o['order_id'],'synthetic not submitted')
        code=p['orders'][0]['ts_code']
        p=self.next_plan('2024-03-01',True,lambda b:b['holds'].update({code:False}))
        order=next(o for o in p['orders'] if o['ts_code']==code)
        self.assertEqual(order['reason'],'hold_false')
        self.assertEqual(order['shares'],self.a.state()['book_positions'][code])

    def test_new_risk_episode_clears_previous_credit(self):
        self.fill_plan(self.next_plan('2024-02-01',True))
        p=self.next_plan('2024-03-01')
        for o in p['orders']:self.a.skip_order(o['order_id'],'synthetic missed')
        epoch=self.a.state()['memory']['monthly_risk']['quota']['epoch']
        self.next_plan('2024-04-01',True)
        q=self.a.state()['memory']['monthly_risk']['quota']
        self.assertEqual(q['epoch'],epoch+1);self.assertEqual(q['shares'],{})

    def test_failed_fill_rolls_back_all_state(self):
        self.fill_plan(self.next_plan('2024-02-01',True))
        p=self.next_plan('2024-03-01');self.a.preflight(p['plan_id'],preflight(p))
        o=p['orders'][0];before=self.a.status()
        with self.assertRaises(ShadowError):self.a.record_fill(dict(execution_id='bad',order_id=o['order_id'],date=o['execution_date'],shares=o['shares']+1,price=10.,fee=5.,evidence='bad fixture'))
        self.assertEqual(before,self.a.status())

    def test_bootstrap_and_corrupt_or_stale_probability(self):
        s,b,cal,cfg=fixture(high=True)
        p=make_plan(s,b,cal,cfg)
        self.assertTrue(p['orders']);self.assertEqual(p['exposure_cap'],.5)
        self.assertLessEqual(sum(o['reserved_cash'] for o in p['orders']),40000.)
        for mutation in ['nan','prob','date','config']:
            bad=deepcopy(b)
            if mutation=='nan':bad['market_risk_history'][0]['features']['vol20']=float('nan')
            if mutation=='prob':bad['market_risk_history'][0]['probability']=.1
            if mutation=='date':bad['market_risk_history'][0]['date']='2024-01-01'
            if mutation=='config':bad['strategy_config']='wrong'
            with self.subTest(mutation=mutation),self.assertRaises(ShadowError):make_plan(s,bad,cal,cfg)

    def test_missed_review_blocks_and_legacy_switch_blocks(self):
        with self.assertRaises(ShadowError):self.next_plan('2024-03-01')
        state=self.a.state();s,b,cal,cfg=fixture('2024-03-01')
        old=yaml.safe_load((ROOT/'configs/shadow_execution.yaml').read_text())
        with self.assertRaises(ShadowError):make_plan(s,b,cal,old,state['memory'])

    def test_nan_stock_recovery_and_skip_mail_do_not_consume_credit(self):
        self.fill_plan(self.next_plan('2024-02-01',True))
        credit=deepcopy(self.a.state()['memory']['monthly_risk']['quota'])
        with self.assertRaises(ValueError):
            self.next_plan('2024-03-01',mutate=lambda b:[v.update(ret20=float('nan')) for v in b['stock_recovery'].values()])
        self.assertEqual(self.a.state()['memory']['monthly_risk']['quota'],credit)

    def test_skipped_sale_has_no_credit_and_company_action_pauses_one_stock(self):
        p=self.next_plan('2024-02-01',True)
        for o in p['orders']:self.a.skip_order(o['order_id'],'synthetic not submitted')
        self.assertEqual(self.a.state()['memory']['monthly_risk']['quota']['shares'],{})
        code=p['orders'][0]['ts_code']
        p=self.next_plan('2024-03-01',True,lambda b:b.update(blocked_next_day=[code]))
        self.assertFalse(any(o['ts_code']==code for o in p['orders']))
        self.assertTrue(any(d.get('ts_code')==code for d in p['decisions']))

    def test_share_adjustment_invalidates_credit(self):
        self.fill_plan(self.next_plan('2024-02-01',True))
        p=self.next_plan('2024-03-01')
        for o in p['orders']:self.a.skip_order(o['order_id'],'synthetic missed')
        code=next(iter(self.a.state()['memory']['monthly_risk']['quota']['shares']))
        item=dict(kind='corporate_action',event_id='share-action',date='2024-03-04',cash_delta=0.,share_delta=100,ts_code=code,evidence='synthetic stock action')
        self.a.adjustment(item)
        self.assertEqual(RestoreQuota(self.a.state()['memory']['monthly_risk']['quota']).available(code),0)
        self.assertEqual(self.a.adjustment(item)['status'],'duplicate')

    def test_prior_probability_revision_and_future_index_receipt_rejected(self):
        def mutate(b):b['market_risk_history'][0]['features']['ret20']+=.1
        with self.assertRaises(ShadowError):self.next_plan('2024-02-01',True,mutate)
        s,b,cal,cfg=fixture('2024-02-01',True)
        b['provenance']['receipt']['market_index']['downloaded_at']='2024-02-02T21:00:00+08:00'
        with self.assertRaises(ShadowError):self.a.prepare(b,cal,cfg,'2024-02-01T21:01:00+08:00')

    def test_restore_cancellation_and_mail_leave_credit_unchanged(self):
        self.fill_plan(self.next_plan('2024-02-01',True));p=self.next_plan('2024-03-01')
        before=deepcopy(self.a.state()['memory']['monthly_risk']['quota'])
        mail=bulletin(p,{},'2024-03-01T21:05:00+08:00',yaml.safe_load((ROOT/'configs/manual_execution.yaml').read_text()))
        self.assertIn('股票预算上限',mail['body']);self.assertFalse(mail['mail_sent'])
        for o in p['orders']:self.a.skip_order(o['order_id'],'synthetic skipped restore')
        self.assertEqual(before,self.a.state()['memory']['monthly_risk']['quota'])

    def test_candidate_mail_queue_is_idempotent_and_does_not_change_account(self):
        from mail_outbox import MailOutbox
        p=self.next_plan('2024-02-01',True);before=self.a.status()
        settings=dict(host='smtp.example.test',port=465,sender='sender@example.test',
            username='fixture',password='synthetic-unused',tls='ssl',recipients=['recipient@example.test'])
        box=MailOutbox(Path(self.tmp.name)/'synthetic_outbox.sqlite')
        try:
            config=yaml.safe_load((ROOT/'configs/manual_execution.yaml').read_text())
            ids=box.enqueue(p,{},settings,'2024-02-01T21:01:00+08:00',config)
            self.assertEqual(ids,box.enqueue(p,{},settings,'2024-02-01T21:02:00+08:00',config))
            self.assertEqual(box.db.execute('SELECT COUNT(*) FROM mail').fetchone()[0],1)
            self.assertEqual(self.a.status(),before)
        finally:box.close()


class RiskInputTests(unittest.TestCase):
    def test_index_prefix_and_gap(self):
        _,_,cal,cfg=fixture()
        dates=pd.DatetimeIndex(cal['sessions']);x=pd.Series(3000*np.exp(np.arange(len(dates))*.001+np.sin(np.arange(len(dates)))*.002),index=dates)
        old=risk_history(x,'2024-03-01',cal,cfg)
        x.loc[x.index>'2024-03-01']*=100
        self.assertEqual(old,risk_history(x,'2024-03-01',cal,cfg))
        with self.assertRaises(ShadowError):risk_history(x.drop(pd.Timestamp('2024-02-06')),'2024-03-01',cal,cfg)
        with self.assertRaises(ShadowError):risk_history(x,'2024-05-07',cal,cfg)
        extra=x.copy();extra.loc[pd.Timestamp('2024-02-10')]=3000.
        with self.assertRaises(ShadowError):risk_history(extra,'2024-03-01',cal,cfg)


if __name__=='__main__':unittest.main()
