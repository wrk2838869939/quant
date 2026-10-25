"""发信状态机使用注入传输测试，不连接真实SMTP。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import smtplib
import yaml

from mail_outbox import MailOutbox,smtp_deliver,settings_from_legacy
from shadow_planner import ShadowError,make_plan,empty_memory
from test_shadow_execution import fixture

ROOT=Path(__file__).resolve().parents[1]


class MailOutboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=self.root/'outbox.sqlite';self.box=MailOutbox(self.path)
        s,b,cal,cfg=fixture();self.plan=make_plan(s,b,cal,cfg,empty_memory())
        self.cfg=yaml.safe_load((ROOT/'configs/manual_execution.yaml').read_text())
        self.settings=dict(host='smtp.example.test',port=465,sender='sender@example.test',username='sender',
            password='synthetic-test-only',recipients=['recipient@example.test'],tls='ssl')
        self.now='2024-01-02T21:01:00+08:00'
        self.key=self.box.enqueue(self.plan,{},self.settings,self.now,self.cfg)[0]

    def tearDown(self):
        self.box.close();self.tmp.cleanup()

    def test_enqueue_idempotent_across_new_timestamp(self):
        keys=self.box.enqueue(self.plan,{},self.settings,'2024-01-02T21:02:00+08:00',self.cfg)
        self.assertEqual(keys,[self.key]);self.assertEqual(len(self.box.status()),1)

    def test_conflicting_content_refused(self):
        plan=deepcopy(self.plan);plan['cash']+=1
        with self.assertRaises(ShadowError):self.box.enqueue(plan,{},self.settings,self.now,self.cfg)

    def test_success_not_repeated_after_restart(self):
        calls=[]
        def send(*args):calls.append(True);return dict(status='accepted')
        self.box.dispatch(self.key,self.settings,self.now,send)
        self.box.close();self.box=MailOutbox(self.path)
        self.assertFalse(self.box.dispatch(self.key,self.settings,self.now,send)['attempted'])
        self.assertEqual(len(calls),1)

    def test_expired_never_calls_transport(self):
        def forbidden(*args):self.fail('expired mail attempted')
        self.assertEqual(self.box.dispatch(self.key,self.settings,'2024-01-03T09:25:00+08:00',forbidden)['status'],'expired')
        with self.assertRaises(ShadowError):self.box.enqueue(self.plan,{},self.settings,'2024-01-03T10:00:00+08:00',self.cfg)

    def test_unknown_requires_resolution(self):
        self.box.dispatch(self.key,self.settings,self.now,lambda *args:dict(status='unknown'))
        self.assertFalse(self.box.dispatch(self.key,self.settings,self.now,lambda *args:self.fail('duplicate'))['attempted'])
        self.box.resolve(self.key,'retry','已核对未收到，明确允许再次尝试')
        self.assertEqual(self.box.dispatch(self.key,self.settings,self.now,lambda *args:dict(status='accepted'))['status'],'accepted')

    def test_active_sender_cannot_be_resolved_or_sent_twice(self):
        other=MailOutbox(self.path)
        try:
            def active(*args):
                with self.assertRaises(ShadowError):other.resolve(self.key,'retry','unsafe while active')
                with self.assertRaises(ShadowError):other.dispatch(self.key,self.settings,self.now)
                return dict(status='accepted')
            self.box.dispatch(self.key,self.settings,self.now,active)
        finally:other.close()

    def test_crash_sending_state_is_not_automatically_retried(self):
        self.box.db.execute("UPDATE mail SET status='sending' WHERE id=?",(self.key,))
        self.assertFalse(self.box.dispatch(self.key,self.settings,self.now,lambda *a:self.fail('duplicate'))['attempted'])
        self.box.resolve(self.key,'accepted','核对后确认已收到')
        self.assertEqual(self.box.status()[0]['status'],'accepted')

    def test_recipient_change_cannot_redirect_existing_mail(self):
        changed=deepcopy(self.settings);changed['recipients']=['other@example.test']
        with self.assertRaises(ShadowError):self.box.dispatch(self.key,changed,self.now)

    def test_legacy_literal_reader_never_executes_script(self):
        path=self.root/'settings.py'
        path.write_text("SMTP_SERVER, SMTP_PORT = 'smtp.example.test', 465\nSENDER='sender@example.test'\nRECEIVERS=['recipient@example.test']\nAUTH_CODE='synthetic'\nraise RuntimeError('must not execute')\n")
        settings=settings_from_legacy(path)
        self.assertEqual(settings['host'],'smtp.example.test')

    def test_transport_distinguishes_before_data_and_ambiguous_disconnect(self):
        with patch('mail_outbox.smtplib.SMTP_SSL',side_effect=TimeoutError):
            self.assertEqual(smtp_deliver(b'x',self.settings['sender'],self.settings['recipients'][0],self.settings)['status'],'failed')
        with patch('mail_outbox.smtplib.SMTP_SSL') as factory:
            client=factory.return_value;client.sendmail.side_effect=smtplib.SMTPServerDisconnected()
            result=smtp_deliver(b'x',self.settings['sender'],self.settings['recipients'][0],self.settings)
            self.assertEqual(result['status'],'unknown')

    def test_success_then_quit_failure_stays_accepted(self):
        with patch('mail_outbox.smtplib.SMTP_SSL') as factory:
            client=factory.return_value;client.sendmail.return_value={}
            client.quit.side_effect=OSError;client.close.side_effect=OSError
            self.assertEqual(smtp_deliver(b'x',self.settings['sender'],self.settings['recipients'][0],self.settings)['status'],'accepted')


if __name__=='__main__':unittest.main()
