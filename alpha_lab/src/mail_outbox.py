"""持久化邮件发件箱；接受确认不是到达收件箱证明，不确定结果不自动重发。"""
import ast
import fcntl
from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
import json
import os
from pathlib import Path
import re
import smtplib
import sqlite3
import ssl

import pandas as pd

from manual_execution import bulletin
from shadow_planner import ShadowError,canonical,fingerprint


def address(value):
    if not isinstance(value,str) or not re.fullmatch(r"[^\s@,;<>]+@[^\s@,;<>]+\.[^\s@,;<>]+",value):
        raise ShadowError('需要单个明确邮箱地址，不接受显示名或多地址拼接')
    return value


def settings_from_env():
    settings=dict(host=os.environ['ALPHA_SMTP_HOST'],port=int(os.environ.get('ALPHA_SMTP_PORT','465')),
        sender=os.environ['ALPHA_SMTP_SENDER'],username=os.environ['ALPHA_SMTP_USER'],
        password=os.environ['ALPHA_SMTP_PASSWORD'],tls=os.environ.get('ALPHA_SMTP_TLS','ssl'),
        recipients=[os.environ['ALPHA_MAIL_TO']])
    validate_settings(settings)
    return settings


def settings_from_legacy(path):
    """只解析既有配置的字面量，不导入会启动调度或产生其他副作用的脚本。"""
    values={}
    wanted={'SMTP_SERVER','SMTP_PORT','SENDER','RECEIVERS','AUTH_CODE'}
    for node in ast.parse(Path(path).read_text()).body:
        if not isinstance(node,ast.Assign):continue
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id in wanted:
                values[target.id]=ast.literal_eval(node.value)
            if isinstance(target,ast.Tuple) and all(isinstance(n,ast.Name) and n.id in wanted for n in target.elts):
                resolved=ast.literal_eval(node.value)
                values.update({n.id:v for n,v in zip(target.elts,resolved)})
    settings=dict(host=values['SMTP_SERVER'],port=values['SMTP_PORT'],sender=values['SENDER'],
        username=values['SENDER'],password=values['AUTH_CODE'],recipients=values['RECEIVERS'],tls='ssl')
    validate_settings(settings)
    return settings


def validate_settings(settings):
    address(settings['sender'])
    if not isinstance(settings['recipients'],list) or not settings['recipients']:raise ShadowError('缺收件人')
    if len(set(settings['recipients']))!=len(settings['recipients']):raise ShadowError('收件人重复')
    for recipient in settings['recipients']:address(recipient)
    if settings['tls'] not in ['ssl','starttls'] or not 1<=int(settings['port'])<=65535:
        raise ShadowError('SMTP需SSL或STARTTLS和有效端口')
    if not all(isinstance(settings[k],str) and settings[k] for k in ['host','username','password']):
        raise ShadowError('缺SMTP配置')


def smtp_deliver(payload, sender, recipient, settings):
    """单收件人传输；不打印服务器异常正文或认证信息。"""
    client=None;phase='connect'
    try:
        context=ssl.create_default_context()
        if settings['tls']=='ssl':
            client=smtplib.SMTP_SSL(settings['host'],settings['port'],timeout=20,context=context)
        else:
            client=smtplib.SMTP(settings['host'],settings['port'],timeout=20)
            client.ehlo();client.starttls(context=context);client.ehlo()
        client.login(settings['username'],settings['password'])
        phase='data'
        refused=client.sendmail(sender,[recipient],payload)
        return dict(status='failed' if refused else 'accepted',error_type='RecipientRefused' if refused else None)
    except (smtplib.SMTPRecipientsRefused,smtplib.SMTPSenderRefused,smtplib.SMTPDataError) as exc:
        return dict(status='failed',error_type=type(exc).__name__)
    except Exception as exc:
        return dict(status='unknown' if phase=='data' else 'failed',error_type=type(exc).__name__)
    finally:
        if client is not None:
            try:client.quit()
            except Exception:
                try:client.close()
                except Exception:pass


class MailOutbox:
    def __init__(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.lock_path=Path(str(path)+'.send.lock')
        self.db=sqlite3.connect(path,timeout=30,isolation_level=None)
        path.chmod(0o600)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS mail (
                id TEXT PRIMARY KEY, semantic_hash TEXT, sender TEXT, recipient TEXT,
                deadline TEXT, payload BLOB, status TEXT, attempts INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, id TEXT, kind TEXT, detail TEXT);
        ''')

    def close(self):self.db.close()

    @contextmanager
    def delivery_lock(self):
        with self.lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ShadowError('发送进程仍在工作，不能同时发送或改判结果')
            try:yield
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK');raise

    def event(self,key,kind,detail):
        self.db.execute('INSERT INTO events(id,kind,detail) VALUES(?,?,?)',(key,kind,canonical(detail)))

    def enqueue(self,plan,metadata,settings,now,config):
        from execution_clock import operation_config
        config=operation_config(plan, config)
        validate_settings(settings)
        draft=bulletin(plan,metadata,now,config)
        if draft['historical']:raise ShadowError('过期/历史清单不进入实际发件队列')
        deadline=pd.Timestamp(plan['execution_date']+' '+config['execution_end'],tz='Asia/Shanghai').isoformat()
        semantic=fingerprint(dict(subject=draft['subject'],body=draft['body']))
        ids=[]
        with self.transaction():
            for recipient in settings['recipients']:
                key=fingerprint(dict(plan=plan['plan_id'],sender=settings['sender'],recipient=recipient,purpose='manual_briefing'))
                old=self.db.execute('SELECT semantic_hash FROM mail WHERE id=?',(key,)).fetchone()
                if old:
                    if old[0]!=semantic:raise ShadowError('同计划邮件内容改变，不能静默覆盖已入队清单')
                else:
                    message=BytesParser(policy=policy.SMTP).parsebytes(draft['eml'])
                    message['From']=settings['sender'];message['To']=recipient
                    message.replace_header('Message-ID','<'+key+'@alpha-lab.local>')
                    self.db.execute('INSERT INTO mail VALUES(?,?,?,?,?,?,?,0)',
                        (key,semantic,settings['sender'],recipient,deadline,message.as_bytes(),'pending'))
                    self.event(key,'enqueued',dict(plan_id=plan['plan_id'],at=str(now)))
                ids.append(key)
        return ids

    def dispatch(self,key,settings,now,deliver=smtp_deliver):
        with self.delivery_lock():
            return self._dispatch(key,settings,now,deliver)

    def _dispatch(self,key,settings,now,deliver):
        validate_settings(settings)
        stamp=pd.Timestamp(now)
        if stamp.tzinfo is None:raise ShadowError('发送时刻缺时区')
        with self.transaction():
            row=self.db.execute('SELECT sender,recipient,deadline,payload,status FROM mail WHERE id=?',(key,)).fetchone()
            if row is None:raise ShadowError('发件箱中没有此邮件')
            sender,recipient,deadline,payload,status=row
            if sender!=settings['sender'] or recipient not in settings['recipients']:
                raise ShadowError('发送配置与冻结收件人不同')
            if status not in ['pending','failed']:return dict(status=status,attempted=False)
            if stamp>=pd.Timestamp(deadline):
                self.db.execute("UPDATE mail SET status='expired' WHERE id=?",(key,))
                self.event(key,'expired',dict(at=str(now)))
                return dict(status='expired',attempted=False)
            self.db.execute("UPDATE mail SET status='sending',attempts=attempts+1 WHERE id=?",(key,))
            self.event(key,'attempt_started',dict(at=str(now)))
        try:result=deliver(payload,sender,recipient,settings)
        except Exception as exc:result=dict(status='unknown',error_type=type(exc).__name__)
        if result.get('status') not in ['accepted','failed','unknown']:
            result=dict(status='unknown',error_type='InvalidTransportResult')
        with self.transaction():
            self.db.execute('UPDATE mail SET status=? WHERE id=?',(result['status'],key))
            self.event(key,'attempt_finished',dict(status=result['status'],error_type=result.get('error_type')))
        return dict(status=result['status'],attempted=True)

    def resolve(self,key,outcome,evidence):
        """进程中断或收件结果未知时，由核对结果决定是否允许再次尝试。"""
        with self.delivery_lock():
            return self._resolve(key,outcome,evidence)

    def _resolve(self,key,outcome,evidence):
        if outcome not in ['accepted','retry'] or not evidence:raise ShadowError('需要核对结果和依据')
        with self.transaction():
            row=self.db.execute('SELECT status FROM mail WHERE id=?',(key,)).fetchone()
            if row is None or row[0] not in ['sending','unknown']:raise ShadowError('仅处理发送结果不确定的邮件')
            state='accepted' if outcome=='accepted' else 'failed'
            self.db.execute('UPDATE mail SET status=? WHERE id=?',(state,key))
            self.event(key,'manually_resolved',dict(outcome=outcome,evidence=evidence))

    def status(self):
        return [dict(id=r[0],status=r[1],attempts=r[2],deadline=r[3]) for r in self.db.execute(
            'SELECT id,status,attempts,deadline FROM mail ORDER BY rowid')]
