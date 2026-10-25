"""用合成券商凭据验证导入边界；不是实盘成交证明。"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import StringIO
from pathlib import Path
import csv
import json
import tempfile
import unittest

import yaml

from broker_csv import BrokerAccount, KINDS, import_csv_batch, parse_files
from shadow_planner import ShadowError
from test_shadow_execution import fixture, preflight

ROOT=Path(__file__).resolve().parents[1]


def exports(plan, encoding='utf-8-sig'):
    profile=yaml.safe_load((ROOT/'configs/broker_csv_template.yaml').read_text())
    profile['encoding']=encoding
    records={kind:[] for kind in KINDS}
    cash=plan['cash']; equity=plan['cash']
    for index,order in enumerate(plan['orders']):
        common=dict(account_id='test',date=order['execution_date'],broker_order_id='000'+str(index),
            code=order['ts_code'][:6],market='沪A' if order['ts_code'].endswith('SH') else '深A',side='买入')
        records['bindings'].append(dict(common,order_id=order['order_id'],shares=order['shares'],price=order['limit_price']))
        # 50股部分成交后撤单，检验部分成交不被整手申报规则错误拒绝。
        records['fills'].append(dict(common,execution_id='000'+str(index),shares=50,price=10,gross=500,fee=5))
        records['orders'].append(dict(common,status='部撤',filled_shares=50))
        records['positions'].append(dict(common,shares=50,sellable_next_day=50,close=10))
        cash-=505; equity-=5
    records['account']=[dict(account_id='test',date=plan['execution_date'],cash=cash,equity=equity,
        receivables=0,frozen_cash=0,open_order_count=0)]
    return profile,records


def encode(profile,records):
    files={}
    for kind in KINDS:
        stream=StringIO(newline='');mapping=profile['columns'][kind]
        writer=csv.DictWriter(stream,fieldnames=list(mapping.values()));writer.writeheader()
        for row in records[kind]:writer.writerow({column:row[key] for key,column in mapping.items()})
        files[kind]=stream.getvalue().encode(profile['encoding'])
    return yaml.safe_dump(profile,allow_unicode=True).encode(),files


class BrokerCSVTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.db=self.root/'account.sqlite';self.a=BrokerAccount(self.db)
        s,b,cal,cfg=fixture();self.a.import_snapshot(s)
        self.plan=self.a.plan(b,cal,cfg);self.a.preflight(self.plan['plan_id'],preflight(self.plan))
        self.profile,self.rows=exports(self.plan)

    def tearDown(self):
        self.a.close();self.temp.cleanup()

    def parsed(self):return parse_files(*encode(self.profile,self.rows))

    def run_batch(self,identity='test-batch'):
        return self.a.import_batch(identity,self.parsed(),{'test':True})

    def unchanged_after_failure(self):
        before=self.a.status()
        with self.assertRaises(ShadowError):self.run_batch()
        self.assertEqual(self.a.status(),before)
        self.assertEqual(self.a.db.execute('SELECT COUNT(*) FROM broker_bindings').fetchone()[0],0)
        self.assertEqual(self.a.db.execute('SELECT COUNT(*) FROM broker_batches').fetchone()[0],0)

    def test_partial_cancel_reconcile_restart_and_repeat(self):
        result=self.run_batch();self.assertEqual(result['status'],'reconciled')
        self.assertTrue(all(o['status']=='cancelled' and o['filled_shares']==50 for o in self.a.status()['orders']))
        before=self.a.status();self.a.close();self.a=BrokerAccount(self.db)
        self.assertEqual(self.run_batch()['status'],'duplicate');self.assertEqual(self.a.status(),before)

    def test_reexport_same_rows_new_batch_no_double_booking(self):
        self.run_batch();before=self.a.state()
        self.run_batch('second-export');self.assertEqual(self.a.state(),before)

    def test_wrong_cash_rolls_back_fills_terminals_and_bindings(self):
        self.rows['account'][0]['cash']+=1;self.rows['account'][0]['equity']+=1
        self.unchanged_after_failure()

    def test_missing_fill_rolls_back_earlier_valid_fill(self):
        self.rows['fills'].pop();self.unchanged_after_failure()

    def test_wrong_order_binding_rejected(self):
        self.rows['bindings'][-1]['price']+=.01;self.unchanged_after_failure()

    def test_side_mismatch_rolls_back(self):
        self.rows['fills'][-1]['side']='卖出';self.unchanged_after_failure()

    def test_missing_terminal_rejected(self):
        self.rows['orders'].pop();self.unchanged_after_failure()

    def test_receivable_change_not_silently_accepted(self):
        self.rows['account'][0]['receivables']=10;self.rows['account'][0]['equity']+=10
        self.unchanged_after_failure()

    def test_conflicting_execution_id_after_success(self):
        self.run_batch();before=self.a.status()
        self.rows['fills'][0]['fee']=6
        with self.assertRaises(ShadowError):self.run_batch('conflict')
        self.assertEqual(before,self.a.status())

    def test_missing_fee_and_unknown_side_fail(self):
        for field,value in [('fee',''),('side','融资买入'),('price','NaN'),('shares','0.1')]:
            with self.subTest(field=field):
                rows=deepcopy(self.rows);rows['fills'][0][field]=value
                with self.assertRaises(ShadowError):parse_files(*encode(self.profile,rows))

    def test_duplicate_execution_within_file_rejected(self):
        self.rows['fills'].append(deepcopy(self.rows['fills'][0]))
        with self.assertRaises(ShadowError):self.parsed()

    def test_mixed_account_or_date_rejected(self):
        for key,value in [('account_id','other'),('date','2024-01-04')]:
            rows=deepcopy(self.rows);rows['fills'][0][key]=value
            with self.assertRaises(ShadowError):parse_files(*encode(self.profile,rows))

    def test_total_fee_and_gross_no_implicit_conversion(self):
        self.rows['fills'][0]['gross']=501
        with self.assertRaises(ShadowError):self.parsed()

    def test_utf8_bom_gb18030_and_leading_zero(self):
        for encoding in ['utf-8-sig','gb18030']:
            self.profile['encoding']=encoding
            rows=deepcopy(self.rows)
            for kind in ['fills','orders','bindings','positions']:
                rows[kind][0].update(code='000001',market='深A')
            parsed=parse_files(*encode(self.profile,rows))
            self.assertEqual(parsed['fills'][0]['ts_code'],'000001.SZ')
            self.assertEqual(parsed['fills'][0]['execution_id'],'0000')

    def test_same_batch_conflicting_normalized_data(self):
        self.run_batch();self.rows['fills'][0]['fee']=6
        with self.assertRaises(ShadowError):self.run_batch()

    def test_no_preflight_cannot_import_fills(self):
        for row in self.a.status()['orders']:
            row['status']='planned';self.a._order(row)
        self.unchanged_after_failure()

    def test_concurrent_batch_commits_once(self):
        data=self.parsed()
        def work(_):
            a=BrokerAccount(self.db)
            try:return a.import_batch('concurrent',data,{'test':True})['status']
            finally:a.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses=list(pool.map(work,range(2)))
        self.assertEqual(sorted(statuses),['duplicate','reconciled'])

    def test_archived_bytes_and_failure_survive_rollback(self):
        self.rows['account'][0]['cash']+=1;self.rows['account'][0]['equity']+=1
        profile,files=encode(self.profile,self.rows)
        p=self.root/'profile.yaml';p.write_bytes(profile)
        paths={}
        for kind,raw in files.items():
            path=self.root/(kind+'.csv');path.write_bytes(raw);paths[kind]=path
        before=self.a.status()
        with self.assertRaises(ShadowError):import_csv_batch(self.db,p,paths,self.root/'archive')
        self.assertEqual(before,self.a.status())
        archive=next((self.root/'archive').iterdir())
        self.assertTrue((archive/'failure.json').exists())
        for kind,raw in files.items():self.assertEqual((archive/(kind+'.csv')).read_bytes(),raw)


if __name__=='__main__':unittest.main()
