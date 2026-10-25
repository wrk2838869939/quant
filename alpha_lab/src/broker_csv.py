"""显式CSV列映射与原子导入；只读取凭据，不连接券商或发送订单。"""
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import uuid

import yaml

from shadow_account import ShadowAccount
from shadow_planner import ShadowError, canonical, fingerprint, validate_snapshot

KINDS = ('bindings', 'fills', 'orders', 'account', 'positions')
REQUIRED = {
    'bindings': {'account_id','date','broker_order_id','order_id','code','market','side','shares','price'},
    'fills': {'account_id','date','broker_order_id','execution_id','code','market','side','shares','price','gross','fee'},
    'orders': {'account_id','date','broker_order_id','code','market','side','status','filled_shares'},
    'account': {'account_id','date','cash','receivables','equity','frozen_cash','open_order_count'},
    'positions': {'account_id','date','code','market','shares','sellable_next_day','close'},
}


def decimal(value):
    """禁止空值、千位分隔、科学计数、负数和隐式费用补零。"""
    if not re.fullmatch(r'\d+(?:\.\d+)?', value):
        raise ShadowError('数值必须为显式非负十进制，费用不可留空')
    try:
        result = Decimal(value)
        if not result.is_finite() or result > Decimal('1e15'):
            raise ShadowError('数值超出支持范围')
        return result
    except InvalidOperation as exc:
        raise ShadowError('无效数值') from exc


def parse_files(profile_bytes, files):
    """五份同账户、同日、已结算CSV；返回与文件路径无关的规范记录。"""
    profile = yaml.safe_load(profile_bytes)
    if not profile.get('profile_id') or set(files) != set(KINDS):
        raise ShadowError('缺格式标识或五份必需CSV')
    parsed = {}
    for kind in KINDS:
        mapping = profile['columns'][kind]
        if set(mapping) != REQUIRED[kind] or len(set(mapping.values())) != len(mapping):
            raise ShadowError(kind+'列映射必须明确完整且不重复')
        reader = csv.DictReader(StringIO(files[kind].decode(profile['encoding']), newline=''))
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or not set(mapping.values()).issubset(header):
            raise ShadowError(kind+'表头缺失或重复')
        rows = []
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ShadowError(kind+'CSV列数不一致')
            x = {key: row[col].strip() for key,col in mapping.items()}
            if any(not v for v in x.values()):
                raise ShadowError(kind+'必需字段为空')
            stamp = datetime.strptime(x['date'], profile['date_format'])
            if stamp.strftime(profile['date_format']) != x['date']:
                raise ShadowError('日期与声明格式不一致')
            x['date'] = stamp.strftime('%Y-%m-%d')
            if 'code' in x:
                market = profile['markets'].get(x.pop('market'))
                code = x.pop('code')
                if market not in ['SH','SZ'] or not re.fullmatch(r'\d{6}', code):
                    raise ShadowError('证券代码或市场不明确；必须保留前导零')
                x['ts_code'] = code+'.'+market
            if 'side' in x:
                x['side'] = profile['sides'].get(x['side'])
                if x['side'] not in ['buy','sell']: raise ShadowError('未知交易方向')
            if 'status' in x:
                x['status'] = profile['statuses'].get(x['status'])
                if x['status'] not in ['filled','cancelled','rejected','expired']:
                    raise ShadowError('委托尚未终结或状态未声明')
            for key in ['shares','filled_shares','sellable_next_day','open_order_count']:
                if key in x:
                    value = decimal(x[key])
                    if value != int(value): raise ShadowError('股数及委托数必须为整数')
                    x[key] = int(value)
            for key in ['price','gross','fee','cash','receivables','equity','frozen_cash','close']:
                if key in x: x[key] = float(decimal(x[key]))
            if kind == 'fills':
                if x['shares'] <= 0 or x['price'] <= 0:
                    raise ShadowError('成交股数和价格须为正')
                if abs(Decimal(str(x['price']))*x['shares']-Decimal(str(x['gross']))) > Decimal('.01'):
                    raise ShadowError('成交金额与数量乘价格不符')
            rows.append(x)
        parsed[kind] = rows
    if len(parsed['account']) != 1: raise ShadowError('账户汇总必须恰好一行')
    account = parsed['account'][0]
    identity = (account['account_id'],account['date'])
    for rows in parsed.values():
        if any((r['account_id'],r['date']) != identity for r in rows):
            raise ShadowError('批次混合账户或日期')
    for kind,key in [('bindings','broker_order_id'),('fills','execution_id'),('orders','broker_order_id'),('positions','ts_code')]:
        keys = [r[key] for r in parsed[kind]]
        if len(keys) != len(set(keys)): raise ShadowError(kind+'批内标识重复')
    snapshot = {k:account[k] for k in ['account_id','cash','receivables','equity','frozen_cash']}
    snapshot.update(asof=account['date'], source='broker-csv:'+profile['profile_id'], open_orders=[], positions={})
    if account['open_order_count']: raise ShadowError('收盘仍有未结委托')
    for row in parsed['positions']:
        snapshot['positions'][row['ts_code']] = {k:row[k] for k in ['shares','sellable_next_day','close']}
    validate_snapshot(snapshot)
    parsed['snapshot'] = snapshot
    return parsed


def archive_inputs(profile, files, directory):
    """读取一次后封存同一字节，解析不再回读易变的原文件。"""
    raw = {'profile.yaml':Path(profile).read_bytes()}
    raw.update({kind+'.csv':Path(files[kind]).read_bytes() for kind in KINDS})
    hashes = {name:hashlib.sha256(data).hexdigest() for name,data in raw.items()}
    batch_id = fingerprint(hashes)
    # 每次尝试独立目录，包括失败/重复尝试；避免并发写同一路径。
    out = Path(directory)/(batch_id[:16]+'_'+uuid.uuid4().hex)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name,data in raw.items():
        path = out/name
        with path.open('xb') as sink:
            sink.write(data); sink.flush(); os.fsync(sink.fileno())
        path.chmod(0o600)
    (out/'manifest.json').write_text(canonical(dict(batch_id=batch_id,sha256=hashes)))
    return batch_id,raw,out


class BrokerAccount(ShadowAccount):
    """在冻结账本外增加SAVEPOINT、显式券商订单绑定和整批提交。"""
    def __init__(self, path):
        super().__init__(path)
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS broker_bindings (
                account TEXT, date TEXT, broker_id TEXT, order_id TEXT UNIQUE, payload TEXT,
                PRIMARY KEY(account,date,broker_id));
            CREATE TABLE IF NOT EXISTS broker_batches (id TEXT PRIMARY KEY, payload TEXT);
        ''')

    @contextmanager
    def transaction(self):
        if not self.db.in_transaction:
            with super().transaction(): yield
            return
        name = 'batch_'+uuid.uuid4().hex
        self.db.execute('SAVEPOINT '+name)
        try:
            yield
            self.db.execute('RELEASE '+name)
        except BaseException:
            self.db.execute('ROLLBACK TO '+name)
            self.db.execute('RELEASE '+name)
            raise

    def _match(self, row):
        match = self.db.execute('SELECT order_id FROM broker_bindings WHERE account=? AND date=? AND broker_id=?',
            (row['account_id'],row['date'],row['broker_order_id'])).fetchone()
        if match is None: raise ShadowError('券商委托没有显式计划绑定')
        order = self.order(match[0])
        if row['ts_code'] != order['ts_code'] or row['side'] != order['side']:
            raise ShadowError('券商证券或方向与绑定订单不符')
        return order

    def import_batch(self, batch_id, data, provenance):
        with self.transaction():
            digest = fingerprint(data)
            old = self.db.execute('SELECT payload FROM broker_batches WHERE id=?',(batch_id,)).fetchone()
            if old:
                if json.loads(old[0])['normalized_sha256'] != digest: raise ShadowError('相同批次内容冲突')
                return dict(status='duplicate',batch_id=batch_id,live_submission=False)
            state = self.state()
            snapshot = data['snapshot']
            if snapshot['account_id'] != state['snapshot']['account_id']:
                raise ShadowError('导入账户与当前账本不同')
            if snapshot['receivables'] != state['snapshot']['receivables']:
                raise ShadowError('应收款变化尚无独立流水支持，需先扩展事件对账，不能静默接受')
            for row in data['bindings']:
                order = self.order(row['order_id'])
                if (order['execution_date'] != row['date'] or order['ts_code'] != row['ts_code']
                        or order['side'] != row['side'] or order['shares'] != row['shares']
                        or abs(order['limit_price']-row['price']) > 1e-8):
                    raise ShadowError('绑定与冻结计划日期、证券、方向、股数或限价不符')
                identity = (row['account_id'],row['date'],row['broker_order_id'])
                old = self.db.execute('SELECT payload FROM broker_bindings WHERE account=? AND date=? AND broker_id=?',identity).fetchone()
                if old:
                    if old[0] != canonical(row): raise ShadowError('券商委托绑定冲突')
                else:
                    self.db.execute('INSERT INTO broker_bindings VALUES(?,?,?,?,?)',identity+(row['order_id'],canonical(row)))
            # 不推测盘中成交先后：本策略不预支同日卖款，先验证全部买入也能结算。
            for row in sorted(data['fills'], key=lambda r:(r['side'] != 'buy',r['execution_id'])):
                order = self._match(row)
                fill = dict(execution_id='broker:'+fingerprint([row['account_id'],row['date'],row['execution_id']]),
                    order_id=order['order_id'],date=row['date'],shares=row['shares'],price=row['price'],fee=row['fee'],
                    evidence='broker-row:'+fingerprint(row))
                self.record_fill(fill)
            for row in data['orders']:
                order = self._match(row)
                if order['filled_shares'] != row['filled_shares']:
                    raise ShadowError('委托累计成交与逐笔成交不符')
                if row['status'] == 'filled':
                    if order['status'] != 'filled': raise ShadowError('委托报告已成但成交未齐')
                else:
                    self.terminal(order['order_id'],row['status'],'broker-row:'+fingerprint(row))
            reported = {self._match(row)['order_id'] for row in data['orders']}
            expected = {json.loads(r[0])['order_id'] for r in self.db.execute('SELECT payload FROM orders')
                        if json.loads(r[0])['execution_date'] == snapshot['asof']}
            if reported != expected:
                raise ShadowError('当天计划委托必须逐一提供最终状态，不从成交推测终态')
            self.import_snapshot(snapshot)
            result = dict(status='reconciled',batch_id=batch_id,fills=len(data['fills']),orders=len(data['orders']),
                normalized_sha256=digest,provenance=provenance,live_submission=False)
            self.db.execute('INSERT INTO broker_batches VALUES(?,?)',(batch_id,canonical(result)))
            self._audit('broker_batch_reconciled',result)
            return result


def import_csv_batch(db, profile, files, archive):
    batch_id,raw,out = archive_inputs(profile,files,archive)
    account = None; result = None
    try:
        data = parse_files(raw['profile.yaml'],{k:raw[k+'.csv'] for k in KINDS})
        account = BrokerAccount(db)
        result = account.import_batch(batch_id,data,dict(archive=str(out.resolve()),
            manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()))
        (out/'result.json').write_text(canonical(result))
        return dict(status=result['status'],batch_id=batch_id,archive=str(out),live_submission=False)
    except Exception as exc:
        (out/'failure.json').write_text(canonical(dict(status='ledger_committed_receipt_write_failed' if result is not None else 'failed',error_type=type(exc).__name__,
            message=str(exc) if isinstance(exc,ShadowError) else '格式、编码或数据库异常；请本地核对原始文件')))
        raise
    finally:
        if account is not None: account.close()
