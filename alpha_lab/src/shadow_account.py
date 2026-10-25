"""SQLite事务账本：导入真实回报、幂等去重与收盘核对；没有券商下单能力。"""
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

import pandas as pd

from shadow_planner import (ShadowError, canonical, fingerprint, integer, number,
                            validate_snapshot, make_plan, empty_memory)


TERMINAL = {'filled', 'cancelled', 'rejected', 'expired'}


class ShadowAccount:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS plans (date TEXT PRIMARY KEY, id TEXT UNIQUE, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS receipts (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS snapshots (date TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS snapshot_history (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, payload TEXT NOT NULL);
        ''')

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def state(self):
        row = self.db.execute('SELECT payload FROM state WHERE id=1').fetchone()
        if row is None:
            raise ShadowError('先导入账户初始快照')
        return json.loads(row[0])

    def _save(self, state):
        self.db.execute('INSERT OR REPLACE INTO state VALUES(1,?)', (canonical(state),))

    def _audit(self, kind, data):
        self.db.execute('INSERT INTO audit(kind,payload) VALUES(?,?)', (kind, canonical(data)))

    def order(self, order_id):
        row = self.db.execute('SELECT payload FROM orders WHERE id=?', (order_id,)).fetchone()
        if row is None:
            raise ShadowError('成交未匹配计划订单，须先核对券商流水')
        return json.loads(row[0])

    def _order(self, order):
        self.db.execute('UPDATE orders SET payload=? WHERE id=?', (canonical(order), order['order_id']))

    def import_snapshot(self, snapshot):
        validate_snapshot(snapshot)
        from execution_clock import snapshot_stamp
        snapshot_key = snapshot_stamp(snapshot).isoformat()
        with self.transaction():
            old = self.db.execute('SELECT payload FROM snapshots WHERE date=?', (snapshot['asof'],)).fetchone()
            version = self.db.execute('SELECT payload FROM snapshot_history WHERE id=?', (snapshot_key,)).fetchone()
            if version:
                if version[0] != canonical(snapshot):
                    raise ShadowError('同一时刻账户快照内容冲突')
                return dict(status='duplicate', snapshot_hash=fingerprint(snapshot))
            transition = old and json.loads(old[0]).get('phase') == 'intraday' and snapshot.get('phase') == 'eod'
            if old and not transition:
                if old[0] != canonical(snapshot):
                    raise ShadowError('同日快照内容改变；保留旧快照并停止，不能静默覆盖')
                return dict(status='duplicate', snapshot_hash=fingerprint(snapshot))
            exists = self.db.execute('SELECT COUNT(*) FROM state').fetchone()[0]
            if exists:
                state = self.state()
                if transition and state['snapshot'].get('phase') != 'intraday':
                    raise ShadowError('当天收盘已核对，不能重复提交另一版本收盘快照')
                if snapshot['account_id'] != state['snapshot']['account_id'] or snapshot_stamp(snapshot) <= snapshot_stamp(state['snapshot']):
                    raise ShadowError('账户不同或快照时间倒退')
                if snapshot['asof'] < state['latest_event_date']:
                    raise ShadowError('快照早于已入账流水，禁止把未来资金带回过去')
                if snapshot.get('phase') == 'eod' and snapshot_stamp(snapshot).strftime('%H:%M:%S') < '15:00:00':
                    raise ShadowError('收盘核对不得早于15:00')
                if state.get('latest_event_at') and snapshot_stamp(snapshot) < pd.Timestamp(state['latest_event_at']):
                    raise ShadowError('快照早于已入账流水时刻')
                for (payload,) in self.db.execute('SELECT payload FROM orders'):
                    order = json.loads(payload)
                    if order['execution_date'] <= snapshot['asof'] and order['status'] not in TERMINAL:
                        raise ShadowError('仍有未终结委托，需先导入终态')
                quantities = {c: p['shares'] for c, p in snapshot['positions'].items()}
                expected = {c: q for c, q in state['book_positions'].items() if q}
                if quantities != expected or abs(snapshot['cash'] - state['book_cash']) > .02:
                    raise ShadowError('券商快照与逐笔账本不符；缺失成交或资金/公司行动流水')
            else:
                state = dict(book_cash=snapshot['cash'], book_positions={c:p['shares'] for c,p in snapshot['positions'].items()},
                             memory=empty_memory(), latest_event_date=snapshot['asof'])
            state['snapshot'] = deepcopy(snapshot)
            state['book_cash'] = snapshot['cash']
            self._save(state)
            self.db.execute('INSERT OR IGNORE INTO snapshots VALUES(?,?)', (snapshot['asof'], canonical(snapshot)))
            self.db.execute('INSERT INTO snapshot_history VALUES(?,?)', (snapshot_key, canonical(snapshot)))
            self._audit('snapshot_reconciled', snapshot)
        return dict(status='reconciled', snapshot_hash=fingerprint(snapshot))

    def plan(self, bundle, calendar, config):
        with self.transaction():
            state = self.state()
            old = self.db.execute('SELECT payload FROM plans WHERE date=?', (bundle['asof'],)).fetchone()
            if old:
                plan = json.loads(old[0])
                inputs = plan['provenance']
                if (inputs['bundle'] != fingerprint(bundle) or inputs['calendar'] != fingerprint(calendar)
                        or inputs['config'] != fingerprint(config)
                        or inputs['snapshot'] != fingerprint(state['snapshot'])):
                    raise ShadowError('同一决策日输入改变，不覆盖已冻结计划')
                return plan
            if state['snapshot']['asof'] != bundle['asof']:
                raise ShadowError('必须先核对当日账户快照')
            if abs(state['book_cash']-state['snapshot']['cash']) > .02 or state['book_positions'] != {
                    c:p['shares'] for c,p in state['snapshot']['positions'].items()}:
                raise ShadowError('快照后出现成交/调整，需等下一次收盘核对')
            plan = make_plan(state['snapshot'], bundle, calendar, config, state['memory'])
            self.db.execute('INSERT INTO plans VALUES(?,?,?)', (bundle['asof'], plan['plan_id'], canonical(plan)))
            for order in plan['orders']:
                order = dict(order, filled_shares=0, filled_gross=0., total_fee=0.)
                self.db.execute('INSERT INTO orders VALUES(?,?)', (order['order_id'], canonical(order)))
            state['memory'] = plan['next_memory']
            self._save(state)
            self._audit('plan_frozen', dict(plan_id=plan['plan_id'], provenance=plan['provenance']))
            return plan

    def preflight(self, plan_id, market):
        """只校验盘前条件并标记ready，绝不发送委托。"""
        with self.transaction():
            row = self.db.execute('SELECT payload FROM plans WHERE id=?', (plan_id,)).fetchone()
            if not row:
                raise ShadowError('未知计划')
            plan = json.loads(row[0])
            stamp = pd.Timestamp(market['timestamp'])
            if stamp.tzinfo is None or not market.get('source') or not market.get('corporate_actions_checked'):
                raise ShadowError('盘前检查缺来源、时区或公司行动确认')
            local = stamp.tz_convert('Asia/Shanghai')
            from execution_clock import timing
            window = timing(plan.get('session', 'opening_auction'))
            if str(local.date()) != plan['execution_date'] or not window['execution_start'] <= local.strftime('%H:%M:%S') < window['execution_end']:
                raise ShadowError('本计划仅适用于对应执行日执行窗口')
            if market.get('session') != plan.get('session', 'opening_auction'):
                raise ShadowError('委托时段不匹配')
            if (market.get('account_id') != plan['account_id'] or market.get('positions') != plan['snapshot_positions']
                    or market.get('open_orders') != []):
                raise ShadowError('盘前账户/持仓变化或存在未结委托，不能沿用昨晚计划')
            for spec in plan['orders']:
                order = self.order(spec['order_id'])
                if order['status'] not in ['planned', 'ready']:
                    raise ShadowError('订单已推进，不能重复盘前放行')
                quote = market['securities'].get(order['ts_code'])
                if (not quote or quote.get('suspended') or quote.get('corporate_action')
                        or not quote.get('broker_accepts_limit')
                        or quote.get('account_sellable', 0) < (order['shares'] if order['side'] == 'sell' else 0)
                        or (order['side'] == 'buy' and not quote.get('buy_allowed'))):
                    raise ShadowError('盘前状态、可卖量或券商限价合法性未通过')
                low, high = quote.get('down_limit'), quote.get('up_limit')
                if not 0 < number(low, 'down_limit') <= order['limit_price'] <= number(high, 'up_limit'):
                    raise ShadowError('预定限价不在官方当日价格范围内，不自动改价')
            needed = sum(o['reserved_cash'] for o in plan['orders'])
            if number(market['available_cash'], 'available_cash') + .01 < needed:
                raise ShadowError('盘前可用资金不足')
            for spec in plan['orders']:
                order = self.order(spec['order_id'])
                order.update(status='ready', preflight=fingerprint(market))
                self._order(order)
            self._audit('preflight_ready_no_submission', dict(plan_id=plan_id, market=market))
            return dict(status='ready', plan_id=plan_id, orders=len(plan['orders']), submitted=False)

    def record_fill(self, fill):
        """真实成交逐笔导入，按成交标识去重，部分买入成交可不是100股整数。"""
        if not fill.get('execution_id') or not fill.get('evidence'):
            raise ShadowError('成交需要唯一回报号及原始凭据引用')
        quantity = integer(fill['shares'], 'shares')
        price, fee = number(fill['price'], 'price'), number(fill['fee'], 'fee')
        if quantity <= 0 or price <= 0:
            raise ShadowError('成交数量与价格必须为正')
        with self.transaction():
            old = self.db.execute('SELECT payload FROM receipts WHERE id=?', (fill['execution_id'],)).fetchone()
            if old:
                if old[0] != canonical(fill):
                    raise ShadowError('同一成交号内容冲突')
                return dict(status='duplicate')
            order = self.order(fill['order_id'])
            if order['status'] not in ['ready', 'partial'] or fill['date'] != order['execution_date']:
                raise ShadowError('订单未放行、已终结或成交日期不一致')
            if quantity + order['filled_shares'] > order['shares']:
                raise ShadowError('累计成交超过委托数量')
            if (order['side']=='buy' and price > order['limit_price']+1e-8) or (
                    order['side']=='sell' and price < order['limit_price']-1e-8):
                raise ShadowError('成交突破原限价，需要核对订单映射')
            state = self.state()
            from execution_clock import event_after_snapshot, local_stamp
            event_after_snapshot(fill, state['snapshot'])
            if order['session'] == 'closing_auction':
                at = local_stamp(fill.get('timestamp'))
                if str(at.date()) != fill['date'] or at.strftime('%H:%M:%S') != '15:00:00':
                    raise ShadowError('需券商收盘集合竞价实际成交时刻15:00:00，不能用回报到达时刻')
                if order['side'] == 'sell':
                    available = state['snapshot']['positions'].get(order['ts_code'], {}).get('sellable_today', 0)
                    if order['filled_shares'] + quantity > available:
                        raise ShadowError('当日实际可卖量不足，不能出售今日买入股数')
            gross = float(Decimal(str(price)) * quantity)
            signed = quantity if order['side'] == 'buy' else -quantity
            code = order['ts_code']
            balance = state['book_positions'].get(code, 0) + signed
            delta = -gross-fee if order['side']=='buy' else gross-fee
            cash = float(Decimal(str(state['book_cash'])) + Decimal(str(delta)))
            if balance < 0 or cash < -.01:
                raise ShadowError('成交导致负现金或负持仓，请核对完整流水')
            if order['side'] == 'buy' and order['filled_gross'] + gross + order['total_fee'] + fee > order['reserved_cash'] + .01:
                raise ShadowError('实际成本超过冻结预算，请核对费用/原始委托')
            if 'monthly_risk' in state['memory']:
                from monthly_risk_live import apply_fill
                apply_fill(state, order, fill, balance)
            state['book_cash'] = cash
            state['latest_event_date'] = max(state['latest_event_date'], fill['date'])
            if fill.get('timestamp'):
                state['latest_event_at'] = max(state.get('latest_event_at', ''), local_stamp(fill['timestamp']).isoformat())
            if balance:
                state['book_positions'][code] = balance
            else:
                state['book_positions'].pop(code, None)
            if order['side'] == 'sell':
                state['memory'].update(last_sale_date=fill['date'], last_sale_signal=order['signal_date'])
            order['filled_shares'] += quantity
            order['filled_gross'] += gross
            order['total_fee'] += fee
            order['status'] = 'filled' if order['filled_shares']==order['shares'] else 'partial'
            self._order(order); self._save(state)
            self.db.execute('INSERT INTO receipts VALUES(?,?)', (fill['execution_id'], canonical(fill)))
            self._audit('fill', fill)
            return dict(status=order['status'], remaining=order['shares']-order['filled_shares'])

    def terminal(self, order_id, status, evidence):
        if status not in TERMINAL - {'filled'} or not evidence:
            raise ShadowError('需提供取消/拒绝/失效终态及券商凭据，不能自行假定失效')
        with self.transaction():
            order = self.order(order_id)
            if order['status'] in TERMINAL:
                if order['status'] != status or order.get('terminal_evidence') != evidence:
                    raise ShadowError('终态回报冲突')
                return order
            order.update(status=status, terminal_evidence=evidence)
            self._order(order)
            self._audit('terminal', dict(order_id=order_id, status=status, evidence=evidence))
            return order

    def adjustment(self, item):
        """凭资金/公司行动流水核对账本，不能静默以快照覆盖差异。"""
        if item['kind'] not in ['deposit', 'withdrawal', 'dividend', 'tax', 'fee', 'corporate_action'] or not item.get('evidence'):
            raise ShadowError('调整需明确类型及凭据')
        number(item['cash_delta'], 'cash_delta', -1e12)
        delta = item.get('share_delta', 0)
        if type(delta) is not int or (delta and item['kind'] != 'corporate_action'):
            raise ShadowError('股数变化仅能由有凭据公司行动调整')
        with self.transaction():
            old = self.db.execute('SELECT payload FROM receipts WHERE id=?', (item['event_id'],)).fetchone()
            if old:
                if old[0] != canonical(item):
                    raise ShadowError('调整流水号冲突')
                return dict(status='duplicate')
            state = self.state()
            from execution_clock import event_after_snapshot, local_stamp
            event_after_snapshot(item, state['snapshot'])
            if item.get('timestamp'):
                state['latest_event_at'] = max(state.get('latest_event_at', ''), local_stamp(item['timestamp']).isoformat())
            state['book_cash'] += item['cash_delta']
            state['latest_event_date'] = max(state['latest_event_date'], item['date'])
            if delta:
                code = item['ts_code']
                if code in state['memory']['pending_sells'] and state['memory']['pending_sells'][code]['target_shares'] != 0:
                    raise ShadowError('公司行动涉及待执行风险减仓目标，需要先核对目标股数；暂不自动换算')
                state['book_positions'][code] = state['book_positions'].get(code, 0) + delta
                if state['book_positions'][code] < 0:
                    raise ShadowError('公司行动后股数为负')
                if not state['book_positions'][code]:
                    del state['book_positions'][code]
                if 'monthly_risk' in state['memory']:
                    from restore_quota import RestoreQuota
                    risk = state['memory']['monthly_risk']
                    quota = RestoreQuota(risk['quota']); quota.clear(code)
                    risk['quota'] = quota.snapshot()
            if state['book_cash'] < -.01:
                raise ShadowError('调整后现金为负')
            self._save(state)
            self.db.execute('INSERT INTO receipts VALUES(?,?)', (item['event_id'], canonical(item)))
            self._audit('adjustment', item)
            return dict(status='recorded')

    def status(self):
        return dict(state=self.state(), orders=[json.loads(r[0]) for r in self.db.execute('SELECT payload FROM orders')],
                    event_count=self.db.execute('SELECT COUNT(*) FROM audit').fetchone()[0], live_submission=False)
