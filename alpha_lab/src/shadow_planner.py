"""截至收盘的独立计划接口：输入真实账户，不读取任何次日行情。"""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import re

import pandas as pd

from selection_first import plan_selection_first


class ShadowError(ValueError):
    """输入或账户不一致，停止生成或推进计划。"""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ShadowError(f'{name}必须为有限数值且不小于{minimum}')
    return value


def integer(value, name):
    number(value, name)
    if int(value) != value:
        raise ShadowError(f'{name}必须为整数')
    return int(value)


def iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ShadowError('日期必须为YYYY-MM-DD')
    pd.Timestamp(value)
    return value


def validate_calendar(calendar, date):
    dates = calendar['sessions']
    if not calendar.get('source') or dates != sorted(set(dates)):
        raise ShadowError('交易日历需要来源且唯一递增')
    for day in dates:
        iso_date(day)
    if date not in dates or dates.index(date) + 1 == len(dates):
        raise ShadowError('交易日历缺决策日或下一交易日')
    return dates[dates.index(date) + 1]


def validate_snapshot(snapshot):
    iso_date(snapshot['asof'])
    if not snapshot.get('account_id') or not snapshot.get('source'):
        raise ShadowError('账户快照缺账户标识或来源')
    for key in ['cash', 'receivables', 'equity', 'frozen_cash']:
        number(snapshot[key], key)
    if snapshot['frozen_cash'] or snapshot.get('open_orders'):
        raise ShadowError('收盘仍有冻结资金或未结委托，须先核对')
    if snapshot.get('phase', 'eod') not in ['eod', 'intraday']:
        raise ShadowError('未知账户快照阶段')
    if snapshot.get('timestamp'):
        from execution_clock import snapshot_stamp
        if str(snapshot_stamp(snapshot).date()) != snapshot['asof']:
            raise ShadowError('账户快照时刻与日期不一致')
        if snapshot.get('phase', 'eod') == 'eod' and snapshot_stamp(snapshot).strftime('%H:%M:%S') < '15:00:00':
            raise ShadowError('收盘快照早于15:00')
    intraday = snapshot.get('phase') == 'intraday'
    if intraday:
        from execution_clock import snapshot_stamp
        at = snapshot_stamp(snapshot)
        if str(at.date()) != snapshot['asof'] or not '14:30:00' <= at.strftime('%H:%M:%S') < '14:55:00':
            raise ShadowError('盘中账户快照须在当日14:30至14:55之间')
    value = snapshot['cash'] + snapshot['receivables']
    for code, pos in snapshot['positions'].items():
        if not re.fullmatch(r'(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)', code):
            raise ShadowError('当前计划器只接收沪深主板股票')
        shares = integer(pos['shares'], 'shares')
        available = integer(pos['sellable_today' if intraday else 'sellable_next_day'], 'sellable')
        if not shares or available > shares or number(pos['close'], 'close') <= 0:
            raise ShadowError('持仓或次日可卖数量非法')
        value += shares * pos['close']
    if abs(value - snapshot['equity']) > .02:
        raise ShadowError('现金、应收及持仓估值与净资产不一致')


def empty_memory():
    return dict(signal_date=None, pending_sells={}, supplementary_used=False,
                last_sale_date=None, last_sale_signal=None)


def make_plan(snapshot, bundle, calendar, cfg, memory=None):
    """同一快照及输入必得同一计划；所有输出订单仍需盘前适配检查。"""
    validate_snapshot(snapshot)
    date = snapshot['asof']
    next_date = validate_calendar(calendar, date)
    closing = cfg['execution_session'] == 'closing_auction'
    if closing:
        next_date = date
        from execution_clock import check_proxy, snapshot_stamp, local_stamp
        check_proxy(bundle.get('provenance', {}).get('receipt', {}), date)
        if bundle['provenance']['receipt'].get('synthetic') or bundle.get('synthetic'):
            raise ShadowError('合成盘中观察仅可验证计算，不能生成执行计划')
        if snapshot.get('phase') != 'intraday' or snapshot_stamp(snapshot) < local_stamp(bundle['provenance']['receipt']['downloaded_at']):
            raise ShadowError('收盘竞价须用行情获取后核对的当日盘中账户快照')
    elif snapshot.get('phase') == 'intraday':
        raise ShadowError('盘中账户快照不能用于次日开盘计划')
    predicted = cfg['strategy'] == 'monthly_predicted_risk'
    if ((not predicted and cfg['strategy'] != 'hybrid_full_risk') or cfg['restore'] != predicted or cfg['lot_grace']
            or cfg['execution_session'] not in ['opening_auction', 'opening_continuous', 'closing_auction'] or cfg['top_n'] != 5):
        raise ShadowError('当前只实现冻结full_risk、五股及开盘/收盘执行计划')
    for key in ['buy_cost','sell_cost','min_fee','solver_relative_gap']:
        number(cfg[key], key)
    if (not 0 < number(cfg['max_stock_fraction'],'max_stock_fraction') <= .25
            or not 0 < number(cfg['max_volume_participation'],'max_volume_participation') <= 1
            or cfg['price_tick'] != .01 or integer(cfg['candidate_pool'],'candidate_pool') < cfg['top_n']):
        raise ShadowError('单股上限、容量、报价单位或候选池配置非法')
    for key in ['buy_limit_buffer','sell_limit_buffer','st_sell_limit_buffer']:
        if not 0 < number(cfg[key],key) < 1:
            raise ShadowError('限价缓冲配置非法')
    if (bundle['asof'] != date or not bundle.get('provenance')
            or (not closing and not bundle.get('eod_complete'))
            or (closing and (bundle.get('eod_complete') is not False or bundle.get('input_kind') != 'intraday_proxy'))):
        raise ShadowError('信号日期、完整性或来源不符合账户决策日')
    if bundle['model_cutoff'] >= f'{pd.Timestamp(date).year}-01-01':
        raise ShadowError('风险模型训练截止越过年度边界')
    if int(bundle['model_year']) != pd.Timestamp(bundle['signal_date']).year:
        raise ShadowError('风险模型年份与信号日期不一致')
    if bundle['signal_date'] > date:
        raise ShadowError('信号来自未来')
    positions = snapshot['positions']
    if len(positions) > cfg['top_n']:
        raise ShadowError('实际持仓超过五只，需要先处理账户差异')
    state = deepcopy(memory or empty_memory())
    prior_state = deepcopy(state)
    if not predicted and 'monthly_risk' in state:
        raise ShadowError('候选账户不能静默退回旧策略')
    exposure_cap, quota = 1., None
    if predicted:
        from monthly_risk_live import advance
        exposure_cap, quota = advance(state, bundle, calendar, cfg, positions)
    review = bundle['is_review']
    if type(review) is not bool or (review and bundle['signal_date'] != date):
        raise ShadowError('复核日与信号日期矛盾')
    if review:
        state['signal_date'] = date
        state['supplementary_used'] = False
    elif state['signal_date'] not in [None, bundle['signal_date']]:
        raise ShadowError('活动信号期与账户状态不一致')
    equity, cash = snapshot['equity'], snapshot['cash']
    stock_value = sum(p['shares'] * p['close'] for p in positions.values())
    buy_budget = min(cash, max(0., equity * exposure_cap - stock_value)) if predicted else cash
    orders, decisions = [], []
    quotes = bundle['quotes']
    signal = bundle['signals']
    holds = bundle['holds']
    risks = bundle['risk_discounts']
    st = set(bundle['st_codes'])
    blocked = set(bundle['blocked_next_day'])
    for code, pos in positions.items():
        if code not in quotes or abs(quotes[code]['close'] - pos['close']) > cfg['price_tick'] / 2:
            raise ShadowError(f'{code}账户估值与当日行情不一致')
    def quote(code):
        if code not in quotes:
            raise ShadowError(f'{code}缺当日报价')
        q = quotes[code]
        if q['date'] != date or number(q['close'], 'close') <= 0:
            raise ShadowError(f'{code}缺当日价格，不能用旧行情生成计划')
        number(q['volume'], 'volume')
        return q
    def risk(code):
        value = risks.get(code)
        if value is None or not 0 < number(value, 'risk_discount') <= 1:
            raise ShadowError(f'{code}缺风险折扣')
        return value
    def limit_price(reference, multiplier):
        tick = Decimal(str(cfg['price_tick']))
        return float((Decimal(str(reference)) * Decimal(str(multiplier)) / tick).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP) * tick)
    def sell_quantity(desired, available, capacity):
        maximum = int(min(desired, available, capacity))
        odd = available % 100
        options = [maximum // 100 * 100]
        if odd and maximum >= odd:
            options.append(odd + (maximum - odd) // 100 * 100)
        return max(options)
    for code in list(state['pending_sells']):
        intent = state['pending_sells'][code]
        if code not in positions or positions[code]['shares'] <= intent['target_shares']:
            del state['pending_sells'][code]
    for code in positions:
        if code in st:
            state['pending_sells'][code] = dict(target_shares=0, signal_date=date, reason='st')
    if review:
        for code, pos in positions.items():
            quote(code)
            if predicted and holds.get(code) is False:
                state['pending_sells'][code] = dict(target_shares=0, signal_date=date, reason='hold_false')
                quota.clear(code)
                continue
            if code in state['pending_sells']:
                continue
            if code not in holds or holds[code] not in [True, False, None]:
                raise ShadowError(f'{code}缺持有状态')
            if holds[code] is False:
                state['pending_sells'][code] = dict(target_shares=0, signal_date=date, reason='hold_false')
                continue
            cap = equity * cfg['max_stock_fraction'] * risk(code)
            available = pos['sellable_today' if closing else 'sellable_next_day']
            locked = pos['shares'] - available
            cap_shares = max(locked, min(pos['shares'], math.floor(cap / pos['close'] + 1e-9)))
            required = pos['shares'] - cap_shares
            odd = available % 100
            candidates = [offset + max(0, math.ceil((required-offset)/100))*100 for offset in {0, odd}]
            reduction = min(q for q in candidates if required <= q <= available)
            if reduction:
                state['pending_sells'][code] = dict(target_shares=pos['shares']-reduction,
                    signal_date=date, reason='single_stock_cap')
    if predicted:
        for code in st:
            quota.clear(code)
        if review and stock_value > equity * exposure_cap + 1e-7:
            ratio = max(0., equity * exposure_cap / stock_value)
            for code, pos in positions.items():
                target = math.floor(pos['shares'] * ratio / 100) * 100
                old = state['pending_sells'].get(code)
                if old is None or target < old['target_shares']:
                    state['pending_sells'][code] = dict(target_shares=target,
                        signal_date=date, reason='portfolio_cash_cap', restore_epoch=quota.data['epoch'])
    def add_order(code, side, quantity, price, reference, signal_date, reason, reserved=0):
        orders.append(dict(ts_code=code, side=side, shares=int(quantity), limit_price=price,
            reference_price=reference, signal_date=signal_date, reason=reason,
            reserved_cash=reserved, decision_date=date, execution_date=next_date,
            session=cfg['execution_session'], status='planned'))
        if predicted:
            orders[-1]['restore_epoch'] = quota.data['epoch']
    for code, intent in state['pending_sells'].items():
        if code in blocked:
            decisions.append(dict(ts_code=code, reason='next_day_corporate_action_or_ineligible'))
            continue
        pos, q = positions[code], quote(code)
        quantity = sell_quantity(max(0, pos['shares']-intent['target_shares']), pos['sellable_today' if closing else 'sellable_next_day'],
                                 q['volume'] * cfg['max_volume_participation'])
        if quantity:
            buffer = cfg['st_sell_limit_buffer'] if code in st else cfg['sell_limit_buffer']
            add_order(code, 'sell', quantity, limit_price(q['close'], 1-buffer), q['close'],
                      intent['signal_date'], intent['reason'])
    settled_sale_date = calendar['sessions'][calendar['sessions'].index(date)-1] if closing and calendar['sessions'].index(date)>0 else date
    supplement = (not review and state['signal_date'] is not None and not state['supplementary_used']
        and state['last_sale_date'] == settled_sale_date and state['last_sale_signal'] == state['signal_date'])
    if supplement:
        state['supplementary_used'] = True
    restore_reserved = 0.
    if predicted and review and not state['monthly_risk']['active']:
        for code in sorted(positions, key=lambda c: (-quota.available(c)*positions[c]['close'], c)):
            credit = quota.available(code)
            if credit < 100 or code in state['pending_sells'] or holds.get(code) is not True or code in st or code in blocked:
                continue
            recovery = bundle.get('stock_recovery', {}).get(code, {})
            ret20 = recovery.get('ret20')
            if (recovery.get('known') is not True or recovery.get('weak') is not False
                    or not isinstance(ret20, (int,float)) or not math.isfinite(ret20) or ret20 <= 0):
                decisions.append(dict(ts_code=code, reason='stock_not_recovered'))
                continue
            q = quote(code)
            price = limit_price(q['close'], 1+cfg['buy_limit_buffer'])
            room = max(0., equity * cfg['max_stock_fraction'] * risk(code) - positions[code]['shares'] * price)
            budget = buy_budget - restore_reserved
            maximum = min(credit, room/price, q['volume']*cfg['max_volume_participation'],
                          budget/(price*(1+cfg['buy_cost'])), (budget-cfg['min_fee'])/price)
            quantity = max(0, math.floor(maximum/100)*100)
            gross = quantity * price
            net = gross * number(bundle['expected_return'], 'expected_return', -1) - max(cfg['min_fee'], gross*cfg['buy_cost']) - max(cfg['min_fee'], gross*cfg['sell_cost'])
            if quantity and net > 0:
                reserved = gross + max(cfg['min_fee'], gross*cfg['buy_cost'])
                add_order(code, 'buy', quantity, price, q['close'], date, 'targeted_restore', reserved)
                restore_reserved += reserved
    if (review or supplement) and state['signal_date'] is not None and len(positions) < cfg['top_n']:
        ranked = sorted(signal, key=lambda c: (-number(signal[c], 'score', -1), c))[:cfg['candidate_pool']]
        records = []
        for code in ranked:
            if code in positions:
                continue
            if code not in quotes:
                decisions.append(dict(ts_code=code,reason='no_known_closing_reference'))
                continue
            q = quote(code)
            price = limit_price(q['close'], 1+cfg['buy_limit_buffer'])
            cap = min(equity * cfg['max_stock_fraction'] * risk(code),
                      q['volume'] * cfg['max_volume_participation'] * price)
            eligible = bundle['eligible'][code] and code not in st and code not in blocked
            records.append(dict(ts_code=code, price=price, expected_return=bundle['expected_return'],
                shares=0, available=0, min_buy=100, buy_step=100, sell_step=100, max_notional=cap,
                min_target_shares=0, can_buy=eligible, can_sell=False))
        if records:
            allocation = plan_selection_first(pd.DataFrame(records), priorities=signal, cash=max(0.,buy_budget-restore_reserved),
                buy_rate=cfg['buy_cost'], sell_rate=cfg['sell_cost'], min_fee=cfg['min_fee'],
                forecast_kind='expected_simple_return', max_positions=cfg['top_n']-len(positions),
                solver_relative_gap=cfg['solver_relative_gap'])
            decisions += allocation['selection_audit'].to_dict('records')
            for row in allocation['plan'].itertuples():
                if row.buy_shares:
                    gross = row.buy_shares * row.price
                    reserved = gross + max(cfg['min_fee'], gross * cfg['buy_cost'])
                    add_order(row.ts_code, 'buy', row.buy_shares, row.price, quotes[row.ts_code]['close'],
                        state['signal_date'], 'monthly_entry' if review else 'sale_cash_supplement', reserved)
    if sum(x['reserved_cash'] for x in orders) > cash + 1e-7:
        raise ShadowError('买单预算超过决策日现金')
    if predicted:
        if sum(x['reserved_cash'] for x in orders) > buy_budget + 1e-7:
            raise ShadowError('恢复与新买合计超过总股票预算')
        state['monthly_risk']['quota'] = quota.snapshot()
    provenance = dict(snapshot=fingerprint(snapshot), bundle=fingerprint(bundle),
                      calendar=fingerprint(calendar), config=fingerprint(cfg), memory=fingerprint(prior_state))
    plan_id = fingerprint(dict(account=snapshot['account_id'], date=date, inputs=provenance))[:24]
    for index, order in enumerate(orders):
        order['order_id'] = f'{plan_id}-{index+1:02d}'
    # 未入选候选没有预约现金/净利：审计缺值序列化为null，不放松订单数值校验。
    decisions = [{k:None if isinstance(v,float) and not math.isfinite(v) else v
                  for k,v in row.items()} for row in decisions]
    result = dict(schema_version=1, mode='shadow', account_id=snapshot['account_id'], plan_id=plan_id,
        decision_date=date, execution_date=next_date, session=cfg['execution_session'],
        input_kind=bundle.get('input_kind', 'eod'), orders=orders, decisions=decisions,
        provenance=provenance, next_memory=state, requires_preflight=True,
        equity=equity, cash=cash, snapshot_positions={c:p['shares'] for c,p in positions.items()}, live_submission=False)
    if closing:
        result['quote_captured_at'] = bundle['provenance']['receipt']['downloaded_at']
    if predicted:
        result.update(strategy=cfg['strategy'], exposure_cap=exposure_cap,
            risk_probability=state['monthly_risk']['probability'], risk_review_date=state['monthly_risk']['last_review'],
            risk_active=state['monthly_risk']['active'], restore_epoch=quota.data['epoch'])
    return result
