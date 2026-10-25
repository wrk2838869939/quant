"""因果日线研究账本：收盘冻结DAY限价单，现金/股数/公司行动独立记账。"""
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd


class AccountingError(RuntimeError):
    """缺少关键执行或公司行动数据时停止，不返回看似有效的收益。"""


@dataclass
class Position:
    shares: int
    last_price: float
    last_adj: float
    locks: list = field(default_factory=list)
    stale_days: int = 0
    action_since_quote: bool = False

    def available(self, date):
        return self.shares - sum(q for q, release in self.locks if date < release)


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    events: pd.DataFrame
    holdings: pd.DataFrame
    summary: dict


def board_of(code):
    if code.endswith(".BJ"):
        return "beijing"
    if code.startswith("68"):
        return "star"
    if code.startswith("30"):
        return "chinext"
    return "main"


def affordable_shares(budget, price, rate, minimum_fee, rule, volume_cap=np.inf, extra_rate=0.):
    """手续费包含在预算内，按板块起买量和步长取整。"""
    if not np.isfinite(price) or price <= 0 or budget <= minimum_fee:
        return 0
    maximum = min(budget / (price * (1 + rate + extra_rate)),
                  (budget - minimum_fee) / (price * (1 + extra_rate)), volume_cap)
    if maximum < rule["min_buy"]:
        return 0
    return int(rule["min_buy"] + math.floor((maximum - rule["min_buy"]) / rule["buy_step"]) * rule["buy_step"])


def daily_metrics(equity, capital, annual_days):
    """按实际观察的交易日数年化；回撤峰值包含初始资金。"""
    returns = equity.pct_change(fill_method=None)
    returns.iloc[0] = equity.iloc[0] / capital - 1
    peaks = np.maximum.accumulate(np.r_[capital, equity.to_numpy()])[1:]
    total = equity.iloc[-1] / capital - 1
    std = returns.std(ddof=1)
    return dict(total=float(total), ann=float((1 + total) ** (annual_days / len(equity)) - 1)
                if total > -1 else -1.0,
                mdd=float(np.min(equity.to_numpy() / peaks - 1)),
                sharpe=float(returns.mean() / std * np.sqrt(annual_days)) if std > 0 else np.nan,
                n_days=len(equity)), returns


def run_backtest(signals, market, basics, st_intervals, limits, actions, start, end, cfg,
                 rights_issues=(), *, risk_allowed=None, risk_signal_max_age=None,
                 retail_forecasts=None, retail_config=None, extra_execution_delay=0,
                 reentry_signals=None, reentry_forecasts=None, slippage_rate=0.,
                 reentry_on_schedule_only=False, hold_allowed=None, causal_config=None):
    """收盘冻结DAY限价单；当日卖款不得支持已冻结买单。

    仅实现当前主板/月度/先选股研究契约。日线穿价是成交假设，不保证队列成交。
    次日OHLC和官方限价仅用于冻结单撮合；次日全日成交量不参与该撮合。
    """
    from decimal import Decimal, ROUND_HALF_UP
    from selection_first import plan_selection_first
    cc = dict(buy_limit_buffer=.02, max_entry_attempts=2, sell_limit_buffer=.10,
              st_sell_limit_buffer=.05, cancel_on_ex_date=True)
    if causal_config:
        unknown = set(causal_config) - set(cc)
        if unknown:
            raise ValueError('未知因果执行参数: ' + str(sorted(unknown)))
        cc.update(causal_config)
    if (not 0 <= cc['buy_limit_buffer'] < 1 or not 0 < cc['sell_limit_buffer'] < 1
            or not 0 < cc['st_sell_limit_buffer'] < 1
            or cc['max_entry_attempts'] not in (1, 2) or cc['cancel_on_ex_date'] is not True):
        raise ValueError('限价/执行次数配置非法')
    if (risk_allowed is not None or reentry_signals is not None or reentry_forecasts is not None
            or extra_execution_delay != 0 or reentry_on_schedule_only):
        raise ValueError('因果引擎仅支持固定月度主板持有策略；外部择时/恢复/延迟须另行实现')
    if (retail_forecasts is None or retail_config is None or hold_allowed is None
            or not retail_config.get('selection_first') or not 1 <= cfg['top_n'] <= 5
            or retail_config.get('max_positions') != cfg['top_n']):
        raise ValueError('因果引擎需要明确先选股、持有许可和最多5只限制')
    if not np.isfinite(slippage_rate) or slippage_rate < 0:
        raise ValueError('附加费用必须非负有限')
    for matrix in [signals, retail_forecasts, hold_allowed]:
        if matrix.index.has_duplicates or matrix.columns.has_duplicates or not matrix.index.is_monotonic_increasing:
            raise AccountingError('矩阵日期/股票重复或日期未排序')
    if not (signals.index.equals(retail_forecasts.index) and signals.columns.equals(retail_forecasts.columns)
            and signals.index.equals(hold_allowed.index) and signals.columns.equals(hold_allowed.columns)):
        raise ValueError('信号、预测、持有许可必须对齐')
    if not (hold_allowed.isna() | hold_allowed.isin([True, False])).all().all():
        raise ValueError('持有许可只能为布尔或未知')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    market = market[(market.date >= start) & (market.date <= end)].copy()
    if market.duplicated(['date', 'ts_code']).any() or limits.duplicated(['date', 'ts_code']).any():
        raise AccountingError('行情或限价重复')
    fields = ['open', 'high', 'low', 'close', 'pre_close', 'adj_factor', 'vol_share']
    if not set(fields).issubset(market):
        raise ValueError('DAY限价撮合需要原始high/low及原账本行情列')
    dates = pd.DatetimeIndex(sorted(market.date.unique()))
    if not len(dates) or cfg['capital'] <= 0:
        raise ValueError('窗口为空或本金非正')
    panels = {col: market.pivot(index='date', columns='ts_code', values=col).reindex(dates) for col in fields}
    arrays = {k: v.to_numpy(dtype=float) for k, v in panels.items()}
    code_idx = {code: j for j, code in enumerate(panels['close'].columns)}
    day_idx = {date: i for i, date in enumerate(dates)}
    price_limits = limits.set_index(['date', 'ts_code'])[['up_limit', 'down_limit']].to_dict('index')
    listings = basics.set_index('ts_code')['list_date'].to_dict()
    delistings = basics.dropna(subset=['delist_date']).set_index('ts_code')['delist_date'].to_dict()
    delist_known = (pd.to_datetime(basics.set_index('ts_code')['delist_known_date'], errors='coerce').dropna().to_dict()
                    if 'delist_known_date' in basics else {})
    if len(actions) and 'known_date' not in actions:
        raise AccountingError('非空公司行动缺少known_date，停止认证')
    if len(actions) and (pd.to_datetime(actions.record_date, errors='coerce').isna().any()
                         or pd.to_datetime(actions.ex_date, errors='coerce').isna().any()
                         or (pd.to_datetime(actions.record_date) >= pd.to_datetime(actions.ex_date)).any()):
        raise AccountingError('公司行动必须登记日在除权日之前')
    if 'known_date' in actions:
        known = pd.to_datetime(actions.known_date, errors='coerce')
        if known.isna().any() or (known > pd.to_datetime(actions.record_date)).any():
            raise AccountingError('公司行动缺失登记日前可知公告')
    st_intervals = st_intervals.copy()
    if len(st_intervals) and not {'known_date', 'ann_date'}.intersection(st_intervals.columns):
        raise AccountingError('非空ST状态缺少公告可知日期，停止认证')
    if 'known_date' in st_intervals:
        known = pd.to_datetime(st_intervals['known_date'], errors='coerce')
        if known.isna().any():
            raise AccountingError('ST状态known_date缺失')
        if 'ann_date' in st_intervals:
            announced = pd.to_datetime(st_intervals['ann_date'], errors='coerce')
            if announced.isna().any():
                raise AccountingError('ST状态ann_date缺失')
            st_intervals['ann_date'] = pd.concat([known, announced], axis=1).max(axis=1)
        else:
            st_intervals['ann_date'] = known
    if 'ann_date' in st_intervals and pd.to_datetime(st_intervals.ann_date, errors='coerce').isna().any():
        raise AccountingError('ST状态公告日缺失')
    record_events, ex_events, rights_by_date = {}, {}, {}
    for event_id, row in actions.iterrows():
        ev = row.to_dict(); ev['id'] = event_id
        record_events.setdefault(ev['record_date'], []).append(ev)
        ex_events.setdefault(ev['ex_date'], []).append(ev)
    for right in rights_issues:
        if right['policy'] != 'decline':
            raise AccountingError('仅实现不参与配股')
        rights_by_date.setdefault(pd.Timestamp(right['ex_date']), []).append(right)
    cash = float(cfg['capital'])
    positions, pending_sell, entitlements = {}, {}, {}
    confirmed_hold_exits = set()
    receivables, daily, trades, events, holdings = [], [], [], [], []
    queue = []
    order_counter = 0
    active_signal_date = None
    supplementary_used = False
    monthly_execution_count = 0
    top_n = cfg['top_n']

    def event(date, code, kind, **detail):
        events.append(dict(date=date, ts_code=code, kind=kind, **detail))

    def quote(i, code, col):
        j = code_idx.get(code)
        return arrays[col][i, j] if j is not None else np.nan

    def st_set(date, asof=None):
        asof = date if asof is None else asof
        # 没有公告元数据时只按当时已生效状态，不把未来生效区间当作提前已知。
        effective = date if 'ann_date' in st_intervals else min(date, asof)
        active = ((st_intervals.start <= effective) & (st_intervals.end.isna() | (st_intervals.end >= effective)))
        if 'ann_date' in st_intervals:
            active &= pd.to_datetime(st_intervals.ann_date) <= asof
        return set(st_intervals.loc[active, 'ts_code'])

    def tick_price(reference, multiplier):
        tick = Decimal(str(cfg['price_tick']))
        value = Decimal(str(reference)) * Decimal(str(multiplier))
        return float((value / tick).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * tick)

    def sell_quantity(desired, available, capacity, step):
        maximum = int(min(desired, available, capacity))
        odd = int(available) % step
        options = [int(maximum // step) * step]
        if odd and maximum >= odd:
            options.append(odd + int((maximum - odd) // step) * step)
        return max(options)

    def order_event(date, order, kind, **detail):
        info = {k: v for k, v in order.items() if k not in ('ts_code', 'kind')}
        info.update(detail)
        event(date, order['ts_code'], kind, **info)

    def create_order(date, execute_date, code, side, quantity, reference, limit, signal_date, reason,
                     available_cash, plan_equity, reserved=0., attempt=1):
        nonlocal order_counter
        order_counter += 1
        order = dict(order_id=f'O{order_counter:06d}', ts_code=code, side=side, shares=int(quantity),
                     decision_date=date, reference_date=date, execution_date=execute_date,
                     signal_date=signal_date, reference_price=float(reference), limit_price=float(limit),
                     reserved_cash=float(reserved), cash_available=float(available_cash),
                     cash_at_decision=float(available_cash), plan_equity=float(plan_equity),
                     budget_source='closing_cash', reason=reason, entry_attempt=attempt)
        order_event(date, order, 'order_planned')
        return order

    def match_order(i, date, order):
        """此函数不改变股票/股数/限价，仅用订单之后的市场结果撮合。"""
        code, side, limit = order['ts_code'], order['side'], order['limit_price']
        if date >= delistings.get(code, pd.Timestamp.max):
            return None, 'delisted_at_execution'
        if side == 'buy':
            active_st = (st_intervals.ts_code.eq(code) & (st_intervals.start <= date)
                         & (st_intervals.end.isna() | (st_intervals.end >= date)))
            if active_st.any():
                # 整日公告日期不能证明盘前可用；发现真实下单路径时停止认证，不事后撤单美化。
                if 'ann_date' not in st_intervals or (
                        pd.to_datetime(st_intervals.loc[active_st, 'ann_date']) > order['decision_date']).any():
                    raise AccountingError(f'{date} {code}: ST生效但冻结订单前公告时点不可认证')
            if code in st_set(date, asof=order['decision_date']):
                return None, 'st_known_before_order'
        bound = price_limits.get((date, code))
        if bound is None or not all(np.isfinite(v) and v > 0 for v in bound.values()):
            event(date, code, 'missing_limit', side=side)
            return None, 'missing_limit'
        if limit < bound['down_limit'] - 1e-8 or limit > bound['up_limit'] + 1e-8:
            return None, 'frozen_limit_outside_official_bounds'
        op, high, low = (quote(i, code, k) for k in ['open', 'high', 'low'])
        if not np.isfinite(op) or op <= 0:
            return None, 'no_open_quote'
        if side == 'buy':
            if op <= limit + 1e-8 and op < bound['up_limit'] - cfg['price_tick'] / 2:
                return op, 'open'
            if not np.isfinite(low):
                raise AccountingError(f'{date} {code}: 未成交买单缺少DAY最低价')
            if low < limit - 1e-8:
                return limit, 'strict_intraday_cross'
        else:
            if op >= limit - 1e-8 and op > bound['down_limit'] + cfg['price_tick'] / 2:
                return op, 'open'
            if not np.isfinite(high):
                raise AccountingError(f'{date} {code}: 未成交卖单缺少DAY最高价')
            if high > limit + 1e-8:
                return limit, 'strict_intraday_cross'
        return None, 'day_limit_not_crossed'

    for i, date in enumerate(dates):
        # 已确认的权益与支付独立：卖出股票不会丢失登记日获得的分红。
        for action in ex_events.get(date, []):
            code = action["ts_code"]
            shares = entitlements.pop(action["id"], 0)
            if not shares:
                continue
            div, bonus = action["cash_div_tax"], action["stk_div"]
            if not np.isfinite(div) or not np.isfinite(bonus) or min(div, bonus) < 0:
                raise AccountingError(f"{date.date()} {code}: 分红字段缺失或为负")
            pos = positions.get(code)
            if pos is None:
                raise AccountingError(f"{date.date()} {code}: 登记后除权前已卖出，需独立红股权益处理")
            reference_fraction = action.get("reference_share_fraction", 1.0)
            if not np.isfinite(reference_fraction) or not 0 < reference_fraction <= 1:
                raise AccountingError(f"{date.date()} {code}: 除权参考参数无效")
            reference_div = div * reference_fraction
            reference_bonus = bonus * reference_fraction
            reference = (pos.last_price - reference_div) / (1 + reference_bonus)
            pc = quote(i, code, "pre_close")
            if np.isfinite(pc) and not np.isclose(reference, pc, atol=cfg["action_price_atol"], rtol=cfg["action_price_rtol"]):
                raise AccountingError(f"{date.date()} {code}: 分红金额与除权参考价不符 ({reference:.4f} vs {pc:.4f})")
            pos.last_price = reference
            pos.action_since_quote = True
            if bonus:
                release = action["div_listdate"]
                if pd.isna(release) or release < date:
                    raise AccountingError(f"{date.date()} {code}: 红股可卖日期未知")
                exact = shares * bonus
                added = int(math.floor(exact + 1e-8))
                pos.shares += added
                pos.locks.append((added, release))
                event(date, code, "bonus_shares", shares=added, fractional_discard=exact - added)
            if div:
                pay = action["pay_date"]
                if pd.isna(pay) or pay < date:
                    raise AccountingError(f"{date.date()} {code}: 分红到账日期未知")
                amount = shares * div * (1 - cfg["cash_dividend_haircut"])
                receivables.append(dict(ts_code=code, amount=amount, pay_date=pay))
                event(date, code, "dividend_receivable", amount=amount)
        waiting = []
        for claim in receivables:
            if claim["pay_date"] <= date:
                cash += claim["amount"]
                event(date, claim["ts_code"], "dividend_paid", amount=claim["amount"])
            else:
                waiting.append(claim)
        receivables = waiting

        for right in rights_by_date.get(date, []):
            code = right["ts_code"]
            pos = positions.get(code)
            if pos is None:
                continue
            reference = (pos.last_price + right["ratio"] * right["subscription_price"]) / (1 + right["ratio"])
            pc = quote(i, code, "pre_close")
            if not np.isfinite(pc) or not np.isclose(reference, pc, atol=cfg["action_price_atol"], rtol=cfg["action_price_rtol"]):
                raise AccountingError(f"{date.date()} {code}: 配股方案与除权参考价不符")
            event(date, code, "rights_declined", reference_loss=pos.shares * (pos.last_price - pc))
            pos.last_price = pc
            pos.action_since_quote = True

        for code, pos in list(positions.items()):
            if date >= delistings.get(code, pd.Timestamp.max):
                recovery = pos.shares * pos.last_price * cfg["delist_recovery"]
                cash += recovery
                event(date, code, "delist_writeoff", shares=pos.shares,
                      amount=pos.shares * pos.last_price - recovery, cash_recovery=recovery)
                del positions[code]
                pending_sell.pop(code, None)
                confirmed_hold_exits.discard(code)
                continue
            adj = quote(i, code, "adj_factor")
            reference = quote(i, code, "pre_close")
            atol = cfg["action_price_atol"] if pos.action_since_quote else cfg["price_tick"] / 2
            rtol = cfg["action_price_rtol"] if pos.action_since_quote else 0
            if np.isfinite(reference) and not np.isclose(reference, pos.last_price, atol=atol, rtol=rtol):
                # 即使复权列漏更新，原始参考价的无解释跳变仍须拦截。
                if cfg["strict_actions"]:
                    raise AccountingError(f"{date.date()} {code}: 前收参考价与持仓估值不连续")
                event(date, code, "unexplained_adjustment", previous_price=pos.last_price, reference=reference)
            if np.isfinite(adj) and adj > 0:
                changed = not np.isclose(adj, pos.last_adj, rtol=cfg["adjustment_rtol"], atol=0)
                if changed and not pos.action_since_quote:
                    if np.isfinite(reference) and np.isclose(reference, pos.last_price,
                                                             atol=cfg["price_tick"] / 2, rtol=0):
                        # 原始参考价连续：只记录上游复权元数据变动，不改变现金或股数。
                        event(date, code, "adjustment_metadata_change", old_adj=pos.last_adj, new_adj=adj)
                    else:
                        event(date, code, "unexplained_adjustment", old_adj=pos.last_adj, new_adj=adj)
                        if cfg["strict_actions"]:
                            raise AccountingError(f"{date.date()} {code}: 参考价不连续且无对应公司行动")
                pos.last_adj = adj
                pos.action_since_quote = False

        action_codes = {a['ts_code'] for a in ex_events.get(date, [])}
        action_codes.update(a['ts_code'] for a in rights_by_date.get(date, []))
        monthly_sale_filled = False
        # 卖出与买入可不同日内时点；买单已由昨日现金独立足额冻结，顺序不释放新预算。
        for order in sorted(queue, key=lambda x: x['side'] != 'sell'):
            code, quantity, side = order['ts_code'], order['shares'], order['side']
            assert order['decision_date'] < date
            if code in action_codes:
                order_event(date, order, 'order_cancelled', cancellation_reason='corporate_action_ex_date')
                if side == 'sell':
                    event(date, code, 'sell_deferred', reason='corporate_action_ex_date')
                continue
            if side == 'sell' and (code not in positions or positions[code].available(date) < quantity):
                order_event(date, order, 'order_cancelled', cancellation_reason='frozen_quantity_not_available')
                event(date, code, 'sell_deferred', reason='locked_or_missing_position')
                continue
            if side == 'buy' and (code in positions or len(positions) >= top_n):
                order_event(date, order, 'order_cancelled', cancellation_reason='position_slot_unavailable')
                continue
            price, execution_kind = match_order(i, date, order)
            if price is None:
                order_event(date, order, 'order_cancelled' if side == 'buy' else 'order_deferred',
                            cancellation_reason=execution_kind)
                if side == 'sell':
                    event(date, code, 'sell_deferred', reason=execution_kind)
                else:
                    event(date, code, 'buy_skipped', reason=execution_kind)
                continue
            gross = quantity * price
            rate = cfg['buy_cost'] if side == 'buy' else cfg['sell_cost']
            fee = max(cfg['min_fee'], gross * rate) + gross * slippage_rate
            if side == 'sell':
                fee = min(gross, fee)
                cash += gross - fee
                positions[code].shares -= quantity
                intent = pending_sell.get(code)
                if intent and intent['signal_date'] == active_signal_date:
                    monthly_sale_filled = True
                if positions[code].shares == 0:
                    del positions[code]
                    pending_sell.pop(code, None)
                    confirmed_hold_exits.discard(code)
                elif intent and positions[code].shares <= intent['target_shares']:
                    pending_sell.pop(code, None)
            else:
                if gross + fee > order['reserved_cash'] + 1e-7 or gross + fee > cash + 1e-7:
                    raise AccountingError('冻结买单预算不覆盖成交成本；禁止挪用当日卖款')
                cash -= gross + fee
                adj = quote(i, code, 'adj_factor')
                if not np.isfinite(adj) or adj <= 0:
                    raise AccountingError('已成交证券缺失复权校验元数据')
                release = dates[i + 1] if i + 1 < len(dates) else date + pd.Timedelta(days=1)
                positions[code] = Position(quantity, price, adj, locks=[(quantity, release)])
            trades.append(dict(date=date, signal_date=order['signal_date'], ts_code=code, side=side,
                               shares=quantity, price=price, gross=gross, fee=fee, cash_after=cash,
                               order_id=order['order_id'], decision_date=order['decision_date'],
                               reference_date=order['reference_date'], reference_price=order['reference_price'],
                               limit_price=order['limit_price'], execution_kind=execution_kind,
                               reserved_cash=order['reserved_cash'], budget_source='closing_cash'))
            if slippage_rate:
                trades[-1]['slippage_cost'] = gross * slippage_rate
            order_event(date, order, 'order_filled', fill_price=price, fill_gross=gross,
                        fill_fee=fee, execution_kind=execution_kind)
        queue = []
        stock_value = 0.
        for code, pos in positions.items():
            close = quote(i, code, 'close')
            if np.isfinite(close) and close > 0:
                pos.last_price = close; pos.stale_days = 0
            else:
                pos.stale_days += 1
                if pos.stale_days == cfg['stale_warning_days']:
                    event(date, code, 'stale_valuation', days=pos.stale_days)
            stock_value += pos.shares * pos.last_price
            holdings.append(dict(date=date, ts_code=code, shares=pos.shares, available=pos.available(date),
                                 price=pos.last_price, value=pos.shares * pos.last_price, stale_days=pos.stale_days))
        claims = sum(claim['amount'] for claim in receivables)
        equity = cash + stock_value + claims
        if cash < -1e-6 or not np.isfinite(equity) or equity < 0 or len(positions) > top_n:
            raise AccountingError('每日现金/净值/最多5只约束失效')
        daily.append(dict(date=date, cash=cash, stock_value=stock_value, receivables=claims,
                          equity=equity, n_hold=len(positions), pending_sell=len(pending_sell)))
        for action in record_events.get(date, []):
            pos = positions.get(action['ts_code'])
            entitlements[action['id']] = pos.shares if pos else 0
        if i + 1 == len(dates):
            continue
        next_date = dates[i + 1]
        review = date in signals.index
        st_today = st_set(date)
        if review:
            active_signal_date = date
            supplementary_used = False
            monthly_execution_count += 1
        # ST公告、生效状态在收盘被获知；次日才允许发出卖单，不能用当日状态倒填开盘。
        for code, pos in positions.items():
            if code in st_today and code not in confirmed_hold_exits:
                confirmed_hold_exits.add(code)
                pending_sell[code] = dict(target_shares=0, signal_date=date, reason='st')
                event(date, code, 'hold_exit_requested', signal_date=date, reason='st', old_shares=pos.shares)
        if review:
            for code, pos in positions.items():
                state = hold_allowed.loc[date].get(code, np.nan)
                mu = retail_forecasts.loc[date].get(code, np.nan)
                if code in pending_sell or code in confirmed_hold_exits:
                    continue
                if pd.notna(state) and not bool(state):
                    confirmed_hold_exits.add(code)
                    pending_sell[code] = dict(target_shares=0, signal_date=date, reason='hold_false')
                    event(date, code, 'hold_exit_requested', signal_date=date, reason='hold_false', old_shares=pos.shares)
                    continue
                reference = pos.last_price
                cap = equity * retail_config['max_stock_fraction']
                available = pos.available(next_date)
                locked = pos.shares - available
                cap_shares = max(locked, min(pos.shares, int(math.floor(cap / reference + 1e-9))))
                required = pos.shares - cap_shares
                step = cfg['boards'][board_of(code)]['buy_step']
                odd = available % step
                options = [offset + max(0, math.ceil((required-offset)/step))*step for offset in {0, odd}]
                reduction = min(q for q in options if required <= q <= available)
                target = pos.shares - reduction
                event(date, code, 'hold_retained', signal_date=date,
                      reason='unknown' if pd.isna(state) else 'trend_allowed', old_shares=pos.shares,
                      min_target_shares=target, risk_cap=cap, unavoidable_locked_excess=max(0., locked*reference-cap))
                if reduction:
                    pending_sell[code] = dict(target_shares=target, signal_date=date, reason='single_stock_cap')
                    event(date, code, 'hold_risk_cap_reduction', signal_date=date, old_shares=pos.shares,
                          target_shares=target, sell_shares=reduction, reason='single_stock_cap')
        # 明确退出与风险减仓均用已知收盘股数/成交量冻结次日数量，失败次日晚重新冻结。
        for code, intent in list(pending_sell.items()):
            pos = positions.get(code)
            if pos is None:
                pending_sell.pop(code, None); confirmed_hold_exits.discard(code)
                continue
            if board_of(code) != 'main':
                raise ValueError('当前DAY卖限价规则仅定义主板')
            available = pos.available(next_date)
            vol = quote(i, code, 'vol_share')
            capacity = max(0., vol * cfg['max_volume_participation']) if np.isfinite(vol) else 0.
            desired = max(0, pos.shares - intent['target_shares'])
            quantity = sell_quantity(desired, available, capacity, cfg['boards']['main']['buy_step'])
            if not quantity:
                event(date, code, 'sell_planning_deferred', reason='known_volume_or_lock_capacity',
                      decision_date=date, signal_date=intent['signal_date'], desired_shares=desired)
                continue
            reference = pos.last_price
            buffer = cc['st_sell_limit_buffer'] if code in st_today else cc['sell_limit_buffer']
            limit = max(cfg['price_tick'], tick_price(reference, 1-buffer))
            queue.append(create_order(date, next_date, code, 'sell', quantity, reference, limit,
                         intent['signal_date'], intent['reason'], cash, equity))
            event(date, code, 'allocation_decision', signal_date=intent['signal_date'], decision_date=date,
                  expected_return=retail_forecasts.loc[active_signal_date].get(code, 0.) if active_signal_date is not None else 0.,
                  old_shares=pos.shares, target_shares=pos.shares-quantity, buy_shares=0, sell_shares=quantity,
                  risk_cap=0. if intent['target_shares']==0 else equity*retail_config['max_stock_fraction'],
                  min_target_shares=intent['target_shares'], unavoidable_locked_excess=0., solver_gap=0., solver_absolute_gap=0.)
        # 月度初次规划，以及首次月度卖出实际到账后的唯一补位。新月覆盖旧候选意图。
        supplement = (not review and monthly_sale_filled and active_signal_date is not None
                      and not supplementary_used and cc['max_entry_attempts'] == 2)
        if supplement:
            supplementary_used = True
        if not (review or supplement) or active_signal_date is None:
            continue
        attempt = 1 if review else 2
        rank = signals.loc[active_signal_date].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False, kind='stable')
        candidates = list(rank.head(retail_config['candidate_pool']).index)
        slots = max(0, top_n - len(positions))  # 仍持有、含退出待成交仓位，始终占名额。
        event(date, '__portfolio__', 'entry_planning_attempt', signal_date=active_signal_date,
              decision_date=date, execution_date=next_date, entry_attempt=attempt,
              cash_available=cash, plan_equity=equity, available_slots=slots, budget_source='closing_cash')
        records = []
        for code in candidates:
            if code in positions:
                continue
            if board_of(code) != 'main':
                raise ValueError('当前因果执行仅允许主板候选')
            reference = quote(i, code, 'close')
            if not np.isfinite(reference) or reference <= 0:
                event(date, code, 'selection_decision', signal_date=active_signal_date,
                      reason='no_known_closing_reference', priority=rank.get(code))
                continue
            rule = cfg['boards']['main']
            listing = listings.get(code, pd.NaT)
            eligible = (pd.notna(listing) and (active_signal_date-listing).days >= cfg['min_listing_calendar_days']
                        and date >= pd.Timestamp(rule['start']) and date < delistings.get(code, pd.Timestamp.max)
                        and code not in st_today and code not in st_set(active_signal_date)
                        and code not in st_set(next_date, asof=date))
            announced_delist = (delist_known.get(code, pd.Timestamp.max) <= date
                                and next_date >= delistings.get(code, pd.Timestamp.max))
            if announced_delist:
                eligible = False
                event(date, code, 'known_next_day_ineligible', signal_date=active_signal_date,
                      decision_date=date, execution_date=next_date, reason='announced_delisting_next_day')
            if code in st_set(next_date, asof=date):
                event(date, code, 'known_next_day_ineligible', signal_date=active_signal_date,
                      decision_date=date, execution_date=next_date, reason='announced_st_next_day')
            mu = retail_forecasts.loc[active_signal_date].get(code, np.nan)
            vol = quote(i, code, 'vol_share')
            capacity = max(0., vol * cfg['max_volume_participation']) if np.isfinite(vol) else 0.
            limit = max(cfg['price_tick'], tick_price(reference, 1+cc['buy_limit_buffer']))
            cap = min(equity * retail_config['max_stock_fraction'], capacity * limit)
            records.append(dict(ts_code=code, price=limit, expected_return=float(mu) if np.isfinite(mu) else 0.,
                                shares=0, available=0, min_buy=rule['min_buy'], buy_step=rule['buy_step'],
                                sell_step=rule['buy_step'], max_notional=cap, min_target_shares=0,
                                can_buy=bool(eligible and np.isfinite(mu) and capacity >= rule['min_buy']), can_sell=False,
                                reference_price=reference))
        if not records:
            continue
        allocation = plan_selection_first(pd.DataFrame(records), priorities=rank.to_dict(), cash=cash,
                        buy_rate=cfg['buy_cost'], sell_rate=cfg['sell_cost'], min_fee=cfg['min_fee'],
                        forecast_kind='expected_simple_return', slippage_rate=slippage_rate,
                        solver_relative_gap=retail_config.get('solver_relative_gap', .001), max_positions=slots)
        for decision in allocation['selection_audit'].to_dict('records'):
            code = decision.pop('ts_code')
            event(date, code, 'selection_decision', signal_date=active_signal_date, decision_date=date,
                  entry_attempt=attempt, **decision)
        references = {r['ts_code']: r['reference_price'] for r in records}
        reserved_total = 0.
        for row in allocation['plan'].itertuples():
            event(date, row.ts_code, 'allocation_decision', signal_date=active_signal_date, decision_date=date,
                  expected_return=row.expected_return, old_shares=0, target_shares=row.target_shares,
                  buy_shares=row.buy_shares, sell_shares=0, risk_cap=row.max_notional, min_target_shares=0,
                  unavoidable_locked_excess=0., solver_gap=allocation['solver_gap'], solver_absolute_gap=allocation['solver_absolute_gap'])
            if row.buy_shares:
                gross = row.buy_shares * row.price
                reserved = gross + max(cfg['min_fee'], gross * cfg['buy_cost']) + gross * slippage_rate
                reserved_total += reserved
                queue.append(create_order(date, next_date, row.ts_code, 'buy', row.buy_shares,
                                          references[row.ts_code], row.price, active_signal_date,
                                          'monthly_entry' if review else 'sale_cash_supplement',
                                          cash, equity, reserved, attempt))
        if reserved_total > cash + 1e-7:
            raise AccountingError('已冻结买单合计超过收盘现金；禁止预支任何卖款')
    eq = pd.DataFrame(daily).set_index('date')
    summary, eq['ret'] = daily_metrics(eq.equity, cfg['capital'], cfg['annual_days'])
    eq['nav'] = eq.equity / cfg['capital']
    summary.update(capital=cfg['capital'], top_n=top_n, n_trades=len(trades),
                   total_fees=sum(t['fee'] for t in trades), mean_invested=float((eq.stock_value/eq.equity).mean()),
                   missing_limit_events=sum(e['kind']=='missing_limit' for e in events),
                   stale_warnings=sum(e['kind']=='stale_valuation' for e in events),
                   delist_events=sum(e['kind']=='delist_writeoff' for e in events),
                   adjustment_metadata_events=sum(e['kind']=='adjustment_metadata_change' for e in events),
                   rights_declined_events=sum(e['kind']=='rights_declined' for e in events),
                   n_rebalances=monthly_execution_count, causal_config=cc,
                   matching_assumption='precommitted_DAY_limits_open_or_strict_intraday_cross_no_queue_guarantee',
                   corporate_action_known_date_provided='known_date' in actions,
                   st_announcement_date_provided='ann_date' in st_intervals or 'known_date' in st_intervals,
                   delisting_announcement_date_provided='delist_known_date' in basics)
    summary['execution_data_sufficient'] = bool(summary['missing_limit_events']==0 and summary['stale_warnings']==0
        and not any(e['kind']=='unexplained_adjustment' for e in events))
    trade_frame = pd.DataFrame(trades).reindex(columns=['date', 'signal_date', 'ts_code', 'side', 'shares', 'price', 'gross', 'fee', 'cash_after', 'order_id', 'decision_date', 'reference_date', 'reference_price', 'limit_price', 'execution_kind', 'reserved_cash', 'budget_source'] + (['slippage_cost'] if slippage_rate else []))
    event_frame = pd.DataFrame(events)
    for field in ['date', 'ts_code', 'kind', 'order_id', 'decision_date', 'reference_date', 'execution_date', 'signal_date', 'side', 'shares', 'reference_price', 'limit_price', 'cash_at_decision', 'cash_available', 'reserved_cash', 'budget_source', 'reason']:
        if field not in event_frame:
            event_frame[field] = pd.Series(index=event_frame.index, dtype=object)
    holding_frame = pd.DataFrame(holdings).reindex(columns=['date', 'ts_code', 'shares', 'available', 'price', 'value', 'stale_days'])
    return BacktestResult(eq, trade_frame, event_frame, holding_frame, summary)
