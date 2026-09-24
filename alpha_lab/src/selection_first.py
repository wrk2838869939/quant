"""研究用先选名单、后定股数；排名与真实收益预测承担不同职责。"""
import math

import numpy as np
import pandas as pd

from retail_rebalance import plan_rebalance


def plan_selection_first(candidates, *, priorities, cash, buy_rate, sell_rate,
                         min_fee, forecast_kind, max_positions=None,
                         slippage_rate=0., solver_relative_gap=.001):
    """保留旧仓，按显式优先级预约新仓最小净利为正的合法股数，再优化。

    priorities 只决定新仓先后；盈利门槛与股数目标始终使用 expected_return。
    同分按股票代码排序，不强制满仓，也不按平均席位预算排除贵股。
    旧仓只按 max_notional 必要减仓；明确退出应传入零 max_notional。
    min_target_shares 与不可卖股数仍为硬下界。
    """
    required = ['ts_code', 'price', 'expected_return', 'shares', 'available',
                'min_buy', 'buy_step', 'sell_step', 'max_notional',
                'can_buy', 'can_sell']
    if forecast_kind != 'expected_simple_return' or not set(required).issubset(candidates):
        raise ValueError('需要真实简单收益与完整交易约束，排名不能替代收益')
    x = candidates.copy().sort_values('ts_code', kind='stable').reset_index(drop=True)
    if x.ts_code.duplicated().any():
        raise ValueError('股票代码必须唯一')
    if 'min_target_shares' not in x:
        x['min_target_shares'] = 0
    numeric = x[required[1:] + ['min_target_shares']].to_numpy(float)
    scalars = [cash, buy_rate, sell_rate, min_fee, slippage_rate]
    if not np.isfinite(numeric).all() or not np.isfinite(scalars).all() or min(scalars) < 0:
        raise ValueError('资金、费率与交易字段必须为有效数值')
    for field in ['shares', 'available', 'min_buy', 'buy_step', 'sell_step', 'min_target_shares']:
        if ((x[field] < 0) | (x[field] != np.floor(x[field]))).any():
            raise ValueError('股数和步长必须为非负整数')
    if ((x.price <= 0).any() or (x[['min_buy', 'buy_step', 'sell_step']] <= 0).any().any()
            or (x.available > x.shares).any() or (x.max_notional < 0).any()
            or (x.expected_return < -1).any() or (x.min_target_shares > x.shares).any()
            or not x.can_buy.isin([0, 1]).all() or not x.can_sell.isin([0, 1]).all()):
        raise ValueError('价格、股数、风险上限或交易许可非法')
    if max_positions is not None and (not np.isfinite(max_positions)
            or int(max_positions) != max_positions or max_positions < 0):
        raise ValueError('持仓上限非法')
    position_limit = len(x) if max_positions is None else int(max_positions)
    original_caps = x.max_notional.copy()
    original_minima = x.min_target_shares.copy()
    reserved_cash = float(cash)
    retained = []
    old_decisions = []
    for i, row in x[x.shares.gt(0)].iterrows():
        available = int(row.available) if row.can_sell else 0
        locked = int(row.shares) - available
        cap_shares = max(locked, min(int(row.shares), int(math.floor(row.max_notional / row.price + 1e-9))))
        required_sale = int(row.shares) - cap_shares
        step = int(row.sell_step)
        odd = available % step
        possible = [offset + max(0, math.ceil((required_sale - offset) / step)) * step
                    for offset in {0, odd}]
        possible = [q for q in possible if required_sale <= q <= available
                    and row.shares - q >= row.min_target_shares]
        if not possible:
            raise ValueError('旧仓保留股数与必要风险减仓冲突')
        sold = min(possible)
        target = int(row.shares) - sold
        fee = max(sold * row.price * sell_rate, min_fee) if sold else 0.
        reserved_cash += sold * row.price * (1 - slippage_rate) - fee
        x.loc[i, 'can_buy'] = False
        x.loc[i, 'min_target_shares'] = target
        # 同时固定上下界，旧仓不因排名或新股的收益预测而被替换。
        x.loc[i, 'max_notional'] = target * row.price
        if target:
            retained.append(row.ts_code)
        old_decisions.append(dict(ts_code=row.ts_code, old_shares=int(row.shares),
                                  target_shares=target, sell_shares=sold))
    if len(retained) > position_limit:
        raise ValueError('不可替换旧仓已超过持仓上限')
    if reserved_cash < -1e-8:
        raise ValueError('必要卖出的净佣金缺口超过可用现金')
    reservation_start = reserved_cash
    selected = []
    reserved_shares = {}
    decisions = []
    new = x[x.shares.eq(0)].copy()
    new['priority'] = [priorities.get(code, np.nan) for code in new.ts_code]
    new = new.sort_values(['priority', 'ts_code'], ascending=[False, True], kind='stable')
    for i, row in new.iterrows():
        notional = float(row.min_buy * row.price)
        buy_fee = max(notional * buy_rate, min_fee)
        exit_fee = max(notional * sell_rate, min_fee)
        entry_cash = notional * (1 + slippage_rate) + buy_fee
        net_profit = notional * row.expected_return - buy_fee - exit_fee - 2 * notional * slippage_rate
        economic_shares = 0
        economic_cash = np.nan
        economic_profit = np.nan

        def economics(steps):
            quantity = int(row.min_buy + steps * row.buy_step)
            value = quantity * row.price
            buy_fee = max(value * buy_rate, min_fee)
            profit = value * (row.expected_return - 2 * slippage_rate) - buy_fee - max(value * sell_rate, min_fee)
            return quantity, value * (1 + slippage_rate) + buy_fee, profit

        if not np.isfinite(row.priority):
            reason = 'missing_priority'
        elif not row.can_buy:
            reason = 'buy_not_allowed'
        elif notional > row.max_notional + 1e-8:
            reason = 'minimum_lot_exceeds_stock_cap'
        elif len(retained) + len(selected) >= position_limit:
            reason = 'position_limit'
        elif entry_cash > reserved_cash + 1e-8:
            reason = 'insufficient_cash_for_minimum_lot'
        elif row.expected_return <= buy_rate + sell_rate + 2 * slippage_rate:
            reason = 'nonpositive_net_margin'
        else:
            # 买费=max(比例费,最低费)，两个现金上界都须满足；不使用平均席位预算。
            max_value = min(row.max_notional,
                            (reserved_cash - min_fee) / (1 + slippage_rate),
                            reserved_cash / (1 + slippage_rate + buy_rate))
            high = int(math.floor((max_value / row.price - row.min_buy) / row.buy_step + 1e-9))
            if high < 0 or economics(high)[2] <= 0:
                reason = 'no_positive_net_profit_within_cap_and_cash'
            else:
                # 已确认边际净收益为正，股数增加时净利单调，二分最小经济可行块。
                low = 0
                while low < high:
                    middle = (low + high) // 2
                    if economics(middle)[2] > 0:
                        high = middle
                    else:
                        low = middle + 1
                economic_shares, economic_cash, economic_profit = economics(low)
                reason = 'selected'
                selected.append(row.ts_code)
                reserved_shares[row.ts_code] = economic_shares
                reserved_cash -= economic_cash
        x.loc[i, 'can_buy'] = reason == 'selected'
        decisions.append(dict(ts_code=row.ts_code, priority=row.priority, reason=reason,
                              expected_return=row.expected_return,
                              minimum_lot_net_profit=net_profit,
                              minimum_lot_entry_cash=entry_cash,
                              reserved_shares=economic_shares,
                              reserved_entry_cash=economic_cash,
                              reserved_net_profit=economic_profit,
                              reservation_cash_after=reserved_cash))
    result = plan_rebalance(x, cash=cash, buy_rate=buy_rate, sell_rate=sell_rate,
                           min_fee=min_fee, forecast_kind=forecast_kind,
                           max_positions=max_positions, slippage_rate=slippage_rate,
                           solver_relative_gap=solver_relative_gap,
                           required_new_codes=selected,
                           required_new_shares=reserved_shares)
    plan = result['plan']
    # 保留调用方原始风险口径，内部固定旧仓的约束不伪装成新的风控参数。
    plan['max_notional'] = original_caps.to_numpy()
    plan['min_target_shares'] = original_minima.to_numpy()
    locked = plan.shares - np.where(plan.can_sell, plan.available, 0)
    plan['unavoidable_locked_excess'] = np.maximum(0., locked * plan.price - plan.max_notional)
    result.update(selected_new_codes=selected, selected_codes=selected,
                  retained_codes=retained, selection_audit=pd.DataFrame(decisions),
                  selection_decisions=pd.DataFrame(decisions),
                  old_position_decisions=pd.DataFrame(old_decisions),
                  reserved_shares=reserved_shares,
                  cash_after_required_sales=reservation_start,
                  cash_after_minimum_reservations=reserved_cash)
    return result
