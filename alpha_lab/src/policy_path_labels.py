"""冻结政策的统一观察窗研究标签；成交许可由外部真实交易资料提供。"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PolicyPathResult:
    """标签与可逐日前缀核验的证据、待卖和净值轨迹。"""
    labels: pd.DataFrame
    marked_nav: np.ndarray
    failure_counts: np.ndarray
    pending_after_open: np.ndarray


def policy_path_labels(adjusted_open, adjusted_close, review_mask, trend, observable,
                       entry_allowed, sell_allowed, delist_step, *,
                       observation_horizon, policy, confirmations, fixed_exit_step=None,
                       risk_exit_review=None, risk_exit_open=None):
    """计算统一成熟终点的无费用、后复权单位路径代理。

    Parameters
    ----------
    adjusted_open : ndarray, shape (observed_steps + 1, n)
        第0行是信号次日开盘；第H行是原信号T+H+1开盘。
    adjusted_close : ndarray, shape (observed_steps, n)
        第j行是第j次开盘之后的收盘；终点只观察开盘，不读未来收盘。
    review_mask, trend, observable : ndarray
        与收盘对齐的月度复核、趋势、可观察状态。未知沿用原规则清零计数。
    entry_allowed : ndarray, shape (n,)
        1=已验证可买，0=不可买，NaN=证据不足。只尝试第0行，不补造延期买入。
    sell_allowed : ndarray, shape (observed_steps + 1, n)
        1=已验证开盘可卖，0=不可卖，NaN=证据不足。不能由估值是否存在替代。
        调用方须审计原价、成交量、官方涨跌停及其既定交易资格。
    delist_step : ndarray, shape (n,)
        首个退市生效的开盘索引；无事件为inf。持有期间退市直接归零。
    observation_horizon : int
        统一观察上限；研究登记120。不足上限的前缀仍返回事件，但成熟标签为NaN。
    policy : {'fixed', 'monthly_two_failures'}
        固定退出或月度失败确认；没有C1记忆保留、未知超时、择优卖点或再投资。
    confirmations : int
        冻结的失败确认次数，研究登记2。
    fixed_exit_step : int, optional
        固定退出在第s次开盘尝试，s-1收盘提交；研究对照60/120。
    risk_exit_review : ndarray, optional
        与收盘对齐、当时已知的ST风险，仅在review_mask为真时产生退出请求。
        保留为显式收盘风险接口；实际ST复核应使用risk_exit_open，本输入全false。
    risk_exit_open : ndarray, optional
        与开盘对齐的已知ST退出事件，由调用方用前日月度复核和当日ST状态构造。
        开盘先提交，再在同次开盘尝试卖；各方案共用，退市先于ST请求。

    Notes
    -----
    默认假设合格入场信号已明确恢复趋势，入场后失败计数从0开始。
    三时钟分别是首次失败复核收盘、不可撤销退出请求时点、实际许可开盘。
    ST开盘请求单列exit_request_open_step，不能编造前一日收盘请求。
    第0日不可卖，符合T+1；缺价、停牌、跌停或交易证据不足均不得标记成交。
    后复权价格比不是整手账户的实际现金收益，未含费用或独立公司行动账本。
    不足共同上限时，即便先已退出也不允许提前生成成熟监督标签。
    """
    op = np.asarray(adjusted_open, dtype=float)
    cl = np.asarray(adjusted_close, dtype=float)
    if op.ndim != 2 or len(op) < 1:
        raise ValueError('开盘矩阵必须有入场行')
    steps, n = len(op)-1, op.shape[1]
    review = np.asarray(review_mask)
    good = np.asarray(trend, dtype=float)
    known = np.asarray(observable)
    buy = np.asarray(entry_allowed, dtype=float)
    sell = np.asarray(sell_allowed, dtype=float)
    death = np.asarray(delist_step, dtype=float)
    risk = np.zeros_like(cl,dtype=bool) if risk_exit_review is None else np.asarray(risk_exit_review)
    open_risk = np.zeros_like(op,dtype=bool) if risk_exit_open is None else np.asarray(risk_exit_open)
    if (cl.shape != (steps,n) or review.shape != cl.shape or good.shape != cl.shape
            or known.shape != cl.shape or buy.shape != (n,) or sell.shape != op.shape
            or death.shape != (n,) or risk.shape != cl.shape or open_risk.shape != op.shape):
        raise ValueError('数组形状不一致')
    if (not isinstance(observation_horizon,int) or isinstance(observation_horizon,bool)
            or observation_horizon < 1 or steps > observation_horizon
            or not isinstance(confirmations,int) or isinstance(confirmations,bool) or confirmations < 1):
        raise ValueError('观察窗或确认次数非法')
    if policy not in {'fixed','monthly_two_failures'}:
        raise ValueError('未知冻结政策')
    if policy == 'fixed' and (not isinstance(fixed_exit_step,int)
            or isinstance(fixed_exit_step,bool) or not 1 <= fixed_exit_step <= observation_horizon):
        raise ValueError('固定退出时钟非法')
    if policy != 'fixed' and fixed_exit_step is not None:
        raise ValueError('月频退出不能混入固定到期强制卖出')
    for name, value in [('review_mask',review),('observable',known),('risk_exit_review',risk),
                        ('risk_exit_open',open_risk)]:
        if not np.isin(value,[0,1]).all():
            raise ValueError(f'{name}必须是明确布尔状态')
    for name, value in [('entry_allowed',buy),('sell_allowed',sell)]:
        if not (np.isnan(value) | np.isin(value,[0.,1.])).all():
            raise ValueError(f'{name}只能为0、1或未知')
    review, known, risk = review.astype(bool), known.astype(bool), risk.astype(bool)
    open_risk = open_risk.astype(bool)
    if not np.isin(good[known & review],[0.,1.]).all():
        raise ValueError('可观察复核必须有明确趋势')
    if np.isnan(death).any() or ((np.isfinite(death)) & ((death < 0) | (death != np.floor(death)))).any():
        raise ValueError('退市时钟非法')

    valid_entry_price = np.isfinite(op[0]) & (op[0] > 0)
    entered = (buy == 1) & valid_entry_price & (death > 0)
    active = entered.copy()
    pending = np.zeros(n,dtype=bool)
    cash_value = np.ones(n)
    mark = np.ones(n)
    counts = np.zeros(n,dtype=int)
    first_failure = np.full(n,np.nan)
    requested = np.full(n,np.nan)
    requested_open = np.full(n,np.nan)
    earliest = np.full(n,np.nan)
    filled = np.full(n,np.nan)
    written_off = np.full(n,np.nan)
    request_mark = np.full(n,np.nan)
    request_reason = np.full(n,'',dtype=object)
    fill_return = np.full(n,np.nan)
    missing_marks = np.zeros(n,dtype=int)
    blocked_days = np.zeros(n,dtype=int)
    execution_unknown_days = np.zeros(n,dtype=int)
    unknown_reviews = np.zeros(n,dtype=int)
    failure_trace = np.zeros((steps,n),dtype=int)
    pending_trace = np.zeros((steps+1,n),dtype=bool)
    # 首行入场净值，随后observed_steps个收盘，最后一个开盘标记。
    nav = np.ones((steps+2,n))

    for step in range(steps+1):
        die = active & (death <= step)
        written_off[die] = step
        cash_value[die], mark[die] = 0., 0.
        active[die], pending[die] = False, False
        quote_ok = np.isfinite(op[step]) & (op[step] > 0)
        st_open_request = active & ~pending & open_risk[step]
        requested_open[st_open_request] = step
        earliest[st_open_request] = max(step,1)
        request_reason[st_open_request] = 'st'
        # 开盘请求的价格观察用当时开盘，缺失则仅保留旧估值，不冒充成交。
        request_mark[st_open_request] = np.where(quote_ok[st_open_request],
            op[step,st_open_request]/op[0,st_open_request],mark[st_open_request])-1
        pending[st_open_request] = True
        can_fill = active & pending & (step >= 1) & (sell[step] == 1) & quote_ok
        fill_return[can_fill] = op[step,can_fill]/op[0,can_fill]-1
        cash_value[can_fill] = fill_return[can_fill]+1
        mark[can_fill] = cash_value[can_fill]
        filled[can_fill] = step
        active[can_fill], pending[can_fill] = False, False
        waiting = active & pending & (step >= 1)
        blocked_days += waiting
        # 明确停牌/跌停是已知不可成交，不得按未来停牌把训练样本归为数据未知。
        execution_unknown_days += waiting & (np.isnan(sell[step])
            | ((sell[step] == 1) & ~quote_ok))
        pending_trace[step] = pending
        if step == steps:
            quoted = active & quote_ok
            mark[quoted] = op[step,quoted]/op[0,quoted]
            missing_marks += active & ~quote_ok
            nav[-1] = np.where(active,mark,cash_value)
            break

        quoted = active & np.isfinite(cl[step]) & (cl[step] > 0)
        mark[quoted] = cl[step,quoted]/op[0,quoted]
        missing_marks += active & ~quoted
        nav[step+1] = np.where(active,mark,cash_value)
        reviewed = active & review[step]
        unknown_reviews += reviewed & ~known[step]
        failed = reviewed & known[step] & (good[step] == 0)
        first_failure[failed & np.isnan(first_failure)] = step
        # 原规则：未知也清零；已提交退出请求另存，不受计数清零影响。
        counts = np.where(reviewed, np.where(failed,counts+1,0), counts)
        failure_trace[step] = counts
        trigger = (active & (step == fixed_exit_step-1) if policy == 'fixed'
                   else reviewed & (counts >= confirmations))
        st_exit = reviewed & risk[step]
        trigger |= st_exit
        new_request = trigger & ~pending
        requested[new_request] = step
        earliest[new_request] = step+1
        request_mark[new_request] = mark[new_request]-1
        request_reason[new_request] = np.where(st_exit[new_request],'st',
            'fixed_horizon' if policy == 'fixed' else 'trend_failure')
        pending[new_request] = True

    mature = steps == observation_horizon
    final = nav[-1]-1
    mae = nav.min(axis=0)-1
    labels = pd.DataFrame(dict(entered=entered,no_entry=~entered,
        entry_execution_unknown=np.isnan(buy),entry_blocked=buy==0,
        entry_price_missing=~valid_entry_price,
        mature=np.repeat(mature,n),observed_steps=np.repeat(steps,n),
        first_failure_review_step=first_failure,exit_request_close_step=requested,
        exit_request_open_step=requested_open,
        exit_request_reason=request_reason,
        earliest_sell_open_step=earliest,actual_sell_open_step=filled,
        delist_writeoff_step=written_off,request_mark_return=request_mark,
        filled_return_process_only=fill_return,
        request_to_fill_intervals=filled-earliest,
        cash_occupation_intervals=np.where(np.isfinite(filled),filled,
            np.where(np.isfinite(written_off),written_off, np.where(entered,steps,0))),
        pending_exit=active & pending,open_at_cutoff=active,
        no_request_administrative_censor=active & ~pending,
        terminal_is_unsettled_mark=active,
        terminal_quote_missing=active & ~(np.isfinite(op[-1]) & (op[-1] > 0)),
        missing_hold_marks=missing_marks,blocked_sell_days=blocked_days,
        execution_unknown_sell_days=execution_unknown_days,unknown_reviews=unknown_reviews,
        label_return=np.where(mature,final,np.nan),label_mae=np.where(mature,mae,np.nan),
        settled_label_return=np.where(mature & ~active,final,np.nan),
        observed_marked_return_process_only=final,
        is_actual_account_net_return=np.zeros(n,dtype=bool)))
    return PolicyPathResult(labels,nav,failure_trace,pending_trace)
