"""邮件准备清单与逐单人工核对；没有邮件发送或券商下单能力。"""
from email.message import EmailMessage
from email.policy import SMTP

import pandas as pd

from shadow_account import ShadowAccount
from execution_clock import operation_config, timing, prepare_cutoff, check_proxy
from shadow_planner import ShadowError, fingerprint, number, validate_calendar


REASONS={'st':'ST状态触发退出','hold_false':'持有条件失效','single_stock_cap':'降低单股风险敞口',
         'selection_first':'月度选股/空位配置', 'monthly_entry':'月度新建仓',
         'sale_cash_supplement':'卖出实际到账后补位', 'portfolio_cash_cap':'降低组合股票仓位至风险预算',
         'targeted_restore':'风险解除后恢复此前实际减掉的旧仓'}


def bulletin(plan, metadata, now, config, names=None, backfill=False):
    """已冻结计划只作清单；异常代码单列，未知状态要求人工核对。"""
    config=operation_config(plan, config)
    names=names or {}
    stamp=pd.Timestamp(now)
    if stamp.tzinfo is None:raise ShadowError('播报时刻需时区')
    local=stamp.tz_convert('Asia/Shanghai')
    start=pd.Timestamp(plan['decision_date']+' '+config['prepare_after'],tz='Asia/Shanghai')
    end=pd.Timestamp(plan['execution_date']+' '+config['execution_end'],tz='Asia/Shanghai')
    if local<start:raise ShadowError('收盘输入尚未到准备时刻')
    historical=backfill or local>=end
    mode='历史核对，勿执行' if historical else '准备清单，逐项核对后手动执行'
    subject=config['email_subject_prefix']+' '+plan['execution_date']+' '+mode
    lines=[subject,'',f"计划编号：{plan['plan_id']}",
           f"信号依据：{plan['decision_date']}；操作窗口：{plan['execution_date']} "+config['execution_start']+'—'+config['execution_end']+'（北京时间）',
           f"现金：{plan['cash']:.2f}元。买入预算不计尚未收到的卖出款。",'',
           '操作清单：']
    if plan.get('session') == 'closing_auction':
        lines.extend(['盘中代理快照：'+plan['quote_captured_at']+'；不是最终收盘数据。',
            '14:57进入收盘集合竞价后提交限价委托；该时段不能撤单。不要提前在连续竞价提交。',
            '限价是可接受价格边界，并非成交价；能参与撮合的成交按收盘竞价价格，以券商回报为准。',
            '未成交资金保留现金，不假定成交，不自动延续到次日。'])
    if plan.get('session') == 'opening_continuous':
        lines.extend(['开盘执行方式：'+plan['execution_date']+' 09:30开盘后连续竞价提交限价委托，限在09:30—09:35窗口内完成。',
            '连续竞价委托可以撤单；限价是可接受价格边界，成交价以券商回报为准，不以播报价为成交依据。',
            '开盘价格可能跳空；限价不满足就不成交，不撤单改价追买。',
            '未成交资金保留现金，不假定成交；窗口结束后未提交的订单按未执行回报，不自动延续到当日晚些时候。'])
    if not plan['orders']:lines.append('无新增买卖计划；保持已核对持仓。')
    if plan.get('strategy') == 'monthly_predicted_risk':
        lines.extend([f"策略：月度风险预算；股票预算上限{plan['exposure_cap']:.0%}。",
            f"最近风险复核：{plan['risk_review_date']}；模型风险概率{plan['risk_probability']:.1%}（非收益预测）。",
            '恢复买入仅限此前实际减仓额度；未成交部分不计入。'])
    for order in plan['orders']:
        side='买入' if order['side']=='buy' else '卖出'
        relation='不高于' if order['side']=='buy' else '不低于'
        lines.append(f"{order['order_id']} | {order['ts_code']} {names.get(order['ts_code'],'')} | {side} {order['shares']}股 | 限价{relation}{order['limit_price']:.2f}元 | 待人工核对")
        lines.append('原因：'+REASONS.get(order['reason'],order['reason'])+
            (f"；含费预留{order['reserved_cash']:.2f}元" if order['side']=='buy' else ''))
    deferred=[d for d in plan.get('decisions',[]) if d.get('reason')=='next_day_corporate_action_or_ineligible']
    if deferred:
        lines.extend(['','暂缓股票：']+[d['ts_code']+'：已知状态变化/公司行动，先核对，不沿用原限价。' for d in deferred])
    lines.extend(['','执行前在券商客户端逐股确认：名称/ST状态、是否停牌、当日价格范围，以及是否除权或发生股数变化。',
        '有异常的股票先暂停，其他股票可分别核对。不要求事先自动处理所有罕见公司行动。',
        '不要自行扩大股数或追价；错过窗口后等待下一份计划。未执行、部分成交和全部成交均需反馈。',
        '成交回报填写实际股数、价格和费用；发送邮件或人工确认都不等于已经成交。'])
    if metadata.get('blockers'):
        lines.extend(['','数据提示：当天自动状态接口尚有缺项，需要在客户端核对上述项目。'])
    if not metadata.get('all_corporate_actions_covered'):
        lines.append('公司行动：自动覆盖不完整，采用逐股人工例外处理。')
    if historical:lines.extend(['','本文件仅用于历史核对/演练，不构成当前操作清单。'])
    body='\n'.join(lines)+'\n'
    message=EmailMessage(policy=SMTP)
    message['Subject']=subject
    message['Date']=stamp.to_pydatetime()
    message['Message-ID']='<'+fingerprint(dict(plan=plan['plan_id'],mode=mode))+'@alpha-lab.local>'
    message.set_content(body)
    return dict(subject=subject,body=body,eml=message.as_bytes(),historical=historical,
                mail_sent=False,live_submission=False)


class ManualAccount(ShadowAccount):
    def prepare(self, bundle, calendar, strategy_config, now):
        """人工模式冻结收盘计划；当日券商状态留给逐股人工核对。"""
        stamp=pd.Timestamp(now)
        if stamp.tzinfo is None:raise ShadowError('准备时刻缺时区')
        local=stamp.tz_convert('Asia/Shanghai')
        date=bundle['asof'];execution=validate_calendar(calendar,date)
        if strategy_config['execution_session'] == 'closing_auction':
            if str(local.date()) != date or not '14:30:00' <= local.strftime('%H:%M:%S') < '14:55:00':
                raise ShadowError('收盘计划仅在当日14:30至14:55准备')
            check_proxy(bundle.get('provenance',{}).get('receipt',{}),date,stamp)
            from execution_clock import snapshot_stamp
            if snapshot_stamp(self.state()['snapshot']) > local:
                raise ShadowError('账户快照来自未来')
            return self.plan(bundle,calendar,strategy_config)
        if local>=pd.Timestamp(execution+' '+prepare_cutoff(strategy_config['execution_session']),tz='Asia/Shanghai'):
            raise ShadowError('计划执行窗口已过期')
        if local<pd.Timestamp(date+' 21:00:00',tz='Asia/Shanghai'):
            raise ShadowError('须等待收盘输入更新')
        receipt=bundle.get('provenance',{}).get('receipt',{})
        if receipt.get('complete') is not True or receipt.get('asof')!=date:
            raise ShadowError('缺收盘输入完整性回执')
        received=pd.Timestamp(receipt.get('downloaded_at'))
        if pd.isna(received) or received.tzinfo is None or received>stamp:
            raise ShadowError('数据获取时刻不可验证')
        if received.tz_convert('Asia/Shanghai')<pd.Timestamp(date+' 15:00:00',tz='Asia/Shanghai'):
            raise ShadowError('数据回执早于该日收盘')
        if receipt.get('historical_backfill') or bundle.get('provenance',{}).get('historical_backfill'):
            raise ShadowError('历史回补包只能核对，不能产生当前订单')
        if strategy_config['strategy'] == 'monthly_predicted_risk':
            from monthly_risk_live import check_index_receipt
            check_index_receipt(receipt, date)
            if pd.Timestamp(receipt['market_index']['downloaded_at']) > stamp:
                raise ShadowError('指数数据获取时刻晚于计划时刻')
        return self.plan(bundle,calendar,strategy_config)

    def review_order(self, order_id, review):
        """一只股票核对失败不影响其他订单；人工确认仅改变可执行状态。"""
        if not review.get('source') or not review.get('evidence'):
            raise ShadowError('人工核对须有来源和记录')
        stamp=pd.Timestamp(review['timestamp'])
        if stamp.tzinfo is None:raise ShadowError('核对时刻缺时区')
        local=stamp.tz_convert('Asia/Shanghai')
        with self.transaction():
            order=self.order(order_id)
            window=timing(order['session'])
            if str(local.date())!=order['execution_date'] or not window['execution_start']<=local.strftime('%H:%M:%S')<window['execution_end']:
                raise ShadowError('核对已超出本计划集合竞价执行窗口')
            if order['status'] not in ['planned','ready']:raise ShadowError('订单已经推进，不重复核对')
            state=self.state()
            if review.get('account_id')!=state['snapshot']['account_id'] or review.get('positions')!=state['book_positions']:
                raise ShadowError('人工持仓与已核对账本不一致，先补齐实际流水')
            if review.get('unresolved_external_orders') is not False:
                raise ShadowError('存在未核对的其他委托')
            if review.get('ts_code')!=order['ts_code']:raise ShadowError('核对股票与订单不同')
            for key in ['status_checked','corporate_actions_checked','broker_accepts_limit']:
                if review.get(key) is not True:raise ShadowError('该股票人工核对尚未完成')
            if review.get('suspended') is not False or review.get('corporate_action') is not False:
                raise ShadowError('该股票状态/公司行动有异常，暂停该单并核对')
            if order['side']=='buy' and (review.get('buy_allowed') is not True or review.get('is_st') is not False):
                raise ShadowError('该股票不允许新买入')
            low=number(review.get('down_limit'),'down_limit');high=number(review.get('up_limit'),'up_limit')
            if not 0<low<=order['limit_price']<=high:raise ShadowError('原限价不在客户端显示的当日范围内')
            if order['side']=='sell' and number(review.get('sellable_shares'),'sellable_shares')<order['shares']:
                raise ShadowError('实际可卖股份不足')
            if order['side']=='buy' and number(review.get('available_cash'),'available_cash')<order['reserved_cash']:
                raise ShadowError('实际可用资金不足该单预留预算')
            if order['status']=='ready' and order.get('manual_review')!=fingerprint(review):
                raise ShadowError('同一订单核对内容改变，先检查原委托状态')
            if order['status']=='ready':return dict(status='duplicate',submitted=False)
            order.update(status='ready',manual_review=fingerprint(review))
            self._order(order);self._audit('manual_order_review_no_submission',dict(order_id=order_id,review=review))
            return dict(status='ready',submitted=False)

    def skip_order(self, order_id, evidence):
        """明确记录未提交，不伪造券商拒单或实际成交。"""
        if not evidence:raise ShadowError('未执行需要原因记录')
        with self.transaction():
            order=self.order(order_id)
            if order.get('manual_not_submitted'):
                if order.get('terminal_evidence')!=evidence:raise ShadowError('未提交回报冲突')
                return dict(status='duplicate',submitted=False)
            if order['status'] not in ['planned','ready'] or order['filled_shares']:
                raise ShadowError('有成交或已经终结，须按真实成交/撤单处理')
            order.update(status='cancelled',manual_not_submitted=True,terminal_evidence=evidence)
            self._order(order)
            self._audit('manual_not_submitted',dict(order_id=order_id,evidence=evidence))
            return dict(status='not_submitted',submitted=False)
