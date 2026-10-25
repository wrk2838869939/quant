"""开盘与收盘执行的时间契约；盘中代理始终不冒充正式收盘数据。"""
import pandas as pd


def local_stamp(value):
    from shadow_planner import ShadowError
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ShadowError('时刻必须有时区')
    return stamp.tz_convert('Asia/Shanghai')


def timing(session):
    from shadow_planner import ShadowError
    if session == 'closing_auction':
        return dict(prepare_after='14:30:00', prepare_before='14:55:00',
                    execution_start='14:57:00', execution_end='15:00:00')
    if session == 'opening_auction':
        return dict(prepare_after='21:00:00', execution_start='09:15:00', execution_end='09:25:00')
    if session == 'opening_continuous':
        return dict(prepare_after='21:00:00', execution_start='09:30:00', execution_end='09:35:00')
    raise ShadowError('未知执行时段')


def prepare_cutoff(session):
    """开盘类计划冻结截止：进入执行窗口后不再接受新的冻结；旧口径保持不变。"""
    from shadow_planner import ShadowError
    if session == 'opening_auction':
        return '09:25:00'
    if session == 'opening_continuous':
        return '09:30:00'
    if session == 'closing_auction':
        return '14:55:00'
    raise ShadowError('未知执行时段')


def operation_config(plan, config):
    return dict(config, **timing(plan.get('session', 'opening_auction')))


def check_proxy(receipt, date, now=None):
    """原始快照须在14:30后取得，全部报价与采集结束相差不超过120秒。"""
    from shadow_planner import ShadowError
    if (receipt.get('asof') != date or receipt.get('kind') != 'intraday_proxy'
            or receipt.get('complete') is not True or not receipt.get('source')
            or not receipt.get('sha256') or receipt.get('historical_backfill')):
        raise ShadowError('缺真实盘中快照及来源回执')
    start = local_stamp(receipt.get('quote_min_at'))
    end = local_stamp(receipt.get('downloaded_at'))
    latest = local_stamp(receipt.get('quote_max_at'))
    if (str(start.date()) != date or str(end.date()) != date
            or start.strftime('%H:%M:%S') < '14:30:00'
            or end.strftime('%H:%M:%S') >= '14:55:00'
            or not start <= latest <= end or (end-start).total_seconds() > 120):
        raise ShadowError('盘中快照过早、过晚、陈旧或存在未来报价')
    if now is not None:
        at = local_stamp(now)
        if at < end or (at-end).total_seconds() > 600:
            raise ShadowError('计算/计划超过快照有效期10分钟或获取时刻来自未来')
    index = receipt.get('market_index', {})
    if (index.get('code') != '000300.SH' or not index.get('sha256')
            or not start <= local_stamp(index.get('timestamp')) <= end):
        raise ShadowError('缺同期沪深300实时快照')


def snapshot_stamp(snapshot):
    """老式无时刻快照仍按当天结束处理，不能被拿来支持当日成交。"""
    return local_stamp(snapshot.get('timestamp', snapshot['asof']+'T23:59:59+08:00'))


def event_after_snapshot(item, snapshot):
    from shadow_planner import ShadowError
    if item['date'] > snapshot['asof']:
        return
    if (item['date'] != snapshot['asof'] or not snapshot.get('phase') == 'intraday'
            or not item.get('timestamp')):
        raise ShadowError('流水已在已核对快照时间内')
    at = local_stamp(item['timestamp'])
    if str(at.date()) != item['date'] or at <= snapshot_stamp(snapshot):
        raise ShadowError('流水时刻不在盘中账户快照之后')
