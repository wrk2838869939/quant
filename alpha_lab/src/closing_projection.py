"""盘中价格投影：只追加实时行，不覆盖历史决策；不估算最后半小时成交量。"""
import re

import numpy as np
import pandas as pd

from execution_clock import check_proxy, local_stamp
from shadow_planner import ShadowError, fingerprint


def parse_sina(text):
    """保留源报价时刻，股票成交量使用股；拒绝无时间的旧接口降级。"""
    rows = []
    for symbol, payload in re.findall(r'hq_str_([a-z]{2}\d{6})="([^"]*)"', text):
        fields = payload.split(',')
        if len(fields) < 32:
            continue
        rows.append(dict(ts_code=symbol[2:]+'.'+symbol[:2].upper(), name=fields[0],
            pre_close=float(fields[2]), close=float(fields[3]), vol_share=float(fields[8]),
            timestamp=fields[30]+'T'+fields[31]+'+08:00'))
    return pd.DataFrame(rows)


def project(history, index_history, basis, quotes, receipt, calendar, suspended=()):
    """basis为截至前一交易日已知的价格、复权系数和总/流通股本，单位股。"""
    date = receipt['asof']; day = pd.Timestamp(date)
    check_proxy(receipt, date)
    if (history.date.max() >= day or index_history.index.max() >= day
            or basis.date.max() >= day):
        raise ShadowError('投影输入含当日或未来正式日线/股本，不可用于盘中决策')
    prior = pd.Timestamp(calendar['sessions'][calendar['sessions'].index(date)-1])
    if history.date.max() != prior or index_history.index.max() != prior:
        raise ShadowError('盘中历史未连续更新到前一交易日')
    if quotes.ts_code.duplicated().any() or basis.ts_code.duplicated().any():
        raise ShadowError('实时行情或股本依据重复')
    if fingerprint(quotes.to_dict('records')) != receipt['sha256']:
        raise ShadowError('实时快照内容与回执哈希不同')
    q = quotes.set_index('ts_code'); b = basis.set_index('ts_code')
    required = set(b.index) - set(suspended)
    if set(suspended) - set(b.index) or set(q.index) != required | {'000300.SH'}:
        raise ShadowError('全A实时截面覆盖不完整或含未知代码，不按静默删股继续')
    stamps = q.timestamp.map(local_stamp)
    if (stamps.min() != local_stamp(receipt['quote_min_at'])
            or stamps.max() != local_stamp(receipt['quote_max_at'])):
        raise ShadowError('报价时间范围与回执不同')
    idx = q.loc['000300.SH']
    if (fingerprint(idx.to_dict()) != receipt['market_index']['sha256']
            or idx.timestamp != receipt['market_index']['timestamp']):
        raise ShadowError('指数原始报价与回执不同')
    if abs(float(idx.pre_close)-float(index_history.iloc[-1])) > .02 and not receipt.get('prior_index_is_proxy'):
        raise ShadowError('实时指数昨收与历史不接续')
    # 市值和换手率不用当日daily_basic；不把尚未发生的成交量外推成事实。
    active = q.loc[sorted(required)].copy()
    reference = b.loc[active.index]
    if not np.isfinite(active[['pre_close','close','vol_share']]).all().all() or (active.close<=0).any() or (active.vol_share<=0).any():
        raise ShadowError('非停牌股票报价或累计量缺失/非法')
    for key in ['close','hfq_close','total_mv_yuan','total_share','float_share']:
        if not np.isfinite(reference[key]).all() or not reference[key].gt(0).all():
            raise ShadowError('盘前股本/复权基准缺失，禁止用当日最终值补齐')
    if reference.float_share.gt(reference.total_share).any():
        raise ShadowError('流通股本大于总股本，检查单位')
    # 发生除权时昨收基准改变，必须先明确换算，不沿用昨天价格/股本。
    changed = (active.pre_close-reference.close).abs() > .011
    if changed.any():
        raise ShadowError('昨收/除权基准变化，需核对：'+','.join(active.index[changed]))
    active['date'] = day
    active['hfq_close'] = active.close * reference.hfq_close / reference.close
    active['total_mv_yuan'] = reference.total_mv_yuan * active.close / reference.close
    active['turnover_rate'] = active.vol_share / reference.float_share * 100
    current = active.reset_index()
    projected = pd.concat([history, current[history.columns]], ignore_index=True)
    index = pd.concat([index_history, pd.Series([float(idx.close)], index=pd.DatetimeIndex([day]))])
    if not np.isfinite(index.iloc[-1]) or index.iloc[-1] <= 0:
        raise ShadowError('指数实时价格非法')
    return projected, index, current[['date','ts_code','close','vol_share']]


def proxy_receipt(quotes, date, downloaded_at, source):
    stamps = quotes.timestamp.map(local_stamp)
    idx = quotes.set_index('ts_code').loc['000300.SH']
    result = dict(asof=date, kind='intraday_proxy', complete=True, source=source,
        downloaded_at=str(downloaded_at), quote_min_at=stamps.min().isoformat(),
        quote_max_at=stamps.max().isoformat(), sha256=fingerprint(quotes.to_dict('records')),
        expected_rows=len(quotes)-1,
        market_index=dict(code='000300.SH', timestamp=idx.timestamp, sha256=fingerprint(idx.to_dict())))
    check_proxy(result, date)
    return result


def mark_account(broker, quotes, receipt, now):
    """数量/现金来自券商，估值统一到冻结快照；保留券商原件摘要。"""
    from copy import deepcopy
    from shadow_planner import validate_snapshot
    check_proxy(receipt, receipt['asof'], now)
    at=local_stamp(broker.get('timestamp'))
    if (not broker.get('evidence') or not broker.get('source')
            or not local_stamp(receipt['downloaded_at']) <= at <= local_stamp(now)
            or broker.get('asof') != receipt['asof']):
        raise ShadowError('需要行情获取后核对的当日券商持仓、现金及原始凭据')
    if fingerprint(quotes.to_dict('records')) != receipt['sha256']:
        raise ShadowError('账户估值行情哈希不同')
    snapshot=deepcopy(broker)
    snapshot.update(phase='intraday', broker_snapshot_hash=fingerprint(broker),
        broker_reported_equity=broker.get('equity'), valuation_quote_hash=receipt['sha256'])
    q=quotes.set_index('ts_code')
    equity=snapshot['cash']+snapshot['receivables']
    for code, position in snapshot['positions'].items():
        if code not in q.index:raise ShadowError('持仓缺实时估值，先核对停牌/缺数')
        position['close']=float(q.at[code,'close'])
        equity+=position['shares']*position['close']
    snapshot['equity']=equity
    validate_snapshot(snapshot)
    return snapshot
