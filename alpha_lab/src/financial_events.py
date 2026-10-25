"""不可变的字段级披露事件；已核验原件不等于已收齐历史全部更正。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['ts_code', 'period_end', 'field', 'scope']
REQUIRED = KEY + ['published_date', 'value', 'unit', 'basis_id', 'source_path',
                  'source_sha256', 'pdf_page', 'source_token', 'source_url', 'event_role']
CUMULATIVE_FLOWS = {'cfo_ytd', 'revenue_ytd'}


def normalize_events(frame):
    """保留全部披露事实，生成内容地址；比较表调整前列不能再次作为新更新。"""
    if not set(REQUIRED).issubset(frame.columns):
        raise ValueError('披露事件缺少字段或原件出处')
    result = frame[REQUIRED].copy()
    for col in ['period_end', 'published_date']:
        result[col] = pd.to_datetime(result[col], errors='raise').dt.normalize()
    strings = [c for c in REQUIRED if c not in ['period_end', 'published_date', 'value', 'pdf_page']]
    if result[strings].isna().any().any() or any(
            not result[col].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
            for col in strings):
        raise ValueError('事件身份、口径或出处不能为空')
    if result[['period_end', 'published_date']].isna().any().any() or (result.published_date <= result.period_end).any():
        raise ValueError('事件日期缺失或披露不晚于报告期')
    result['value'] = pd.to_numeric(result.value, errors='raise').astype(float)
    if np.isinf(result.value).any():
        raise ValueError('不能记录无穷值')
    if not result.event_role.isin(['disclosed_value', 'historical_comparator']).all():
        raise ValueError('必须区分当次披露值与调整前历史比较列')
    if not result.pdf_page.map(lambda x: isinstance(x, (int, np.integer)) and x > 0).all():
        raise ValueError('PDF页码必须为正整数')
    if not result.source_sha256.str.fullmatch('[0-9a-f]{64}').all():
        raise ValueError('原件SHA256格式无效')
    ids = []
    for row in result.to_dict('records'):
        payload = {key: (None if pd.isna(value) else str(value)) for key, value in row.items()}
        ids.append(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
    result['event_id'] = ids
    if 'event_id' in frame and not frame.event_id.reset_index(drop=True).eq(pd.Series(ids)).all():
        raise ValueError('事件内容已变，不能沿用旧event_id')
    return result.drop_duplicates('event_id').sort_values(['published_date', 'event_id']).reset_index(drop=True)


def append_events(existing, additions):
    """追加不覆盖；原版本与新版本均留在账本。"""
    left = normalize_events(existing)
    right = normalize_events(additions)
    return normalize_events(pd.concat([left, right], ignore_index=True))


def verify_event_sources(events):
    """校验原件哈希与指明页的数字；数字所在行/列仍由人工原件审阅确认。"""
    import pymupdf
    records = normalize_events(events)
    for path, group in records.groupby('source_path'):
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if not group.source_sha256.eq(digest).all():
            raise ValueError('原件哈希不符')
        with pymupdf.open(stream=raw, filetype='pdf') as document:
            for row in group.itertuples():
                if row.pdf_page > len(document):
                    raise ValueError('原件页码越界')
                text = ''.join(document[row.pdf_page-1].get_text().split())
                if ''.join(row.source_token.split()) not in text:
                    raise ValueError('指定页没有记录中的原始数字')
    return len(records)


def asof_events(events, requests):
    """字段单独前向选版本；同日冲突输出未知，同日新公告不用于当日决策。

    history_complete始终为False：本函数只查询提交的事件集合，不宣称
    上游收齐全部公告，也不提供修改现有训练认证标识的捷径。
    """
    ledger = normalize_events(events)
    query = requests[['date', *KEY]].copy()
    query['date'] = pd.to_datetime(query.date, errors='raise').dt.normalize()
    query['period_end'] = pd.to_datetime(query.period_end, errors='raise').dt.normalize()
    if query.duplicated(['date', *KEY]).any() or query.isna().any().any():
        raise ValueError('查询身份重复或缺失')
    groups = {key: group for key, group in ledger[ledger.event_role.eq('disclosed_value')].groupby(KEY)}
    rows = []
    for request in query.to_dict('records'):
        group = groups.get(tuple(request[k] for k in KEY))
        row = dict(request, status='unavailable', value=np.nan, unit='', basis_id='',
                   published_date=pd.NaT, lineage='', history_complete=False)
        if group is not None:
            visible = group[group.published_date.lt(request['date'])]
            if len(visible):
                latest = visible[visible.published_date.eq(visible.published_date.max())]
                row.update(published_date=latest.published_date.iloc[0], lineage='|'.join(sorted(latest.event_id)))
                meanings = latest[['value', 'unit', 'basis_id']].drop_duplicates()
                if len(meanings) != 1:
                    row['status'] = 'same_day_conflict'
                else:
                    item = meanings.iloc[0]
                    row.update(value=item.value, unit=item.unit, basis_id=item.basis_id,
                               status='available' if pd.notna(item.value) else 'disclosed_missing')
        rows.append(row)
    return pd.DataFrame(rows)


def quarter_difference(current, previous):
    """累计差分需同公司、字段、单位、合并范围、已审阅会计口径和相邻季度。"""
    if current['status'] != 'available' or previous['status'] != 'available':
        raise ValueError('累计值不可用')
    if current['field'] not in CUMULATIVE_FLOWS or previous['field'] not in CUMULATIVE_FLOWS:
        raise ValueError('仅允许已登记累计流量，不能对资产余额等存量作单季差分')
    if current['basis_id'].strip().lower() in {'unknown', 'unverified'}:
        raise ValueError('累计差分的会计口径未核验')
    for key in ['ts_code', 'field', 'unit', 'scope', 'basis_id']:
        if current[key] != previous[key]:
            raise ValueError('不能拼接不同口径累计值')
    curr = pd.Timestamp(current['period_end'])
    prev = pd.Timestamp(previous['period_end'])
    if not curr.is_quarter_end or not prev.is_quarter_end or curr.year != prev.year or curr.to_period('Q')-1 != prev.to_period('Q'):
        raise ValueError('累计差分必须是同年相邻标准季度')
    return dict(value=current['value']-previous['value'], unit=current['unit'],
                available_after=max(current['published_date'], previous['published_date']),
                lineage=current['lineage']+'|'+previous['lineage'], history_complete=False)
