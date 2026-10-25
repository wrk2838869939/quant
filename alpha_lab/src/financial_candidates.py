"""公告时序财务候选连接；时序正确不代表原始快照具有完整修订历史。"""
import numpy as np
import pandas as pd


FIELDS = ('q_sales_yoy', 'q_ocf_to_sales')


def join_financial_candidates(opportunities, reports):
    """只使用复核日前公告，当前期与上年同期均取当时可见版本。

    返回未经源版本认证的数值候选及完整来源日期。晚到的旧报告不能
    覆盖较新报告，最新报告缺字段时不偷偷回填旧值。不会读取收益标签。
    """
    identities = ['date', 'ts_code']
    if opportunities.duplicated(identities).any():
        raise ValueError('机会身份重复')
    work = opportunities[identities].copy()
    work['date'] = pd.to_datetime(work.date, errors='raise').dt.normalize()
    data = reports[['ts_code', 'ann_date', 'end_date', *FIELDS]].copy()
    for col in ('ann_date', 'end_date'):
        data[col] = pd.to_datetime(data[col], errors='raise').dt.normalize()
    if work[identities].isna().any().any() or data.iloc[:, :3].isna().any().any():
        raise ValueError('身份或来源日期缺失')
    if (data.ann_date <= data.end_date).any():
        raise ValueError('公告必须晚于报告期末')
    keys = ['ts_code', 'ann_date', 'end_date']
    data = data.drop_duplicates()
    if data.duplicated(keys).any():
        raise ValueError('同公告同报告期数值冲突，不能按文件顺序选版本')
    for field in FIELDS:
        data[field] = pd.to_numeric(data[field], errors='raise')
        if np.isinf(data[field]).any():
            raise ValueError('财务指标不能为无穷值')
    by_code = {code: frame.sort_values(['ann_date', 'end_date']).to_dict('records')
               for code, frame in data.groupby('ts_code', sort=False)}
    rows = []
    for code, group in work.groupby('ts_code', sort=False):
        events = by_code.get(code, [])
        seen, cursor = {}, 0
        for opportunity in group.sort_values('date').itertuples():
            date = opportunity.date
            while cursor < len(events) and events[cursor]['ann_date'] < date:
                item = events[cursor]
                seen[item['end_date']] = item
                cursor += 1
            row = dict(date=date, ts_code=code, report_end=pd.NaT, report_ann=pd.NaT,
                       prior_end=pd.NaT, prior_ann=pd.NaT, report_age_days=np.nan,
                       q_sales_yoy=np.nan, q_ocf_to_sales=np.nan,
                       sales_growth_change_pp=np.nan, cash_conversion_change_pp=np.nan,
                       standard_quarter=False, numeric_available=False, source_history_verified=False)
            if seen:
                current = seen[max(seen)]
                previous_end = current['end_date'] - pd.DateOffset(years=1)
                prior = seen.get(previous_end)
                row.update(report_end=current['end_date'], report_ann=current['ann_date'],
                           report_age_days=(date-current['end_date']).days,
                           standard_quarter=bool(current['end_date'].is_quarter_end),
                           **{field: current[field] for field in FIELDS})
                if prior is not None and row['standard_quarter']:
                    row.update(prior_end=prior['end_date'], prior_ann=prior['ann_date'],
                               sales_growth_change_pp=current[FIELDS[0]]-prior[FIELDS[0]],
                               cash_conversion_change_pp=current[FIELDS[1]]-prior[FIELDS[1]])
                    row['numeric_available'] = bool(np.isfinite([
                        row['sales_growth_change_pp'], row['cash_conversion_change_pp']]).all())
            rows.append(row)
    result = pd.DataFrame(rows)
    return work.merge(result, on=identities, how='left', validate='one_to_one')


def require_verified_financial_history(frame):
    """阻止将仅有ann_date的现代快照候选误送入历史训练。"""
    if 'source_history_verified' not in frame or not frame.source_history_verified.eq(True).all():
        raise ValueError('财务源修订历史尚未认证，不能启动历史增量模型')
