"""多窗口首次信号按回落过程因果去重。"""
import pandas as pd


def merge_signals(signals):
    """依时间保留首次信号；前次确认后出现新峰值，才允许新事件。

    同日按预设窗口顺序确定代表，不以未来收益或更晚信号选代表。
    后续出现的旧峰回落归入最近已确认事件，不计为新机会。
    """
    frame = signals.sort_values(['date', 'priority', 'sector_code', 'event_id']).copy()
    previous, rows = {}, []
    for r in frame.to_dict('records'):
        key = (r['sector_code'], r['segment_id'])
        old = previous.get(key)
        accepted = old is None or r['peak_date'] > old['date']
        if accepted:
            group_id = f"{r['sector_code']}_s{r['segment_id']}_{r['date']:%Y%m%d}"
            previous[key] = dict(date=r['date'], group_id=group_id)
        else:
            group_id = old['group_id']
        rows.append({**r, 'accepted': accepted, 'merged_event_id': group_id})
    return pd.DataFrame(rows).reset_index(drop=True)
