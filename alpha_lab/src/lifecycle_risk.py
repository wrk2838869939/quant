"""从候选完整生命周期采样可知状态，并将风险估计转换为原上限的折扣。"""
import numpy as np
import pandas as pd


def lifecycle_reviews(signals, holds, top_n):
    active, rows = {}, []
    for date, row in signals.iterrows():
        snapshot = {}
        for code in list(active):
            state = holds.at[date, code]
            kind = 'exit' if pd.notna(state) and state == 0 else 'hold'
            snapshot[code] = dict(date=date, ts_code=code, entry_date=active[code], phase=kind)
            if kind == 'exit':
                active.pop(code)
        for code in row.dropna().sort_values(ascending=False, kind='stable').head(top_n).index:
            if code not in active:
                if holds.at[date, code] != 1:
                    raise ValueError('入场与持有许可不一致')
                active[code] = date
                snapshot[code] = dict(date=date, ts_code=code, entry_date=date, phase='entry')
        rows.extend(snapshot.values())
    return pd.DataFrame(rows).sort_values(['date', 'ts_code']).reset_index(drop=True)


def risk_discount(prediction, reference_risk):
    """MAE为非正数；风险不高于历史中位数时不加仓，高于时按比例缩小。"""
    if not np.isfinite(reference_risk) or reference_risk <= 0:
        raise ValueError('历史参考风险必须为正')
    values = np.asarray(prediction, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('未知风险须明确回退，不隐含当作零风险')
    risk = np.maximum(-values, reference_risk)
    return reference_risk / risk
