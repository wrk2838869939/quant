"""独立候选生命周期与严格成熟标签预测；不模拟资金或真实交易。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def candidate_spells(signals, holds, top_n):
    """每股最多一个研究生命周期，未知持有许可不作为退出。"""
    if not signals.index.equals(holds.index) or not signals.columns.equals(holds.columns):
        raise ValueError('候选与持有许可未对齐')
    active = {}
    events, entries = [], []
    for date, row in signals.iterrows():
        for code in list(active):
            if pd.notna(holds.at[date, code]) and holds.at[date, code] == 0:
                events.append(dict(date=date, ts_code=code, entry_date=active.pop(code)))
        selected = row.dropna().sort_values(ascending=False, kind='stable').head(top_n)
        for code in selected.index:
            if code not in active:
                if holds.at[date, code] != 1:
                    raise ValueError('入场信号与持有许可矛盾')
                active[code] = date
                entries.append(dict(date=date, ts_code=code))
    return (pd.DataFrame(events, columns=['date', 'ts_code', 'entry_date']),
            pd.DataFrame(entries, columns=['date', 'ts_code']),
            pd.DataFrame([dict(ts_code=c, entry_date=d) for c, d in active.items()]))


def annual_fit(frame, year, horizon, target, cfg):
    """只使用年初已成熟的训练标签，缺特征预测回退历史均值。"""
    cutoff = pd.Timestamp(f'{year}-01-01')
    train = frame[frame.date.lt(cutoff) & frame[f'label_end_{horizon}'].lt(cutoff)
                  & frame[f'{target}_{horizon}'].notna() & frame.features_complete].copy()
    mean = float(train[f'{target}_{horizon}'].mean()) if len(train) else 0.0
    model = None
    threshold = mean
    if len(train) >= cfg['model']['min_train_events'] and train.date.nunique() >= cfg['model']['min_train_dates']:
        model = make_pipeline(StandardScaler(), Ridge(alpha=cfg['model']['alpha']))
        model.fit(train[cfg['features']], train[f'{target}_{horizon}'])
        threshold = float(np.median(model.predict(train[cfg['features']])))
    return train, model, mean, threshold


def predict_frame(frame, model, mean, features):
    result = np.full(len(frame), mean)
    valid = frame.features_complete.to_numpy()
    if model is not None and valid.any():
        result[valid] = model.predict(frame.loc[valid, features])
    return result
