"""按历史板块通过事件训练个股截面排序，禁止未来标签参与。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def cross_section_features(frame, cfg):
    """每个已知事件独立排序并去均值，未来事件不改变历史截面。"""
    result = frame[['event_id', 'date', 'ts_code']].copy()
    columns = []
    for field, higher_better in cfg['rank_features'].items():
        values = frame[field].replace([np.inf, -np.inf], np.nan)
        rank = values.groupby(frame.event_id).rank(method='average', pct=True, ascending=higher_better)
        name = field + '_rank'
        result[name], result[field + '_missing'] = rank.fillna(.5), values.isna().astype(float)
        columns.extend([name, field + '_missing'])
    for field in cfg['binary_features']:
        result[field + '_flag'] = frame[field].astype(float)
        columns.append(field + '_flag')
    if cfg['include_interaction']:
        result['trend_price'] = (frame.trend & frame.price).astype(float)
        columns.append('trend_price')
    result[columns] = result[columns] - result.groupby('event_id')[columns].transform('mean')
    assert np.isfinite(result[columns].to_numpy()).all()
    return result, columns


def equal_date_weights(frame):
    """日期等权、同日事件等权、同事件股票等权。"""
    nstock = frame.groupby('event_id').ts_code.transform('size')
    nevent = frame.date.map(frame[['date', 'event_id']].drop_duplicates().groupby('date').size())
    weights = 1 / (nstock * nevent)
    return weights.to_numpy() / weights.mean()


def mature_history(data, start, calendar, cfg):
    """先以标签完成时点截断，再取得可训练数据。"""
    start = pd.Timestamp(start)
    index = calendar.searchsorted(start) - cfg['embargo_sessions']
    if index < 0:
        return data.iloc[:0].copy(), pd.NaT
    cutoff = calendar[index]
    lower = start - pd.DateOffset(years=cfg['training_years'])
    past = data[data.date.lt(start) & data.date.ge(lower) & data.exit_date.lt(cutoff)].copy()
    past = past[past.label_complete.eq(True) & past.target.notna()]
    return past, cutoff


def fit_stock_model(data, start, calendar, cfg, features):
    """历史内层按日期选正则参数，不接受测试年标签。"""
    train, cutoff = mature_history(data, start, calendar, cfg)
    dates = np.sort(train.date.unique())
    count = max(cfg['minimum_validation_dates'], int(np.ceil(len(dates) * cfg['validation_fraction'])))
    if count >= len(dates):
        raise ValueError('历史日期不足')
    valid_start = pd.Timestamp(dates[-count])
    fit, inner_cutoff = mature_history(train, valid_start, calendar, cfg)
    valid = train[train.date.ge(valid_start)]
    if fit.event_id.nunique() < cfg['minimum_fit_events'] or valid.event_id.nunique() < cfg['minimum_validation_events']:
        raise ValueError(f'历史拟合/验证事件不足: {fit.event_id.nunique()}/{valid.event_id.nunique()}')
    assert fit.exit_date.lt(inner_cutoff).all() and train.exit_date.lt(cutoff).all()
    assert set(fit.date).isdisjoint(valid.date)
    fw, vw = equal_date_weights(fit), equal_date_weights(valid)
    candidates = []
    for alpha in cfg['alphas']:
        model = Ridge(alpha=alpha, fit_intercept=False).fit(fit[features], fit.target, sample_weight=fw)
        loss = float(np.average((model.predict(valid[features]) - valid.target.to_numpy()) ** 2, weights=vw))
        candidates.append(dict(alpha=alpha, loss=loss))
    chosen = min(candidates, key=lambda r: (r['loss'], -r['alpha']))
    model = Ridge(alpha=chosen['alpha'], fit_intercept=False).fit(train[features], train.target, sample_weight=equal_date_weights(train))
    audit = dict(start=pd.Timestamp(start), cutoff=cutoff, inner_cutoff=inner_cutoff,
                 train_events=sorted(train.event_id.unique()), fit_events=sorted(fit.event_id.unique()),
                 valid_events=sorted(valid.event_id.unique()), train_last_label=train.exit_date.max(),
                 fit_last_label=fit.exit_date.max(), valid_start=valid_start,
                 train_rows=len(train), alpha=chosen['alpha'], features=features, search=candidates)
    return model, audit


def select_positive_top(frame, score_column, cfg):
    """价格形态池内正预测才入选；分数并列按代码固定排序。"""
    eligible = frame[frame.price & frame[score_column].gt(cfg['model']['minimum_predicted_excess'])]
    ordered = eligible.sort_values(['event_id', score_column, 'ts_code'], ascending=[True, False, True])
    return ordered.groupby('event_id').head(cfg['top_n']).index
