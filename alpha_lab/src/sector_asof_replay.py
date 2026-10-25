"""按决策时点隔离训练标签、名称公告和模型拟合。"""
import numpy as np
import pandas as pd


def history_before(data, start, calendar, cfg):
    """先截断时间，再读取成熟标签；未知路径不进入训练。"""
    start = pd.Timestamp(start)
    index = calendar.searchsorted(start) - cfg['embargo_sessions']
    if index < 0:
        return data.iloc[:0].copy(), pd.NaT
    cutoff = calendar[index]
    lower = start - pd.DateOffset(years=cfg['training_years'])
    past = data[data.date.ge(lower) & data.date.lt(start)].copy()
    return past[past.end_date.lt(cutoff) & past.complete.eq(True) & past.target.notna()].copy(), cutoff


def training_plan(data, start, calendar, cfg):
    """外层与内层均按完整日期切分，标签结束而非起点决定可用性。"""
    train, cutoff = history_before(data, start, calendar, cfg)
    dates = np.sort(train.date.unique())
    n = max(cfg['minimum_validation_dates'], int(np.ceil(len(dates) * cfg['validation_fraction'])))
    if n >= len(dates):
        raise ValueError('历史验证日期不足')
    validation_start = pd.Timestamp(dates[-n])
    fit, inner_cutoff = history_before(train, validation_start, calendar, cfg)
    valid = train[train.date.ge(validation_start)].copy()
    if len(fit) < cfg['minimum_train_events'] or len(valid) < cfg['minimum_validation_events']:
        raise ValueError(f'历史拟合/验证样本不足: {len(fit)}/{len(valid)}')
    assert fit.end_date.lt(inner_cutoff).all() and train.end_date.lt(cutoff).all()
    assert set(fit.date).isdisjoint(valid.date)
    return train, fit, valid, cutoff, inner_cutoff


def announced_normal_names(names, date, codes):
    """名称有效区间与公告可知时间同时满足；未知、冲突、ST均拒绝。"""
    active = names[names.ts_code.isin(codes) & names.start_date.le(date)
                   & (names.end_date.isna() | names.end_date.ge(date))]
    known = active.ann_date.notna() & active.ann_date.lt(date)
    normal = ~active['name'].fillna('ST').str.upper().str.contains('ST|退', regex=True)
    count = active.groupby('ts_code').size()
    passed = active.assign(passed=known & normal).groupby('ts_code').passed.all() & count.eq(1)
    return passed.reindex(codes, fill_value=False)


def fit_asof(data, start, calendar, cfg, model_cfg):
    """仅历史内层选参，不接受外层目标特征或标签。"""
    import torch
    from model import TrainScaler, fit_network, predict_network, sample_weights
    torch.set_num_threads(1)
    train, fit, valid, cutoff, inner_cutoff = training_plan(data, start, calendar, cfg)
    features = model_cfg['features']
    network_cfg = {**model_cfg['fnn'], 'maximum_epochs': cfg['maximum_epochs']}
    scaler = TrainScaler(model_cfg).fit(fit[features])
    x, vx = scaler.transform(fit[features]), scaler.transform(valid[features])
    y, vy = fit.target.to_numpy() / model_cfg['target_scale'], valid.target.to_numpy() / model_cfg['target_scale']
    weights = sample_weights(fit.date, valid.date.min(), cfg['half_life_days'])
    vw = sample_weights(valid.date, start, None)
    search, candidates = [], []
    for rules in cfg['rules']:
        for lr in cfg['learning_rates']:
            losses, epochs = [], []
            for seed in cfg['tuning_seeds']:
                _, epoch, loss, _ = fit_network(x, y, weights, rules, lr, seed, network_cfg, validation=(vx, vy, vw))
                losses.append(loss)
                epochs.append(epoch)
                search.append(dict(rules=rules, learning_rate=lr, seed=seed, epochs=epoch, validation_loss=loss))
            candidates.append((float(np.mean(losses)), rules, lr, int(np.median(epochs))))
    loss, rules, lr, epochs = min(candidates)
    final_scaler = TrainScaler(model_cfg).fit(train[features])
    tx = final_scaler.transform(train[features])
    ty = train.target.to_numpy() / model_cfg['target_scale']
    tw = sample_weights(train.date, start, cfg['half_life_days'])
    networks, predictions = [], []
    for seed in cfg['ensemble_seeds']:
        network, _, _, _ = fit_network(tx, ty, tw, rules, lr, seed, network_cfg, epochs=epochs)
        networks.append(network)
        predictions.append(predict_network(network, tx) * model_cfg['target_scale'])
    threshold = float(np.quantile(np.mean(predictions, axis=0), 1 - cfg['retention']))
    audit = dict(start=start, cutoff=cutoff, inner_cutoff=inner_cutoff,
                 train_ids=train.event_id.tolist(), fit_ids=fit.event_id.tolist(), valid_ids=valid.event_id.tolist(),
                 train_last_date=train.date.max(), train_last_label=train.end_date.max(),
                 fit_last_label=fit.end_date.max(), validation_start=valid.date.min(),
                 rules=rules, learning_rate=lr, epochs=epochs, threshold=threshold, validation_loss=loss)
    return dict(networks=networks, scaler=final_scaler, audit=audit, search=search)


def predict_asof(fitted, features, model_cfg):
    """仅取特征白名单，输出不读取目标标签。"""
    from model import predict_network
    x = fitted['scaler'].transform(features[model_cfg['features']])
    score = np.mean([predict_network(net, x) for net in fitted['networks']], axis=0) * model_cfg['target_scale']
    return score, score >= fitted['audit']['threshold']
