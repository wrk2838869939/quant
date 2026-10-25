"""第二批学习器：冻结特征、成熟标签、状态/分组/多任务模型及风险门控。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from learning_search import training_mask


def attach_context(data, wide, eligible):
    """市场状态仅用当日及过去的收益与广度；不使用全期估计量。"""
    close = wide['hfq_close'].reindex(columns=data['codes'])
    returns = close.pct_change(fill_method=None)
    valid = eligible.reindex(returns.index, fill_value=False)
    r = returns.where(valid)
    mean = r.mean(axis=1)
    context = pd.DataFrame({
        'mean_return': mean,
        'mean_return_5': mean.rolling(5, min_periods=5).mean(),
        'mean_return_20': mean.rolling(20, min_periods=20).mean(),
        'market_vol_20': mean.rolling(20, min_periods=20).std(),
        'breadth': r.gt(0).sum(axis=1) / r.notna().sum(axis=1).replace(0, np.nan),
        'dispersion': r.std(axis=1),
    }).reindex(data['dates'])
    data['context'] = context.fillna(0).to_numpy(dtype=np.float64)
    data['stock_returns'] = r.reindex(data['dates']).to_numpy()
    return context


def host_scores(data):
    """预先固定的23因子等权宿主；不读取上一轮胜者。"""
    return np.where(data['allowed'], data['ranks'].mean(axis=2), np.nan)


def positive_probability(model, x):
    probabilities = model.predict_proba(x)
    matches = np.flatnonzero(model.classes_ == 1)
    return probabilities[:, matches[0]] if len(matches) else np.zeros(len(x))


def fit_bundle(data, p, cfg, seed, at):
    """所有模型和标准化器只在本次成熟训练窗估计。"""
    date = data['dates'][at]
    mask = training_mask(data, p, date, cfg)
    if p['family'] == 'multi_horizon':
        for h in cfg['joint_horizons']:
            mask &= training_mask(data, dict(p, horizon=h), date, cfg)
    if p['family'] == 'trade_filter':
        host = pd.DataFrame(host_scores(data)).rank(axis=1, ascending=False, method='first').to_numpy()
        mask &= host <= cfg['host_candidate_count']
    used = np.flatnonzero(mask.any(axis=1))
    row_count = int(mask.sum())
    rec = dict(fit_date=str(date.date()), rows=row_count, dates=len(used), params=p)
    minimum = cfg['filter_min_rows'] if p['family'] == 'trade_filter' else cfg['min_rows']
    if len(used) < cfg['min_dates'] or row_count < minimum:
        rec['status'] = 'insufficient_training'
        return None, rec
    horizons = cfg['joint_horizons'] if p['family'] == 'multi_horizon' else [p['horizon']]
    max_end = max(pd.Timestamp(data['labels'][h]['end'][used].max()) for h in horizons)
    assert max_end < date
    rec.update(status='ok', first_signal_date=str(data['dates'][used[0]].date()),
               last_signal_date=str(data['dates'][used[-1]].date()), max_label_end=str(max_end.date()),
               training_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    bundle = dict(family=p['family'], params=p)
    x = data['x'][mask]
    label = data['labels'][p['horizon']]
    y = label['returns'][mask]
    weights = np.broadcast_to(1 / np.maximum(mask.sum(axis=1), 1)[:, None], mask.shape)[mask]
    weights *= len(weights) / weights.sum()
    family = p['family']
    if family == 'neural':
        # 固定训练轮数是显式预算，不用未来验证集早停。
        model = MLPRegressor(hidden_layer_sizes=(p['width'],), activation=p['variant'],
                             alpha=p['alpha'], learning_rate_init=p['learning_rate'],
                             batch_size=cfg['neural_batch_size'], random_state=seed, shuffle=True,
                             max_iter=1, early_stopping=False)
        scale = cfg['neural_target_scale']
        for _ in range(p['epochs']):
            model.partial_fit(x, y * scale, sample_weight=weights)
        bundle.update(model=model, target_scale=scale)
        rec['training_loss'] = float(model.loss_)
    elif family == 'state_experts':
        scaler = StandardScaler().fit(data['context'][used])
        state = KMeans(n_clusters=p['states'], n_init=cfg['kmeans_n_init'], random_state=seed).fit(scaler.transform(data['context'][used]))
        states = state.predict(scaler.transform(data['context']))
        pooled = Ridge(alpha=p['alpha']).fit(x, y, sample_weight=weights)
        models = {}
        for group in range(p['states']):
            subset = states[np.nonzero(mask)[0]] == group
            if subset.sum() >= cfg['expert_min_rows']:
                models[group] = Ridge(alpha=p['alpha']).fit(x[subset], y[subset], sample_weight=weights[subset])
        bundle.update(model=pooled, experts=models, scaler=scaler, state=state)
        rec['expert_rows'] = {str(i): int((mask & (states == i)[:, None]).sum()) for i in range(p['states'])}
    elif family == 'adaptive_ic':
        ic = label['ic'][used]
        if p['variant'] == 'ewma':
            values = pd.DataFrame(ic).ewm(halflife=p['half_life'], adjust=False).mean().iloc[-1].to_numpy()
        else:
            values = np.zeros(ic.shape[1])
            variance = np.full(ic.shape[1], cfg['kalman_initial_variance'])
            for observation in ic:
                variance += p['process_noise']
                valid = np.isfinite(observation)
                gain = variance[valid] / (variance[valid] + p['observation_noise'])
                values[valid] += gain * (observation[valid] - values[valid])
                variance[valid] *= 1 - gain
        values = np.nan_to_num(values)
        total = np.abs(values).sum()
        values = values / total if total else values
        values = (1 - p['shrinkage']) * values + p['shrinkage'] / len(values)
        bundle['weights'] = values
        rec['weights'] = dict(zip(data['names'], values.tolist()))
    elif family == 'trade_filter':
        model_type = DecisionTreeClassifier if p['variant'] == 'tree' else RandomForestClassifier
        kwargs = dict(max_depth=p['depth'], min_samples_leaf=p['min_leaf'], random_state=seed)
        if p['variant'] == 'forest':
            kwargs.update(n_estimators=cfg['filter_forest_trees'], n_jobs=1)
        model = model_type(**kwargs).fit(x, (y > cfg['filter_return_hurdle']).astype(int), sample_weight=weights)
        bundle['model'] = model
        bundle['host_candidate_count'] = cfg['host_candidate_count']
        rec['positive_rate'] = float((y > cfg['filter_return_hurdle']).mean())
    elif family == 'multi_horizon':
        targets = np.column_stack([data['labels'][h]['returns'][mask] / np.sqrt(h) for h in cfg['joint_horizons']])
        if p['variant'] == 'relative':
            targets = np.column_stack([pd.DataFrame(data['labels'][h]['returns']).where(mask).rank(axis=1, pct=True).to_numpy()[mask] - .5 for h in cfg['joint_horizons']])
        bundle['model'] = Ridge(alpha=p['alpha']).fit(x, targets, sample_weight=weights)
        bundle['blend'] = np.asarray(cfg['horizon_blends'][p['blend']])
    elif family == 'correlation_experts':
        start = max(0, at - p['correlation_window'])
        returns = pd.DataFrame(data['stock_returns'][start:at])
        corr = returns.corr(min_periods=cfg['correlation_min_days']).fillna(0).clip(-1, 1).to_numpy()
        np.fill_diagonal(corr, 1.)
        distance = np.sqrt(np.maximum(0., 2 - 2 * corr))
        groups = AgglomerativeClustering(n_clusters=p['clusters'], metric='precomputed', linkage='average').fit_predict(distance)
        pooled = Ridge(alpha=p['alpha']).fit(x, y, sample_weight=weights)
        models = {}
        stocks = np.nonzero(mask)[1]
        for group in range(p['clusters']):
            subset = groups[stocks] == group
            if subset.sum() >= cfg['expert_min_rows']:
                models[group] = Ridge(alpha=p['alpha']).fit(x[subset], y[subset], sample_weight=weights[subset])
        bundle.update(model=pooled, experts=models, groups=groups)
        rec['stock_clusters'] = dict(zip(data['codes'], groups.tolist()))
    elif family == 'market_gate':
        # 标签是当日可选股票的等权未来收益；与账户回撤明确区分。
        market_target = np.array([np.mean(label['returns'][i, mask[i]]) for i in used])
        scaler = StandardScaler().fit(data['context'][used])
        z = scaler.transform(data['context'][used])
        event = (market_target < -p['loss_threshold']).astype(int)
        if p['variant'] == 'logistic' and len(np.unique(event)) > 1:
            model = LogisticRegression(C=p['c'], max_iter=cfg['logistic_max_iter'], random_state=seed).fit(z, event)
            assert model.n_iter_.max() < cfg['logistic_max_iter'], '逻辑回归未收敛'
        else:
            model = DecisionTreeClassifier(max_depth=p['depth'], min_samples_leaf=cfg['market_min_leaf'], random_state=seed).fit(z, event)
        bundle.update(model=model, scaler=scaler)
        rec['positive_rate'] = float(event.mean())
    else:
        raise ValueError(family)
    return bundle, rec


def infer_bundle(bundle, data, at):
    """单日推断；返回原始选股分数和可审计的门控概率。"""
    p = bundle['params']
    family = p['family']
    valid = data['allowed'][at]
    x = data['x'][at, valid]
    pred = np.full(len(valid), np.nan)
    risk = pred.copy()
    if not valid.any():
        return pred, risk
    if family == 'adaptive_ic':
        pred[valid] = data['ranks'][at, valid] @ bundle['weights']
    elif family in ['trade_filter', 'market_gate']:
        pred = host_scores_day(data, at)
        if family == 'trade_filter':
            # 仅筛选固定宿主前15名，不能用未训练的尾部股票补位。
            candidates = pd.Series(pred).rank(ascending=False, method='first').to_numpy() <= bundle['host_candidate_count']
            pred[~candidates] = np.nan
            risk[valid] = positive_probability(bundle['model'], x)
        else:
            z = bundle['scaler'].transform(data['context'][at:at + 1])
            risk[valid] = positive_probability(bundle['model'], z)[0]
    elif family == 'state_experts':
        group = int(bundle['state'].predict(bundle['scaler'].transform(data['context'][at:at + 1]))[0])
        pool = bundle['model'].predict(x)
        local = bundle['experts'].get(group, bundle['model']).predict(x)
        pred[valid] = (1 - p['expert_weight']) * pool + p['expert_weight'] * local
    elif family == 'correlation_experts':
        pool = bundle['model'].predict(x)
        local = pool.copy()
        groups = bundle['groups'][valid]
        for group, model in bundle['experts'].items():
            selected = groups == group
            if selected.any():
                local[selected] = model.predict(x[selected])
        pred[valid] = (1 - p['expert_weight']) * pool + p['expert_weight'] * local
    elif family == 'multi_horizon':
        pred[valid] = bundle['model'].predict(x) @ bundle['blend']
    elif family == 'neural':
        pred[valid] = bundle['model'].predict(x) / bundle['target_scale']
    else:
        raise ValueError(family)
    return pred, risk


def host_scores_day(data, at):
    return np.where(data['allowed'][at], data['ranks'][at].mean(axis=1), np.nan)


def score_predictions(pred, risk, params, remove_risk=False):
    if remove_risk:
        return pred
    if params['family'] == 'trade_filter':
        return pred.where(risk >= params['probability_cut'])
    if params['family'] == 'market_gate':
        return pred.where(risk <= params['probability_cut'])
    return pred


def predict_walk_forward(data, params, cfg, seed, start, end, folder=None):
    dates, codes = data['dates'], data['codes']
    evaluation = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    prediction = np.full(data['allowed'].shape, np.nan)
    risk = prediction.copy()
    block = (dates.year * 12 + dates.month - 1) // params['retrain_months']
    previous = None
    bundle = None
    fits = []
    for at in evaluation:
        if previous != block[at]:
            previous = block[at]
            bundle, record = fit_bundle(data, params, cfg, seed, at)
            if bundle is not None and folder:
                path = Path(folder)
                path.mkdir(parents=True, exist_ok=True)
                artifact = path / f'model_{dates[at]:%Y%m%d}.joblib'
                joblib.dump(bundle, artifact)
                record.update(model_file=artifact.name, model_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
            fits.append(record)
        if bundle is not None:
            prediction[at], risk[at] = infer_bundle(bundle, data, at)
    return (pd.DataFrame(prediction[evaluation], index=dates[evaluation], columns=codes),
            pd.DataFrame(risk[evaluation], index=dates[evaluation], columns=codes), fits)
