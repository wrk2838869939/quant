"""冻结样本上的逐时点学习：因果特征、成熟标签、联合采样及滚动预测。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMRegressor, LGBMRanker

from rule_search import RuleSignals, config_id


def build_features(wide, eligible, calendar, specs):
    """只按当日有资格截面标准化；不估计全时期均值或填补参数。"""
    model = RuleSignals(wide, eligible, calendar)
    ranks = {name: model.score(p).rank(axis=1, pct=True) - .5 for name, p in specs.items()}
    return ranks


def build_labels(wide, dates, horizons):
    """标签始于次日开盘；缺一日价格则该路径无标签，结束日标记可得时点。"""
    close = wide['hfq_close'].reindex(dates)
    op = (wide['open'] * wide['hfq_close'] / wide['close']).reindex(dates)
    low = wide['hfq_low'].reindex(dates)
    result = {}
    for horizon in horizons:
        entry = op.shift(-1)
        valid = entry.gt(0)
        trough = low.shift(-1).copy()
        for step in range(1, horizon + 1):
            valid &= low.shift(-step).gt(0) & close.shift(-step).gt(0)
            trough = pd.DataFrame(np.minimum(trough, low.shift(-step)), index=dates, columns=close.columns)
        returns = (close.shift(-horizon) / entry - 1).where(valid)
        risk = (1 - trough / entry).clip(lower=0).where(valid)
        end = pd.Series(dates, index=dates).shift(-horizon)
        result[horizon] = dict(returns=returns, risk=risk, end=end)
    return result


def daily_ic(features, target, minimum):
    """按特征可得样本重新排名，避免缺失值填零造成伪相关。"""
    values = []
    for frame in features.values():
        valid = frame.notna() & target.notna()
        x = frame.where(valid).rank(axis=1, pct=True)
        y = target.where(valid).rank(axis=1, pct=True)
        xm, ym = x.sub(x.mean(axis=1), axis=0), y.sub(y.mean(axis=1), axis=0)
        denom = np.sqrt((xm * xm).sum(axis=1) * (ym * ym).sum(axis=1))
        values.append(((xm * ym).sum(axis=1) / denom.where(denom > 0)).where(valid.sum(axis=1) >= minimum))
    return pd.concat(values, axis=1, keys=features.keys())


def prepare_arrays(features, eligible, labels, cfg):
    ranks = np.stack([f.to_numpy() for f in features.values()], axis=2)
    missing = ~np.isfinite(ranks)
    x = np.concatenate([np.nan_to_num(ranks), missing.astype(float)], axis=2).astype(np.float32)
    allowed = eligible.to_numpy() & ((~missing).sum(axis=2) >= cfg['min_features'])
    return dict(dates=eligible.index, codes=eligible.columns, names=list(features), x=x,
                ranks=np.nan_to_num(ranks).astype(np.float32), allowed=allowed,
                labels={h: dict(returns=d['returns'].to_numpy(), risk=d['risk'].to_numpy(),
                                end=d['end'].to_numpy(), ic=daily_ic(features, d['returns'].where(allowed), cfg['min_ic_stocks']).to_numpy())
                        for h, d in labels.items()})


def training_mask(data, params, fit_date, cfg):
    dates = data['dates']
    at = dates.searchsorted(pd.Timestamp(fit_date))
    label = data['labels'][params['horizon']]
    past = (np.arange(len(dates)) >= max(0, at - params['train_window'])) & (dates < fit_date)
    mature = ~pd.isna(label['end']) & (label['end'] < np.datetime64(fit_date))
    mask = data['allowed'] & (past & mature)[:, None]
    mask &= np.isfinite(label['returns']) & np.isfinite(label['risk'])
    return mask


def make_model(params, cfg, seed, risk=False):
    variant = params['variant']
    if variant == 'ridge':
        return Ridge(alpha=params['alpha'])
    if variant == 'elastic_net':
        return ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], max_iter=cfg['elastic_max_iter'],
                          tol=cfg['elastic_tolerance'], selection='cyclic')
    kwargs = {k: params[k] for k in ['n_estimators', 'num_leaves', 'min_child_samples', 'learning_rate', 'reg_lambda']}
    kwargs.update(n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True, random_state=seed)
    if variant == 'ranking' and not risk:
        return LGBMRanker(**kwargs, label_gain=list(range(cfg['rank_levels'])))
    return LGBMRegressor(**kwargs)


def predict_walk_forward(data, params, cfg, seed, start, end, folder=None):
    """每次拟合严格排除尚未成熟标签；月内模型冻结，可按1或3个月更新。"""
    dates, codes = data['dates'], data['codes']
    evaluation = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    pred = np.full(data['allowed'].shape, np.nan)
    risk_pred = np.full_like(pred, np.nan)
    fits = []
    block = (dates.year * 12 + dates.month - 1) // params['retrain_months']
    model = risk_model = weights = None
    last_block = None
    label = data['labels'][params['horizon']]
    for at in evaluation:
        if last_block != block[at]:
            last_block = block[at]
            mask = training_mask(data, params, dates[at], cfg)
            used_dates = np.flatnonzero(mask.any(axis=1))
            record = dict(fit_date=str(dates[at].date()), rows=int(mask.sum()), dates=len(used_dates), params=params)
            model = risk_model = weights = None
            if len(used_dates) < cfg['min_dates'] or mask.sum() < cfg['min_rows']:
                record['status'] = 'insufficient_training'
                fits.append(record)
                continue
            latest_end = pd.Timestamp(label['end'][used_dates].max())
            assert latest_end < dates[at]
            record.update(status='ok', first_signal_date=str(dates[used_dates[0]].date()),
                          last_signal_date=str(dates[used_dates[-1]].date()),
                          max_label_end=str(latest_end.date()),
                          training_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
            if params['family'] == 'factor_weight':
                ic = label['ic'][used_dates]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    weights = np.nanmean(ic, axis=0)
                    if params['variant'] == 'icir':
                        std = np.nanstd(ic, axis=0, ddof=1)
                        weights = np.divide(weights, std, out=np.zeros_like(weights), where=std > 1e-8)
                weights = np.nan_to_num(weights)
                keep = params['keep_features']
                if keep:
                    order = np.argsort(-np.abs(weights), kind='stable')
                    weights[order[keep:]] = 0
                total = np.abs(weights).sum()
                weights = weights / total if total else np.zeros_like(weights)
                weights = (1 - params['shrinkage']) * weights + params['shrinkage'] / len(weights)
                record['weights'] = dict(zip(data['names'], weights.tolist()))
            else:
                x, y = data['x'][mask], label['returns'][mask]
                row_weights = np.broadcast_to(1 / np.maximum(mask.sum(axis=1), 1)[:, None], mask.shape)[mask]
                row_weights *= len(row_weights) / row_weights.sum()
                model = make_model(params, cfg, seed)
                fit_kwargs = dict(sample_weight=row_weights)
                if params['variant'] == 'ranking':
                    target = pd.DataFrame(label['returns']).where(mask).rank(axis=1, pct=True).to_numpy()
                    y = np.minimum((target[mask] * cfg['rank_levels']).astype(int), cfg['rank_levels'] - 1)
                    fit_kwargs['group'] = mask.sum(axis=1)[used_dates].tolist()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    model.fit(x, y, **fit_kwargs)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError('线性模型未收敛，不能静默计为成功')
                if params['family'] == 'dual':
                    risk_model = make_model(params, cfg, seed, risk=True)
                    risk_model.fit(x, label['risk'][mask], sample_weight=row_weights)
                importance = getattr(model, 'coef_', getattr(model, 'feature_importances_', None))
                if importance is not None:
                    record['importance'] = np.asarray(importance).ravel().tolist()
            if folder:
                path = Path(folder)
                path.mkdir(parents=True, exist_ok=True)
                artifact = path / f'model_{dates[at]:%Y%m%d}.joblib'
                joblib.dump(dict(model=model, risk_model=risk_model, weights=weights), artifact)
                record['model_file'] = artifact.name
                record['model_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            fits.append(record)
        valid = data['allowed'][at]
        if weights is not None:
            pred[at, valid] = data['ranks'][at, valid] @ weights
        elif model is not None and valid.any():
            # LightGBM的predict只使用数值列，不依赖自动生成的特征名。
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                pred[at, valid] = model.predict(data['x'][at, valid])
                if risk_model is not None:
                    risk_pred[at, valid] = np.maximum(0., risk_model.predict(data['x'][at, valid]))
    return (pd.DataFrame(pred[evaluation], index=dates[evaluation], columns=codes),
            pd.DataFrame(risk_pred[evaluation], index=dates[evaluation], columns=codes), fits)


def score_predictions(pred, risk, params, remove_risk=False):
    if params['family'] != 'dual' or remove_risk:
        return pred
    return (pred - params['risk_aversion'] * risk).where(risk <= params['risk_cap'])


def axes_for(plan, family, variant):
    return {**plan['common_axes'], **plan['families'][family][variant]}


def sample_parameters(plan, family, count, seed, seen=(), centers=None):
    """对结构、交易频率、持股数分层；局部变动一或两轴，不推断单调性。"""
    if count <= 0:
        return []
    rng = np.random.default_rng(seed)
    variants = list(plan['families'][family])
    categories = [(v, f, n) for v in variants for f in plan['common_axes']['frequency'] for n in plan['common_axes']['top_n']]
    rng.shuffle(categories)
    result, seen = [], set(seen)
    for attempt in range(max(1000, count * 200)):
        if centers:
            p = dict(centers[attempt % len(centers)])
            axes = axes_for(plan, family, p['variant'])
            keys = list(axes)
            for key in rng.choice(keys, size=min(len(keys), 1 + attempt % 2), replace=False):
                axis = axes[key]
                old = axis.index(p[key])
                p[key] = axis[max(0, min(len(axis) - 1, old + int(rng.choice([-1, 1]))))]
        else:
            variant, frequency, top_n = categories[attempt % len(categories)]
            axes = axes_for(plan, family, variant)
            p = dict(family=family, variant=variant, **{k: v[int(rng.integers(len(v)))] for k, v in axes.items()})
            p.update(frequency=frequency, top_n=top_n)
        cid = config_id(p)
        if cid not in seen:
            seen.add(cid)
            result.append(p)
            if len(result) == count:
                return result
    if len(result) != count:
        raise ValueError('参数空间去重后不足预算')
    return result
