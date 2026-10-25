"""保留原样宿主的监督排名增量；直接模型为成对对照。"""
from pathlib import Path
import hashlib
import itertools
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from rule_search import config_id


def model_grid(plan):
    result = []
    cfg = plan['training']['models']
    for horizon in plan['training']['horizons']:
        result.extend(dict(model='ridge', horizon=horizon, alpha=a) for a in cfg['ridge']['alpha'])
        fixed = {k: v for k, v in cfg['lgbm'].items() if k != 'num_leaves'}
        result.extend(dict(model='lgbm', horizon=horizon, num_leaves=n, **fixed) for n in cfg['lgbm']['num_leaves'])
    return result


def prepare_data(scores, host, labels, min_stocks):
    """不以特征完整数额外缩减宿主股票资格；缺失指标随样本保存。"""
    ranks = [s.reindex_like(host).rank(axis=1, pct=True) - .5 for s in scores]
    prior = host.rank(axis=1, pct=True)
    ranks.append(prior - .5)
    raw = np.stack([x.to_numpy() for x in ranks], axis=2)
    missing = ~np.isfinite(raw)
    x = np.concatenate([np.nan_to_num(raw), missing.astype(float)], axis=2)
    allowed = host.notna().to_numpy()
    targets = {}
    for horizon, label in labels.items():
        returns = label['returns'].reindex_like(host).where(host.notna())
        target = returns.rank(axis=1, pct=True).where(returns.notna().sum(axis=1) >= min_stocks, axis=0)
        targets[horizon] = dict(direct=target.to_numpy(), residual=(target - prior).to_numpy(),
                                end=label['end'].reindex(host.index).to_numpy())
    return dict(dates=host.index, codes=host.columns, x=x, allowed=allowed,
                prior=prior.to_numpy(), labels=targets)


def training_mask(data, params, fit_date, train_window):
    dates = data['dates']
    at = dates.searchsorted(pd.Timestamp(fit_date))
    label = data['labels'][params['horizon']]
    mature = ~pd.isna(label['end']) & (label['end'] < np.datetime64(fit_date))
    past = (np.arange(len(dates)) >= max(0, at - train_window)) & (dates < fit_date)
    return data['allowed'] & (past & mature)[:, None] & np.isfinite(label['direct']) & np.isfinite(label['residual'])


def make_model(params, seed):
    if params['model'] == 'ridge':
        return Ridge(alpha=params['alpha'])
    if params['model'] != 'lgbm':
        raise ValueError('未知增量模型')
    p = {k: v for k, v in params.items() if k not in ['model', 'horizon']}
    return LGBMRegressor(**p, n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True, random_state=seed)


def predict_pair(data, params, plan, start, end, folder=None):
    """按季度对直接目标/增量目标成对拟合，严格屏蔽未成熟标签。"""
    cfg = plan['training']
    dates, codes = data['dates'], data['codes']
    at_eval = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    output = {k: np.full(data['allowed'].shape, np.nan) for k in ['direct', 'residual']}
    blocks = (dates.year * 12 + dates.month - 1) // cfg['retrain_months']
    fits, models, last = [], None, None
    for at in at_eval:
        if blocks[at] != last:
            last, models = blocks[at], None
            mask = training_mask(data, params, dates[at], cfg['train_window'])
            used = np.flatnonzero(mask.any(axis=1))
            record = dict(fit_date=str(dates[at].date()), rows=int(mask.sum()), training_dates=len(used),
                          mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), params=params)
            if len(used) < cfg['min_dates'] or mask.sum() < cfg['min_rows']:
                record['status'] = 'insufficient_history_base_fallback'
            else:
                label = data['labels'][params['horizon']]
                latest = pd.Timestamp(label['end'][used].max())
                assert latest < dates[at]
                x = data['x'][mask]
                weights = np.broadcast_to(1 / np.maximum(mask.sum(axis=1), 1)[:, None], mask.shape)[mask]
                weights *= len(weights) / weights.sum()
                models = {}
                for mode in ['direct', 'residual']:
                    model = make_model(params, plan['seed'])
                    model.fit(x, label[mode][mask], sample_weight=weights)
                    models[mode] = model
                record.update(status='ok', first_training_date=str(dates[used[0]].date()),
                              last_training_date=str(dates[used[-1]].date()), max_label_end=str(latest.date()),
                              importance={k: np.asarray(getattr(m, 'coef_', getattr(m, 'feature_importances_', []))).tolist()
                                          for k, m in models.items()})
                if folder:
                    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f'model_{dates[at]:%Y%m%d}.joblib'
                    joblib.dump(models, path)
                    reloaded = joblib.load(path)
                    first = data['x'][at, data['allowed'][at]]
                    if len(first):
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='X does not have valid feature names')
                            for mode in models:
                                np.testing.assert_allclose(models[mode].predict(first), reloaded[mode].predict(first), rtol=0, atol=0)
                    record.update(model_file=path.name, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  reload_verified=bool(len(first)))
            fits.append(record)
        valid = data['allowed'][at]
        if models is not None and valid.any():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                for mode, model in models.items():
                    output[mode][at, valid] = model.predict(data['x'][at, valid])
    frames = {k: pd.DataFrame(v[at_eval], index=dates[at_eval], columns=codes) for k, v in output.items()}
    for frame in frames.values():
        if np.isinf(frame.to_numpy()).any():
            raise ValueError('模型输出无穷值')
    return frames, fits


def compose_score(host, prediction, kind, amplitude, clip):
    """零增量原样返回；非零增量保持宿主资格，模型缺失退回宿主排名。"""
    if kind == 'baseline' or (kind == 'residual' and amplitude == 0):
        return host.copy()
    if kind not in ['residual', 'direct'] or amplitude < 0 or clip <= 0:
        raise ValueError('非法增量定义')
    pred = prediction.reindex_like(host)
    if np.isinf(pred.to_numpy()).any():
        raise ValueError('模型输出无穷值')
    prior = host.rank(axis=1, pct=True)
    score = (prior + amplitude * pred.clip(-clip, clip).fillna(0) if kind == 'residual'
             else pred.where(pred.notna(), prior))
    return score.where(host.notna())


def selection_key(row):
    amplitude = row['params'].get('amplitude', 1.)
    return (-row['metrics']['median_ann'], amplitude, row['metrics']['turnover'], row['id'])


def select_paths(rows, plan):
    """原样宿主是明示候选，稳健选择也不得跌破其内层目标值。"""
    base = next(r for r in rows if r['params']['kind'] == 'baseline')
    good = [r for r in rows if r['metrics']['feasible']]
    best = lambda items: min(items, key=selection_key) if items else None
    residual = [r for r in good if r['params']['kind'] in ['baseline', 'residual']]
    direct = [r for r in good if r['params']['kind'] == 'direct']
    chosen = best(residual)
    floor = base['metrics']['median_ann'] if base['metrics']['feasible'] else -np.inf
    stable = None
    if chosen:
        near = [r for r in residual if r['metrics']['median_ann'] >= max(floor, chosen['metrics']['median_ann'] - plan['search']['stable_tie_band'])]
        if all(r.get('neighbor_q25') is not None for r in near):
            stable = min(near, key=lambda r: (-r['neighbor_q25'], selection_key(r)))
        else:
            stable = min(near, key=lambda r: (r['params'].get('amplitude', 1.), r['metrics']['turnover'], selection_key(r)))
    result = dict(baseline=base, residual_best=chosen, residual_stable=stable, direct_best=best(direct),
                  direct_with_fallback=best(direct + ([base] if base['metrics']['feasible'] else [])))
    if base['metrics']['feasible']:
        for key in ['residual_best', 'residual_stable', 'direct_with_fallback']:
            assert result[key]['metrics']['median_ann'] >= floor
    return result


def amplitude_neighbors(rows, plan):
    """同一模型规格上的实际相邻幅度；零点共用原样宿主账户。"""
    base = next(r for r in rows if r['params']['kind'] == 'baseline')
    curves, triplets = [], []
    for spec in model_grid(plan):
        curve = [base] + [next(r for r in rows if r['params']['kind'] == 'residual'
                              and r['params']['model_spec'] == spec and r['params']['amplitude'] == amplitude)
                          for amplitude in plan['amplitudes'] if amplitude > 0]
        curves.append(curve)
        for i in range(1, len(curve) - 1):
            lo, mid, hi = curve[i - 1:i + 2]
            x, y, z = [r['metrics']['median_ann'] for r in [lo, mid, hi]]
            triplets.append(dict(model_spec=spec, left=lo['id'], middle=mid['id'], right=hi['id'],
                                 amplitudes=plan['amplitudes'][i - 1:i + 2], left_value=x, middle_value=y, right_value=z,
                                 nonmonotonic=bool((y - x) * (z - y) < 0)))
    for row in rows:
        neighbors = {}
        for curve in curves:
            for at, r in enumerate(curve):
                if r['id'] == row['id']:
                    for i in [at - 1, at + 1]:
                        if 0 <= i < len(curve):
                            neighbors[curve[i]['id']] = curve[i]
        values = [r['metrics']['median_ann'] for r in neighbors.values()]
        row['neighbor_count'] = len(values)
        row['neighbor_q25'] = float(np.quantile(values, .25)) if len(values) >= plan['search']['min_neighbors'] else None
    return triplets
