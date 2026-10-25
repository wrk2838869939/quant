"""独立矩阵解和逐事件行情复算，并检查生命周期来自当时候选。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from research_exit_history import ROOT, verify, digest, save

run = Path(sys.argv[1]).resolve()
cfg = verify(run)
all_events = pd.read_parquet(run / 'data/events.parquet')
frame = all_events[all_events.nonoverlap_primary]
original = pd.read_parquet(ROOT / cfg['parent'] / 'data/exit_events.parquet')
original = original[original.nonoverlap_primary]
pred = pd.read_csv(run / 'models/predictions.csv', parse_dates=['date'])
close = pd.read_parquet(run / 'data/close.parquet')
sig = pd.read_parquet(run / 'data/signals.parquet')
hold = pd.read_parquet(run / 'data/hold.parquet')
entries = pd.read_csv(run / 'data/entries.csv', parse_dates=['date'])
for date, g in entries.groupby('date'):
    selected = sig.loc[date].dropna().sort_values(ascending=False, kind='stable').head(cfg['top_n'])
    assert set(g.ts_code) <= set(selected.index)
for r in all_events.itertuples():
    assert r.entry_date < r.date and hold.at[r.date, r.ts_code] == 0
    assert not hold.loc[(hold.index > r.entry_date) & (hold.index < r.date), r.ts_code].eq(0).any()
    for h in cfg['horizons']:
        start = close.index.get_loc(r.date) + 1
        end = start + h
        if end >= len(close):
            assert getattr(r, f'right_censored_{h}') and np.isnan(getattr(r, f'ret_{h}'))
            continue
        values = close.iloc[start:end+1][r.ts_code].to_numpy()
        if not np.isfinite(values).all() or (values <= 0).any():
            assert np.isnan(getattr(r, f'ret_{h}'))
            continue
        path = values / values[0] - 1
        assert abs(path[-1] - getattr(r, f'ret_{h}')) < 1e-12
        assert abs(path.min() - getattr(r, f'mae_{h}')) < 1e-12
        assert getattr(r, f'label_end_{h}') == close.index[end]
for code, g in frame.groupby('ts_code'):
    assert (np.diff(close.index.get_indexer(g.sort_values('date').date)) > cfg['nonoverlap_days']).all()

records = []
for (target, h, year), group in pred.groupby(['target', 'horizon', 'year']):
    cutoff = pd.Timestamp(f'{year}-01-01')
    train = frame[frame.date.lt(cutoff) & frame[f'label_end_{h}'].lt(cutoff)
                  & frame[f'{target}_{h}'].notna() & frame.features_complete]
    saved = pd.read_csv(run / 'models' / f'{target}_h{h}_{year}_train.csv', parse_dates=['date', f'label_end_{h}'])
    assert set(zip(train.date, train.ts_code)) == set(zip(saved.date, saved.ts_code))
    assert saved[f'label_end_{h}'].lt(cutoff).all()
    y = train[f'{target}_{h}'].to_numpy()
    mean = y.mean() if len(y) else 0.
    fit = len(train) >= cfg['model']['min_train_events'] and train.date.nunique() >= cfg['model']['min_train_dates']
    if fit:
        x = train[cfg['features']].to_numpy()
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale == 0] = 1.
        z = (x-center)/scale
        zmean = z.mean(axis=0)
        z -= zmean
        coef = np.linalg.solve(z.T @ z + cfg['model']['alpha'] * np.eye(z.shape[1]), z.T @ (y-mean))
    max_error = 0.
    for cohort, p in group.groupby('cohort'):
        data = frame if cohort == 'shadow' else original
        test = data.set_index(['date', 'ts_code']).loc[p.set_index(['date', 'ts_code']).index]
        predicted = np.full(len(p), mean)
        if fit:
            valid = test.features_complete.to_numpy()
            predicted[valid] = ((test.loc[test.features_complete, cfg['features']].to_numpy()-center)/scale-zmean) @ coef + mean
        error = float(np.max(np.abs(predicted-p.prediction))) if len(p) else 0.
        assert error < 1e-10
        max_error = max(max_error, error)
    records.append(dict(target=target, horizon=int(h), year=int(year), train_events=len(train),
                        model_fitted=bool(fit), manual_prediction_max_error=max_error))
save(run / 'independent_verification.json', dict(status='passed', source_sha256=digest(__file__),
    exits_verified=len(all_events), labels_and_mae_recomputed=True, entries_verified=len(entries),
    nonoverlap_checked=True, models=records))
print('Verified', len(all_events), 'exit paths;', len(entries), 'entries;', len(records), 'manual prediction reconstructions')
