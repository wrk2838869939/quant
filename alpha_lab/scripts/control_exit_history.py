"""结果后提出的复杂度对照：波动率及波动率加回撤，不搜索阈值。"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from research_exit_history import ROOT, verify, save, digest
from exit_history import annual_fit, predict_frame

run = Path(sys.argv[1]).resolve()
cfg = verify(run)
out = run / 'controls'
out.mkdir()
controls = {'volatility': ['volatility20'], 'volatility_drawdown': ['volatility20', 'drawdown120']}
save(out / 'protocol.json', dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
    posthoc=True, reason='主实验结果后检查九特征风险模型是否优于简单风险状态',
    target='mae', horizon=60, features=controls, alpha=10.0, years=cfg['prediction_years'],
    no_threshold_search=True, no_strategy_admission=True, source_sha256=digest(__file__)))
frame = pd.read_parquet(run / 'data/events.parquet')
frame = frame[frame.nonoverlap_primary]
original = pd.read_parquet(ROOT / cfg['parent'] / 'data/exit_events.parquet')
original = original[original.nonoverlap_primary]
rows, checks = [], []
for name, features in controls.items():
    spec = deepcopy(cfg)
    spec['features'] = features
    for year in cfg['prediction_years']:
        train, model, mean, threshold = annual_fit(frame, year, 60, 'mae', spec)
        # 独立矩阵解复算；训练事件依然要求原九特征完整，保持样本一致。
        assert model is not None and train.label_end_60.lt(pd.Timestamp(f'{year}-01-01')).all()
        x = train[features].to_numpy()
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale == 0] = 1.
        z = (x-center)/scale
        zmean = z.mean(axis=0)
        z -= zmean
        coef = np.linalg.solve(z.T @ z + spec['model']['alpha'] * np.eye(len(features)), z.T @ (train.mae_60-mean))
        error = 0.
        for cohort, data in [('shadow', frame)] + ([('original_transfer', original)] if year >= 2024 else []):
            test = data[data.date.dt.year.eq(year)]
            predicted = predict_frame(test, model, mean, features)
            manual = np.full(len(test), mean)
            valid = test.features_complete
            manual[valid] = ((test.loc[valid, features].to_numpy()-center)/scale-zmean) @ coef + mean
            if len(test):
                error = max(error, float(np.max(np.abs(manual-predicted))))
            assert error < 1e-10
            for (_, r), value in zip(test.iterrows(), predicted):
                rows.append(dict(model=name, cohort=cohort, date=r.date, ts_code=r.ts_code, year=year,
                    prediction=value, baseline=mean, actual=r.mae_60, terminal=r.ret_60,
                    high_group=value > threshold))
        checks.append(dict(model=name, year=year, train_events=len(train), manual_error=error))
controls_pred = pd.DataFrame(rows)
controls_pred.to_csv(out / 'predictions.csv', index=False)
full = pd.read_csv(run / 'models/predictions.csv', parse_dates=['date'])
full = full[full.target.eq('mae') & full.horizon.eq(60)].assign(model='nine_features')
all_pred = pd.concat([controls_pred, full], ignore_index=True)
metrics = []
for (model, cohort), raw in all_pred.groupby(['model', 'cohort']):
    groups = [('2018_2023', raw[raw.year < 2024]), ('2024_2026', raw[raw.year >= 2024])]
    groups += [(str(year), group) for year, group in raw.groupby('year')]
    for period, g in groups:
        g = g[g.actual.notna()].copy()
        if g.empty:
            continue
        within = g.groupby('date').filter(lambda group: len(group) >= cfg['min_date_group']).copy()
        x = within.prediction - within.groupby('date').prediction.transform('mean')
        y = within.actual - within.groupby('date').actual.transform('mean')
        hi, lo = g[g.high_group], g[~g.high_group]
        metrics.append(dict(model=model, cohort=cohort, period=period, n=len(g),
            r2=1-((g.actual-g.prediction)**2).sum()/((g.actual-g.baseline)**2).sum(),
            ic=float(spearmanr(g.prediction, g.actual).statistic),
            within_date_ic=float(spearmanr(x, y).statistic) if len(within) > 2 else np.nan,
            high_mae=hi.actual.mean(), low_mae=lo.actual.mean(),
            high_deep_loss=(hi.actual <= cfg['deep_loss']).mean(), low_deep_loss=(lo.actual <= cfg['deep_loss']).mean()))
pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
save(out / 'completion.json', dict(status='complete', posthoc=True, manual_checks=checks,
    source_sha256=digest(__file__), artifacts={p.name: digest(p) for p in out.iterdir()}))
print(pd.DataFrame(metrics)[lambda f: f.period.isin(['2018_2023', '2024_2026'])].round(4).to_string(index=False))
