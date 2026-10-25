"""固定预测的分期、同日横截面、路径风险及缺失敏感性分析。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from research_exit_history import ROOT, verify, save, digest

run = Path(sys.argv[1]).resolve()
cfg = verify(run)
out = run / 'analysis'
out.mkdir()
predictions = pd.read_csv(run / 'models/predictions.csv', parse_dates=['date', 'label_end'])


def correlation(x, y):
    return float(spearmanr(x, y).statistic) if len(x) >= 3 and x.nunique() > 1 and y.nunique() > 1 else np.nan


def metrics(raw):
    g = raw[raw.actual.notna()].copy()
    if g.empty:
        return dict(n=0)
    g['se'] = (g.actual - g.prediction) ** 2
    g['base_se'] = (g.actual - g.baseline) ** 2
    per_date = g.groupby('date')[['se', 'base_se']].mean()
    dates = g.groupby('date').filter(lambda group: len(group) >= cfg['min_date_group']).copy()
    dates['pred_resid'] = dates.prediction - dates.groupby('date').prediction.transform('mean')
    dates['actual_resid'] = dates.actual - dates.groupby('date').actual.transform('mean')
    high, low = g[g.high_group], g[~g.high_group]
    return dict(n=len(g), dates=g.date.nunique(), missing_not_censored=int((raw.actual.isna() & ~raw.right_censored).sum()),
        censored=int(raw.right_censored.sum()), r2=1-g.se.sum()/g.base_se.sum(), ic=correlation(g.prediction, g.actual),
        date_balanced_r2=1-per_date.se.mean()/per_date.base_se.mean(),
        within_date_n=len(dates), within_date_dates=dates.date.nunique(),
        within_date_ic=correlation(dates.pred_resid, dates.actual_resid),
        mean_actual=g.actual.mean(), high_n=len(high), low_n=len(low),
        high_actual=high.actual.mean(), low_actual=low.actual.mean(),
        high_terminal=high.terminal.mean(), low_terminal=low.terminal.mean(),
        high_mae=high.mae.mean(), low_mae=low.mae.mean(),
        high_deep_loss=(high.mae <= cfg['deep_loss']).mean(), low_deep_loss=(low.mae <= cfg['deep_loss']).mean(),
        high_negative_terminal=(high.terminal < 0).mean(), low_negative_terminal=(low.terminal < 0).mean())


summary, yearly, pressure, influence, centered = [], [], [], [], []
for (cohort, target, horizon), raw in predictions.groupby(['cohort', 'target', 'horizon']):
    identity = dict(cohort=cohort, target=target, horizon=horizon)
    for period, group in [('all', raw), ('2018_2023', raw[raw.year < 2024]), ('2024_2026', raw[raw.year >= 2024])]:
        if group.empty:
            continue
        summary.append(dict(**identity, period=period, **metrics(group)))
        stress = group.copy()
        missing = stress.actual.isna() & ~stress.right_censored
        stress.loc[missing, ['actual', 'terminal', 'mae']] = -1.
        pressure.append(dict(**identity, period=period, assumed_total_losses=int(missing.sum()), **metrics(stress)))
        for year in sorted(group.year.unique()):
            influence.append(dict(**identity, period=period, omitted_year=int(year), **metrics(group[group.year.ne(year)])))
    for year, group in raw.groupby('year'):
        yearly.append(dict(**identity, year=int(year), **metrics(group)))
    complete = raw[raw.actual.notna()]
    for date, group in complete.groupby('date'):
        if len(group) >= cfg['min_date_group']:
            centered.append(dict(**identity, date=date, n=len(group),
                within_date_ic=correlation(group.prediction, group.actual),
                mean_prediction=group.prediction.mean(), mean_actual=group.actual.mean()))

for name, rows in [('metrics', summary), ('yearly_metrics', yearly), ('missing_price_stress', pressure),
                   ('leave_year_out', influence), ('same_date_groups', centered)]:
    pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)

# 两个独立目标在相同事件上的联合状态，只作固定预测的描述。
returns = predictions[predictions.target.eq('ret')].copy()
risk = predictions[predictions.target.eq('mae')][['cohort', 'horizon', 'date', 'ts_code', 'high_group']]
joint = returns.merge(risk.rename(columns={'high_group': 'predicted_safer'}),
                      on=['cohort', 'horizon', 'date', 'ts_code'], validate='one_to_one')
joint['period'] = np.where(joint.year < 2024, '2018_2023', '2024_2026')
rows = []
for keys, g in joint[joint.actual.notna()].groupby(['cohort', 'horizon', 'period', 'high_group', 'predicted_safer']):
    rows.append(dict(zip(['cohort', 'horizon', 'period', 'higher_return_prediction', 'safer_prediction'], keys),
        n=len(g), terminal_mean=g.terminal.mean(), mae_mean=g.mae.mean(),
        deep_loss_rate=(g.mae <= cfg['deep_loss']).mean(), loss_rate=(g.terminal < 0).mean()))
pd.DataFrame(rows).to_csv(out / 'joint_return_risk.csv', index=False)

events = pd.read_parquet(run / 'data/events.parquet')
coverage = []
for year, group in events.groupby(events.date.dt.year):
    primary = group[group.nonoverlap_primary]
    coverage.append(dict(year=int(year), exits=len(group), primary=len(primary), stocks=primary.ts_code.nunique(),
        missing_features=int((~primary.features_complete).sum()), complete20=int(primary.ret_20.notna().sum()),
        complete60=int(primary.ret_60.notna().sum()), missing20=int((primary.ret_20.isna() & ~primary.right_censored_20).sum()),
        missing60=int((primary.ret_60.isna() & ~primary.right_censored_60).sum())))
pd.DataFrame(coverage).to_csv(out / 'sample_by_year.csv', index=False)
save(out / 'completion.json', dict(status='complete', source_sha256=digest(__file__),
    artifacts={p.name: digest(p) for p in out.iterdir()}))
print(pd.DataFrame(summary)[['cohort', 'target', 'horizon', 'period', 'n', 'r2', 'ic', 'within_date_ic']].round(4).to_string(index=False))
