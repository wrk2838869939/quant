#!/usr/bin/env python3
"""固定板块确认日比较个股门槛，并只在历史窗口内训练排序。"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'alpha_lab/src'), str(ROOT / 'alpha_lab/scripts')]
from sector_dataset import digest
from stock_selection_asof import cross_section_features, fit_stock_model, select_positive_top
from report_sector_asof_replay import cluster_interval


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')


def evaluate_baskets(frame, obs, variants, events, scopes, cfg):
    """没有选股时保留空信号；未知价格路径不当作零收益。"""
    baseline = obs.groupby(['event_id', 'horizon']).agg(base_n=('ts_code', 'size'),
                   base_complete=('complete', 'all'), base_return=('price_return', 'mean')).reset_index()
    baseline.loc[~baseline.base_complete, 'base_return'] = np.nan
    rows = []
    for scope, ids in scopes.items():
        scope_frame = frame[frame.event_id.isin(ids)]
        for variant in variants:
            chosen = scope_frame[scope_frame[variant]]
            values = chosen[['event_id', 'ts_code']].merge(obs, on=['event_id', 'ts_code'], validate='one_to_many')
            for e in events[events.event_id.isin(ids)].itertuples():
                for horizon in cfg['horizons']:
                    part = values[values.event_id.eq(e.event_id) & values.horizon.eq(horizon)]
                    b = baseline[baseline.event_id.eq(e.event_id) & baseline.horizon.eq(horizon)].iloc[0]
                    n = len(part)
                    complete = bool(n and part.complete.all())
                    ret = part.price_return.mean() if complete else np.nan
                    # 每股最多20%事件额度，五股以上仍等权；余下为零收益现金。
                    exposure = min(1., n * cfg['statistics']['maximum_primary_stock_weight'])
                    capped = ret * exposure if complete else 0. if n == 0 else np.nan
                    row = dict(scope=scope, variant=variant, event_id=e.event_id, date=e.date, horizon=horizon,
                               stocks=n, signaled=bool(n), complete=complete, price_return=ret,
                               base_complete=bool(b.base_complete), base_return=b.base_return,
                               excess=ret - b.base_return, capped_exposure=exposure, capped_return=capped,
                               # 有信号但缺价格时保持未知，无信号空仓为0。
                               slot_return=ret if n else 0.)
                    for stress in cfg['statistics']['missing_stress_returns']:
                        key = f'stress_{abs(int(stress * 100))}'
                        row[key] = part.price_return.where(part.complete, stress).mean() if n else 0.
                    rows.append(row)
    return pd.DataFrame(rows), baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'alpha_lab/configs/stock_selection_asof_v2.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    parent = ROOT / cfg['parent_run']
    old_manifest = json.loads((parent / 'manifest.json').read_text())
    assert old_manifest['status'] == 'complete' and not old_manifest['data_vintage_certified']
    filenames = ['stock_decisions_BEFORE_EVALUATION.parquet', 'predictions_BEFORE_EVALUATION.parquet',
                 'price_observations_NOT_ACCOUNT.parquet', 'observation_hfq_open.parquet']
    for name in filenames:
        assert digest(parent / name) == old_manifest['artifacts'][name], name
    out = ROOT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    (out / 'sources').mkdir()
    (out / 'models').mkdir()
    sources = [Path(__file__), args.config, ROOT / 'alpha_lab/src/stock_selection_asof.py',
               ROOT / 'alpha_lab/tests/test_stock_selection_asof.py', ROOT / 'alpha_lab/scripts/report_sector_asof_replay.py',
               ROOT / 'alpha_lab/docs/current_strategy_20261013/STOCK_SELECTION_V2_PREREG.md']
    for source in sources:
        shutil.copy2(source, out / 'sources' / source.name)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    manifest = dict(status='running', parent=str(parent), data_vintage_certified=False, account_backtest=False,
                    sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
                    inputs={name: digest(parent / name) for name in filenames})
    save_json(out / 'manifest.json', manifest)
    print('RUN', out, flush=True)
    raw = pd.read_parquet(parent / filenames[0])
    frame = raw[raw.date.eq(raw.confirmation_date) & raw.qualified].copy().reset_index(drop=True)
    pred = pd.read_parquet(parent / filenames[1])
    events = pred[pred.selected][['event_id', 'date', 'sector_code']].copy()
    assert set(frame.event_id) == set(events.event_id) and not frame.duplicated(['event_id', 'ts_code']).any()
    features, columns = cross_section_features(frame, cfg['model'])
    frame = frame.merge(features.drop(columns='date'), on=['event_id', 'ts_code'], validate='one_to_one')
    frame['all_members'] = True
    for variant, field, ascending in [('price_strength5', 'strength', False), ('price_lowvol5', 'volatility', True)]:
        frame[variant] = False
        group = frame[frame.price & frame[field].notna()].sort_values(['event_id', field, 'ts_code'], ascending=[True, ascending, True])
        frame.loc[group.groupby('event_id').head(cfg['top_n']).index, variant] = True
    frame.to_parquet(out / 'fixed_rules_and_features_BEFORE_EVALUATION.parquet', index=False)
    observations = pd.read_parquet(parent / filenames[2])
    obs = observations.merge(events[['event_id', 'date']], on=['event_id', 'date'], validate='many_to_one')
    assert len(obs) == len(frame) * len(cfg['horizons'])
    calendar = pd.read_parquet(parent / filenames[3], columns=[]).index
    label = obs[obs.horizon.eq(cfg['primary_horizon'])].copy()
    group_complete = label.groupby('event_id').complete.transform('all')
    group_mean = label.groupby('event_id').price_return.transform('mean')
    label['target'] = (label.price_return - group_mean).where(group_complete)
    label['label_complete'] = group_complete
    data = frame.merge(label[['event_id', 'ts_code', 'exit_date', 'target', 'label_complete']], on=['event_id', 'ts_code'], validate='one_to_one')
    predictions, fits = [], []
    for year in cfg['model']['test_years']:
        start = pd.Timestamp(year=year, month=1, day=1)
        target = frame[frame.date.dt.year.eq(year)].copy()
        try:
            model, audit = fit_stock_model(data, start, calendar, cfg['model'], columns)
        except ValueError as error:
            fits.append(dict(year=year, status='skipped', reason=str(error)))
            print('SKIP', year, str(error), flush=True)
            continue
        target['model_score'] = model.predict(target[columns])
        target['model_start'] = start
        target['ridge_price5'] = False
        target.loc[select_positive_top(target, 'model_score', cfg), 'ridge_price5'] = True
        # 删除未来记录后，真实年度参数与权重必须逐值一致。
        prefix, paudit = fit_stock_model(data[data.date.lt(start)], start, calendar, cfg['model'], columns)
        assert audit == paudit
        np.testing.assert_array_equal(model.coef_, prefix.coef_)
        predictions.append(target[['event_id', 'ts_code', 'model_score', 'model_start', 'ridge_price5']])
        fits.append(dict(year=year, status='complete', prefix_exact=True, **audit))
        np.savez(out / 'models' / f'{year}.npz', coef=model.coef_, features=columns, alpha=audit['alpha'])
        print('MODEL', year, 'train events', len(audit['train_events']), 'fit', len(audit['fit_events']),
              'valid', len(audit['valid_events']), 'alpha', audit['alpha'], 'selected', int(target.ridge_price5.sum()), flush=True)
    scores = pd.concat(predictions, ignore_index=True)
    scores.to_parquet(out / 'stock_scores_BEFORE_EVALUATION.parquet', index=False)
    save_json(out / 'fits.json', fits)
    data.to_parquet(out / 'research_labels_WITH_FUTURE.parquet', index=False)
    frame = frame.merge(scores, on=['event_id', 'ts_code'], how='left', validate='one_to_one')
    frame['ridge_price5'] = frame.ridge_price5.fillna(False).astype(bool)
    variants = ['original', 'no_expansion', 'price_trend', 'trend_only', 'price_only', 'all_members',
                'price_strength5', 'price_lowvol5', 'ridge_price5']
    frame.to_parquet(out / 'all_decisions_BEFORE_EVALUATION.parquet', index=False)
    scopes = dict(all_sector_pass=set(events.event_id), stock_model_available=set(scores.event_id))
    result, baseline = evaluate_baskets(frame, obs, variants, events, scopes, cfg)
    # 模型不可评分年份不视为模型空仓表现，模型结果只报告共同可评分期。
    result = result[~(result.variant.eq('ridge_price5') & result.scope.eq('all_sector_pass'))].copy()
    result.to_csv(out / 'event_baskets_NOT_ACCOUNT.csv', index=False)
    summary, annual = [], []
    for (scope, variant, horizon), group in result.groupby(['scope', 'variant', 'horizon']):
        signals = group[group.signaled]
        complete = signals[signals.complete]
        paired = complete[complete.base_complete]
        summary.append(dict(scope=scope, variant=variant, horizon=horizon, total_events=len(group),
                            signals=len(signals), complete_signals=len(complete), paired_events=len(paired),
                            stocks=int(signals.stocks.sum()), mean=complete.price_return.mean(), median=complete.price_return.median(),
                            q10=complete.price_return.quantile(.1), positive=complete.price_return.gt(0).mean(),
                            mean_excess=paired.excess.mean(), mean_slot=group.slot_return.mean(),
                            capped_slot_mean=group.capped_return.mean(), **cluster_interval(paired, calendar, cfg['statistics'])))
        for year, part in group.groupby(group.date.dt.year):
            annual.append(dict(scope=scope, variant=variant, horizon=horizon, year=year, events=len(part),
                               signals=int(part.signaled.sum()), mean=part.loc[part.complete, 'price_return'].mean(),
                               mean_excess=part.excess.mean(), mean_slot=part.slot_return.mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'summary_NOT_ACCOUNT.csv', index=False)
    pd.DataFrame(annual).to_csv(out / 'annual_NOT_ACCOUNT.csv', index=False)
    # 互斥组仅用于机制诊断，不作为新增可部署策略。
    primary = obs[obs.horizon.eq(cfg['primary_horizon'])]
    diagnostic = frame.merge(primary, on=['event_id', 'ts_code', 'date'], validate='one_to_one')
    diagnostic['stratum'] = np.select([diagnostic.trend & diagnostic.price, diagnostic.trend,
                                       diagnostic.price], ['trend_and_price', 'trend_without_price', 'price_without_trend'], default='neither')
    stratum = diagnostic.groupby(['event_id', 'date', 'stratum']).agg(stocks=('ts_code', 'size'), complete=('complete', 'all'), value=('price_return', 'mean')).reset_index()
    stratum.loc[~stratum.complete, 'value'] = np.nan
    stratum.to_csv(out / 'trend_price_strata.csv', index=False)
    comparisons = []
    for a, b in [('trend_and_price', 'price_without_trend'), ('trend_and_price', 'trend_without_price'),
                 ('price_without_trend', 'neither'), ('trend_without_price', 'neither')]:
        wide = stratum.pivot(index=['event_id', 'date'], columns='stratum', values='value')[[a, b]].dropna()
        paired = (wide[a] - wide[b]).rename('excess').reset_index()
        comparisons.append(dict(comparison=a + '_minus_' + b, paired_events=len(paired), mean=paired.excess.mean(),
                                median=paired.excess.median(), **cluster_interval(paired, calendar, cfg['statistics'])))
    pd.DataFrame(comparisons).to_csv(out / 'gate_paired_comparisons.csv', index=False)
    # 在个股模型可评分期、模型相同入场事件比较价格池，隔离板块与日期。
    price_base = result[(result.scope.eq('stock_model_available')) & result.variant.eq('price_only')][['event_id', 'horizon', 'price_return']].rename(columns={'price_return': 'price_pool_return'})
    ml = result[result.scope.eq('stock_model_available') & result.variant.eq('ridge_price5')].merge(price_base, on=['event_id', 'horizon'], validate='one_to_one')
    ml['price_pool_excess'] = ml.price_return - ml.price_pool_return
    ml.to_csv(out / 'ridge_vs_same_date_price_pool.csv', index=False)
    pair_rows = []
    for horizon, g in ml.groupby('horizon'):
        paired = g.dropna(subset=['price_pool_excess']).copy()
        paired['excess'] = paired.price_pool_excess
        pair_rows.append(dict(horizon=horizon, pairs=len(paired), mean=paired.excess.mean(), median=paired.excess.median(),
                              **cluster_interval(paired, calendar, cfg['statistics'])))
    pd.DataFrame(pair_rows).to_csv(out / 'ridge_vs_price_pool_summary.csv', index=False)
    # 所有同年统计和缺失压力情景完整保存，不按最佳持有期重新选择。
    sensitivity = []
    for (scope, variant), g in result[result.horizon.eq(cfg['primary_horizon'])].groupby(['scope', 'variant']):
        paired = g.dropna(subset=['excess'])
        largest = paired.excess.idxmax() if len(paired) else None
        for name, part in [('all', paired), ('exclude_2022', paired[paired.date.dt.year.ne(2022)]),
                           ('exclude_largest_positive', paired.drop(index=largest) if largest is not None else paired)]:
            sensitivity.append(dict(scope=scope, variant=variant, sensitivity=name, events=len(part), mean_excess=part.excess.mean()))
    pd.DataFrame(sensitivity).to_csv(out / 'sensitivity.csv', index=False)
    result.groupby(['scope', 'variant', 'horizon'])[['stress_20', 'stress_100']].mean().reset_index().to_csv(out / 'missing_path_stress_NOT_ACCOUNT.csv', index=False)
    coefs = []
    for fit in fits:
        if fit['status'] != 'complete':
            continue
        archive = np.load(out / 'models' / f"{fit['year']}.npz")
        coefs.extend(dict(year=fit['year'], feature=k, coefficient=float(v)) for k, v in zip(columns, archive['coef']))
    pd.DataFrame(coefs).to_csv(out / 'model_coefficients.csv', index=False)
    frame[frame.ridge_price5].sort_values(['date', 'model_score'], ascending=[False, False]).to_csv(out / 'ridge_historical_candidates.csv', index=False)
    print(summary[summary.horizon.eq(cfg['primary_horizon'])].to_string(index=False), flush=True)
    print('GATE_COMPARISONS', pd.DataFrame(comparisons).to_string(index=False), flush=True)
    print('RIDGE_VS_PRICE', pd.DataFrame(pair_rows).to_string(index=False), flush=True)
    manifest.update(status='complete', sector_events=len(events), stock_rows=len(frame), stock_model_events=len(scopes['stock_model_available']),
                    prefix_checks_passed=True,
                    artifacts={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'})
    save_json(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
