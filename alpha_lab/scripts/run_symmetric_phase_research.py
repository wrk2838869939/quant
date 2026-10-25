#!/usr/bin/env python3
"""固定主板对称分治模型与选择后诊断；不读TEST、不搜索阈值。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from divide_selection import phase_features
from value_forecast import fit_value_model
from symmetric_phase import fit_symmetric


def predict_bundle(frame, models, cfg, prefix=''):
    """优先级和费前收益分开；只在新仓score上应用现有门槛。"""
    expanded, _ = phase_features(frame, cfg['features'])
    result = expanded[['date', 'ts_code', 'phase', 'trend']].copy()
    for policy in cfg['policies']:
        pred = models[policy].predict(frame)
        name = prefix + policy
        for column in pred:
            result[name+'__'+column] = pred[column]
        score = -frame.x_turnover if policy == 'low_turnover' else pred.expected_return
        gate = (frame.trend & pred.expected_return.gt(cfg['entry']['min_expected_return'])
                & pred.expected_adverse.le(cfg['entry']['max_expected_adverse']))
        result[name+'__score'] = score.where(gate)
    return result


def diagnostics(predictions, cfg, out):
    """日期等权、Top-1/5校准与同阶段近邻；近邻不是可交易组合。"""
    names = cfg['policies'] + cfg['execution_transfer_policies']
    error_rows, selected_rows, opportunity_rows = [], [], []
    neighbors = cfg['diagnostics']['match_neighbors']
    fields = cfg['diagnostics']['match_features']
    for date, group in predictions.groupby('date'):
        realized = group.ret_60
        phase_means = group.groupby('phase').ret_60.mean()
        for name in names:
            mu = group[name+'__expected_return']
            for phase, part in [('all', group), *list(group.groupby('phase'))]:
                residual = part[name+'__expected_return'] - part.ret_60
                error_rows.append(dict(date=date, year=date.year, policy=name, phase=phase,
                    n=len(part), mse=(residual**2).mean(), bias=residual.mean(),
                    time_mse=residual.mean()**2,
                    within_mse=((residual-residual.mean())**2).mean()))
            admitted = group[group[name+'__score'].notna()]
            ranked = admitted.sort_values([name+'__score', 'ts_code'], ascending=[False, True])
            for k in cfg['diagnostics']['top_k']:
                chosen = ranked.head(k)
                matches = []
                for _, stock in chosen.iterrows():
                    pool = group[group.phase.eq(stock.phase) & ~group.ts_code.isin(chosen.ts_code)]
                    matched_return, distance = np.nan, np.nan
                    if len(pool) >= neighbors:
                        distances = cdist(stock[fields].to_numpy(float)[None, :], pool[fields].to_numpy(float))[0]
                        order = np.lexsort((pool.ts_code.to_numpy(), distances))[:neighbors]
                        matched_return = pool.iloc[order].ret_60.mean()
                        distance = distances[order].mean()
                    matches.append(matched_return)
                    selected_rows.append(dict(date=date, year=date.year, policy=name, k=k,
                        ts_code=stock.ts_code, phase=stock.phase, predicted=stock[name+'__expected_return'],
                        realized=stock.ret_60, mae=stock.mae_60, entry_missing=stock.entry_missing_60,
                        matched_return=matched_return, match_distance=distance,
                        matched_excess=stock.ret_60-matched_return))
                phase_return = chosen.phase.map(phase_means)
                matched = pd.Series(matches, dtype=float)
                opportunity_rows.append(dict(date=date, year=date.year, policy=name, k=k,
                    n=len(chosen), admitted=len(admitted),
                    predicted=chosen[name+'__expected_return'].mean(), realized=chosen.ret_60.mean(),
                    optimism=(chosen[name+'__expected_return']-chosen.ret_60).mean(),
                    mae=chosen.mae_60.mean(), pool_return=realized.mean(),
                    between_phase=(phase_return-realized.mean()).mean(),
                    within_phase=(chosen.ret_60-phase_return).mean(),
                    matched_excess=(chosen.ret_60.reset_index(drop=True)-matched).mean(),
                    match_coverage=matched.notna().mean(), no_entry=int(chosen.entry_missing_60.sum())))
    for label, rows, keys in [('errors', error_rows, ['policy','phase']),
                             ('opportunities', opportunity_rows, ['policy','k'])]:
        table = pd.DataFrame(rows)
        table.to_csv(out/(label+'_dates.csv'), index=False)
        numeric = table.select_dtypes(include=np.number).columns.difference(['year','k'])
        table.groupby(keys)[numeric].mean().to_csv(out/(label+'_summary.csv'))
        table.groupby(['year']+keys)[numeric].mean().to_csv(out/(label+'_yearly.csv'))
    pd.DataFrame(selected_rows).to_csv(out/'selected_opportunities.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args();out = args.run.resolve()
    cfg = yaml.safe_load((out/'config.yaml').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['status'] != 'registered':
        raise ValueError('仅允许运行尚未拟合的已登记实验')
    source_paths = [ROOT/cfg['source_panel'], ROOT/cfg['execution_features']]
    source_hashes = {}
    for path in [Path(__file__), ROOT/'src/symmetric_phase.py', ROOT/'src/value_forecast.py',
                 ROOT/'src/divide_selection.py', ROOT/'configs/symmetric_phase_research.yaml']:
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = out/'sources'/path.relative_to(ROOT);dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    manifest['source_hashes'] = source_hashes
    manifest['input_hashes'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    panel = pd.read_parquet(source_paths[0])
    if panel.date.max() >= pd.Timestamp('2022-01-01'):
        raise ValueError('训练/诊断源越过既有2014—2021边界')
    panel = panel[panel.monthly & panel.trend].copy()
    execution = pd.read_parquet(source_paths[1])
    if execution.date.max() >= pd.Timestamp('2024-01-01'):
        raise ValueError('禁止TEST执行特征')
    execution = execution[execution.monthly & execution.ts_code.str.match(cfg['mainboard_pattern'])].copy()
    evaluation = panel[panel.ts_code.str.match(cfg['mainboard_pattern']) &
        panel.date.between(*map(pd.Timestamp, cfg['development'])) &
        panel.label_end_60.notna() & panel.label_end_60.le(pd.Timestamp(cfg['development'][1]))].copy()
    all_predictions = {'development': None, 'execution': None}
    metadata = {}
    for universe in cfg['training_universes']:
        train = panel[panel.ts_code.str.match(cfg['mainboard_pattern'])] if universe == 'mainboard' else panel
        for stage, cutoff, frame in [('development', cfg['fit_end'], evaluation),
                                     ('execution', cfg['final_fit_end'], execution)]:
            models = {
                'low_turnover':fit_value_model(train, [], asof=cutoff, **cfg['forecast']),
                'global_return':fit_value_model(train, cfg['features'], asof=cutoff, **cfg['forecast']),
                'symmetric_intercept':fit_symmetric(train, cfg['features'], phases=cfg['phases'], slopes=False,
                    asof=cutoff, **cfg['forecast']),
                'symmetric_slopes':fit_symmetric(train, cfg['features'], phases=cfg['phases'], slopes=True,
                    asof=cutoff, **cfg['forecast']),
            }
            prefix = '' if universe == 'mainboard' else 'transfer_'
            pred = predict_bundle(frame, models, cfg, prefix)
            current = all_predictions[stage]
            all_predictions[stage] = pred if current is None else pd.concat(
                [current, pred.drop(columns=['date','ts_code','phase','trend'])], axis=1)
            metadata[universe+'_'+stage] = {name:model.metadata for name,model in models.items()}
            joblib.dump(models, out/(universe+'_'+stage+'_models.joblib'))
            print(universe, stage, 'models frozen', flush=True)
    for stage, pred in all_predictions.items():
        if stage == 'development':
            for name in ['ret_60','mae_60','entry_missing_60','label_end_60']+cfg['diagnostics']['match_features']:
                pred[name] = evaluation[name]
        pred.to_parquet(out/(stage+'_predictions.parquet'), index=False)
    (out/'model_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('execution_predictions.parquet ready', flush=True)
    diagnostics(all_predictions['development'], cfg, out)
    manifest.update(status='models_and_diagnostics_complete',
        model_count=16, evaluation_dates=evaluation.date.nunique(),
        evaluation_universe='mainboard', feature_rank_universe=cfg['feature_rank_universe'],
        diagnostic_semantics='overlapping_60d_opportunities_not_portfolio; matched_neighbors_not_investable')
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('diagnostics complete', flush=True)


if __name__ == '__main__':
    main()
