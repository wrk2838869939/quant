#!/usr/bin/env python3
"""预登记收益幅度与不确定性诊断；只使用 2021 年及以前路径标签。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from value_forecast import fit_value_model


def prediction_parts(actual, predicted):
    """同日期均方误差拆为日期均值偏差与截面偏差。"""
    error = np.asarray(predicted) - np.asarray(actual)
    time = float(error.mean() ** 2)
    within = float(np.mean((error - error.mean()) ** 2))
    total = float(np.mean(error ** 2))
    assert np.isclose(total, time + within)
    return total, time, within


def chunk_predict(model, frame):
    return pd.concat([model.predict(frame.iloc[i:i + 20000])
                      for i in range(0, len(frame), 20000)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    source = ROOT / cfg['source_panel']
    panel = pd.read_parquet(source)
    assert panel.date.max() < pd.Timestamp('2022-01-01')
    train = panel[panel.monthly & panel.trend].copy()
    model_dir = out / 'models'
    model_dir.mkdir(exist_ok=True)
    features = {'constant': [], 'style': cfg['style_features'], 'full': cfg['features']}
    metadata = {}
    # 2022-2023 用最终成熟开发样本拟合；在查看这些执行结果前冻结模型。
    for name, columns in features.items():
        model = fit_value_model(train, columns, asof=cfg['history'][1], **cfg['forecast'])
        joblib.dump(model, model_dir / f'execution_{name}.joblib')
        metadata['execution_' + name] = model.metadata
        model.coefficients.to_csv(out / f'execution_{name}_coefficients.csv')
    (out / 'execution_models_ready.json').write_text(json.dumps(metadata, indent=2))
    print('execution models frozen', flush=True)
    evaluation = panel[panel.trend & panel.date.between(*map(pd.Timestamp, cfg['development']))
                       & panel.label_end_60.notna()].copy()
    # 不因未来缺成交或缺标记剔除评估池。筛除仅为训练标签删失敏感性诊断。
    variants = {
        'constant': (train, []), 'style': (train, cfg['style_features']),
        'full': (train, cfg['features']),
        'full_known_exit': (train[train.exit_marked_60.eq(False)], cfg['features']),
        'full_complete_path': (train[train.exit_marked_60.eq(False)
                                     & train.missing_marks_60.eq(0)], cfg['features']),
    }
    predictions = evaluation[['date', 'ts_code', 'monthly', 'weekly', 'ret_60', 'mae_60',
                              'entry_missing_60', 'exit_marked_60', 'missing_marks_60',
                              'turnover', 'x_turnover']].copy()
    coefficients = []
    for name, (fit, columns) in variants.items():
        model = fit_value_model(fit, columns, asof=cfg['fit_end'], **cfg['forecast'])
        joblib.dump(model, model_dir / f'development_{name}.joblib')
        pred = chunk_predict(model, evaluation)
        for column in pred:
            predictions[name + '__' + column] = pred[column]
        metadata['development_' + name] = model.metadata
        co = model.coefficients.reset_index(names='feature')
        co['model'] = name
        coefficients.append(co)
    predictions.to_parquet(out / 'development_predictions.parquet', index=False)
    pd.concat(coefficients).to_csv(out / 'development_coefficients.csv', index=False)
    errors, paths, admissions, sensitivity = [], [], [], []
    for frequency in cfg['review_frequencies']:
        for date, group in predictions[predictions[frequency]].groupby('date'):
            truth = group.ret_60
            for name in variants:
                pred = group[name + '__expected_return']
                total, time, within = prediction_parts(truth, pred)
                errors.append(dict(frequency=frequency, date=date, year=date.year,
                                   model=name, n=len(group), mse=total, time_mse=time,
                                   within_mse=within, rank_ic=pred.corr(truth, method='spearman')
                                   if pred.nunique() > 1 else np.nan,
                                   mean_prediction=pred.mean(), mean_return=truth.mean(),
                                   adverse_mae=np.mean(abs(group[name+'__expected_adverse']+group.mae_60))))
            point = group.full__expected_return > cfg['entry']['mean_lower_buffer']
            confidence = (group.full__mean_lower > cfg['entry']['mean_lower_buffer']) & (
                group.full__expected_adverse <= cfg['entry']['max_expected_adverse'])
            pools = {'trend_pool': group,
                     'low_turnover': group.sort_values(['turnover', 'ts_code']).head(5),
                     'style_top5': group.sort_values(['style__expected_return','ts_code'], ascending=[False,True]).head(5),
                     'full_top5': group.sort_values(['full__expected_return','ts_code'], ascending=[False,True]).head(5),
                     'full_point': group[point].sort_values(['full__expected_return','ts_code'], ascending=[False,True]).head(5),
                     'full_confidence': group[confidence].sort_values(['full__expected_return','ts_code'], ascending=[False,True]).head(5)}
            for policy, chosen in pools.items():
                n = len(chosen)
                # 新开仓机会诊断，缺席的五个名额计现金；不是有旧仓的组合回测。
                opportunity_return = chosen.ret_60.mean() if n else np.nan
                cash_slot_return = chosen.ret_60.sum()/5 if policy != 'trend_pool' else truth.mean()
                paths.append(dict(frequency=frequency, date=date, year=date.year, policy=policy,
                                  n=n, opportunity_return=opportunity_return,
                                  entry_observed_return=chosen.loc[chosen.entry_missing_60.eq(False), "ret_60"].mean(),
                                  cash_slot_return=cash_slot_return,
                                  mae=chosen.mae_60.mean(), mean_prediction=chosen.full__expected_return.mean(),
                                  pool_return=truth.mean(), missing_entries=chosen.entry_missing_60.sum()))
            for gate, mask in [('point', point), ('confidence', confidence)]:
                admissions.append(dict(frequency=frequency, date=date, gate=gate, eligible=len(group),
                    admitted=int(mask.sum()), fraction=mask.mean(), none_admitted=not mask.any(),
                    admitted_return=truth[mask].mean(), rejected_return=truth[~mask].mean(),
                    missed_positive=int(((~mask)&truth.gt(0)).sum()), total_positive=int(truth.gt(0).sum())))
            for name in ['full_known_exit', 'full_complete_path']:
                p = group[name+'__expected_return']; baseline = group.full__expected_return
                sensitivity.append(dict(frequency=frequency,date=date,model=name,
                    mean_abs_prediction_change=abs(p-baseline).mean(),
                    max_abs_prediction_change=abs(p-baseline).max(),
                    top5_overlap=len(set(group.loc[p.nlargest(5).index,'ts_code']) &
                                     set(group.loc[baseline.nlargest(5).index,'ts_code']))/5))
    tables = {'error_dates':pd.DataFrame(errors), 'entry_path_dates':pd.DataFrame(paths),
              'admission_dates':pd.DataFrame(admissions), 'sensitivity_dates':pd.DataFrame(sensitivity)}
    for name, table in tables.items():
        table.to_csv(out / (name+'.csv'), index=False)
        category = 'model' if name in ['error_dates','sensitivity_dates'] else 'policy' if name=='entry_path_dates' else 'gate'
        numeric = table.select_dtypes(include=[np.number,bool]).columns.difference(['year'])
        table.groupby(['frequency',category])[numeric].mean().to_csv(out/(name.replace('_dates','_summary')+'.csv'))
        if 'year' in table:
            table.groupby(['frequency','year',category])[numeric].mean().to_csv(out/(name.replace('_dates','_yearly')+'.csv'))
    (out/'model_metadata.json').write_text(json.dumps(metadata,indent=2))
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''): digest.update(block)
    (out/'round12_verification.json').write_text(json.dumps(dict(status='complete',source_sha256=digest.hexdigest(),
        source_last_date=str(panel.date.max()),evaluated_rows=len(evaluation),
        no_entry_rows_kept_in_evaluation=int(evaluation.entry_missing_60.sum()),
        decomposition_checked=True,all_variants_retained=list(variants),
        primary_error_semantics='opportunity_pool_including_no_entry_cash_zero; fit_conditions_on_observed_entry',
        fresh_holdout=False,interval_is_outcome_prediction_interval=False,
        sensitivities_are_censoring_diagnostics_not_strategy_candidates=True),indent=2))
    print(tables['error_dates'].groupby(['frequency','model'])[['mse','time_mse','within_mse','rank_ic']].mean().to_string(),flush=True)
    print(tables['entry_path_dates'].groupby(['frequency','policy'])[['n','opportunity_return','cash_slot_return','mae']].mean().to_string(),flush=True)


if __name__ == '__main__':
    main()
