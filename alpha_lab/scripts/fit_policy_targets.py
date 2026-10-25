#!/usr/bin/env python3
"""同样本、同容量比较三个毛收益目标；未知执行状态不用于训练与校准。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from value_forecast import fit_value_model


TARGETS = ('fixed60', 'fixed120', 'patient120')
POLICIES = ('low_turnover',) + TARGETS
OUTPUTS = ('expected_return', 'expected_adverse', 'mean_lower', 'mean_upper')
LABELS = [name + '_return' for name in TARGETS] + ['patient120_mae']


def check_inputs(frame, cfg):
    """拒绝缺失状态、重复机会、未来窗口和未登记模型容量。"""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError('标签面板必须为列名唯一的DataFrame')
    required = ['date', 'ts_code', 'trend', 'monthly', 'label_end_120',
                'entry_missing_120', 'execution_unknown_120'] + cfg['features'] + LABELS
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'缺少标签或状态字段: {sorted(missing)}')
    if (tuple(cfg['targets']) != TARGETS or len(cfg['features']) != 9
            or len(set(cfg['features'])) != 9 or not all(f.startswith('x_') for f in cfg['features'])
            or cfg['forecast']['horizon'] != 120 or cfg['forecast']['alpha'] != 1.0
            or cfg['common_risk_target'] != 'patient120_mae'):
        raise ValueError('必须保持三个登记目标、同9特征、alpha=1及共同patient120风险')
    work = frame.copy()
    for col in ['date', 'label_end_120']:
        work[col] = pd.to_datetime(work[col], utc=True).dt.tz_convert(None).dt.normalize()
    if (work.date.isna().any() or work.date.ge(pd.Timestamp('2024-01-01')).any()
            or work.date.lt(pd.Timestamp(cfg['history'][0])).any()
            or work.label_end_120.ge(pd.Timestamp('2024-01-01')).any()):
        raise ValueError('标签面板越过2014—2023已授权窗口或日期缺失')
    if work.duplicated(['date', 'ts_code']).any():
        raise ValueError('date+ts_code必须唯一')
    for col in ['trend', 'monthly', 'entry_missing_120', 'execution_unknown_120']:
        if not work[col].isin([True, False]).all():
            raise ValueError(f'{col}必须为明确布尔状态；未知不得伪装成可执行')
        work[col] = work[col].astype(bool)
    if not np.isfinite(work[cfg['features']].to_numpy(float)).all():
        raise ValueError('所有候选预测特征必须有限，不能根据未来可成交性删候选')
    return work.loc[work.monthly & work.ts_code.str.match(cfg['mainboard_pattern'])].copy().reset_index(drop=True)


def common_sample(frame, cutoff):
    """统一120日成熟边界；按共同标签可用性给出显式排除状态。"""
    end = pd.Timestamp(cutoff).normalize()
    mature = frame.label_end_120.notna() & frame.label_end_120.le(end)
    finite = pd.Series(np.isfinite(frame[LABELS].to_numpy(float)).all(axis=1), index=frame.index)
    observable = mature & ~frame.execution_unknown_120 & finite
    eligible = observable & ~frame.entry_missing_120
    return eligible, pd.DataFrame({
        'mature_120': mature, 'execution_unknown_120': frame.execution_unknown_120,
        'entry_missing_120': frame.entry_missing_120, 'finite_common_targets': finite,
        'common_label_available': observable, 'common_training_available': eligible,
    }, index=frame.index)


def cohort_hash(frame, features):
    """只比较共同身份与预测特征，不要求不同收益目标的训练hash相同。"""
    columns = ['date', 'ts_code', 'label_end_120'] + list(features)
    canonical = frame[columns].sort_values(['date', 'ts_code']).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes()).hexdigest()


def fit_models(frame, cfg, cutoff):
    """四模型共享成熟行；三个global只改变收益目标，风险列完全共享。"""
    available, flags = common_sample(frame, cutoff)
    available &= frame.trend
    cohort = frame.loc[available].copy()
    if cohort.empty:
        raise ValueError('共同成熟、可执行、有限且trend的训练样本为空')
    models = {}
    for policy in POLICIES:
        native = 'patient120' if policy == 'low_turnover' else policy
        training = cohort.assign(ret_120=cohort[native + '_return'], mae_120=cohort.patient120_mae)
        features = [] if policy == 'low_turnover' else cfg['features']
        models[policy] = fit_value_model(training, features, asof=cutoff, **cfg['forecast'])
    # 相同X和MAE的正规方程一致。目标排序可能产生末位舍入差，固定为同一风险估计。
    shared = models['patient120']
    maximum_risk_difference = 0.0
    for policy in TARGETS:
        model = models[policy]
        for actual, expected in [
            (model.coefficients.expected_adverse, shared.coefficients.expected_adverse),
            (model.bootstrap_coefficients[:, :, 1], shared.bootstrap_coefficients[:, :, 1]),
        ]:
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            maximum_risk_difference = max(maximum_risk_difference, float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))))
        model.coefficients['expected_adverse'] = shared.coefficients.expected_adverse.copy()
        model.bootstrap_coefficients[:, :, 1] = shared.bootstrap_coefficients[:, :, 1]
    sample_hash = cohort_hash(cohort, cfg['features'])
    for policy, model in models.items():
        model.metadata.update({
            'policy': policy, 'return_target': 'patient120_return' if policy == 'low_turnover' else policy + '_return',
            'risk_target': 'patient120_mae', 'risk_estimation': 'constant' if policy == 'low_turnover' else 'shared_global9',
            'common_cohort_hash': sample_hash, 'common_maturity': 'T_plus_121_open_for_all_targets',
            'return_semantics': 'gross_adjusted_unit_path_not_actual_trade_net_profit',
        })
    metadata = dict(cutoff=str(cutoff), input_rows=len(frame), trend_rows=int(frame.trend.sum()),
        common_training_rows=len(cohort), common_training_dates=int(cohort.date.nunique()),
        common_cohort_hash=sample_hash, risk_max_difference_before_sharing=maximum_risk_difference,
        exclusion_counts_overlapping={col: int((frame.trend & flags[col]).sum()) for col in
            ['execution_unknown_120', 'entry_missing_120']},
        immature_trend_rows=int((frame.trend & ~flags.mature_120).sum()),
        nonfinite_common_targets=int((frame.trend & ~flags.finite_common_targets).sum()),
        models={policy: model.metadata for policy, model in models.items()})
    return models, metadata


def predict_models(frame, models, cfg):
    """只用当日特征与预测门槛，未来标签和执行未知状态不能改变分数。"""
    result = frame[['date', 'ts_code', 'trend']].copy()
    for policy in POLICIES:
        prediction = models[policy].predict(frame)
        for column in OUTPUTS:
            result[policy + '__' + column] = prediction[column]
        score = -frame.x_turnover if policy == 'low_turnover' else prediction.expected_return
        gate = frame.trend & prediction.expected_return.gt(cfg['entry']['min_expected_return'])
        gate &= prediction.expected_adverse.le(cfg['entry']['max_expected_adverse'])
        result[policy + '__score'] = score.where(gate)
    for policy in TARGETS[:-1]:
        np.testing.assert_array_equal(result[policy + '__expected_adverse'], result['patient120__expected_adverse'])
    return result


def diagnostics(frame, prediction, cfg, cutoff):
    """先排事前Top-k再核对可评价性；共同目标与原生目标校准分开。"""
    available, flags = common_sample(frame, cutoff)
    work = frame.copy()
    for col in prediction.columns.difference(['date', 'ts_code', 'trend']):
        work[col] = prediction[col]
    for col in flags:
        work[col] = flags[col]
    errors, selection, gate_rows, selected = [], [], [], []
    for date, all_names in work.groupby('date'):
        pool = all_names[all_names.trend]
        evaluated = pool[pool.common_label_available]
        for policy in POLICIES:
            native = 'patient120' if policy == 'low_turnover' else policy
            mu = pool[policy + '__expected_return']
            risk = pool[policy + '__expected_adverse']
            return_ok = mu.gt(cfg['entry']['min_expected_return'])
            risk_ok = risk.le(cfg['entry']['max_expected_adverse'])
            gate_rows.append(dict(date=date, policy=policy, all_names=len(all_names), trend_names=len(pool),
                admitted=int((return_ok & risk_ok).sum()), return_only_rejected=int((~return_ok & risk_ok).sum()),
                risk_only_rejected=int((return_ok & ~risk_ok).sum()), both_rejected=int((~return_ok & ~risk_ok).sum()),
                no_admitted=not (return_ok & risk_ok).any(),
                execution_unknown=int(pool.execution_unknown_120.sum()),
                immature_120=int((~pool.mature_120).sum()),
                semantics='prediction_gate_only_not_account_cash_shortage'))
            for target_kind, target_col in [('common_patient120', 'patient120_return'),
                                              ('native_' + native, native + '_return')]:
                calibration = evaluated if target_kind == 'common_patient120' else evaluated[~evaluated.entry_missing_120]
                residual = calibration[policy + '__expected_return'] - calibration[target_col]
                errors.append(dict(date=date, policy=policy, target_kind=target_kind, n=len(calibration),
                    calibration_population='known_opportunities_including_no_entry_cash' if target_kind == 'common_patient120' else 'entered_only',
                    mse=(residual ** 2).mean(), bias=residual.mean(),
                    time_mse=residual.mean() ** 2,
                    within_mse=((residual - residual.mean()) ** 2).mean()))
            ranked = pool[pool[policy + '__score'].notna()].sort_values(
                [policy + '__score', 'ts_code'], ascending=[False, True])
            for k in (1, 5):
                chosen = ranked.head(k)
                observed = chosen[chosen.common_label_available]
                for _, row in chosen.iterrows():
                    selected.append(dict(date=date, policy=policy, k=k, ts_code=row.ts_code,
                        score=row[policy + '__score'], predicted=row[policy + '__expected_return'],
                        common_label_available=row.common_label_available, execution_unknown_120=row.execution_unknown_120,
                        mature_120=row.mature_120, entry_missing_120=row.entry_missing_120,
                        patient120_return=row.patient120_return if row.common_label_available else np.nan,
                        native_return=row[native + '_return'] if row.common_label_available else np.nan))
                for target_kind, target_col in [('common_patient120', 'patient120_return'),
                                                  ('native_' + native, native + '_return')]:
                    predicted = observed[policy + '__expected_return']
                    selection.append(dict(date=date, policy=policy, k=k, target_kind=target_kind,
                        selected_n=len(chosen), observed_n=len(observed),
                        complete_selected_labels=bool(len(chosen) > 0 and len(chosen) == len(observed)),
                        complete_cash_slots=bool(len(chosen) == len(observed)),
                        coverage=len(observed) / len(chosen) if len(chosen) else np.nan,
                        unknown_selected_n=int(chosen.execution_unknown_120.sum()),
                        immature_selected_n=int((~chosen.mature_120).sum()),
                        no_entry_selected_n=int(chosen.entry_missing_120.sum()),
                        predicted=predicted.mean(), realized=observed[target_col].mean(),
                        optimism=(predicted - observed[target_col]).mean(),
                        selected_mse=((predicted - observed[target_col]) ** 2).mean(),
                        cash_slot_return=observed[target_col].sum() / k if len(chosen) == len(observed) else np.nan,
                        cash_slot_optimism=(predicted - observed[target_col]).sum() / k if len(chosen) == len(observed) else np.nan,
                        pool_return=evaluated[target_col].mean(),
                        excess=(observed[target_col].mean() - evaluated[target_col].mean())))
    return {name: pd.DataFrame(rows) for name, rows in [
        ('errors_dates', errors), ('top_k_dates', selection),
        ('cash_rejection_dates', gate_rows), ('selected_opportunities', selected)]}


def paired_comparisons(tables):
    """仅取三个目标模型共同完整覆盖的日期，显示固定配对差值。"""
    selections = tables['top_k_dates']
    common = selections[selections.target_kind.eq('common_patient120') & selections.policy.isin(TARGETS)]
    rows = []
    errors = tables['errors_dates']
    errors = errors[errors.target_kind.eq('common_patient120')]
    mse = errors.pivot(index='date', columns='policy', values='mse')
    for k in (1, 5):
        part = common[common.k.eq(k)]
        for comparison, coverage_col, fields in [
            ('selected_nonempty', 'complete_selected_labels', ['realized', 'optimism', 'selected_mse', 'excess']),
            ('fixed_k_slots_cash_zero', 'complete_cash_slots', ['cash_slot_return', 'cash_slot_optimism']),
        ]:
            coverage = part.pivot(index='date', columns='policy', values=coverage_col).reindex(columns=TARGETS)
            dates = coverage.index[coverage.fillna(False).all(axis=1)]
            for baseline in TARGETS[:-1]:
                result = dict(k=k, baseline=baseline, comparison=comparison,
                              paired_complete_dates=len(dates), total_dates=len(coverage))
                for field in fields:
                    wide = part.pivot(index='date', columns='policy', values=field).reindex(dates)
                    result['patient_minus_baseline_' + field] = (wide.patient120 - wide[baseline]).mean()
                result['patient_minus_baseline_common_mse'] = (mse.patient120 - mse[baseline]).reindex(dates).mean()
                rows.append(result)
    return pd.DataFrame(rows)


def run_fit(run):
    """读取已构造标签并冻结模型/预测；不下载行情，不启动执行回放。"""
    run = Path(run).resolve()
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    path = run / 'policy_labels.parquet'
    if not path.exists():
        raise FileNotFoundError(f'等待标签构造产物: {path}')
    if any((run / (stage + '_models.joblib')).exists() for stage in ['development', 'execution']):
        raise ValueError('模型已冻结；禁止覆盖正在使用的预测/模型')
    frame = check_inputs(pd.read_parquet(path), cfg)
    metadata, stage_reports = {}, {}
    for stage, cutoff in [('development', cfg['fit_end']), ('execution', cfg['final_fit_end'])]:
        models, info = fit_models(frame, cfg, cutoff)
        query = frame[frame.date.between(*map(pd.Timestamp, cfg[stage]))].copy()
        if query.empty:
            raise ValueError(f'{stage}没有候选，不能输出空评估')
        prediction = predict_models(query, models, cfg)
        joblib.dump(models, run / (stage + '_models.joblib'))
        prediction.to_parquet(run / (stage + '_predictions.parquet'), index=False)
        flags = common_sample(query, cfg[stage][1])[1]
        query[['date', 'ts_code']].join(flags).to_parquet(run / (stage + '_label_coverage.parquet'), index=False)
        tables = diagnostics(query, prediction, cfg, cfg[stage][1])
        folder = run / (stage + '_diagnostics')
        folder.mkdir(exist_ok=True)
        for name, table in tables.items():
            table.to_csv(folder / (name + '.csv'), index=False)
        pairs = paired_comparisons(tables)
        pairs.to_csv(folder / 'paired_comparisons.csv', index=False)
        for name, keys in [('errors_dates', ['policy', 'target_kind']),
                            ('top_k_dates', ['policy', 'k', 'target_kind'])]:
            table = tables[name]
            numeric = table.select_dtypes(include=np.number).columns.difference(keys)
            table.groupby(keys)[numeric].mean().to_csv(folder / name.replace('_dates', '_summary.csv'))
        metadata[stage] = info
        stage_reports[stage] = dict(prediction_rows=len(query), prediction_dates=int(query.date.nunique()),
            future_unknown_candidates_preserved=int(query.execution_unknown_120.sum()),
            paired_comparisons=pairs.to_dict(orient='records'))
        print(stage, 'four frozen models and all monthly predictions saved', flush=True)
    (run / 'policy_model_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    sources = [Path(__file__), ROOT / 'src/value_forecast.py', run / 'config.yaml', path]
    report = dict(status='models_and_diagnostics_complete', model_count=8,
        no_download=True, no_execution_replay=True, no_TEST=True,
        return_semantics='gross_adjusted_unit_path; optimizer_costs_not_deducted_in_labels',
        selection_semantics='rank_before_future_label_filter; paired_comparisons_require_complete_coverage',
        stages=stage_reports, conditional_replay='requires_parent_review_of_registered_stop_rule',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (run / 'fit_policy_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run_fit(args.run)


if __name__ == '__main__':
    main()
