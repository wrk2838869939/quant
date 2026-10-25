#!/usr/bin/env python3
"""独立复算冻结模型与选择诊断；只读既有开发/执行特征，禁止TEST。"""
import argparse
from itertools import permutations
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
GROUPS = ('recovery', 'accelerating', 'decelerating')


def independent_design(frame, features, policy, groups=GROUPS, orthogonal=False):
    """不调用生产设计函数，从两个历史区间独立构造矩阵。"""
    past = (1 + frame.mom120.to_numpy()) / (1 + frame.mom60.to_numpy()) - 1
    phase = np.where(past <= 0, 'recovery',
                     np.where(frame.mom60.to_numpy() > past, 'accelerating', 'decelerating'))
    columns = ['intercept']
    blocks = [np.ones((len(frame), 1))]
    if policy != 'low_turnover':
        blocks.append(frame[features].to_numpy(float))
        columns.extend(features)
    if policy.startswith('symmetric_'):
        coded = np.column_stack([phase == group for group in groups]).astype(float) - 1 / len(groups)
        if orthogonal:
            eye = np.eye(len(groups))
            contrasts, _ = np.linalg.qr(eye[:, :-1] - eye[:, [-1]])
            coded = coded @ contrasts
            names = ['contrast_' + str(i) for i in range(coded.shape[1])]
        else:
            names = ['sym_phase_' + group for group in groups]
        blocks.append(coded)
        columns.extend(names)
        if policy == 'symmetric_slopes':
            for i, name in enumerate(names):
                blocks.append(coded[:, [i]] * frame[features].to_numpy(float))
                columns.extend(name + '__' + feature for feature in features)
    return np.column_stack(blocks), columns, phase


def independent_fit(frame, x):
    """每日期总权重一，截距不罚，其余固定alpha=1。"""
    w = 1 / frame.groupby('date').date.transform('size').to_numpy()
    y = frame[['ret_60', 'mae_60']].to_numpy(float) * [1, -1]
    a = x.T @ (x * w[:, None])
    b = x.T @ (y * w[:, None])
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0
    return np.linalg.solve(a + penalty, b), a, b, penalty


def numeric_error(actual, expected, tolerance):
    """检查缺失模式及有限数值，并记录最大误差。"""
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance, equal_nan=True)
    difference = np.abs(a - b)
    return float(np.nanmax(difference)) if difference.size and np.isfinite(difference).any() else 0.0


def prediction_values(x, model):
    """独立重建点估计、条件均值区间，保留不利幅度截零语义。"""
    point = x @ model.coefficients.to_numpy()
    samples = x @ model.bootstrap_coefficients[:, :, 0].T
    quantiles = np.quantile(samples, [0.1, 0.9], axis=1).T
    return np.column_stack([point[:, 0], np.maximum(point[:, 1], 0), quantiles]), samples


def bootstrap_diagnostics(frame, point, samples, score, policy, universe, stage, top_k):
    """固定点模型准入池，仅诊断估计扰动，不解释为未来盈利概率。"""
    selected_rows, ranking_rows = [], []
    work = frame[['date', 'ts_code', 'phase']].reset_index(drop=True)
    work['score'] = score
    for date, date_frame in work.groupby('date'):
        admitted = date_frame[date_frame.score.notna()].sort_values('ts_code')
        if admitted.empty:
            continue
        ids = admitted.index.to_numpy()
        bootstrap_score = samples[ids].copy()
        if policy == 'low_turnover':
            bootstrap_score[:] = admitted.score.to_numpy()[:, None]
        order = np.argsort(-bootstrap_score, axis=0, kind='stable')
        point_order = np.argsort(-admitted.score.to_numpy(), kind='stable')
        for k in top_k:
            frequency = np.bincount(order[:min(k, len(ids))].ravel(), minlength=len(ids)) / samples.shape[1]
            for rank, j in enumerate(point_order[:k], 1):
                row = admitted.iloc[j]
                selected_rows.append(dict(stage=stage, universe=universe, policy=policy, date=date,
                    k=k, rank=rank, ts_code=row.ts_code, phase=row.phase,
                    fixed_admitted=len(ids), bootstrap_reselection_frequency=frequency[j],
                    semantics='conditional_on_fixed_admitted_pool_not_profit_probability'))
        for phase, part in admitted.groupby('phase'):
            if len(part) < 2:
                continue
            positions = admitted.index.get_indexer(part.index)
            base = rankdata(admitted.loc[part.index, 'score'].to_numpy())
            ranks = rankdata(bootstrap_score[positions], axis=0)
            base = base - base.mean()
            ranks = ranks - ranks.mean(axis=0)
            denominator = np.sqrt((base**2).sum() * (ranks**2).sum(axis=0))
            correlation = np.divide(base @ ranks, denominator,
                out=np.full(samples.shape[1], np.nan), where=denominator > 0)
            finite = correlation[np.isfinite(correlation)]
            ranking_rows.append(dict(stage=stage, universe=universe, policy=policy, date=date,
                phase=phase, fixed_admitted=len(part), valid_replicates=len(finite),
                mean_spearman=float(finite.mean()) if len(finite) else np.nan,
                p10_spearman=float(np.quantile(finite, 0.1)) if len(finite) else np.nan,
                semantics='within_phase_fixed_admitted_pool_estimation_stability'))
    return selected_rows, ranking_rows


def verify_diagnostics(saved, cfg, out, tolerance):
    """不调用runner诊断函数，复算Top-k乐观缺口及阶段分解。"""
    error_table = pd.read_csv(out / 'errors_dates.csv', parse_dates=['date'])
    opportunity_table = pd.read_csv(out / 'opportunities_dates.csv', parse_dates=['date'])
    selected = pd.read_csv(out / 'selected_opportunities.csv', parse_dates=['date'])
    summaries = []
    for (date, policy, k), actual in opportunity_table.groupby(['date', 'policy', 'k']):
        frame = saved[saved.date.eq(date)]
        admitted = frame[frame[policy + '__score'].notna()]
        chosen = admitted.sort_values([policy + '__score', 'ts_code'], ascending=[False, True]).head(k)
        phase_mean = frame.groupby('phase').ret_60.mean()
        ref = chosen.phase.map(phase_mean)
        expected = dict(n=len(chosen), admitted=len(admitted),
            predicted=chosen[policy + '__expected_return'].mean(), realized=chosen.ret_60.mean(),
            optimism=(chosen[policy + '__expected_return'] - chosen.ret_60).mean(),
            pool_return=frame.ret_60.mean(), between_phase=(ref - frame.ret_60.mean()).mean(),
            within_phase=(chosen.ret_60 - ref).mean())
        for key, value in expected.items():
            numeric_error(actual[key], [value], tolerance)
        identity = numeric_error(actual.between_phase + actual.within_phase,
                                 actual.realized - actual.pool_return, tolerance)
        picked = selected[selected.date.eq(date) & selected.policy.eq(policy) & selected.k.eq(k)]
        assert list(picked.ts_code) == list(chosen.ts_code), 'Top-k股票顺序不同'
        summaries.append(dict(date=date, policy=policy, k=k, optimism=expected['optimism'],
                              decomposition_error=identity))
    numeric_error(error_table.mse, error_table.time_mse + error_table.within_mse, tolerance)
    for _, row in error_table.iterrows():
        part = saved[saved.date.eq(row.date)]
        if row.phase != 'all':
            part = part[part.phase.eq(row.phase)]
        residual = part[row.policy + '__expected_return'] - part.ret_60
        for key, value in [('mse', (residual**2).mean()), ('bias', residual.mean()),
                           ('time_mse', residual.mean()**2),
                           ('within_mse', ((residual - residual.mean())**2).mean())]:
            numeric_error([row[key]], [value], tolerance)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    target = out / 'verification'
    target.mkdir(exist_ok=True)
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    tolerance = cfg['diagnostics']['permutation_tolerance']
    assert cfg['forecast']['alpha'] == 1.0
    assert cfg['forecast']['horizon'] == 60
    columns = ['date', 'ts_code', 'monthly', 'trend', 'mom60', 'mom120'] + cfg['features']
    labels = ['label_end_60', 'ret_60', 'mae_60', 'entry_missing_60']
    panel = pd.read_parquet(ROOT / cfg['source_panel'], columns=columns + labels)
    execution = pd.read_parquet(ROOT / cfg['execution_features'], columns=columns)
    assert panel.date.max() < pd.Timestamp('2022-01-01'), '源面板越过允许边界'
    assert execution.date.max() < pd.Timestamp('2024-01-01'), '执行特征越过允许边界'
    panel = panel[panel.monthly & panel.trend].copy()
    execution = execution[execution.monthly & execution.ts_code.str.match(cfg['mainboard_pattern'])].copy()
    development = panel[panel.ts_code.str.match(cfg['mainboard_pattern']) &
        panel.date.between(*map(pd.Timestamp, cfg['development'])) &
        panel.label_end_60.notna() & panel.label_end_60.le(pd.Timestamp(cfg['development'][1]))].copy()
    model_rows, selected_rows, ranking_rows = [], [], []
    for stage, cutoff, frame in [('development', cfg['fit_end'], development),
                                  ('execution', cfg['final_fit_end'], execution)]:
        saved = pd.read_parquet(out / (stage + '_predictions.parquet'))
        pd.testing.assert_frame_equal(saved[['date', 'ts_code', 'trend']],
                                      frame[['date', 'ts_code', 'trend']].reset_index(drop=True))
        for universe in cfg['training_universes']:
            train = panel[panel.ts_code.str.match(cfg['mainboard_pattern'])] if universe == 'mainboard' else panel
            mature = train.label_end_60.notna() & train.label_end_60.le(pd.Timestamp(cutoff))
            eligible = mature & train.entry_missing_60.eq(False) & train[['ret_60', 'mae_60']].notna().all(axis=1)
            fit_frame = train.loc[eligible].copy()
            models = joblib.load(out / (universe + '_' + stage + '_models.joblib'))
            for policy in cfg['policies']:
                wrapper = models[policy]
                model = getattr(wrapper, 'underlying_model', wrapper)
                x, names, _ = independent_design(fit_frame, cfg['features'], policy)
                assert names == list(model.coefficients.index)
                beta, a, b, penalty = independent_fit(fit_frame, x)
                coefficient_error = numeric_error(beta, model.coefficients, tolerance)
                normal_error = numeric_error((a + penalty) @ model.coefficients.to_numpy(), b, tolerance)
                numeric_error(model.metadata['exclusion_counts']['fitted_rows'], eligible.sum(), tolerance)
                assert model.last_label == fit_frame.label_end_60.max() <= pd.Timestamp(cutoff)
                assert list(model.fit_dates) == sorted(fit_frame.date.unique())
                assert model.metadata['date_total_weight'] == 1.0
                assert model.metadata['alpha'] == 1.0
                query, query_names, phase = independent_design(frame, cfg['features'], policy)
                assert query_names == names
                assert np.array_equal(saved.phase.to_numpy(), phase)
                values, samples = prediction_values(query, model)
                prefix = '' if universe == 'mainboard' else 'transfer_'
                name = prefix + policy
                fields = ['expected_return', 'expected_adverse', 'mean_lower', 'mean_upper']
                prediction_error = numeric_error(values, saved[[name + '__' + c for c in fields]], tolerance)
                raw_score = -frame.x_turnover.to_numpy() if policy == 'low_turnover' else values[:, 0]
                gate = frame.trend.to_numpy() & (values[:, 0] > cfg['entry']['min_expected_return']) & (values[:, 1] <= cfg['entry']['max_expected_adverse'])
                score = np.where(gate, raw_score, np.nan)
                numeric_error(score, saved[name + '__score'], tolerance)
                # 实际wrapper在乱序输入上重建特征，检查完整四列而非只检查均值。
                order = np.random.default_rng(cfg['forecast']['seed']).permutation(len(frame))
                numeric_error(wrapper.predict(frame.iloc[order]).to_numpy(), values[order], tolerance)
                orthogonal_error = group_error = permutation_error = 0.0
                if policy.startswith('symmetric_'):
                    ortho_x, _, _ = independent_design(fit_frame, cfg['features'], policy, orthogonal=True)
                    ortho_beta, _, _, _ = independent_fit(fit_frame, ortho_x)
                    ortho_query, _, _ = independent_design(frame, cfg['features'], policy, orthogonal=True)
                    orthogonal_error = numeric_error(query @ beta, ortho_query @ ortho_beta, tolerance)
                    for suffix in [''] + (['__' + c for c in cfg['features']] if policy.endswith('slopes') else []):
                        ids = [names.index('sym_phase_' + group + suffix) for group in GROUPS]
                        group_error = max(group_error, numeric_error(model.coefficients.iloc[ids].sum(axis=0), np.zeros(2), tolerance),
                            numeric_error(model.bootstrap_coefficients[:, ids].sum(axis=1), np.zeros((len(model.bootstrap_coefficients), 2)), tolerance))
                    for groups in permutations(GROUPS):
                        _, reordered_names, _ = independent_design(frame.iloc[:0], cfg['features'], policy, groups)
                        ids = [names.index(c) for c in reordered_names]
                        reordered_beta = np.linalg.solve((a + penalty)[np.ix_(ids, ids)], b[ids])
                        permutation_error = max(permutation_error, numeric_error(query[:, ids] @ reordered_beta, query @ beta, tolerance))
                diag_frame = frame.assign(phase=phase)
                selections, rankings = bootstrap_diagnostics(diag_frame, values, samples, score,
                    policy, universe, stage, cfg['diagnostics']['top_k'])
                selected_rows.extend(selections)
                ranking_rows.extend(rankings)
                model_rows.append(dict(stage=stage, universe=universe, policy=policy,
                    training_rows=len(fit_frame), training_dates=fit_frame.date.nunique(), last_label=str(model.last_label),
                    coefficient_error=coefficient_error, normal_equation_error=normal_error,
                    prediction_error=prediction_error, orthogonal_error=orthogonal_error,
                    group_sum_error=group_error, permutation_error=permutation_error))
                print(stage, universe, policy, 'verified', flush=True)
    development_saved = pd.read_parquet(out / 'development_predictions.parquet')
    diagnostics = verify_diagnostics(development_saved, cfg, out, tolerance)
    pd.DataFrame(model_rows).to_csv(target / 'model_checks.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(target / 'selection_checks.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(target / 'bootstrap_reselection.csv', index=False)
    pd.DataFrame(ranking_rows).to_csv(target / 'bootstrap_within_phase_ranking.csv', index=False)
    report = dict(status='passed', models_verified=len(model_rows), diagnostic_rows=len(diagnostics),
        scope='source_panel_before_2022_execution_features_before_2024_no_TEST',
        independent_checks=['date_equal_normal_equations', 'orthonormal_contrasts',
            'all_bootstrap_group_sums_zero', 'maturity_cutoff', 'prediction_reproduction',
            'row_shuffle', 'all_group_permutations', 'top_k_optimism', 'between_within_identity'],
        bootstrap_semantics='fixed_admitted_pool_estimation_stability_not_future_profit_probability')
    (target / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
