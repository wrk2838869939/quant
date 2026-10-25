#!/usr/bin/env python3
"""两次随机半区互换实验；最终模型始终只拟合各轮训练半区。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import sys
import unittest
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import PROJECT, HERE, save_json, stats, bootstrap_difference, digest, DataHub, assign_clusters, plt
from model import TrainScaler, FuzzyNetwork, fit_network, predict_network, fuzzy_score, sample_weights
from random_split import make_assignments


def fit_round(features, train, valid, cfg, out, fold):
    """此接口不接收测试标签；验证标签仅用于选参和早停。"""
    scaler = TrainScaler(cfg).fit(train[cfg['features']])
    tx, vx, ax = [scaler.transform(f[cfg['features']]) for f in [train, valid, features]]
    ty, vy = [f.target.to_numpy() / cfg['target_scale'] for f in [train, valid]]
    tw, vw = [sample_weights(f.date, None, None) for f in [train, valid]]
    tuning, logs, candidates = [], [], []
    for rules in cfg['fnn']['rules']:
        for lr in cfg['fnn']['learning_rates']:
            losses, epochs = [], []
            for seed in cfg['fnn']['tuning_seeds']:
                _, epoch, loss, trajectory = fit_network(tx, ty, tw, rules, lr, seed, cfg['fnn'], validation=(vx, vy, vw))
                losses.append(loss)
                epochs.append(epoch)
                tuning.append(dict(round=fold, model='fnn', rules=rules, learning_rate=lr,
                                   seed=seed, epochs=epoch, validation_loss=loss))
                logs.extend(dict(round=fold, phase='tuning', rules=rules, learning_rate=lr, seed=seed, **r) for r in trajectory)
            candidates.append((float(np.mean(losses)), rules, lr, int(np.median(epochs))))
    loss, rules, lr, epochs = min(candidates)
    scores, train_scores = {}, {}
    networks, train_networks = [], []
    for seed in cfg['fnn']['ensemble_seeds']:
        network, _, _, trajectory = fit_network(tx, ty, tw, rules, lr, seed, cfg['fnn'], epochs=epochs)
        networks.append(predict_network(network, ax) * cfg['target_scale'])
        train_networks.append(predict_network(network, tx) * cfg['target_scale'])
        logs.extend(dict(round=fold, phase='final_train_only', rules=rules, learning_rate=lr, seed=seed, **r) for r in trajectory)
        path = out / 'models' / f'{fold}_fnn_{seed}.pt'
        torch.save(dict(state_dict=network.state_dict(), rules=rules, seed=seed, epochs=epochs,
                        median=scaler.median, mean=scaler.mean, scale=scaler.scale), path)
        # 重载检查所有事件评分，确保实际报告能由保存模型重放。
        saved = torch.load(path, map_location='cpu')
        replay = FuzzyNetwork(tx, rules, seed, cfg['fnn'])
        replay.load_state_dict(saved['state_dict'])
        np.testing.assert_array_equal(predict_network(replay, ax), predict_network(network, ax))
    scores['fnn'], train_scores['fnn'] = np.mean(networks, axis=0), np.mean(train_networks, axis=0)
    ridge_choices = []
    for alpha in cfg['ridge_alphas']:
        estimator = Ridge(alpha=alpha).fit(tx, ty, sample_weight=tw)
        value = float(np.average((estimator.predict(vx) - vy) ** 2, weights=vw))
        ridge_choices.append((value, alpha))
        tuning.append(dict(round=fold, model='ridge', alpha=alpha, validation_loss=value))
    alpha = min(ridge_choices)[1]
    estimator = Ridge(alpha=alpha).fit(tx, ty, sample_weight=tw)
    scores['ridge'] = estimator.predict(ax) * cfg['target_scale']
    train_scores['ridge'] = estimator.predict(tx) * cfg['target_scale']
    ridge_path = out / 'models' / f'{fold}_ridge.npz'
    np.savez(ridge_path, coef=estimator.coef_, intercept=estimator.intercept_, alpha=alpha,
             median=scaler.median, mean=scaler.mean, scale=scaler.scale)
    saved_ridge = np.load(ridge_path)
    np.testing.assert_allclose(scores['ridge'], (ax @ saved_ridge['coef'] + saved_ridge['intercept']) * cfg['target_scale'])
    medians = train[cfg['features']].median().fillna(0)
    scores['fuzzy'] = fuzzy_score(features, medians, cfg)
    train_scores['fuzzy'] = fuzzy_score(train, medians, cfg)
    rows = []
    for kind in scores:
        for retention in cfg['retention_fractions']:
            threshold = float(np.quantile(train_scores[kind], 1 - retention))
            part = features[['event_id', 'date', 'sector_code']].copy()
            part['round'], part['model'], part['retention'] = fold, kind, retention
            part['score'], part['threshold'] = scores[kind], threshold
            part['selected'] = part.score >= threshold
            rows.append(part)
    info = dict(round=fold, fit_n=len(train), valid_n=len(valid), rules=rules, learning_rate=lr,
                epochs=epochs, validation_loss=loss, ridge_alpha=alpha)
    save_json(out / f'round_{fold}_fit.json', dict(**info, train_ids=train.event_id.tolist(),
              validation_ids=valid.event_id.tolist(), fuzzy_medians=medians.to_dict()))
    return pd.concat(rows, ignore_index=True), info, tuning, logs


def overlap_diagnostics(data, assignments):
    """只诊断，不参与划分、拟合或选参；窗口含确认日和末日。"""
    rows = []
    for fold, a in assignments.groupby('round'):
        test = data.merge(a[a.role == 'test'][['event_id']], on='event_id')
        for role in ['train', 'validation']:
            other = data.merge(a[a.role == role][['event_id']], on='event_id')
            same_date = test.date.isin(other.date)
            overlap, sector_overlap = [], []
            for r in test.itertuples():
                shared = other.date.le(r.end_date) & other.end_date.ge(r.date)
                overlap.append(bool(shared.any()))
                sector_overlap.append(bool((shared & other.sector_code.eq(r.sector_code)).any()))
            rows.append(dict(round=fold, compared_role=role, test_n=len(test),
                same_date_events=int(same_date.sum()), overlapping_window_events=sum(overlap),
                same_sector_overlapping_events=sum(sector_overlap)))
    return pd.DataFrame(rows)


def main():
    cfg = yaml.safe_load((HERE / 'random_config.yaml').read_text())
    assert cfg['split_unit'] == 'event_id' and cfg['rounds'] == 2
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test_*.py')
    tests = unittest.TextTestRunner().run(suite)
    assert tests.wasSuccessful()
    source = PROJECT / cfg['feature_source']
    returns = PROJECT / cfg['returns']
    for directory in [source, returns]:
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        for name, expected in manifest['artifacts'].items():
            assert digest(directory / name) == expected, name
    market = json.loads((PROJECT / cfg['dataset'] / 'manifest.json').read_text())
    assert os.environ['DATAHUB_CONTENT_ID'] == market['market_content_id']
    hub = DataHub()
    assert hub.cfg.root.resolve() == (PROJECT / cfg['release']).resolve()
    calendar = hub.calendar.cn_days(market['config']['start'], market['config']['end'])
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    (out / 'models').mkdir(parents=True)
    for name in ['random_config.yaml', 'RANDOM_PREREG.md', 'run_random.py', 'random_split.py',
                 'test_random_split.py', 'model.py', 'run.py', 'test_model.py', 'config.yaml']:
        shutil.copy2(HERE / name, out / name)
    save_json(out / 'manifest.json', dict(status='building'))
    print('OUTPUT', out, flush=True)
    features = pd.read_parquet(source / 'causal_features.parquet').sort_values('event_id').reset_index(drop=True)
    assert len(features) == 330 and features.event_id.is_unique
    features.to_parquet(out / 'causal_features.parquet', index=False)
    assignments = make_assignments(features.event_id, cfg['split_seed'])
    assignments.to_csv(out / 'split_assignments.csv', index=False)
    # 划分只使用事件编号，在目标标签读取前固定并落盘。
    outcomes = pd.read_parquet(returns / 'sector_event_outcomes.parquet')
    target = outcomes[outcomes.horizon.eq(cfg['target_horizon'])][['event_id', 'end_date', 'end_return', 'complete']].rename(columns={'end_return': 'target'})
    data = features.merge(target, on='event_id', validate='one_to_one')
    predictions, folds, searches, curves, counts = [], [], [], [], []
    for fold, a in assignments.groupby('round'):
        joined = data.merge(a[['event_id', 'role']], on='event_id', validate='one_to_one')
        observed = joined.complete & joined.target.notna()
        train = joined[(joined.role == 'train') & observed].copy()
        valid = joined[(joined.role == 'validation') & observed].copy()
        for role, group in joined.groupby('role'):
            counts.append(dict(round=fold, role=role, assigned=len(group),
                               complete20=int((group.complete & group.target.notna()).sum()), dates=group.date.nunique()))
        # 测试收益污染冲击应完全不影响本轮训练/验证输入。
        changed = joined.copy()
        changed.loc[changed.role == 'test', 'target'] = 999
        pd.testing.assert_frame_equal(train, changed[(changed.role == 'train') & observed])
        pd.testing.assert_frame_equal(valid, changed[(changed.role == 'validation') & observed])
        assert set(train.event_id).isdisjoint(valid.event_id)
        prediction, info, tuning, logs = fit_round(features, train, valid, cfg, out, fold)
        prediction = prediction.merge(a[['event_id', 'role']], on='event_id', validate='many_to_one')
        predictions.append(prediction)
        folds.append(info)
        searches.extend(tuning)
        curves.extend(logs)
        print('ROUND', info, flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(out / 'all_role_scores_before_evaluation.parquet', index=False)
    test_pred = pred[pred.role == 'test'].copy()
    assert not test_pred.duplicated(['event_id', 'model', 'retention']).any()
    test_pred.to_parquet(out / 'test_predictions_before_evaluation.parquet', index=False)
    pd.DataFrame(folds).to_csv(out / 'folds.csv', index=False)
    pd.DataFrame(counts).to_csv(out / 'split_counts.csv', index=False)
    pd.DataFrame(searches).to_csv(out / 'validation_search.csv', index=False)
    pd.DataFrame(curves).to_csv(out / 'training_curves.csv', index=False)
    overlap = overlap_diagnostics(data, assignments)
    overlap.to_csv(out / 'cross_split_overlap.csv', index=False)
    # 至此两轮模型、参数和评分均固定；以下只进行评价。
    clusters = assign_clusters(features.date.unique(), calendar, cfg['cluster_gap_sessions'])
    evaluated = test_pred.merge(outcomes, on=['event_id', 'date', 'sector_code'], validate='many_to_many').merge(clusters, on='date', validate='many_to_one')
    evaluated.to_parquet(out / 'test_outcomes_WITH_FUTURE.parquet', index=False)
    summaries = []
    groups = [('combined', evaluated)] + [(str(k), g) for k, g in evaluated.groupby('round')]
    for fold, group in groups:
        for (kind, retention, horizon), part in group.groupby(['model', 'retention', 'horizon']):
            g = part[part.complete].copy()
            base, chosen = stats(g), stats(g[g.selected])
            row = dict(round=fold, model=kind, retention=retention, horizon=horizon,
                       scored_n=len(part), selected_signals=int(part.selected.sum()),
                       incomplete_n=int((~part.complete).sum()), base_n=base['n'], **chosen)
            for metric in ['mean', 'median', 'q10', 'q90', 'date_mean', 'positive', 'min_return_mean']:
                row['base_' + metric] = base[metric]
                row['delta_' + metric] = chosen[metric] - base[metric]
            row['max_return_mean'] = g.loc[g.selected, 'max_return'].mean()
            row['base_max_return_mean'] = g.max_return.mean()
            row['selected_fraction'] = chosen['n'] / base['n']
            if horizon == cfg['target_horizon']:
                row.update(bootstrap_difference(g, cfg))
                row['descriptive_right_shift'] = bool(chosen['n'] >= cfg['minimum_selected_events'] and chosen['dates'] >= cfg['minimum_selected_dates']
                    and row['selected_fraction'] >= cfg['minimum_selected_fraction'] and all(row['delta_' + m] > 0 for m in ['mean', 'median', 'q10', 'date_mean']))
                row['intervals_support_shift'] = bool(row['descriptive_right_shift'] and all(row['ci_' + m + '_low'] > 0 for m in ['mean', 'median', 'q10']))
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / 'selection_summary.csv', index=False)
    report(out, cfg, evaluated, summary, pd.DataFrame(folds), pd.DataFrame(curves), pd.DataFrame(counts), overlap)
    save_json(out / 'validation.json', dict(unit_tests=tests.testsRun, disjoint_roles=True, swapped_halves=True,
        test_targets_excluded_from_fit_interface=True, test_target_shock_train_valid_unchanged=True,
        scaler_train_only=True, model_checkpoints_replayed=8, source_artifacts_verified=True,
        scored_test_events=test_pred.event_id.nunique(), random_seed=cfg['split_seed']))
    save_json(out / 'manifest.json', dict(status='complete', config=cfg, dataset_content_id=market['content_id'],
        feature_source_manifest_sha256=digest(source / 'manifest.json'), returns_manifest_sha256=digest(returns / 'manifest.json'),
        artifacts={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('COMPLETE', out, flush=True)
    print(summary[(summary.horizon == 20) & (summary.retention == .5)][['round', 'model', 'n', 'base_n', 'mean', 'base_mean', 'delta_mean', 'delta_median', 'delta_q10']].to_string(index=False), flush=True)


def report(out, cfg, evaluated, summary, folds, curves, counts, overlap):
    font_manager.fontManager.addfont(str(PROJECT / cfg['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / cfg['font']).get_name(), 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    names = {'fnn': '模糊神经网络', 'fuzzy': '固定模糊规则', 'ridge': '岭回归'}
    charts = []
    with PdfPages(out / 'random_halves_report.pdf') as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
        for ax, h in zip(axes, cfg['report_horizons']):
            common = evaluated[(evaluated.horizon == h) & evaluated.complete & (evaluated.retention == cfg['primary_retention'])]
            base = common.drop_duplicates('event_id').end_return.sort_values().to_numpy() * 100
            ax.step(base, np.arange(1, len(base) + 1) / len(base), color='#555', label=f'未筛选 n={len(base)}')
            for kind, name in names.items():
                v = common[(common.model == kind) & common.selected].end_return.sort_values().to_numpy() * 100
                if len(v):
                    ax.step(v, np.arange(1, len(v) + 1) / len(v), label=f'{name} n={len(v)}')
            ax.set(title=f'{h} 日收益累计分布', xlabel='收益（%）', ylabel='不超过横轴收益的比例')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle('随机二等分互换：仅合并两轮测试集，主阈值取训练评分中位数')
        fig.savefig(out / 'test_return_ecdf.png', dpi=cfg['dpi']); pdf.savefig(fig); plt.close(fig)
        charts.append('test_return_ecdf.png')
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
        for ax, r in zip(axes, folds.itertuples()):
            chosen = curves[(curves['round'] == r.round) & (curves.phase == 'tuning') & (curves.rules == r.rules) & (curves.learning_rate == r.learning_rate)]
            for seed, g in chosen.groupby('seed'):
                ax.plot(g.epoch, g.train_loss, label=f'训练，种子 {seed}')
                ax.plot(g.epoch, g.validation_loss, '--', label=f'验证，种子 {seed}')
            ax.axvline(r.epochs, color='gray', ls=':', label=f'最终训练 {r.epochs} 轮')
            ax.set(title=f'第 {r.round} 轮：训练 {r.fit_n} / 验证 {r.valid_n}', xlabel='迭代轮数', ylabel='加权 Huber 损失')
            ax.legend(fontsize=8)
        fig.suptitle('仅验证集决定超参数和轮数；最终模型不合并验证数据')
        fig.savefig(out / 'training_curves.png', dpi=cfg['dpi']); pdf.savefig(fig); plt.close(fig)
        charts.append('training_curves.png')
    primary = summary[(summary.retention == cfg['primary_retention']) & (summary.horizon == 20)]
    lines = ['# 随机两半互换实验', '', '按事件随机划分，未使用日期或收益分层。A/B 各 165 条，每轮训练 165、验证 82、测试 83。',
        '缺失 20 日标签的训练/验证事件排除；测试按期限报告完整数。两轮独立选参，不根据测试结果调整。', '',
        '| 轮次 | 方法 | 完整选中/基准 | 基准均值 | 选中均值 | 中位数变化 | q10 变化 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in primary.itertuples():
        lines.append(f'| {r.round} | {names[r.model]} | {r.n}/{r.base_n} | {r.base_mean:.2%} | {r.mean:.2%} | {r.delta_median * 100:+.2f}pp | {r.delta_q10 * 100:+.2f}pp |')
    combined = primary[primary['round'] == 'combined'].set_index('model').loc['fnn']
    lines += ['', f'神经网络合并均值增量 {combined.delta_mean * 100:+.2f}pp，条件于本批预测的行情簇 95% 区间 '
              f'[{combined.ci_mean_low * 100:+.2f}, {combined.ci_mean_high * 100:+.2f}]pp。',
              f'合并描述性右移：{combined.descriptive_right_shift}；区间支持右移：{combined.intervals_support_shift}。', '',
              '## 分组与完整标签', '', counts.to_csv(index=False),
              '## 跨集合关联诊断', '', overlap.to_csv(index=False),
              '以上重叠不参与筛选或调参。窗口包括确认日到第 20 个交易日，同日/共同市场冲击仍可能相关。', '',
              '## 解释边界', '',
              '- 两半交换训练角色，测试合计 166 条；另外 164 条只进入某一轮验证，不能混入测试汇总。',
              '- 两轮分别独立选参，但不是独立重复实验；同一条事件在另一轮可能被训练使用。',
              '- 随机留出允许晚期事件用于模型训练，这是本次实验设计，不是时间样本外交易证据。',
              '- 划分前已看过历史事件规则与总体收益；没有新增未见市场数据。',
              '- 区间为整簇重采样下的描述性不确定性，不涵盖重新抽样训练，也未做多重比较校正。',
              '- 收益是确认收盘至未来收盘的板块统计，期间极值只取逐日收盘；没有费用、可成交入场或资金曲线。', '',
              '## 文件', '',
              '- random_halves_report.pdf / overview.html：测试收益分布与训练/验证损失曲线。',
              '- selection_summary.csv：每轮与合并的 3/10/20 日指标，50% 主阈值与 70% 敏感性全部保留。',
              '- split_assignments.csv / split_counts.csv：所有事件的角色与可用标签数量。',
              '- validation_search.csv / training_curves.csv / models/：全部选参与训练记录。',
              '- all_role_scores_before_evaluation.parquet 含训练/验证分数，禁止将这些行当测试评价。',
              '- test_predictions_before_evaluation.parquet 仅含两轮测试评分；test_outcomes_WITH_FUTURE.parquet 含事后标签。', '']
    (out / 'README.md').write_text('\n'.join(lines))
    (out / 'overview.html').write_text('<!doctype html><meta charset="utf-8"><title>随机两半互换实验</title><style>body{max-width:1400px;margin:auto;font-family:system-ui}img{width:100%}</style><h1>随机两半互换：仅测试集结果</h1><p>事件级随机 50/25/25；两轮独立选参、训练半区互换。属于随机泛化实验。</p>' + ''.join(f'<img src="{p}">' for p in charts))


if __name__ == '__main__':
    main()
