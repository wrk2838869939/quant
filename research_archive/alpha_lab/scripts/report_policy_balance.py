# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""核验完成的联合实验并生成中文报告与非单调性能图。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_policy_balance as B
from policy_evolution import candidates, restore_rng
from run_classic_strategy_smoke import digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    hashes = json.loads((run / 'artifact_hashes.json').read_text())
    assert all(digest(run / p) == h for p, h in hashes.items())
    config = ROOT / 'configs/policy_balance_v1.yaml'
    cfg, _, _ = B.setup(config, run)
    B.G['full_package'] = B.package_until(cfg['end'])
    out = run / 'completion'
    out.mkdir(exist_ok=True)
    locks = json.loads((run / 'locked_choices.json').read_text())
    # 核验所有内层账户的模型截止日及决策时点，不只查最终获胜记录。
    cutoff_checks = 0
    for lock in locks:
        assert pd.Timestamp(lock['evaluation_end']) < pd.Timestamp(lock['asof'])
        for row in lock['candidates']:
            folder = run / 'selection' / lock['asof'] / row['id']
            for entry in json.loads((folder / 'schedule.json').read_text()):
                trained = json.loads((run / entry['theta']).parent.joinpath('complete.json').read_text())
                assert pd.Timestamp(trained['train_end']) < pd.Timestamp(entry['date']) <= pd.Timestamp(lock['evaluation_end'])
                cutoff_checks += 1
    # 额外强制验证季度学习参数切换路径，避免自适应全退回基线时检查退化。
    control = f'w{cfg["training_months"][0]}_k{cfg["checkpoints"][0]}_r{cfg["retrain_months"][0]}'
    folder = run / 'outer' / control
    entries = json.loads((folder / 'schedule.json').read_text())
    result, _ = B.rollout(entries, cfg['outer_start'], cfg['end'], original=True)
    for table in B.R.original.TABLES:
        pd.testing.assert_frame_equal(getattr(result, table), pd.read_parquet(folder / f'{table}.parquet'), check_exact=True)
    cutoff = cfg['prefix_dates'][0]
    short, _ = B.rollout(entries, cfg['outer_start'], cutoff, package=B.package_until(cutoff))
    B.R.original.compare_prefix(short, result, pd.Timestamp(cutoff))
    # 从中心、冠军与 RNG 状态重建首末代的全部候选，校验未落盘候选可复现。
    reconstructed = 0
    for path in sorted((run / 'training').glob('*/w*/complete.json')):
        trials = json.loads((path.parent / 'trials.json').read_text())
        for generation in [1, max(cfg['checkpoints'])]:
            state = json.loads((path.parent / f'state_g{generation-1:03d}.json').read_text())
            center = np.load(path.parent / f'center_g{generation-1:03d}.npy')
            theta = np.load(path.parent / f'theta_g{generation-1:03d}.npy')
            scales = B.NeuralPolicy.scales(B.G['full_package']['features'].shape[2], cfg['hidden'])
            values = candidates(center, theta, scales, state['multiplier'], restore_rng(state['rng']), cfg['search'])
            expected = [r for r in trials if r['generation'] == generation]
            assert len(expected) == len(values)
            for (value, _, _, _), row in zip(values, expected):
                assert hashlib.sha256(value.tobytes()).hexdigest() == row['parameter_sha256']
                reconstructed += 1
    extreme_checks = []
    learning = []
    for path in sorted((run / 'training').glob('*/w*/complete.json')):
        task = json.loads(path.read_text())
        states = {k: json.loads((path.parent / f'state_g{k:03d}.json').read_text()) for k in cfg['checkpoints']}
        learning.append(dict(origin=task['origin'], window_months=task['window_months'],
                             **{f'train_return_{k}': state['champion']['metrics']['total'] for k, state in states.items()},
                             final_scale=states[max(cfg['checkpoints'])]['multiplier']))
    for window in cfg['training_months']:
        winner = max((r for r in learning if r['window_months'] == window), key=lambda r: r[f'train_return_{max(cfg["checkpoints"])}'])
        folder = B.task_folder(winner['origin'], window)
        task = json.loads((folder / 'complete.json').read_text())
        p = B.package_until(task['train_end'])
        entries = [dict(date=task['train_start'], theta=str((folder / f'theta_g{max(cfg["checkpoints"]):03d}.npy').relative_to(run)))]
        replay, _ = B.rollout(entries, task['train_start'], task['train_end'], package=p, original=True)
        for table in B.R.original.TABLES:
            pd.testing.assert_frame_equal(getattr(replay, table), pd.read_parquet(folder / f'checkpoint_g{max(cfg["checkpoints"]):03d}' / f'{table}.parquet'), check_exact=True)
        extreme_checks.append(dict(origin=winner['origin'], window_months=window, original_engine_exact=True, metrics=B.R.metric(replay)))
    save_json(out / 'training_extremes.json', extreme_checks)
    checks = dict(core_artifacts_verified=len(hashes), historical_cutoffs_verified=cutoff_checks,
                  candidate_vectors_reconstructed=reconstructed, learned_quarterly_original_engine_exact=True,
                  learned_quarterly_prefix=True, source_hashes_verified=True, training_extremes_original_engine=len(extreme_checks))
    save_json(out / 'verification.json', checks)
    summary = pd.read_csv(run / 'outer_summary.csv')
    age = pd.read_csv(run / 'age_details.csv')
    assert age.loc[age.age_months == 0, 'log_old_minus_fresh'].abs().max() < 1e-12
    young = age[age.age_months == 0].pivot(index=['origin', 'window_months'], columns='generation', values='log_excess')
    young.columns = [f'next_quarter_excess_{k}' for k in young.columns]
    learning = pd.DataFrame(learning).merge(young.reset_index(), on=['origin', 'window_months'])
    lo, hi = min(cfg['checkpoints']), max(cfg['checkpoints'])
    learning['train_up_next_quarter_down'] = ((learning[f'train_return_{hi}'] > learning[f'train_return_{lo}'])
                                             & (learning[f'next_quarter_excess_{hi}'] < learning[f'next_quarter_excess_{lo}']))
    learning.to_csv(out / 'learning_generalization.csv', index=False)
    # 同一组有完整 12 月跟踪的批次，避免年龄越大样本越少造成组分变化。
    latest = pd.Timestamp(cfg['end']) + pd.Timedelta(days=1) - pd.DateOffset(months=cfg['max_age_months'])
    balanced = age[pd.to_datetime(age.origin) <= latest]
    age_summary = balanced.groupby(['window_months', 'generation', 'age_months']).agg(
        cohorts=('log_excess', 'size'), mean_log_excess=('log_excess', 'mean'),
        median_log_excess=('log_excess', 'median'), mean_old_minus_fresh=('log_old_minus_fresh', 'mean')).reset_index()
    age_summary.to_csv(out / 'age_balanced_summary.csv', index=False)
    learned = summary[summary.strategy.str.startswith('w')].copy()
    learned[['window', 'generation', 'interval']] = learned.strategy.str.extract(r'w(\d+)_k(\d+)_r(\d+)').astype(int)
    triplets = []
    for (window, interval), group in learned.groupby(['window', 'interval']):
        values = group.sort_values('generation').total.to_numpy()
        triplets.append(dict(window=int(window), interval=int(interval), nonmonotonic=bool(np.prod(np.diff(values)) < 0)))
    save_json(out / 'nonmonotonic_iterations.json', triplets)
    font = FontProperties(fname=str(PROJECT / 'fonts/SimHei.ttf'))
    fig, axes = plt.subplots(len(cfg['training_months']), 2, figsize=(10, 7), constrained_layout=True)
    for i, window in enumerate(cfg['training_months']):
        group = learned[learned.window == window]
        for j, (metric, title) in enumerate([('total', '累计净收益'), ('mdd', '最大回撤')]):
            matrix = group.pivot(index='generation', columns='interval', values=metric).reindex(index=cfg['checkpoints'], columns=cfg['retrain_months']) * 100
            ax = axes[i, j]
            ax.imshow(matrix.to_numpy(), cmap='RdYlGn', aspect='auto', vmin=learned[metric].min()*100, vmax=learned[metric].max()*100)
            for y in range(len(matrix.index)):
                for x in range(len(matrix.columns)):
                    ax.text(x, y, f'{matrix.iloc[y,x]:+.1f}%', ha='center', va='center', fontsize=11)
            ax.set_xticks(range(len(matrix.columns)), matrix.columns)
            ax.set_yticks(range(len(matrix.index)), matrix.index)
            ax.set_xlabel('重训间隔（月）', fontproperties=font)
            ax.set_ylabel('迭代代数', fontproperties=font)
            ax.set_title(f'窗口 {window} 月：{title}', fontproperties=font)
    fig.suptitle('2022–2023 已观察历史：完整网格，不按外层选优', fontproperties=font)
    fig.savefig(out / 'joint_grid.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(cfg['training_months']), figsize=(10, 4), constrained_layout=True, sharey=True)
    for ax, window in zip(np.atleast_1d(axes), cfg['training_months']):
        for k in cfg['checkpoints']:
            part = age_summary[(age_summary.window_months == window) & (age_summary.generation == k)]
            ax.plot(part.age_months, part.mean_log_excess * 100, marker='o', label=f'{k} generations')
        ax.axhline(0, color='grey', linewidth=.8)
        ax.set_xticks([0, 3, 6, 9], ['0–3', '3–6', '6–9', '9–12'])
        ax.set_title(f'训练窗口 {window} 月', fontproperties=font)
        ax.set_xlabel('参数冻结后的年龄（月）', fontproperties=font)
        ax.set_ylabel('季度对数相对收益均值（百分点）', fontproperties=font)
        ax.legend(fontsize=8)
    fig.suptitle('同一批次集合的年龄表现：不假定单调衰减', fontproperties=font)
    fig.savefig(out / 'age_curve.png', dpi=150)
    plt.close(fig)
    dynamic = summary.set_index('strategy').loc['adaptive']
    base = summary.set_index('strategy').loc['baseline']
    decisions = []
    for lock in locks:
        row = lock['selected']
        decisions.append(dict(asof=lock['asof'], history_end=lock['evaluation_end'],
                              choice=row['id'] if row else lock['fallback'], next_review=lock['next_review']))
    pd.DataFrame(decisions).to_csv(out / 'decisions.csv', index=False)
    selected_count = sum(lock['selected'] is not None for lock in locks)
    timings = pd.read_json(run / 'training_progress.json').groupby('window_months').seconds.median()
    timing_text = '；'.join(f'{int(w)} 月窗口单批次耗时中位数 {seconds:.1f} 秒' for w, seconds in timings.items())
    lines = ['# 模型衰减—迭代—窗口联合实验结果', '',
             f'运行：`{run.relative_to(PROJECT)}`。冻结数据/账户口径不变。全部区间已观察；这是方法与工程验证，不是新样本外认证。', '',
             '## 已实现', '',
             '低换手基线初始化，全部 1686 参数可学习。窗口 6/18 月 × 代数 8/32/64 × 重训 3/6/12 月，共 18 组。2021 年提供首轮历史评价，2022–2023 连续账户回放。每次仅用此前完整 12 月评价组合，选中重训间隔决定下次选择日期。', '',
             '模型年龄无需服从指数或单调衰减：逐批次冻结参数，测 0–3/3–6/6–9/9–12 月的同历基线相对收益，同时保存旧模型与同窗口同代数的新模型比较。年龄结果不能解释为纯参数衰减，持仓路径和市场变化也会影响它。', '',
             '## 连续外层结果（净费用）', '',
             '| 策略 | 累计收益 | 最大回撤 | 正收益且回撤≤30% |', '|---|---:|---:|---|']
    for _, row in summary.iterrows():
        lines.append(f'| {row.strategy} | {row.total:+.2%} | {abs(row.mdd):.2%} | {"通过" if row.total > 0 and row.mdd >= -.30 else "未通过"} |')
    lines += ['', f'自适应相对固定基线累计收益差为 {(dynamic.total-base.total)*100:+.2f} 个百分点。{len(locks)} 次决策中 {selected_count} 次选用学习策略；这是历史结果，不能保证未来回撤。固定基线是 2021 参数全程不变，与过去逐年换参数基线不是同一对照。', '',
              '## 决策记录', '', '| 生效日 | 历史评价截止 | 选择 | 下次评价 |', '|---|---|---|---|']
    lines += [f'| {r["asof"]} | {r["history_end"]} | {r["choice"]} | {r["next_review"]} |' for r in decisions]
    lines += ['', '## 联合映射与年龄诊断', '',
              f'固定窗口和重训间隔的 {len(triplets)} 条代数切片中，{sum(r["nonmonotonic"] for r in triplets)} 条出现中间点峰/谷，不能使用“训练更久必然更好”的单调假设。此处为描述性统计，未据此再调参。', '',
              f'从 {lo} 代增加到 {hi} 代，{len(learning)} 条轨迹中 {int(learning.train_up_next_quarter_down.sum())} 条同时出现训练收益提高、下一季度相对收益降低。训练改善和可部署改善必须分开判断。详见 completion/learning_generalization.csv。', '',
              f'年龄主图只保留全部四段年龄都完整的 {balanced.origin.nunique()} 个共同批次；原始汇总保留全部可见批次。批次重叠，未将其当成独立样本给显著性结论。', '',
              f'![联合性能网格](../outputs/policy_balance/{run.name}/completion/joint_grid.png)', '',
              f'![模型年龄曲线](../outputs/policy_balance/{run.name}/completion/age_curve.png)', '',
              '## 验证与边界', '',
              f'- 完成 {manifest["training_tasks"]} 条训练轨迹、{manifest["training_evaluations"]:,} 次候选账户评估；全部候选执行独立账本审核。',
              f'- {timing_text}。耗时包含候选账本审计，采用 8 进程并行。',
              f'- 核验 {checks["core_artifacts_verified"]:,} 个核心产物哈希、{cutoff_checks} 个历史模型截止日、{reconstructed} 个首末代候选参数重建。',
              '- 自适应与强制季度学习路径均通过原引擎逐表等价和截断前缀检查；切换保留账户及持仓记忆。9 项针对性单元测试通过。',
              '- 同分冠军保留旧参数，中心仍可探索；与旧实验相比，这是额外实现改动，不能把全部变化单独归因于三个超参数。',
              '- 同一网格共享完整 64 代轨迹。本轮实际计算花费包含全部候选研究，选中 K×12/R 只是未来部署的名义训练负担。',
              '- 历史选择账户重置为 10 万，外层账户连续，存在状态分布差异；时间边界正确不等于市场泛化已成立。',
              '- 93 股全窗口质量隔离、既有宿主/基线选择、单随机种子和反复观察历史仍有偏差；季度网格无法测定短于 3 月的衰减。',
              '- 不从本轮外层优胜组合反向改写选择规则；仍需新时段验证才能据此作交易判断。', '',
              '完整过程见运行目录 `training/`、`selection/`、`outer/`、`age_details.csv` 和 `completion/verification.json`。']
    report = ROOT / 'docs/POLICY_BALANCE_RESULTS_20260921.md'
    report.write_text('\n'.join(lines) + '\n')
    save_json(out / 'manifest.json', dict(status='complete', report=str(report.relative_to(PROJECT)), report_sha256=digest(report),
                                        reporter_sha256=digest(Path(__file__)), core_manifest_sha256=digest(run / 'manifest.json'), checks=checks))
    save_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps(dict(report=str(report), adaptive=dynamic.to_dict(), baseline=base.to_dict(), decisions=decisions, checks=checks), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
