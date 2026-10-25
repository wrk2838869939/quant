#!/usr/bin/env python3
"""汇总监督增量实验，按固定宿主配对，不按外层成绩重新选参数。"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HOSTS = {'B01_low_turnover': '低换手', 'B02_low_risk': '低波/振幅', 'B03_valuation': '估值', 'equal_weight': '三等权'}
METHODS = {'baseline': '原样参照', 'residual_best': '增量最优', 'residual_stable': '增量邻域稳健',
           'direct_best': '直接预测', 'direct_with_fallback': '直接预测加回退'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    summary = pd.read_csv(run / 'summary.csv')
    yearly = pd.read_csv(run / 'yearly.csv')
    choices = json.loads((run / 'all_locked_choices.json').read_text())
    triplets = pd.read_csv(run / 'amplitude_triplets.csv')
    selected = []
    for year, hosts in choices.items():
        for host, methods in hosts.items():
            base = methods['baseline']['metrics']
            for method, row in methods.items():
                r = dict(year=int(year), host=host, method=method, selection_id=row['id'] if row else None)
                if row:
                    p, m = row['params'], row['metrics']
                    r.update(**p, inner_median_ann=m['median_ann'], inner_mdd=m['mdd'],
                             baseline_inner_feasible=base['feasible'], inner_delta_pp=(m['median_ann'] - base['median_ann']) * 100,
                             neighbor_count=row['neighbor_count'], neighbor_q25=row['neighbor_q25'])
                selected.append(r)
    selected = pd.DataFrame(selected)
    selected.to_csv(run / 'selected_policies.csv', index=False)
    for i, row in summary.iterrows():
        base = summary[(summary.host == row.host) & (summary.method == 'baseline')].iloc[0]
        summary.loc[i, 'delta_total_pp'] = (row.total - base.total) * 100
        summary.loc[i, 'delta_mdd_pp'] = (row.mdd - base.mdd) * 100
        summary.loc[i, 'delta_fees_yuan'] = row.total_fees - base.total_fees
    summary.to_csv(run / 'paired_summary.csv', index=False)
    fit_records = [r for p in run.glob('*_models/*/*/*/fits.json') for r in json.loads(p.read_text())]
    audits = [json.loads(p.read_text()) for p in run.glob('search/*/*/*/audit.json')]
    audits += [json.loads(p.read_text()) for p in run.glob('outer/*/*/audit.json')]
    diagnostic = dict(audited_accounts=len(audits), audited_trade_records=sum(r['trades_checked'] for r in audits),
                      cash_max_error=max(r['cash_max_error'] for r in audits),
                      saved_fit_pairs=sum(r['status'] == 'ok' for r in fit_records),
                      saved_models=2 * sum(r['status'] == 'ok' for r in fit_records),
                      immature_labels_used=0, nonmonotonic_triplets=int(triplets.nonmonotonic.sum()),
                      all_triplets=len(triplets), selections={})
    for method in METHODS:
        sub = selected[selected.method == method]
        diagnostic['selections'][method] = dict(host_years=len(sub),
                                                exact_baseline=int(sub.kind.eq('baseline').sum()),
                                                learned=int(sub.kind.isin(['direct', 'residual']).sum()),
                                                cash=int(sub.selection_id.isna().sum()))
    (run / 'diagnostics.json').write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 基线＋监督学习增量研究', '',
             f'运行：`{run.relative_to(ROOT.parent)}`。四个固定宿主、岭回归/浅层LightGBM，直接排名与残差排名成对训练。', '',
             '10万元，2021–2023连续账户，费用、交易约束和冻结93股样本保持一致。此历史已经观察过，不是独立样本外验证。', '',
             '## 配对账户结果', '',
             '| 宿主 | 路径 | 累计净收益 | 年化 | 最大回撤 | 相对宿主收益差(pp) | 成交笔数 | 历史验收 |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for row in summary.itertuples():
        lines.append(f'| {HOSTS[row.host]} | {METHODS[row.method]} | {row.total:.2%} | {row.ann:.2%} | {abs(row.mdd):.2%} | {row.delta_total_pp:+.2f} | {row.n_trades} | {"通过" if row.historical_acceptance else "未通过"} |')
    lines += ['', '## 年度内层锁定', '',
              '| 年度 | 宿主 | 路径 | 模型规格/幅度 | 内层年化中位数 | 相对原宿主(pp) |', '|---|---|---|---|---:|---:|']
    for year, hosts in choices.items():
        for host, methods in hosts.items():
            for method, row in methods.items():
                if method == 'baseline':
                    continue
                if row is None:
                    lines.append(f'| {year} | {HOSTS[host]} | {METHODS[method]} | 无可行候选，现金 | — | — |')
                    continue
                p, m = row['params'], row['metrics']
                description = ('原样宿主，幅度0' if p['kind'] == 'baseline' else
                               json.dumps(p, ensure_ascii=False, sort_keys=True))
                change = (m['median_ann'] - methods['baseline']['metrics']['median_ann']) * 100
                lines.append(f'| {year} | {HOSTS[host]} | {METHODS[method]} | {description} | {m["median_ann"]:.2%} | {change:+.2f} |')
    lines += ['', '## 训练与过程核验', '',
              f'- {manifest["inner_candidates"]}个宿主/年度候选，{manifest["inner_errors"]}个训练/账户错误；内层{manifest["inner_fit_pairs"]}组成对拟合。',
              f'- 内外层共保存{diagnostic["saved_fit_pairs"]}组成对模型，即{diagnostic["saved_models"]}个训练模型，均重载核验。历史不足回退{manifest["insufficient_fit_pairs"]}次。',
              f'- 四个原样宿主与既有信号、持股数和四张账户表一致；{manifest["raw_account_prefix_checks"]}次原始输入重建、重新预测和账户前缀核验；另有{manifest["paired_prediction_prefix_checks"]}次直接/增量预测配对前缀核验。',
              f'- 独立账本核验{diagnostic["audited_accounts"]}个账户、{diagnostic["audited_trade_records"]}条成交记录，现金最大误差{diagnostic["cash_max_error"]:.3g}元。',
              f'- {len(triplets)}组同模型相邻幅度三点中有{int(triplets.nonmonotonic.sum())}组非单调；重叠点不能当成独立统计样本。',
              '- 每次拟合均记录标签结束时间且严格早于拟合日。季度中不更新模型，年度中不重选超参数。',
              '- 幅度0直接返回原始宿主评分；其他特征缺失不改变宿主资格。模型不能预测时回到宿主排序，不偷偷变成现金或删掉股票。',
              '- 直接预测与增量预测来自同一训练掩码、同样本权重、同模型规格与同标签期限；区别仅为监督目标及是否加上基线排名。',
              '- 有可行宿主时，增量最优、增量稳健和直接预测加回退在内层选择目标上均不低于宿主；该约束没有未来保证。',
              '- 基线与等权已按历史挑选；基线年度参数再用于内层拟合，仍有选择偏差。股票池全期质量隔离偏差也未消除。', '',
              '## 分年结果', '', '| 宿主 | 路径 | 年度 | 净收益 | 年内回撤 |', '|---|---|---:|---:|---:|']
    for row in yearly.itertuples():
        lines.append(f'| {HOSTS[row.host]} | {METHODS[row.method]} | {row.year} | {row.total:.2%} | {abs(row.mdd):.2%} |')
    # 显示每个宿主逐年选中增量模型的实测幅度曲线；零点来自独立原样账户。
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, host in zip(axes.flat, HOSTS):
        for year, hosts in choices.items():
            selected_row = hosts[host]['residual_best']
            rows = json.loads((run / f'all_trials_{year}.json').read_text())
            rows = [r for r in rows if r['host'] == host]
            base = next(r for r in rows if r['params']['kind'] == 'baseline')
            if selected_row and selected_row['params']['kind'] == 'residual':
                spec = selected_row['params']['model_spec']
                curve = [base] + [r for r in rows if r['params']['kind'] == 'residual' and r['params']['model_spec'] == spec]
                curve.sort(key=lambda r: r['params']['amplitude'])
                ax.plot([r['params']['amplitude'] for r in curve], [r['metrics']['median_ann'] * 100 for r in curve], 'o-', label=year)
                rejected = [r for r in curve if not r['metrics']['feasible']]
                ax.scatter([r['params']['amplitude'] for r in rejected], [r['metrics']['median_ann'] * 100 for r in rejected],
                           marker='x', color='red', s=65, zorder=5)
                chosen_amplitude = selected_row['params']['amplitude']
                ax.scatter(chosen_amplitude, selected_row['metrics']['median_ann'] * 100, s=140, marker='*', color='black')
            else:
                ax.scatter(0, base['metrics']['median_ann'] * 100, label=year, marker='*', s=100)
        ax.set(title=host, xlabel='Increment amplitude', ylabel='Inner median net annualized return (%)')
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Observed amplitudes: star = selected; red x = risk rejection; lines connect observations only')
    fig.savefig(run / 'selected_amplitude_curves.png', dpi=150)
    plt.close(fig)
    lines += ['', '![实测增量幅度曲线](selected_amplitude_curves.png)', '',
              '完整候选、年度选择、模型和过程表分别见`parameter_surface.csv`、`selected_policies.csv`、`inner_models/`、`outer_models/`、`search/`、`outer/`。', '']
    if (run / 'prediction_objective_diagnosis.csv').exists():
        objective = pd.read_csv(run / 'prediction_objective_diagnosis.csv')
        grouped = objective.groupby(['segment', 'score'])[['mean_ic', 'equal_date_mse']].mean()
        streams = objective[['year', 'host', 'model_spec']].drop_duplicates()
        lines += ['', '## 已选模型的训练目标与紧邻预测段', '',
                  f'只读重载{len(streams)}条已选模型规格的首个实际拟合，核对训练掩码及保存的首段预测；不重新训练或选参。'
                  '预测段截止下次拟合前且只使用届时已成熟标签，短段指标不代表全年表现，也不是相互独立的显著性样本。', '',
                  '| 评分 | 训练平均IC | 紧邻预测段平均IC | 训练MSE | 紧邻段MSE |', '|---|---:|---:|---:|---:|']
        names = {'host': '原宿主排名', 'direct': '直接预测', 'residual_full': '宿主＋幅度1增量', 'constant_half': '常数0.5'}
        for mode, name in names.items():
            train, future = grouped.loc[('train', mode)], grouped.loc[('next_segment', mode)]
            tic = f'{train.mean_ic:.4f}' if mode != 'constant_half' else '无排名能力'
            fic = f'{future.mean_ic:.4f}' if mode != 'constant_half' else '无排名能力'
            lines.append(f'| {name} | {tic} | {fic} | {train.equal_date_mse:.5f} | {future.equal_date_mse:.5f} |')
        lines += ['', 'MSE降低可以仅来自预测向截面均值收缩，因此必须保留常数0.5对照，不能把相对未经校准宿主排名的MSE降低当作选股技能。'
                  '这里的幅度1仅用于诊断残差监督目标，不改变已锁定的交易幅度。', '']
    if (run / 'high_inner_flags.json').exists():
        high = json.loads((run / 'high_inner_flags.json').read_text())
        lines += [f'内层Sharpe>3或月收益>20%的候选共{high["count"]}个，全部包含在独立账本审计中；'
                  '宿主参数选择偏差及股票池质量隔离偏差仍存在，内层高收益不作为未来收益证据。', '']
    (run / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(diagnostic, ensure_ascii=False))


if __name__ == '__main__':
    main()
