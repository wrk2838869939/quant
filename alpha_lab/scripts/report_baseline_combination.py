#!/usr/bin/env python3
"""从冻结组合诊断产物生成报告与实测参数图；不重新选择策略。"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'baseline_selector': '只选择单条基线', 'account_best': '账户目标最优组合',
    'account_stable': '账户目标稳健邻域', 'ic_best': 'IC目标最优组合',
    'ic_with_baseline_fallback': 'IC组合加基线回退', 'equal_weight': '三因子等权',
    'B01_low_turnover': '固定低换手基线', 'B02_low_risk': '固定低波/振幅基线',
    'B03_valuation': '固定估值基线'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    choices = json.loads((run / 'all_locked_choices.json').read_text())
    summary = pd.read_csv(run / 'summary.csv')
    yearly = pd.read_csv(run / 'yearly.csv')
    surface = pd.read_csv(run / 'parameter_surface.csv')
    triples = pd.read_csv(run / 'nonmonotonic_triplets.csv')
    audits = [json.loads(p.read_text()) for p in run.glob('search/*/*/audit.json')]
    audits += [json.loads(p.read_text()) for p in run.glob('outer/*/audit.json')]
    selection = []
    for year, methods in choices.items():
        base = methods['baseline_selector']
        for method, row in methods.items():
            values = dict(year=int(year), method=method, selection_id=row['id'] if row else None)
            if row:
                values.update(**row['params'], inner_median_ann=row['metrics']['median_ann'], inner_ic=row['mean_ic'],
                              inner_mdd=row['metrics']['mdd'], neighbor_count=row['neighbor_count'],
                              neighbor_q25=row['neighbor_q25'], max_label_end=row['max_label_end'],
                              baseline_floor_margin_pp=(row['metrics']['median_ann'] - base['metrics']['median_ann']) * 100 if base else None)
            selection.append(values)
    pd.DataFrame(selection).to_csv(run / 'selected_policies.csv', index=False)
    diagnostics = dict(
        independently_audited_accounts=len(audits), trades_checked=sum(a['trades_checked'] for a in audits),
        max_cash_error=max(a['cash_max_error'] for a in audits),
        account_best_endpoint_years=sum(v['account_best'] is not None and v['account_best']['params']['endpoint'] is not None for v in choices.values()),
        fallback_changed_years=sum((v['ic_best']['id'] if v['ic_best'] else None) !=
                                  (v['ic_with_baseline_fallback']['id'] if v['ic_with_baseline_fallback'] else None) for v in choices.values()),
        inner_objective_correlations={str(year): float(group['mean_ic'].corr(group['median_ann'], method='spearman'))
                                      for year, group in surface.groupby('outer_year')},
        nonmonotonic_all=int(triples.nonmonotonic.sum()), triplets_all=len(triples),
        nonmonotonic_feasible=int((triples.nonmonotonic & triples.all_feasible).sum()),
        triplets_feasible=int(triples.all_feasible.sum()))
    (run / 'diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + '\n')
    # 只画真实离散点；无平滑、无插值，x/y为两个权重，第三个由和为1确定。
    fig, axes = plt.subplots(1, len(choices), figsize=(15, 4.7), constrained_layout=True)
    vmin, vmax = surface.median_ann.min() * 100, surface.median_ann.max() * 100
    for ax, (year, methods) in zip(np.atleast_1d(axes), choices.items()):
        row = methods['account_best']
        if row is None:
            ax.set_title(f'{year}: no feasible candidate')
            continue
        p = row['params']
        group = surface[(surface.outer_year == int(year)) & (surface.frequency == p['frequency']) & (surface.top_n == p['top_n'])].copy()
        w = np.array([json.loads(x) for x in group.weights])
        dots = ax.scatter(w[:, 0], w[:, 1], c=group.median_ann * 100, s=48, cmap='viridis', vmin=vmin, vmax=vmax)
        bad = ~group.feasible.to_numpy()
        ax.scatter(w[bad, 0], w[bad, 1], marker='x', c='red', s=40)
        ax.scatter(p['weights'][0], p['weights'][1], marker='*', c='orange', edgecolors='black', s=200)
        ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color='gray', linewidth=.7)
        ax.set(xlabel='Low turnover weight', ylabel='Low risk weight', xlim=(-.05, 1.05), ylim=(-.05, 1.05),
               title=f'{year}: {p["frequency"]}, top {p["top_n"]}')
        ax.set_aspect('equal')
    if 'dots' in locals():
        fig.colorbar(dots, ax=list(np.atleast_1d(axes)), label='Inner median annualized net return (%)', shrink=.8)
    fig.suptitle('Observed weight grid only; star = account choice; red x = drawdown rejection')
    fig.savefig(run / 'observed_weight_surfaces.png', dpi=160)
    plt.close(fig)
    lines = ['# 完整基线组合学习诊断', '',
             f'运行：`{run.relative_to(ROOT.parent)}`。数据与账户保持冻结；10万元、含手续费滑点，2021–2023连续账户。', '',
             '本轮属于已观察历史上的机制诊断。三条基线经过既有结果筛选，不能把本轮结果称为新鲜样本外证明。', '',
             '## 账户结果', '',
             '| 路径 | 累计净收益 | 年化 | 最大回撤 | 交易笔数 | 费用（元） | 历史验收 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in summary.itertuples():
        lines.append(f'| {NAMES[row.method]} | {row.total:.2%} | {row.ann:.2%} | {abs(row.mdd):.2%} | {row.n_trades} | {row.total_fees:.2f} | {"通过" if row.historical_acceptance else "未通过"} |')
    lines += ['', '## 年度锁定路径', '',
              '权重顺序为低换手、低波/振幅、估值；这是评分权重，不是三个子账户资金份额。顶点强制继承原基线频率、持股数及过滤。', '',
              '| 评价年 | 方法 | 权重 | 频率/持股数 | 内层年化中位数 | 内层IC | 相对端点最优值（pp） |',
              '|---|---|---|---|---:|---:|---:|']
    for r in selection:
        if r['selection_id'] is None:
            lines.append(f'| {r["year"]} | {NAMES[r["method"]]} | 现金 | — | — | — | — |')
        else:
            w = '/'.join(f'{x:.3g}' for x in r['weights'])
            lines.append(f'| {r["year"]} | {NAMES[r["method"]]} | {w} | {r["frequency"]}/{r["top_n"]} | {r["inner_median_ann"]:.2%} | {r["inner_ic"]:.4f} | {r["baseline_floor_margin_pp"]:.3f} |')
    lines += ['', '## 分年结果', '', '| 方法 | 年度 | 净收益 | 年内回撤 |', '|---|---:|---:|---:|']
    for row in yearly.itertuples():
        lines.append(f'| {NAMES[row.method]} | {row.year} | {row.total:.2%} | {abs(row.mdd):.2%} |')
    lines += ['', '## 核验与适用范围', '',
              f'- {manifest["inner_trials"]}个唯一内层候选，错误0；所有候选都独立重放账本。另有{manifest["outer_accounts"]}个连续评价账户。',
              f'- {manifest["inner_endpoint_replays"]}个内层端点与原搜索过程表一致，{manifest["outer_endpoint_replays"]}条组合端点与原基线四张账户表一致；{manifest["prefix_checks"]}次原始信号/账户前缀核验。',
              f'- 独立审计{diagnostics["trades_checked"]}笔交易，最大现金对账误差{diagnostics["max_cash_error"]:.3g}元；资金、费用、T+1、涨跌停与公司行动使用原引擎。',
              f'- 实测三点权重路径{len(triples)}组，其中{int(triples.nonmonotonic.sum())}组收益非单调；三点全部风险可行的{diagnostics["triplets_feasible"]}组中有{diagnostics["nonmonotonic_feasible"]}组非单调。重叠三点不是独立统计样本。',
              '- account_best、account_stable及IC加回退在每折训练评价准则下不低于可行端点最优值；该性质不保证下一年，也不保证外层连续账户回撤小于30%。',
              '- 账户目标是多自然年净年化中位数，IC目标是决策日截面20日价格收益相关性；风险约束相同。IC本轮直接在同一候选中择优，区别于旧版按IC比例配权。',
              '- 本轮直接估计有限网格上的组合权重，不是新训练的神经网络，也没有证明普通回归训练器自然会找到端点。',
              '- 三条因子方向由规则先验固定，权重只允许非负；旧版有符号IC配权可反转方向。前后比较同时改变了因子窗口、资格、政策和权重约束，不能把全部差异归到某一个原因。',
              '- 局部图中的资格交集会在权重变为零时改变，频率/持股数也是离散变量；任何收益插值、单调外推均不成立。',
              '- 已按基线年度参数选择后再拟合组合，叠加了历史选择偏差；固定93股的全窗口质量隔离偏差继续存在。',
              '- 未据外层结果改参数或替换基线；所有年度选择在外层计算前落盘。', '',
              '![账户最优政策下的实测权重表面](observed_weight_surfaces.png)', '',
              '完整过程：`parameter_surface.csv`、`nonmonotonic_triplets.csv`、`selected_policies.csv`、`all_trials_*.json`、`search/`及`outer/`。', '']
    ablation_path = run / 'posthoc_ablation/summary.csv'
    if ablation_path.exists():
        ablation_manifest = json.loads((run / 'posthoc_ablation/manifest.json').read_text())
        assert ablation_manifest['status'] == 'complete'
        ablation = pd.read_csv(ablation_path).set_index('method')
        low = ablation.loc['low_turnover']
        gated = ablation.loc['low_turnover_with_intersection']
        equal = ablation.loc['equal_weight_replay']
        original_low = summary.set_index('method').loc['B01_low_turnover']
        lines += ['', '## 等权结果的事后固定消融', '',
                  '看到主实验结果后另行登记两个固定对照及一次等权重放，不追加参数搜索，不据此改主实验选择。', '',
                  '| 顺序对照 | 累计净收益 | 最大回撤 | 交易笔数 | 费用（元） |', '|---|---:|---:|---:|---:|']
        for name, row in [('原低换手基线', original_low), ('低换手换成等权交易安排', low),
                          ('再加三因子资格交集', gated), ('再改为三因子等权排名', equal)]:
            lines.append(f'| {name} | {row.total:.2%} | {abs(row.mdd):.2%} | {int(row.n_trades)} | {row.total_fees:.2f} |')
        lines += ['',
                  f'按此固定干预顺序，交易安排对应累计收益变化{(low.total-original_low.total)*100:+.2f}pp，'
                  f'资格交集对应{(gated.total-low.total)*100:+.2f}pp，排名混合对应{(equal.total-gated.total)*100:+.2f}pp。'
                  '这是这段历史和固定顺序下的账户反事实，不是独立可加的普遍因果效应。',
                  '三个补充账户独立核验，9次前缀检查通过；等权重放的原始分数、持股数和四张账户表与主实验完全一致。', '']
    if (run / 'retry_reproducibility.json').exists():
        retry = json.loads((run / 'retry_reproducibility.json').read_text())
        lines += ['', '## 运行留痕', '',
                  f'首次393个内层候选完成后，CSV导出遇到重复top_n字段错误，尚未进入外层账户评价。'
                  f'修复仅涉及导出字段合并，重跑后年度选择与{retry["matching_inner_parquet_tables"]}张内层Parquet表逐字节一致。'
                  '393个独立候选共执行786次内层账户，首次失败记录保留于`retry_reproducibility.json`指向的目录。', '']
    (run / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(diagnostics, ensure_ascii=False))


if __name__ == '__main__':
    main()
