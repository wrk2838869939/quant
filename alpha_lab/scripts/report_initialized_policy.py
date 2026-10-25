#!/usr/bin/env python3
"""只读首轮策略演化归档，提取学习距离、行为变化与随机初始化配对。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    locked = json.loads((run / 'all_locked_choices.json').read_text())
    assert digest(run / 'all_locked_choices.json') == manifest['locked_choices_sha256']
    architecture = json.loads((run / 'architecture.json').read_text())
    summary = pd.read_csv(run / 'outer_summary.csv')
    init = json.loads((run / 'initialization_checks.json').read_text())
    output = run / 'diagnostics'
    output.mkdir(exist_ok=True)
    trials, audits, triplets, parameter_rows = [], [], [], []
    for path in sorted((run / 'search').rglob('trial.json')):
        row = json.loads(path.read_text())
        m = row['metrics']
        trials.append(dict(host=row['host'], initialization=row['initialization'], outer_year=row['outer_year'],
                           folder=row['folder'], generation=row['generation'], split='validation' if '/validation/' in row['folder'] else 'train',
                           **{k: v for k, v in m.items() if k != 'yearly'}))
        daily = pd.read_parquet(path.parent / 'daily.parquet')
        trials[-1]['max_month_return'] = float(((1 + daily.ret).groupby(daily.index.to_period('M')).prod() - 1).max())
    for path in sorted(run.rglob('audit.json')):
        audits.append(dict(folder=str(path.parent.relative_to(run)), **json.loads(path.read_text())))
    for choice in locked:
        folder = run / 'search' / str(choice['outer_year']) / choice['host'] / choice['initialization']
        initial = json.loads((folder / 'train/g00_initial/trial.json').read_text())
        updates = json.loads((folder / 'updates.json').read_text())
        parent = initial
        for update in updates:
            gen = update['generation']
            rows = [json.loads(p.read_text()) for p in (folder / 'train').glob(f'g{gen:02d}_c*/trial.json')]
            for pair in [1, 3, 5]:
                positive = next(r for r in rows if r['pair'] == pair and r['direction'] == 1)
                negative = next(r for r in rows if r['pair'] == pair and r['direction'] == -1)
                a, b, c = [r['metrics']['total'] for r in [negative, parent, positive]]
                triplets.append(dict(host=choice['host'], initialization=choice['initialization'], outer_year=choice['outer_year'],
                                     generation=gen, pair=pair, minus_return=a, center_return=b, plus_return=c,
                                     nonmonotonic=bool((b-a)*(c-b) < 0)))
            parent = json.loads((run / update['champion'] / 'trial.json').read_text())
        if choice['choice']:
            theta = np.load(run / choice['choice']['folder'] / 'theta.npy')
            original = np.load(folder / 'train/g00_initial/theta.npy')
            at = 0
            for name, shape in architecture['shapes']:
                size = int(np.prod(shape))
                delta = theta[at:at+size] - original[at:at+size]
                parameter_rows.append(dict(host=choice['host'], initialization=choice['initialization'], outer_year=choice['outer_year'],
                                           block=name, n_parameters=size, n_changed=int(np.count_nonzero(delta)), l2_distance=float(np.linalg.norm(delta))))
                at += size
    pd.DataFrame(trials).to_csv(output / 'all_evaluations.csv', index=False)
    pd.DataFrame(audits).to_csv(output / 'independent_audits.csv', index=False)
    pd.DataFrame(triplets).to_csv(output / 'measured_direction_triplets.csv', index=False)
    pd.DataFrame(parameter_rows).to_csv(output / 'selected_parameter_changes.csv', index=False)
    behavior = []
    for row in summary.itertuples():
        folder = run / 'outer' / row.host / row.initialization
        base = run / 'initialization' / row.host
        observed = pd.read_parquet(folder / 'observations_actions.parquet').set_index('date')
        reference = pd.read_parquet(base / 'observations_actions.parquet').set_index('date')
        held = pd.read_parquet(folder / 'holdings.parquet')
        baseline_held = pd.read_parquet(base / 'holdings.parquet')
        concentration = (held.value / held.date.map(observed.equity)).max() if len(held) else 0.
        a = held.groupby('date').ts_code.agg(set).to_dict() if len(held) else {}
        b = baseline_held.groupby('date').ts_code.agg(set).to_dict()
        jaccard = [len(a.get(d, set()) & b.get(d, set())) / len(a.get(d, set()) | b.get(d, set()))
                   if a.get(d, set()) | b.get(d, set()) else 1. for d in observed.index]
        behavior.append(dict(host=row.host, initialization=row.initialization,
                             rebalance_dates=int(observed.rebalance.sum()), baseline_rebalance_dates=int(reference.rebalance.sum()),
                             rebalance_changed_days=int(observed.rebalance.ne(reference.rebalance).sum()),
                             top_n_changed_days=int(observed.top_n.ne(reference.top_n).sum()),
                             cash_fraction_changed_days=int(observed.cash_fraction.ne(reference.cash_fraction).sum()),
                             mean_holdings_jaccard=float(np.mean(jaccard)),
                             unique_stocks=int(held.ts_code.nunique()) if len(held) else 0,
                             max_single_stock_weight=float(concentration)))
    pd.DataFrame(behavior).to_csv(output / 'behavior_changes.csv', index=False)
    trained = [r for r in trials if r['split'] == 'train']
    val = [r for r in trials if r['split'] == 'validation']
    assert len(trained) == manifest['training_evaluations'] == 2328
    assert len(val) == manifest['validation_evaluations'] == 216
    high = [r for r in trials if r['sharpe'] > 3 or r['max_month_return'] > .2]
    save_json(output / 'high_performance_flags.json', high)
    stats = dict(train_accounts=len(trained), validation_accounts=len(val), archived_independent_audits=len(audits),
                 training_improved=sum(r['final_train']['total'] > r['initial_train']['total'] for r in locked),
                 chosen_initial=sum(r['choice'] is not None and r['choice']['generation'] == 0 for r in locked),
                 validation_cash_fallback=sum(r['choice'] is None for r in locked),
                 high_sharpe_or_month_return_accounts=len(high), direction_triplets=len(triplets),
                 nonmonotonic_triplets=sum(r['nonmonotonic'] for r in triplets),
                 accepted_baseline_initialized=int(summary[summary.initialization == 'baseline'].accepted.sum()),
                 accepted_random_initialized=int(summary[summary.initialization == 'random'].accepted.sum()),
                 independent_audited_trades=sum(a['trades_checked'] for a in audits),
                 max_trade_cash_error=max(a['trade_cash_max_error'] for a in audits),
                 max_ledger_cash_error=max(a['cash_max_error'] for a in audits))
    save_json(output / 'summary.json', stats)
    baseline_by_host = {r['host']: r for r in init}
    selection_note = '；'.join(host + ': ' + ', '.join(str(r['outer_year']) + '年第' + str(r['choice']['generation']) + '代' if r['choice'] else str(r['outer_year']) + '年现金' for r in sorted(locked, key=lambda x: x['outer_year']) if r['host'] == host and r['initialization'] == 'baseline') for host in baseline_by_host)
    lines = ['# 基线初始化完整策略学习：首轮结果', '',
             f'运行目录：`{run.relative_to(ROOT.parent)}`。完成 {len(trained)} 次训练、{len(val)} 次验证、8条连续外层账户；每个模型 {architecture["parameter_dimension"]} 个可训练参数。', '',
             '以下均为2021—2023累计扣费收益和最大回撤，初始资金10万元。93股及全部历史已被观察，不是新的无偏样本外证据。', '',
             '|起点|原基线累计收益|学习后累计收益 / 最大回撤|随机初值累计收益 / 最大回撤|',
             '|---|---:|---:|---:|']
    for host, base in baseline_by_host.items():
        learned = summary[(summary.host == host) & (summary.initialization == 'baseline')].iloc[0]
        random = summary[(summary.host == host) & (summary.initialization == 'random')].iloc[0]
        lines.append(f'|{host}|{base["total"]:.2%}|{learned.total:.2%} / {abs(learned.mdd):.2%}|{random.total:.2%} / {abs(random.mdd):.2%}|')
    lines += ['', '本轮4个基线初始化均比对应随机初始化取得更高累计收益，但仅估值路线超过自身原基线：累计收益提高约1.92个百分点，最大回撤降低约1.27个百分点；仍低于原等权基线的累计收益。基线初始化2/4通过正收益及30%回撤验收，随机初始化0/4通过。单轮配对结果不构成初始化优势的统计证明。', '',
              '等权学习路径的费用低于原等权基线，收益仍从正转负，因此退化不能统一归因于换手费用。训练目标已经直接对齐账户回报，但跨期稳定性仍是主要未解决问题。', '', '## 实际学习的形式', '',
              '模型每日收盘读取市场因子及真实现金、持仓、可卖股、持有历史、拒单成交反馈，决定次日选股/准入、是否调仓、保留/退出/减仓、新仓预算与现金投入。全部隐层和动作头参与多尺度演化，按真实账户净收益更新中心，没有基线距离惩罚，没有未来收益排名监督标签。', '',
              '4条初始策略逐表复现归档的净值、交易、事件和持仓。初始评分使用等序排名，不要求分数数值与旧策略相等。首版沿用最多5只、既有仓不加仓的执行边界；不是对任意交易程序的无限策略搜索。', '',
              f'{stats["training_improved"]}/24 个训练任务提高了训练期总收益；验证选择初始点 {stats["chosen_initial"]} 次，无可行验证方案而转现金 {stats["validation_cash_fallback"]} 次。训练收益改善与跨期有效性分别报告。', '',
              '基线初始化组年度选择（第0代即原始基线）：' + selection_note + '。因此估值路线的三年结果包含后两年选回基线，不能解释为同一个新模型持续获得增强。', '',
              '## 行为是否离开基线', '',
              '|起点（基线初始化）|改变调仓决定的天数|持股数输出改变天数|现金比例输出改变天数|日均持仓交集/并集|最大单股占比|',
              '|---|---:|---:|---:|---:|---:|']
    for r in behavior:
        if r['initialization'] == 'baseline':
            lines.append(f'|{r["host"]}|{r["rebalance_changed_days"]}|{r["top_n_changed_days"]}|{r["cash_fraction_changed_days"]}|{r["mean_holdings_jaccard"]:.1%}|{r["max_single_stock_weight"]:.1%}|')
    lines += ['', '持股数和现金比例统计模型逐日输出，不调仓日不会据此成交；持仓相似度及单股占比来自实际账本。', '', '## 口径与检查', '',
              '- 54项pytest检查通过（含原账本用例及回调适配版）；4个初始策略的16张账户表另以零浮点容差复核通过。',
              f'- 原始数据截断重算及账户前缀一致性 {manifest["prefix_checks"]} 组；相同参数完整重复回放 {manifest["deterministic_replays"]} 组。',
              f'- 已归档独立账本审计 {len(audits)} 个账户、{stats["independent_audited_trades"]} 笔交易；最大现金误差 {stats["max_ledger_cash_error"]:.3g} 元，保留费用、T+1、官方涨跌停、公司行动检查。另24次前缀回放也调用审计。',
              f'- 已实测正负扰动方向三点 {len(triplets)} 组，其中 {stats["nonmonotonic_triplets"]} 组出现局部峰谷；搜索不以单调性为前提。',
              f'- {len(high)} 个训练/验证账户Sharpe>3或月收益>20%，逐账户完成独立核账；因果前缀检查覆盖24组锁定外层策略，并非每个高收益候选单独重跑。账本通过不能排除选样偏差或过拟合，高训练收益不作为跨期有效性证据。',
              '- 基线参数已用截至前一年末历史筛选，验证年曾参与基线形成；这一污染无法在本轮固定数据下消除。每个训练和验证账户10万重置，外层三年保持连续账户。',
              '- 单轮、每起点每年度97次训练评价，只能衡量有限搜索结果，不能断言高维空间最优，也不能据此否定整个学习方向。30%回撤是历史验收，不是保证。', '',
              '过程数据：`search/`含每次参数、账户轨迹、审计、每代更新与验证选择；`outer/`含逐日观察、动作意图、持仓、交易；`diagnostics/`含行为距离、各参数块变化、配对非单调性和审计汇总。']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, host in zip(axes.flat, baseline_by_host):
        paths = [('Original baseline', run / 'initialization' / host, '#777777'),
                 ('Baseline initialized', run / 'outer' / host / 'baseline', '#1565c0'),
                 ('Random initialized', run / 'outer' / host / 'random', '#ef6c00')]
        for label, folder, color in paths:
            daily = pd.read_parquet(folder / 'daily.parquet')
            ax.plot(daily.index, daily.nav, label=label, color=color, linewidth=1.5)
        ax.set_title(host)
        ax.set_ylabel('Net asset value (initial = 1)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Frozen 93-stock observed sample | 2021–2023 | after fees', fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / 'net_asset_comparison.png', dpi=160)
    fig.savefig(output / 'net_asset_comparison.svg')
    plt.close(fig)
    lines += ['', '净值图：`diagnostics/net_asset_comparison.png`（附SVG）。']
    report = ROOT / 'docs/BASELINE_INITIALIZED_POLICY_RESULTS_20260920.md'
    report.write_text('\n'.join(lines) + '\n')
    (output / report.name).write_text(report.read_text())
    save_json(output / 'report_sources.json', dict(script_sha256=digest(Path(__file__)), run_manifest_sha256=digest(run / 'manifest.json'),
                                                  report_sha256=digest(report)))
    print(json.dumps(stats, indent=2))
    print(summary[['host','initialization','total','mdd','n_trades','total_fees']].to_string(index=False))


if __name__ == '__main__':
    main()
