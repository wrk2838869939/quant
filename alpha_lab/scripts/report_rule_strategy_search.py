#!/usr/bin/env python3
"""汇总规则搜索的全部采样点、时间锁定选择与连续账户；不再选参。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    manifest = json.loads((run / 'manifest.json').read_text())
    rows = []
    for path in sorted((run / 'search').glob('*/*/trials.jsonl')):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            metrics = item.get('metrics', {})
            rows.append(dict(family=item['family'], outer_year=item['outer_year'], id=item['id'],
                             phase=item['phase'], order=item['order'], status=item['status'],
                             params_json=json.dumps(item['params'], sort_keys=True),
                             **{f'p_{k}': v for k, v in item['params'].items() if k != 'family'},
                             **{k: v for k, v in metrics.items() if k != 'yearly'},
                             error=item.get('error', '')))
    trials = pd.DataFrame(rows)
    trials.to_csv(run / 'all_trials.csv', index=False)
    trials.to_parquet(run / 'all_trials.parquet', index=False)
    counts = trials.groupby(['outer_year', 'family']).agg(
        trials=('id', 'size'), errors=('status', lambda s: int((s != 'ok').sum())),
        feasible=('feasible', 'sum'), best_inner_median_ann_any=('median_ann', 'max')).reset_index()
    feasible_best = trials[trials.feasible.eq(True)].groupby(['outer_year', 'family']).median_ann.max()
    counts = counts.join(feasible_best.rename('best_feasible_inner_median_ann'), on=['outer_year', 'family'])
    counts.to_csv(run / 'search_counts.csv', index=False)
    selections = json.loads((run / 'all_locked_choices.json').read_text())
    selected = []
    for choice in selections:
        best = choice['selected']
        selected.append(dict(family=choice['family'], outer_year=choice['outer_year'],
                             selected_id=best['id'] if best else '', locked_at=choice['locked_at'],
                             inner_end=choice['inner_end'], feasible_count=choice['feasible'],
                             params_json=json.dumps(best['params'], sort_keys=True) if best else '{}',
                             median_inner_ann=best['metrics']['median_ann'] if best else np.nan,
                             inner_mdd=best['metrics']['mdd'] if best else np.nan,
                             neighbor_count=best.get('neighbor_count', 0) if best else 0,
                             neighbor_q25=best.get('neighbor_q25') if best else None))
    pd.DataFrame(selected).to_csv(run / 'locked_selection_summary.csv', index=False)

    # 保持其他参数完全相同，寻找真实采样点之间的方向反转；不插值、不回归补点。
    examples = []
    for (family, year), group in trials[trials.status.eq('ok')].groupby(['family', 'outer_year']):
        params = {row.id: json.loads(row.params_json) for row in group.itertuples()}
        numeric = sorted({k for p in params.values() for k, v in p.items() if isinstance(v, (int, float))})
        for axis in numeric:
            buckets = {}
            for row in group.itertuples():
                p = params[row.id]
                fixed = json.dumps({k: v for k, v in p.items() if k != axis}, sort_keys=True)
                buckets.setdefault(fixed, []).append((p[axis], row.median_ann, row.id))
            for fixed, points in buckets.items():
                points.sort()
                for a, b, c in zip(points, points[1:], points[2:]):
                    d1, d2 = b[1] - a[1], c[1] - b[1]
                    if abs(d1) > 1e-5 and abs(d2) > 1e-5 and d1 * d2 < 0:
                        examples.append(dict(family=family, outer_year=year, axis=axis, fixed_params=fixed,
                                             x1=a[0], x2=b[0], x3=c[0], y1=a[1], y2=b[1], y3=c[1],
                                             trial1=a[2], trial2=b[2], trial3=c[2]))
    example_frame = pd.DataFrame(examples)
    example_frame.to_csv(run / 'observed_nonmonotonic_triplets.csv', index=False)
    # 同结构/频率的实际数值邻域；缺少采样证据的点明确标未知。
    valid = trials[trials.status.eq('ok')].copy()
    if 'p_variant' not in valid:
        valid['p_variant'] = ''
    neighborhoods = []
    for keys, group in valid.groupby(['family', 'outer_year', 'p_variant', 'p_frequency'], dropna=False):
        param_cols = [k for k in group if k.startswith('p_') and pd.api.types.is_numeric_dtype(group[k]) and group[k].notna().all()]
        values = group[param_cols].to_numpy(dtype=float)
        floors = np.array([.01 if 'quantile' in k else 1. for k in param_cols])
        tol = np.maximum(np.abs(values) * .1, floors)
        for i, row in enumerate(group.itertuples()):
            mask = (np.abs(values - values[i]) <= tol[i] + 1e-12).all(axis=1)
            mask[i] = False
            neighbors = group.iloc[np.flatnonzero(mask)]
            sufficient = len(neighbors) >= 3
            neighborhoods.append(dict(family=keys[0], outer_year=int(keys[1]), id=row.id,
                                      evaluated_neighbors=len(neighbors), sufficient_evidence=sufficient,
                                      neighbor_ann_q25=float(neighbors.median_ann.quantile(.25)) if sufficient else np.nan,
                                      neighbor_risk_pass_fraction=float(neighbors.feasible.mean()) if sufficient else np.nan,
                                      center_ann=row.median_ann, center_mdd=row.mdd))
    pd.DataFrame(neighborhoods).to_csv(run / 'parameter_neighborhoods.csv', index=False)
    summary_path = run / 'outer_summary.csv'
    if not summary_path.exists():
        print(f'仅试跑：{len(trials)}次；没有外层报告')
        return
    summary = pd.read_csv(summary_path)
    controls = pd.read_csv(run / 'control_summary.csv')
    names = {'S01': '低换手', 'S02': '低波/振幅', 'S03': 'Amihud', 'S04': '估值',
             'S05': '短期反转', 'S06': '跳近端动量', 'S07': '趋势质量', 'S10': '资金流',
             'S19': '波动换手状态切换', 'S20': '条件振幅', 'S21': '量价相关差',
             'S22': '换手变化', 'S23': '高点锚定', 'momentum': '动量调参',
             'macd': 'MACD调参', 'moving_average': '双均线调参'}
    nav = {}
    yearly = []
    for row in summary.itertuples():
        d = pd.read_parquet(run / 'outer' / row.family / 'daily.parquet')
        nav[row.family] = d.nav
        for item in json.loads(row.yearly):
            yearly.append(dict(family=row.family, **item))
    pd.DataFrame(yearly).to_csv(run / 'outer_yearly.csv', index=False)
    nav = pd.DataFrame(nav)
    nav.to_csv(run / 'outer_nav.csv')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for name in nav:
        axes[0].plot(nav.index, nav[name], label=name, lw=1)
        peaks = np.maximum.accumulate(np.r_[1., nav[name]])[1:]
        axes[1].plot(nav.index, nav[name].to_numpy() / peaks - 1, lw=1)
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_ylabel('Continuous account NAV')
    axes[1].set_ylabel('Drawdown')
    axes[1].axhline(-.30, color='red', ls='--')
    fig.suptitle('Chronological parameter selection — engineering sample, 2021–2023')
    fig.tight_layout()
    fig.savefig(run / 'outer_nav_drawdown.png', dpi=160)
    plt.close(fig)

    # 选一个低维家族，按频率和持股数分面展示真实点，不能混淆不同切片。
    g = trials[trials.family.eq('S01') & trials.outer_year.eq(2023) & trials.status.eq('ok')]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=True)
    for i, freq in enumerate(['daily', 'weekly', 'monthly']):
        for j, top in enumerate([3, 4, 5]):
            part = g[g.p_frequency.eq(freq) & g.p_top_n.eq(top)]
            points = axes[i, j].scatter(part.p_window, part.median_ann, c=part.mdd, cmap='viridis', vmin=-.3, vmax=0, s=22)
            axes[i, j].set_title(f'{freq}, top {top}, n={len(part)}')
            axes[i, j].axhline(0, color='gray', lw=.5)
            if i == 2:
                axes[i, j].set_xlabel('Turnover lookback')
            if j == 0:
                axes[i, j].set_ylabel('Median inner-year net CAGR')
    fig.suptitle('S01: inner 2019–2022, selection for 2023; observed samples, no interpolation')
    fig.colorbar(points, ax=axes, label='Inner maximum drawdown', shrink=.8)
    fig.savefig(run / 'S01_parameter_samples.png', dpi=160)
    plt.close(fig)

    lines = ['# 无训练规则搜索执行报告', '',
             f"完成状态：{manifest['status']}。共{len(trials):,}次正式内层试验，{int((trials.status!='ok').sum())}次错误，覆盖{trials.family.nunique()}个家族与3个时间折。",
             '', '初始资金100,000元；保持原费用、整手、涨跌停、T+1、公司行动及原始价格成交规则。',
             '股票池为固定93只工程样本，有全期质量隔离偏差；2019—2023已被历史研究使用。以下为按时间选参流程的回放，不是全市场新鲜样本外有效性认证。', '',
             '## 执行方式', '',
             f"- 本运行家族：{', '.join(manifest['families'])}。每家族每折上限{manifest['budget']}次；小空间穷举，较大空间按约一半全局、四分之一局部、其余继续全局分配。",
             '- 13个新增规则家族按登记的整数/阈值范围抽样；动量、MACD、双均线穷举登记的有限离散参数集合，其列表外参数未评估。',
             '- 依次用2019—2020选择2021参数、2019—2021选择2022参数、2019—2022选择2023参数。选择记录先落盘，再生成外层账户。',
             '- 内层选参同时要求连续账户及各年回撤不超过30%；没有可行配置的年份转现金，不能把此类现金结果叫作选股成功。',
             '- 30%是参数筛选与结果验收阈值；后续外层账户仍可能超限，超限按实际结果计为风险不合格。',
             '- 外层2021—2023只初始化一次10万元，跨年保留现金、股数和事件。新参数在当年第一个收盘确认、下个交易日执行。',
             '- 各试验保存全部信号、净值、成交、持仓与事件；每次检查资产恒等式和费用。外层账户及基准另做逐笔独立现金/股数/T+1对账，跨年截断账本一致。',
             '- 参数区间和方向在正式搜索前固定，试跑只验证工程。历史默认点纳入参考，不按试跑表现修改空间。', '',
             '## 连续外层账户', '',
             '| 家族 | 累计净收益 | 年化 | 最大回撤 | 有参数可用年份 | 回撤≤30% |',
             '|---|---:|---:|---:|---:|---|']
    for row in summary.sort_values('family').itertuples():
        flag = '是' if row.risk_pass else '否'
        if row.all_cash_due_to_no_candidate:
            flag = '纯现金，不算策略通过'
        label = row.family + ' ' + names.get(row.family, '')
        lines.append(f'| {label} | {row.total:.2%} | {row.ann:.2%} | {row.mdd:.2%} | {row.selected_years}/3 | {flag} |')
    matched_path = run / 'matched_control_summary.csv'
    if matched_path.exists():
        matched = pd.read_csv(matched_path)
        lines += ['', '### 同窗口口径的固定参数对照', '',
                  '新搜索按市场交易日网格计算窗口；旧缓存按有报价记录数计算窗口，停牌附近可能不同。下面固定参数对照与新搜索共用同一信号实现，原缓存基准另保留，避免将窗口口径变化算成调参贡献。', '',
                  '| 固定参数/频率 | 累计净收益 | 最大回撤 |', '|---|---:|---:|']
        for row in matched.itertuples():
            lines.append(f'| {row.strategy} | {row.total:.2%} | {row.mdd:.2%} |')
    random_controls = controls[controls.strategy.str.startswith('random_')].copy()
    if not random_controls.empty:
        random_controls['frequency'] = random_controls.strategy.str.split('_').str[1]
        lines += ['', '### 固定随机种子对照', '',
                  '每种频率使用事前固定的5个随机种子，展示同账户下随机选股结果的离散程度；这5次对照不构成显著性检验。', '',
                  '| 频率 | 累计净收益中位数 | 最低 | 最高 |', '|---|---:|---:|---:|']
        for frequency, group in random_controls.groupby('frequency'):
            lines.append(f'| {frequency} | {group.total.median():.2%} | {group.total.min():.2%} | {group.total.max():.2%} |')
    lines += ['', '这些结果评价的是“事前搜索并按风险条件选参”的流程，不能反过来从外层收益挑一个赢家再宣称其为预先选定。独立家族全部保留，不输出择优拼接组合。', '',
              '## 非单调性检查', '',
              f'在其他参数完全相同的真实采样切片中，共记录{len(examples)}个三点方向反转案例（不是独立统计样本）。只说明这些已采样点不满足单调顺序，不推断未采样点，也不证明全局最优。']
    if examples:
        lines += ['', '| 家族/折 | 参数 | 三个取值 | 对应内层年化中位数 |', '|---|---|---|---|']
        shown = set()
        for ex in examples:
            if ex['family'] in shown:
                continue
            shown.add(ex['family'])
            lines.append(f"| {ex['family']}/{ex['outer_year']} | {ex['axis']} | {ex['x1']}, {ex['x2']}, {ex['x3']} | {ex['y1']:.2%}, {ex['y2']:.2%}, {ex['y3']:.2%} |")
            if len(shown) == 5:
                break
    lines += ['', '## 文件与边界', '',
              '- [全部试验](all_trials.csv)、[每折搜索计数](search_counts.csv)、[锁定参数](locked_selection_summary.csv)、[完整参数选择原件](all_locked_choices.json)。',
              '- [参数邻域表](parameter_neighborhoods.csv)：仅使用同结构、同频率下真实运行的邻近配置，少于3个邻点时不评为稳定平台。',
              '- [外层汇总](outer_summary.csv)、[逐年结果](outer_yearly.csv)、[固定基准](control_summary.csv)、[非单调三点原件](observed_nonmonotonic_triplets.csv)。',
              '- [同窗口口径固定基准](matched_control_summary.csv)、[窗口定义差异](fixed_formula_calendar_differences.csv)、[真实数据信号前缀检查](real_input_signal_prefix_checks.csv)。',
              '- [连续账户图](outer_nav_drawdown.png)、[参数采样图](S01_parameter_samples.png)。',
              '- search/年份/家族/配置ID/ 保存每次内层信号、账户与事件；outer/家族/ 保存连续外层账户、持股数日程、独立对账和前缀验证。',
              '- 核心搜索只包含上方列明的家族；覆盖层如有完成则在下节独立报告，其余覆盖层和行业/跨市场分支不据此声称已完成。训练与扩展方向未执行。',
              '- 搜索预算有限，未采样区域仍未知；风险不合格或无可行配置只限本次样本、参数域和预算，不等于整个经济机制无效。', '']
    overlay = run / 'overlay_study/outer_summary.csv'
    if overlay.exists():
        ov = pd.read_csv(overlay)
        ov_manifest = json.loads((overlay.parent / 'manifest.json').read_text())
        supplement = ['# 首批规则覆盖层补充', '',
                      f"完成{ov_manifest['trials']}组内层联合枚举。固定两个宿主：20日动量、20日低换手；比较O01计数排雷、O02平滑/排名缓冲、O12现金储备。", '',
                      '本补充用2019—2020一次锁定参数，再完整评价2021—2023；与核心每年滚动选参分开。配对基准使用同一宿主、同一被选频率和同一持股数，只移除覆盖层。不得用这些外层结果重新选覆盖层。', '',
                      '| 模块/宿主 | 净收益 | 最大回撤 | 平均股票仓位 | 无覆盖层收益 | 无覆盖层回撤 |',
                      '|---|---:|---:|---:|---:|---:|']
        for row in ov.itertuples():
            if row.selected:
                supplement.append(f'| {row.module}/{row.host} | {row.total:.2%} | {row.mdd:.2%} | {row.exposure:.1%} | {row.paired_total:.2%} | {row.paired_mdd:.2%} |')
            else:
                supplement.append(f'| {row.module}/{row.host} | 无内层可行解 | — | — | — | — |')
        supplement += ['', 'O12保留的现金始终计入同一10万元账户，成交后现金、净资产和收益均以总账户核算；股票预算变小造成的风险下降不称为选股增强。',
                       'O02的内层排名状态从2019初初始化，启动等待期计入完整内层账户；外层使用此前历史排名热身。所有策略均不删除现金等待日以缩短评价区间。',
                       '候选与配对账户均完成独立现金/股数/T+1核验。参数、所有试验及逐笔过程见本目录CSV、locked_choices.json和inner/outer/paired子目录。', '']
        (overlay.parent / 'REPORT.md').write_text('\n'.join(supplement))
        lines += ['', '## 首批规则覆盖层补充', '',
                  f"另外完成{ov_manifest['trials']}次独立联合枚举，覆盖计数排雷、平滑/排名缓冲、固定现金储备。详见[覆盖层报告](overlay_study/REPORT.md)及[成对结果](overlay_study/outer_summary.csv)。",
                  '这项补充采用一次锁参后的三年评价，不与核心滚动选参混为同一种流程；其他覆盖层及行业/跨市场分支仍未执行。', '']
    (run / 'REPORT.md').write_text('\n'.join(lines))
    print(f'REPORT={run / "REPORT.md"}; trials={len(trials)}; observed_nonmonotonic_triplets={len(examples)}')


if __name__ == '__main__':
    main()
