#!/usr/bin/env python3
"""合并入场有效期和换手消融证据，不按历史最高收益选择生产参数。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    study = yaml.safe_load((run / 'study_config.yaml').read_text())
    names = {'original': '原个股规则', 'no_expansion_gate': '取消放量硬门槛', 'no_turnover_gates': '取消两项换手硬门槛'}
    rows, marginal, execution, gaps = [], [], [], []
    base = pd.read_parquet(run / 'original/first_opportunities.parquet')
    base = base[base.scope.eq('chronological_selected') & base.variant.eq('same_event_today')]
    base_keys = set(zip(base.event_id, base.ts_code))
    for rule in study['variants']:
        folder = run / rule
        assert json.loads((folder / 'manifest.json').read_text())['status'] == 'complete'
        counts = pd.read_csv(folder / 'opportunity_counts.csv')
        observations = pd.read_parquet(folder / 'price_observations_NOT_ACCOUNT.parquet')
        observations = observations[observations.scope.eq('chronological_selected') & observations.horizon.eq(20)]
        for variant in ['same_event_today', 'persistent_5d']:
            count = counts[counts.scope.eq('chronological_selected') & counts.variant.eq(variant)].iloc[0]
            sample = observations[observations.variant.eq(variant)].copy()
            sample['added_to_original'] = [(a, b) not in base_keys for a, b in zip(sample.event_id, sample.ts_code)]
            valid = sample[sample.complete]
            values = valid.cost_adjusted_price_return
            rows.append(dict(stock_rule=rule, entry_rule=variant, event_stocks=int(count.event_stocks),
                             sector_events=int(count.sector_events), first_entry_dates=int(count.first_entry_dates),
                             complete=len(valid), incomplete=len(sample) - len(valid),
                             mean=values.mean(), median=values.median(), positive=values.gt(0).mean(),
                             q10=values.quantile(.1), minimum=values.min(),
                             event_equal_weight_mean=valid.groupby('event_id').cost_adjusted_price_return.mean().mean()))
            extra = sample[sample.added_to_original]
            vextra = extra[extra.complete]
            if len(extra):
                marginal.append(dict(stock_rule=rule, entry_rule=variant, added_event_stocks=len(extra),
                                     events=extra.event_id.nunique(), first_entry_dates=extra.signal_date.nunique(),
                                     complete=len(vextra), mean=vextra.cost_adjusted_price_return.mean(),
                                     median=vextra.cost_adjusted_price_return.median(),
                                     positive=vextra.cost_adjusted_price_return.gt(0).mean(),
                                     q10=vextra.cost_adjusted_price_return.quantile(.1)))
            execution.append(dict(stock_rule=rule, entry_rule=variant, observations=len(sample),
                                  both_open_limit_checks_pass=int((sample.entry_execution_check.eq('open_quote_and_limit_pass_only')
                                                                  & sample.exit_execution_check.eq('open_quote_and_limit_pass_only')).sum()),
                                  missing_limits=int((sample.entry_execution_check.eq('missing_official_limits')
                                                      | sample.exit_execution_check.eq('missing_official_limits')).sum()),
                                  blocked_at_limit=int((sample.entry_execution_check.eq('blocked_at_price_limit')
                                                        | sample.exit_execution_check.eq('blocked_at_price_limit')).sum()),
                                  adjustment_changes=int(sample.adjustment_changes.gt(0).sum())))
            gaps.append(sample[~sample.complete].assign(stock_rule=rule))
    frame, extras, checks = pd.DataFrame(rows), pd.DataFrame(marginal), pd.DataFrame(execution)
    frame.to_csv(run / 'chronological_comparison.csv', index=False)
    extras.to_csv(run / 'marginal_opportunities.csv', index=False)
    checks.to_csv(run / 'execution_checks.csv', index=False)
    pd.concat(gaps, ignore_index=True).to_csv(run / 'incomplete_price_paths.csv', index=False)
    lines = ['# 当前策略优化验证结果', '',
             '**结论：五日资格单独作用很小；换手硬门槛是主要数量瓶颈，但直接放开会降低当前样本的中位表现和正收益比例。两种改法均不足以据此直接升级实盘。**', '',
             '数据截至2026-10-12。331个板块事件逐条复原，覆盖4,156只在相关事件日期属于观察板块的股票。全部股票条件使用当时数据及成员区间。实盘脚本、配置、冻结模型、LATEST指针均未修改。', '',
             '主结果使用2019—2026年的8次年度时间推进FNN：训练只用年初前20个交易日之前已经成熟的标签；4/8规则、100轮、学习率及种子固定继承，不依据本轮结果选参。256个事件可评分，118个通过。2014—2018训练不足60事件，不参与主表。该方法修正模型训练时间方向，但超参数和规则已观察过历史，仍是开发复核，不是全新独立测试。', '',
             '## 1. 两项优化的数量与价格表现', '',
             '以下“机会”是去重的事件×股票候选，不是独立行情或实际成交笔数。日期为各事件×股票首次满足条件的日期去重。价格观察使用信号次日复权开盘至20交易日后复权开盘，施加固定约40bp双边费用及滑点假设；**不是账户收益**。', '',
             '| 个股规则 | 板块入场资格 | 去重机会 | 有候选的板块事件 | 首次出现日期 | 完整/缺口样本 | 均值 | 中位数 | 正收益比例 | 10%分位 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in frame.itertuples():
        entry = '同事件当天确认' if r.entry_rule == 'same_event_today' else '同事件保留5日'
        lines.append(f'| {names[r.stock_rule]} | {entry} | {r.event_stocks} | {r.sector_events} | {r.first_entry_dates} | {r.complete}/{r.incomplete} | {r.mean:+.2%} | {r.median:+.2%} | {r.positive:.1%} | {r.q10:+.2%} |')
    lines += ['', '原规则仅放宽时间资格：26→27个机会、板块事件19→19。新增1只同轮股票，另有1只提前一天出现。不能把板块资格日875→1,282的约47%增长当作最终交易机会也增长47%。', '',
              '只取消放量硬门槛：26→269个机会，但新增243个机会的20日价格观察中位数为负。取消两项换手门槛进一步扩容，但仍不能把平均值为正解释为实盘优势。原规则仅26个样本，其较高均值和正收益比例本身也缺乏充分独立证据。', '',
              '扩展池的机会集中在同一批行情：269个机会对应70个板块事件、69个首次出现日；521个机会对应93个板块事件、102个首次出现日。整个约8年窗口仍不是稳定的每周供给。', '',
              '## 2. 新增部分单独评价', '',
              '相对“原个股规则＋严格同事件当天确认”的新增候选，未将原有优质样本混作新增收益：', '',
              '| 个股规则 | 入场资格 | 新增事件×股票 | 涉及板块事件 | 完整样本 | 均值 | 中位数 | 正收益比例 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in extras.itertuples():
        lines.append(f'| {names[r.stock_rule]} | {r.entry_rule} | {r.added_event_stocks} | {r.events} | {r.complete} | {r.mean:+.2%} | {r.median:+.2%} | {r.positive:.1%} |')
    lines += ['', '新增部分存在同板块和同日相关性；这些数字是描述性统计，没有把逐股票样本量当作独立显著性证据。', '',
              '## 3. 执行与缺口限制', '',
              '```', checks.to_string(index=False), '```', '',
              '已有官方涨跌停缓存不能覆盖所有计划买卖日；放宽两项换手的当天确认版本有4个未来行情路径不完整，其中存在计划退出日缺报价/停牌。它们保留在incomplete_price_paths.csv并排除完整路径统计，没有当成零损失或前向填价。此类排除也会影响分布，因此不能将表格当作无偏交易回测。', '',
              '即使原始开盘与涨跌停检查通过，也未验证排队成交、整手最小费用、10万元现金、持仓重叠、分红到账及送转股可卖时间。复权价格观察不能替代这些账本。现阶段不能确认30%账户回撤目标或声称新增机会已具有净交易优势。', '',
              '## 4. 落地判断', '',
              '1. 五日资格可保留为研究流程改进，但其增量太小，不承担扩大机会的主要目标。必须先绑定同一事件和失效规则，不能简单按板块连接旧模型分数。',
              '2. 不把放量门槛直接从生产删掉。它适合成为分层研究的依据：原完整规则作为核心样本，未放量但仍缩量减速的股票进入扩展观察池；两项换手都不满足的股票作为更宽的研究对照。扩展池尚未获得自动买入资格。',
              '3. 如果下一阶段继续优化，优先比较扩展池内的事前排序与核心池在同一账户中的增量价值，要求时间推进、执行数据完整、新增候选单独归因。不能强制取Top-N，也不能因今天无候选就降低所有条件。',
              '4. 若目标是每周有新机会，当前板块事件族仍是上限，需要另行验证守住支撑的企稳等独立形态；本轮未对新形态做回测，不把它写成已验证有效。', '',
              '## 5. 核验与复现', '',
              '- 原板块331个事件精确复原；原股票96条未叠加板块资格的记录在换手消融程序中逐列精确复原。',
              '- 5项针对性测试通过；6个不同历史日期的加速趋势计算与原函数一致。',
              '- 8次年度拟合的训练标签截止、训练/测试事件互斥、阈值重放检查通过。',
              '- 原事件、模型、收益和扫描文件136项哈希核验通过；全部三种换手方案及三种板块资格方案均披露。', '',
              f'初始研究：`{study["parent_run"]}`。换手对照：`{run}`。', '',
              '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python datahub/scripts/frozen_research.py run --release datahub/releases/selection_live_20261013 alpha_lab/scripts/research_sector_turnover_gates.py',
              '# 对新输出目录的 original、no_expansion_gate、no_turnover_gates 分别运行 report_sector_entry_window.py，再运行本汇总脚本。',
              '```', '']
    report = ROOT / 'alpha_lab/docs/current_strategy_20261013/OPTIMIZATION_RESULTS.md'
    report.write_text('\n'.join(lines))
    shutil.copy2(report, run / 'REPORT.md')
    shutil.copy2(Path(__file__), run / Path(__file__).name)
    manifest = json.loads((run / 'manifest.json').read_text())
    manifest.update(status='complete', report=str(report.relative_to(ROOT)),
                    account_backtest=False, production_modified=False,
                    artifacts={str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in run.rglob('*') if p.is_file() and p != run / 'manifest.json'})
    (run / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(frame.to_string(index=False))
    print('REPORT', report)


if __name__ == '__main__':
    main()
