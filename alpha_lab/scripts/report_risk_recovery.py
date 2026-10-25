"""从逐账户复算结果生成风险恢复研究报告，不手填绩效表。"""
from pathlib import Path
import json
import sys

import pandas as pd

from research_risk_recovery import ROOT, verify


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pct(value):
    return f'{value * 100:.2f}%'


run = Path(sys.argv[1]).resolve()
cfg, manifest = verify(run)
s = pd.read_csv(run / 'analysis/summary.csv')
r = pd.read_csv(run / 'diagnostics/calendar_robustness.csv')
j = pd.read_csv(run / 'analysis/june.csv').set_index('name')
a = json.loads((run / 'analysis/verification.json').read_text())
restores = pd.read_csv(run / 'analysis/actual_restores.csv')
aftermath = pd.read_csv(run / 'diagnostics/restore_next_sale.csv')
names = {'baseline': '上轮风险仓位', 'restore': '恢复旧仓',
         'grace': '整手缓冲', 'combined': '两者组合'}
windows = {'continued': '2022启动', 'cold': '2024启动'}
main = s[s.stress.eq('main') & s.schedule.eq('calendar_01')]
performance = table(['起点', '政策', '累计收益', '年化', '最大回撤', 'Sharpe', '成交笔数 / 交易日'],
    [[windows[x.window], names[x.policy], pct(x.total), pct(x.ann), pct(-x.mdd),
      f'{x.sharpe:.3f}', f'{x.n_trades} / {x.trade_days}'] for x in main.itertuples()])
robustness = table(['起点', '政策', '年化中位数', '回撤中位数', '现金中位数',
                    '收益提高 / 8', '回撤降低 / 8', '收益提高且回撤不增 / 8'],
    [[windows[x.window], names[x.policy], pct(x.ann), pct(-x.mdd), pct(x.cash_fraction),
      x.return_improved if x.policy != 'baseline' else '—',
      x.drawdown_improved if x.policy != 'baseline' else '—',
      x.both_improved if x.policy != 'baseline' else '—'] for x in r.itertuples()])
stress = table(['起点', '压力条件', '政策', '累计收益', '最大回撤', 'Sharpe', '成交笔数'],
    [[windows[x.window], '5万 / 双费用' if x.stress == 'cost2' else '10万参考',
      names[x.policy], pct(x.total), pct(-x.mdd), f'{x.sharpe:.3f}', x.n_trades]
     for x in s[~s.stress.eq('main')].itertuples()])
june = table(['起点', '政策', '五月末现金', '六月收益', '恢复成交', '缓冲事件'],
    [[windows[x.window], names[x.policy], pct(j.at[x.name, 'may_end_cash_fraction']),
      pct(j.at[x.name, 'june_return']), x.restore_fills, x.grace_events] for x in main.itertuples()])
run_rel = str(run.relative_to(ROOT))
text = f'''# 风险恢复与整手缓冲：76账户完整检验

日期：2026-09-14。研究目录：`{run_rel}`。

## 结论

本轮没有产生可替换上轮风险仓位版的通用改进。恢复买入提高资金使用率，但16个主配对中仅2个降低最大回撤；整手缓冲在全部16个配对中减少成交笔数，但近期收益只在3/8日历改善。两者合并也不稳定。保留上轮 `full_risk` 为防守研究对照，新增两项只保留实验开关，不改生产或默认策略。

更接近核心问题的解释是：**风险估计下降不足以证明值得重新增加仓位；整数股减仓还可能把小幅权重变化变成整只股票退出，从而改变未来持仓。** 本轮账本支持这两项机制，不能据此宣称已识别稳定的收益预测信号。

## 固定设计与执行规则

- 主资金5万元；月内1/8/15/22日复核及20交易日滚动0/5/10/15偏移，共8日历。分别2022、2024空仓启动，连续运行到2026-09-08；2022账户后续不重置。两窗口重叠，不是独立样本。
- 8日历×2起点×4政策=64主账户；原月初5万双费用8组、10万参考4组，共76组。2万元本轮不追加优化。
- 原信号、股票排名、持有退出规则、年度风险模型及折扣矩阵均冻结。这里 baseline 是上轮 `full_risk`，不是更早的原始Hybrid。最多5股，100股买入，不新增高价股过滤；原25%单股配置上限继续沿用，它是策略假设。
- **恢复旧仓**：只有实际风险减仓成交、仍有旧股、当次持有与完整入场条件通过，且风险折扣严格高于最近一次实际减仓/恢复成交时，才在原复核日考虑买回。新仓先预约预算，恢复仅用剩余收盘现金；按当前风险上限计算100股整数买量，不限于曾卖出的原股数。未成交不更新参考；全部卖出后取消恢复资格。
- **整手缓冲**：计算所需减仓量大于0且不足100股，并且市值未超过原25%上限时，暂不减仓。达到一手或原上限仍减；趋势/ST退出不受缓冲影响。因此风险折扣上限允许小幅超出，不能称为严格硬上限；25%也只在复核时约束，价格漂移仍会越过。
- 次日DAY限价、数量、含费预算在前晚固定，不预支次日卖款；恢复追加旧仓并单独保留新增股T+1锁定。买/卖成本0.1%/0.2%、最低5元；双费用同时翻倍比例及最低费用。费用已含在收益内，现金不计利息。年化与Sharpe用244交易日，Sharpe无风险利率设0。

有交易成本时设置不交易区间具有理论依据，见 [Leland原始研究](https://escholarship.org/uc/item/0fw6k0hm)。本轮“一手容忍”是结合账户约束提出的独立假设，并非复现论文最优阈值，也不借用其绩效结论。

## 原月初5万元：两个起点同时展示

最大回撤列用正数表示跌幅；成交笔数含买卖，不等于调仓次数。

{performance}

2022月初组合版累计166.31%看似突出，但2024月初累计58.72%，低于上轮59.86%，回撤也扩大。不能以较好的长期起点选定新方案。

## 全部日历的配对稳健性

{robustness}

每一行中位数独立计算，不代表同一个账户；改善次数来自逐日历配对。严格“回撤降低”不计持平，最后一列允许回撤持平。

汇总16主配对：恢复旧仓收益提高10组、回撤降低2组；整手缓冲收益提高11组、回撤降低5组；组合收益提高9组、回撤降低3组。收益提高且回撤不增分别仅1、4、2组。缓冲减少3—11笔成交，但交易日不一定减少：2022月初89→86笔，交易日55→57。

![收益与回撤逐日历配对](../{run_rel}/figures/paired_recovery_tradeoffs.png)

## 过程证据：资金到底去了哪里

### 恢复仓位削弱了六月现金缓冲

{june}

2024月初恢复版全期平均现金9.78%→7.96%，但五月末现金14.33%→3.12%，六月−9.22%→−10.01%。恢复条件获批后，增仓仍会暴露于随后回撤；风险下降不是预期收益上升的同义词。全部76账户首次买入后都没有完整空仓日，不能把本轮称为主动全空仓策略。

恢复共{len(restores)}笔真实成交（{restores[['date', 'ts_code']].drop_duplicates().shape[0]}个不同股票—成交日，跨账户重复并不独立）。其中{int(aftermath.risk_resale_within60.sum())}笔的下一次同股卖出在60个交易日内，原因再次为风险上限减仓；另有{int(aftermath.next_sale_reason.eq('no_observed_sale').sum())}笔到样本末未观察到后续卖出。此计数有右删失，下一次卖出也不一定对应刚买回的同一批股份，不能用作精确往返胜率。

### 整手缓冲的大幅增益来自持仓路径变化

2022月初逐股净损益对账：

- 原版飞科电器 `603868.SH` 在2022-03-02先卖100股，2022-11-02最后100股因上限卖出；11月3日补入中国石化 `600028.SH`。
- 缓冲版3月保留，7月才卖100股，12月按持有规则卖完；2023-01-05补入工业富联 `601138.SH`。工业富联贡献9,400.89元，原版中国石化贡献5,741.59元，飞科电器另增3,310.53元。其余股票与资金路径也改变，全账户净增10,805.01元。
- 原版总费用1,439.27元，缓冲版1,558.20元：成交笔数较少仍可能因成交金额与路径不同支付更多费用。该起点收益提高不能归为手续费节省。

反例在2024月初：中国人寿 `601628.SH` 后一轮原版买400股，随后在43.23、48.56元各减100股，余200股36.85元退出；缓冲版买300股并一直留到36.85元退出。该股全期净损益比原版少1,397.95元，超过账户净落后867.82元，其他股票部分抵消。保留小仓位既可能留住趋势，也可能放弃原有减仓保护。

这些是实际账本净损益差，包含持仓数量、现金和后续选择联动；不是某项单独改动的固定篮子因果收益。`actual_restores.csv`中的20/60日价格路径只用于描述，不含成交成本、实际退出和后续分配，禁止当作策略收益。

## 成本与资金参考

{stress}

双费用没有让近期改动稳定优于原版。10万元组合版2022起点收益略增但回撤、Sharpe恶化；2024起点收益与回撤都退化。费用改变也会改变整数股与买入路径，因此压力账户不是在原净值上直接扣固定费用。

## 核验、复现与边界

- 37项测试通过：6项新增恢复/缓冲、6项风险仓位、25项因果执行对抗测试。覆盖真实减仓记忆、失败订单、冻结现金、旧股保留与新增股T+1、未来折扣扰动等。
- 20个基线账户（16主、4压力）的日账、交易、持仓、事件四类CSV精确复现父实验。
- 恢复/缓冲/组合三种实际账户分别截断至2025-06-30重放；截断前交易、持仓、净值和已决策订单一致。
- {a['cases']}账户、{a['orders']:,}个订单计划、{a['fills']:,}笔成交全部独立对账；核验{sum(x['recovery_plans_checked'] for x in a['checks'])}个恢复计划、{sum(x['grace_checked'] for x in a['checks'])}个缓冲事件。现金非负、最多5股、100股买入、决策日早于执行日成立；逐股含费及分红损益和总资产变化最大误差{max(x['asset_pnl_error'] for x in a['checks']):.2g}元。
- 首次分析脚本因空列表初始化数量错误退出，原源码和错误日志保留于 `analysis_failed_attempt/`；修正分析初始化后完成复算，执行引擎和76账户结果未改。分析/诊断脚本仅接受运行目录参数，不提供 `--help`。
- 完整76账户表为 `analysis/summary.csv`，逐月逐股损益为 `analysis/monthly_asset_pnl.csv`；日历配对与恢复后再次卖出见 `diagnostics/`。运行源码、配置、协议和各级哈希一并封存，最终核验凭据为本轮 `completion.json`。

本轮沿用已观察历史，没有新的未触碰留出集，8日历与两起点不能充当16份独立市场证据。风险模型按年仅使用已成熟标签，但既有原始数据版本、历史复权差异和日线限价队列仍有外部真实性边界；截断通过不等于所有数据已获PIT认证。此次账户没有退市事件，因此未覆盖退市执行压力；也没有补齐2018起资金账户回测。全部结果保持研究状态。

复现入口（从 `alpha_lab/` 运行；准备会创建新目录，执行和分析拒绝覆盖已有目录）：

```bash
python scripts/run_risk_recovery_tests.py
python scripts/research_risk_recovery.py --prepare
python scripts/research_risk_recovery.py --run NEW_RUN --window continued --family calendar
# 两个window：continued/cold；三个family：calendar/rolling/stress，共六次执行
python scripts/analyze_risk_recovery.py NEW_RUN
python scripts/diagnose_risk_recovery.py NEW_RUN
python scripts/verify_recovery_prefixes.py NEW_RUN
python scripts/plot_risk_recovery.py NEW_RUN
python scripts/report_risk_recovery.py NEW_RUN
python scripts/seal_risk_recovery.py NEW_RUN
```

下一研究重点应是同一旧仓在“保持/减仓/恢复”动作下的收益与下行风险差异，以及何时值得彻底退出；当前两特征风险水平不能独立回答恢复买入是否划算。整手缓冲可作为交易频率对照保留，不根据本轮最好起点继续挑阈值。
'''
(ROOT / 'docs/RISK_RECOVERY_RESEARCH_20260914.md').write_text(text)
(run / 'REPORT.md').write_text(text)
print('REPORT', len(text), 'characters')
