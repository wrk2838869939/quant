"""汇总风险折扣的完整执行结果，保留日期差异、资金约束和失败记录。"""
from pathlib import Path
import sys
import json
import pandas as pd
from research_risk_budget import ROOT

run = Path(sys.argv[1]).resolve()
assert not (run / 'completion.json').exists()
s = pd.read_csv(run / 'analysis/summary.csv')
r = pd.read_csv(run / 'diagnostics/calendar_robustness.csv')
j = pd.read_csv(run / 'diagnostics/june_summary.csv').set_index('name')
yearly = pd.read_csv(run / 'analysis/yearly.csv')
checks = json.loads((run / 'analysis/verification.json').read_text())
assert len(s) == 76
labels = {'baseline': '原Hybrid', 'uniform': '统一降仓', 'entry_risk': '仅新买入风险折扣', 'full_risk': '新旧仓风险折扣'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
                      + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


main_rows = []
for window, label in [('cold', '2024空仓启动'), ('continued', '2022连续账户')]:
    for policy in ['baseline', 'uniform', 'entry_risk', 'full_risk']:
        row = s[s.name.eq(f'{window}_calendar_01_{policy}')].iloc[0]
        main_rows.append([label, labels[policy], f'{row.total:+.2%}', f'{row.ann:.2%}',
                          f'{-row.mdd:.2%}', f'{row.sharpe:.3f}', int(row.n_trades), int(row.trade_days)])
robust = []
for row in r.itertuples():
    robust.append(['2024启动' if row.window == 'cold' else '2022启动', labels[row.policy],
        f'{row.ann_median:.2%}', f'{-row.mdd_median:.2%}', f'{row.ann_min:.2%}—{row.ann_max:.2%}',
        f'{row.return_improved}/8', f'{row.drawdown_improved}/8'])
stress = []
for window in ['cold', 'continued']:
    for test, label in [('small', '2万元'), ('large', '10万元'), ('cost2', '5万元双费用')]:
        pair = s[s.window.eq(window) & s.stress.eq(test)].set_index('policy')
        stress.append(['2024启动' if window == 'cold' else '2022启动', label,
            f"{pair.at['baseline','total']:+.2%} → {pair.at['full_risk','total']:+.2%}",
            f"{-pair.at['baseline','mdd']:.2%} → {-pair.at['full_risk','mdd']:.2%}"])
annual_rows = []
for year in [2024, 2025, 2026]:
    group = yearly[yearly.year.eq(year)].set_index('name')
    annual_rows.append([year, f"{group.at['cold_calendar_01_baseline','total']:+.2%}",
                       f"{group.at['cold_calendar_01_full_risk','total']:+.2%}"])

text = f'''# 风险仓位折扣：76账户完整执行检验

报告日期2026-09-14；行情截至2026-09-08。运行目录时间戳沿用执行环境的UTC时钟。全部历史已观察，未进行实盘交易或修改默认生产策略。

## 结论

保留“Hybrid＋新旧仓风险折扣”（`full_risk`）为防守研究候选。8日历×2起点的16组主对照，全部最大回撤降低，10组累计收益同时提高；相对统一降仓对照，16组回撤降低、14组收益提高。

这不是全面替换基线的理由。2022起2万元账户累计收益从+141.52%降为+112.28%，回撤仅从14.11%改善到13.84%；长期资金约束下的代价明显。风险减仓增加成交笔数、形成难以补回的现金，并未实现主动全部空仓。

## 为什么重建训练样本

上一轮模型仅在退出状态训练，直接用于正常持仓会发生样本迁移。本轮从历史当时Top20候选的完整生命周期提取月度入场、持有、退出状态，共7,714条；同股间隔严格超过60交易日后保留2,470条，其中入场1,024条、持有1,243条、退出203条。

研究生命周期不受资金和总持仓数限制，只提供训练状态；执行账户仍最多5只。候选源于每个历史月份当时全市场排名，不用后来选中的股票倒推早年股票池。

输入只保留20日波动率、距120日高点的回撤。目标为下一交易日收盘开始、未来60交易日内相对起点的最大浮亏MAE，并非途中峰值计算的账户最大回撤。复权价格用于训练代理；交易评价另走真实价格、整手、现金与公司行动账本。

每年初拟合标准化Ridge，alpha固定10，训练标签结束时间严格早于年初。2022—2026分别使用1,335/1,531/1,732/1,891/2,055条成熟完整训练状态，原执行账户不进入训练。模型更新不增加交易日期。

## 风险如何转成仓位

每年参考风险取成熟训练标签的最大浮亏绝对值中位数，2022—2026约为7.37%、7.42%、7.19%、6.99%、6.92%。

```text
预测风险 = max(-预测MAE, 0)
单股上限折扣 = min(1, 参考风险 / max(预测风险, 参考风险))
单股市值上限 = 当时账户资产 × 原25%上限 × 折扣
```

例如参考风险6.92%、预测风险13.84%，折扣为0.5，原25%单股上限缩为12.5%。该参考值不是账户损失保证。未知特征回退当年统一折扣，不当作零风险；此次实际决策中12条使用了该回退。

四项政策在回放前固定：

- `baseline`：原Hybrid及25%单股上限。
- `uniform`：新旧仓统一折扣，折扣为年度训练样本预测折扣的均值，约0.79—0.80。
- `entry_risk`：仅新买入使用个股风险折扣，旧仓仍执行原规则。
- `full_risk`：新旧仓都使用个股风险折扣，旧仓只在原复核日减仓。

保留原候选排序、Top20池、100股买入单位、最多5只、DAY预定限价、不能预支卖款和原分红假设。没有另外排除高价股；买不起一手仍由真实预算约束决定。风险降低后不主动补足旧仓，沿用原无补仓逻辑，这一点会影响结果。

## 原月初、5万元账户

收益为账户资金口径，已计买卖费用和模型分红税；最大回撤列用损失幅度正数表示。

{table(['启动方式', '政策', '累计收益', '年化', '最大回撤', 'Sharpe', '成交笔数', '成交日数'], main_rows)}

近期新旧仓风险折扣的优势主要来自防守：平均投入比例97.38%→90.22%，期末现金1.08%→19.57%；持有期间中位数仍为138个交易日。成交由41笔增至63笔，但成交日期只从32天增至36天，仍按原低频复核及一次补位执行。

2022连续账户累计收益有所下降；年化21.50%→20.20%，回撤15.97%→11.21%，Sharpe1.112→1.251。这是明确的收益与防守取舍。

### 近期逐年收益

{table(['年份', '原Hybrid', '新旧仓风险折扣'], annual_rows)}

2026不是完整年度。2024、2025单独看收益都略有损失，近期累计优势主要在2026形成；不能仅凭总收益把每个时期都描述为改善。

## 日期偏移稳健性

月内自然日1/8/15/22之后首个交易日，以及固定20交易日间隔的0/5/10/15偏移，全部报告，不挑最好日期。

{table(['窗口', '政策', '年化中位数', '回撤中位数', '年化最小—最大', '收益优于基线', '回撤优于基线'], robust)}

新旧仓风险折扣的近期年化跨度从9.17个百分点收窄至5.53个百分点；连续账户从8.74收窄至6.53个百分点。日期敏感性有所减轻，并没有消失。

连续账户虽然年化中位数略有提高，16组中的该窗口仍有4/8累计收益下降，其中最差一组年化下降约3.80个百分点。中位数不能代替逐一配对结果。

统一降仓不是按事后实现仓位精确匹配的对照。新旧仓风险折扣的实际现金略多，且整数配置改变了后续持仓路径；优于统一降仓不能直接被命名为独立选股alpha。

## 六月防守的具体过程

2024启动的月初账户，6月收益从−12.93%改善为−9.22%；统一降仓为−12.04%，仅新买入风险折扣为−13.35%。风险必须作用到旧仓，才能影响已累积的暴露。

减仓并非直到六月下跌后才发生：新旧仓折扣账户在2月、3月已将中石油1,800股减至1,300股、再减至900股；桂冠电力3月从2,000股减至1,100股，4月再减至900股；龙源电力4月从1,100股减至700股。

五月最后交易日，两账户的部分持仓为：

| 股票 | 原账户股数/权重 | 风险折扣账户股数/权重 |
| --- | --- | --- |
| 中石油601857 | 1,500 / 19.48% | 900 / 11.68% |
| 桂冠电力600236 | 1,300 / 18.07% | 900 / 12.51% |
| 龙源电力001289 | 900 / 20.15% | 700 / 15.67% |
| 中国银行601988 | 2,600 / 18.20% | 3,200 / 22.40% |

现金比例已从基线的0.69%提高到14.33%。六月按逐股现金流和市值变化对账，桂冠电力少亏约1,355.80元、中石油少亏1,194元、龙源电力少亏612元，中国银行反而多亏84元。风险配置改变了实际持仓权重，而不只是修改一个最终统计指标。

六月账户损益已经发生，可以检验上述减亏；6月退出事件的完整60日预测标签仍未全部成熟，不能把执行改善当作该批预测已经准确。回放仍为项目已观察历史。

## 小资金与成本压力

只在原月初比较基线和新旧仓风险折扣，两起点×三个压力条件共12账户，与64主账户合计76。

{table(['窗口', '条件', '累计收益：基线→风险折扣', '最大回撤：基线→风险折扣'], stress)}

2万元连续账户是重要反例：平均现金从4.36%提高到18.02%，却只获得很小的最大回撤改善。逐股全期损益差额中，新华文轩少赚约2,294.50元、桃李面包少赚1,722.68元，部分防守收益被长期持仓利润损失抵消。这些是配置路径的净影响，不等于证明某笔减仓单独造成全部差额。

继承的25%单股上限是原研究参数，不是交易所规定。本轮保留它用于隔离风险折扣的影响；硬上限与100股最小单位、只减不补规则叠加，是否适合2万元账户仍待解决，不能直接固化为个人实盘规则。

费用翻倍包括买费率、卖费率及最低佣金一并翻倍。费用会改变能买的整数股和后续路径，因此结果不等于在同一条净值曲线上简单扣掉额外手续费。

## 现金和整手的后续问题

所有76账户在首次买入之后均没有真正全部空仓。当前机制解决的是暴露大小，不是市场风险开关。

新旧仓风险折扣在全部主/压力账户共出现7次单股上限要求完全退出的请求，基线1次；这是请求数，不等于全部已经成交。100股最小单位会放大某些减仓幅度，不能假设仓位可连续微调。

对满5只、没有当日记录的待卖、存在至少一手预算空间且现金足以覆盖限价和费用的复核日作诊断：近期月初新旧仓折扣账户有29次、连续月初账户55次。其规则不补足旧仓，故这些现金没有自动投入原持仓。该统计只描述价格/预算空间，没有假设订单一定成交，也没有证明补仓必定有利。

因此后续优先检验风险下降后的恢复配置与整手减仓缓冲，仍需保留本轮防守能力和低频约束；不因为看到了踏空案例便临时改阈值。

## 验证、失败记录与限制

- 6项新增风险行为测试、25项原因果撮合对抗测试通过；16个主基线的每日资产、成交、持仓、事件四类账本逐格复现。
- 两次生命周期和真实特征截断检查；5次年度未来标签扰动不改既有预测，独立矩阵解复算误差小于1e-10。
- 2,470条主状态的MAE、成熟边界和同股间隔独立复核；三种风险政策作截至2025-06-30的真实账户前缀回放，资产/成交/持仓/冻结订单完全一致。
- 76账户、{checks['orders']:,}个计划订单、{checks['fills']:,}笔成交完成现金、整手、次日执行和逐股逐日损益对账；风险比例及未知回退逐项核验。
- 首次目录`outputs/risk_budget/20260913T160033_640474Z`在汇总阶段因重复`capital`参数中止，未生成完成账户回执；保留原源码、部分CSV和失败说明。修复仅为字典汇总，随后在当前新目录重建风险、完整重跑，未删除失败记录。
- 原封存数据及AGENTS.md保持哈希不变。此次验证没有消除原始历史版本、停牌/退市缺口、公司行动覆盖、日线限价穿越不保证排队成交等既有限制。
- 训练风险代理不包含真实成交成本；只有后续执行回放才计这些约束。2018等更早年份没有在本轮做完整现金执行检验；不得用训练历史长度冒充策略回测长度。
- 没有新鲜留出数据，不把多次研究后的历史顺序评价称为最终样本外认证；未实盘准入。

[Moreira与Muir的原研究](https://www.nber.org/papers/w22208)提供了高波动时缩小风险暴露的研究依据。本轮是A股五股整手账户上的独立机制实验，没有逐式复现其因子组合，也不引用其收益作为本策略证据。

## 复现与产物

目录：`{run.relative_to(ROOT)}`。`risk/`保存生命周期、逐年训练样本、参考值与预测；`execution/`保存全部76账户；`analysis/`保存逐股逐月归因、配对差额、整手减仓和现金空间；`diagnostics/`保存六月过程。

[近期月初净值、回撤及现金图](../{run.relative_to(ROOT)}/figures/monthly_account_risk_budget.png)。

```bash
cd alpha_lab
python scripts/run_risk_budget_tests.py
python scripts/research_risk_budget.py --stage prepare
# RUN为新目录；不可覆盖封存目录。
python scripts/research_risk_budget.py --stage build --run "$RUN"
# 两window(continued/cold)与三family(calendar/rolling/stress)分别执行。
python scripts/research_risk_budget.py --stage execute --run "$RUN" --window continued --family calendar
python scripts/analyze_risk_budget.py "$RUN"
python scripts/verify_risk_budget_prefixes.py "$RUN"
python scripts/diagnose_risk_budget.py "$RUN"
python scripts/plot_risk_budget.py "$RUN"
python scripts/report_risk_budget.py "$RUN"
python scripts/seal_risk_budget.py "$RUN"
```
'''
(ROOT / 'docs/RISK_BUDGET_RESEARCH_20260914.md').write_text(text)
(run / 'REPORT.md').write_text(text)
print('Report generated')
