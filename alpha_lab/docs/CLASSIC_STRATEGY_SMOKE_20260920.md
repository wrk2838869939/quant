# 经典策略工程验收与固定输入入口

结论：**冻结行情 → 经典信号 → 10万元账户模拟 → 过程对账 → 结果导出已跑通；本次六个经典算法版本均未满足最大回撤30%的要求。**
不进行参数寻优，不把历史反复使用的窗口视为新的样本外证据。

## 实测结果

2019-01-01～2023-12-31，1,214个交易日；初始资金10万元，最多5只股票，计入手续费、最低费用和滑点。
从2018年成交额中位数前100只主板股票出发；7只因测试期价格/公司行动资料不一致隔离，最终所有策略共用93只。
**全期质量隔离存在事后样本选择偏差，结果限于工程验收样本，不代表全市场可交易策略有效性。**

| 算法 | 日频累计收益 | 日频最大回撤 | 月频累计收益 | 月频最大回撤 |
|---|---:|---:|---:|---:|
| 20日动量 | -57.17% | -73.57% | +2.66% | -51.86% |
| MACD 12/26/9 | -13.51% | -71.76% | +8.64% | -60.70% |
| MA20/MA60 | -47.64% | -74.41% | +0.38% | -57.80% |
| 固定种子随机选股 | -94.29% | -95.45% | -9.50% | -45.88% |

现金基准收益及回撤均为0。经典指标采用正趋势过滤后排名，MACD为趋势状态策略，不限于金叉当天；月末收盘决策、次日开盘模拟成交。
旧持仓仍在前5名则继续持有，不强制恢复等权；单股权重随价格变化可能超过初始约20%。

日频动量交易2,676笔，五年费用含滑点约87,143元；月频为491笔、约22,390元。
MACD日频账户曾达到约29.57万元，最终约8.65万元，因此大回撤与终值并不矛盾。
高换手和持仓风险有实际成本；不能把负收益直接归因为数据错误，也不能把月频略正收益理解为策略已可用。

## 验证证据

- 经典信号与既有账本相关27项测试通过；包含日期边界、正负方向、随机前缀稳定、现金不足、T+1、分红及退市等测试。
- 指标与独立pandas公式逐股核对；所有成交的价格、费用、整手与可卖数量核验通过。
- 现金流水重建最大误差小于4×10⁻¹⁰元，股数误差0；没有负现金或超5只持仓。
- 用原始收盘价重算有报价持仓价值，独立重算日收益和最大回撤，确认买入在信号后下一交易日。
- 截断至2021年末重新生成信号及回测，过去净值与长窗口一致；该项不消除共同样本隔离偏差。
- 首次运行和直接复用归档输入的第二次运行，9个组合的汇总与45份信号/净值/成交/事件/持仓表逐值完全一致。
- 涨停买入跳过、跌停/停牌卖出顺延、买不起整手跳过、分红到账、红股锁定等都有事件记录。全部9个组合的执行资料充分标记通过。

## 固定入口

策略入口：`scripts/run_classic_strategy_smoke.py`；策略信号模块：`src/classic_smoke.py`；通用账本：`src/portfolio.py`，本次未修改。
配置：`configs/classic_strategy_smoke.yaml` 和 `configs/selection_account_v1.yaml`。
准备完的执行资料独立保存在 `data/selection_execution_smoke_v1/`；冻结 datahub 行情仍为 `selection_v1_20260920`，未修改。

项目根目录下复用已准备资料：

```bash
python datahub/scripts/frozen_research.py run \
  --release datahub/releases/selection_v1_20260920 \
  alpha_lab/scripts/run_classic_strategy_smoke.py \
  --prepared alpha_lab/data/selection_execution_smoke_v1
```

这一入口不再扫描活动执行缓存，不联网下载。改变日期范围、股票池或已有预计算指标参数时会明确拒绝混用旧资料。
增加新自定义因子仍可从冻结行情派生，但应明确准备新特征产物，不能在回测过程中修原行情或改变执行口径。
当前验收范围是固定93只股票/2019～2023年，不意味着全市场任意股票与日期均已认证。

## 报告与过程数据

- [完整报告](../outputs/classic_strategy_smoke/20260920T092937_046916Z/REPORT.md)
- [净值与回撤图](../outputs/classic_strategy_smoke/20260920T092937_046916Z/nav_drawdown.png)
- [结果汇总](../outputs/classic_strategy_smoke/20260920T092937_046916Z/summary.csv)
- [逐年结果](../outputs/classic_strategy_smoke/20260920T092937_046916Z/yearly.csv)
- [过程检查](../outputs/classic_strategy_smoke/20260920T092937_046916Z/process_checks.csv)
- [独立估值与指标检查](../outputs/classic_strategy_smoke/20260920T092937_046916Z/independent_valuation_checks.csv)
- [精确重跑比较](../outputs/classic_strategy_smoke/20260920T092937_046916Z/replay_comparison.json)

各策略子目录保留 `signals.parquet`、`targets.csv`、`daily.csv`、`trades.csv`、`holdings.csv`、`events.csv`、`daily_diagnostics.csv` 和 `audit.json`。
