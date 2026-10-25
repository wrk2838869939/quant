# 近期扩窗执行的独立工程审查

本次用户已授权 2024 年至当前已准备可用日的冻结规则账户检验。只改新入口的研究窗口，原开发期隔离和策略代码保持冻结。已读代码、配置及面板元数据，未读取或打印 ts_client 源码、凭证，未下载数据或调参。

## 可复用接口与硬边界

| 接口 | 扩窗使用方式 |
|---|---|
| `audit_monthly_portfolio.load_inputs(prepared,cfg)` | 读取 `cfg.valid`；第 73 行硬拒绝 2024+，不能直接调用。新 driver 显式读取同样原价字段、上市退市/ST 历史、分红与官方限价，保留旧加载器。 |
| `prepare_monthly_audit.fetch_execution_data(codes,dates,cfg)` | 函数本身没有 2024 硬上限，可复用；主程序另有开发期上限，不能直接复用 main。配置 `data_workers=2`，错误只记录类型，空响应与 None 区分，临时文件原子替换。 |
| `portfolio.normalize_actions` | 可继续复用，保持组成事件、源修正、差异化除权参考的严格检查；不能盲目去重求和或停用异常。 |
| `portfolio.run_backtest` | 无固定年份上限。接口位置参数：signals、market、basics、st、limits、actions、start、end、portfolio_config。已核对新 driver 顺序正确。 |
| `audit_monthly_portfolio.reconcile`、既有过程/持仓段归因 | 可复用流水守恒和归因核对，但应增加独立状态机及源报价检查。 |
| `verify_divide_selection_execution.verify_case` | 第 47 行硬拒绝 2024+。新 `verify_recent_low_turnover.py` 复制其状态机，仅把该日期断言改为明确的 case.start 至 manifest.end，运行时逐字验证唯一改动及复制来源哈希。旧模块不改。 |

旧 divide/symmetric/cash 的 prepare 和矩阵生成入口也有 2024 上限，不应通过篡改日期或 monkeypatch require 来复用。新 driver 将范围与授权直接写入 manifest。

## 执行前检查清单

1. **固定终点。** 已读面板 Parquet 元数据：12,275,558 行，12 row groups，最大日期是 **2026-09-08**。文件存在或“当前日期”不能证明更晚交易日已完整。当前登记终点与该数据资产一致；不是已核实外部数据源只能提供到这一天。
2. **按新窗口补缓存。** 旧缓存目录只有 `20220101_20231231`。fetch 函数使用日期命名目录，但对存在文件直接跳过，没有内部覆盖校验；旧 dividend 文件还已按 ex_date 裁掉窗口外事件。新 run 使用 2022-01-01 至 2026-09-08 的完整独立目录，不能把旧缓存指针当新数据。需逐代码核对官方限价日期、重复键、有限正值，零行响应不能单凭文件存在认证覆盖；失败 CSV 不代表之后重试已成功，应以当前完整覆盖核验为准。
3. **固定候选并集。** 数据代码并集来自冻结月度前 20 候选，不按后来收益或实际成交选股票，覆盖所有资本情景与旧持仓来源。候选行情日期并集仍需与预期交易日历对照，不能在全候选当日缺行时默默跳过交易日。新 driver 已对全部已有行情行检查官方限价，而非仅成交行。
4. **保留全部必要公司行动。** 新缓存从 2022 年起，能够保留 600600 的 2023 精确修正行：源修正代码只要看见该股票就要求该完整键存在，单独裁成 2024+ 会因缺旧行报错。完整缓存避免此问题，不必改旧修正规则。多项分红、回购股不参与的参考价、送股上市日、登记/除息/付款日和配股另查；旧配置只覆盖已登记的历史事实，新窗口异常要凭发行人公告补证，不能关闭 strict_actions。
5. **冻结在首个案例之前。** 需冻结源代码、配置、预测、signals、forecasts、hold、market、limits、公司行动及股票/ST 元数据。`data_manifest.hashes` 纳入 `execution_config.yaml`；`data_manifest.prediction_hash` 绑定 run 根预测。新 verifier 强制比对这些哈希，并要求每 case 参数仅有登记的本金、五仓上限和费用倍数变化。买费、卖费、最低费三项同时乘压力倍数。
6. **冷启动与续跑分别解释。** 2024 冷启动不能继承 2023 持仓或以前信号直接开仓。续跑从 2022 原始资本记账，2024 以后收益分母是 2023 最后净值，不能再除原始本金。相同冻结版本须将续跑 2022–2023 的 8 个执行 CSV 前缀与 P3 `base_50000_none` 对账。新增事件字段仅允许旧前缀全空，经济值不允许容差。
7. **完成独立账本及源报价核验。** 新入口逐股逐日重建现金、股数、应收、总资产和持仓段 PNL，核对预测分数与点预测分离、禁止旧仓补仓、原持仓下界；附加成交原始开盘价、官方限价、主板整手、买入 T+1、冻结量能上限和有效原始收盘估值检查。倒序情景逐个 CSV 严格核对。实际公司行动金额的外部公告真实性仍需执行准备的公司行动证据支持，流水恒等式不能独自认证上游输入。

## 独立入口及测试

```bash
python scripts/verify_recent_low_turnover.py --run outputs/recent_low_turnover/20260913T085313_818456Z
python -m unittest discover -s tests -p test_verify_recent_low_turnover.py -v
```

6 项新测试通过：复制函数唯一边界改动、明确近期窗口通过/旧窗口拒绝、现金偏差仍拒绝、涨停开盘买入拒绝、旧前缀全空新增事件列通过、旧前缀经济值微小漂移拒绝。本环境当前 python 没有 pytest，测试使用标准库 unittest，也兼容 pytest 收集。未运行全套测试。

运行成功将输出 `execution/independent_recent_verification.json` 与 `execution/independent_fill_checks.csv`。执行尚未完成时本文只是准备审查，不宣告六个账户已经通过。
