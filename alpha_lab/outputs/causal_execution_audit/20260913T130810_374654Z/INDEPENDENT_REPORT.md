# 因果执行真实账户独立复验

审查日期：2026-09-13。审查人：`audit_future_signals`，独立于新引擎作者。对象为 `outputs/causal_execution/20260913T130810_374654Z` 的20组已完成账户。没有修改封存引擎或任何结果。

## 裁决

**20/20组通过本项实际订单、资金时点、DAY成交约束与独立股数核验；没有复现旧引擎的下一日开盘配单或全日量决定开盘成交问题。** 本结论认证的是已登记撮合假设下的时间一致性，不保证实际队列成交，也不证明策略样本外有效。

本次独立核验1,001个冻结订单、994笔成交、15,924个账户交易日；其中965笔按开盘价、29笔按严格盘中穿越限价的假设成交。7个买单因整日未穿限价取消，0个卖单延迟。共有385次原月度卖出实际到账后的补位规划，均核对当日确有同一原信号的卖出；同一原信号最多一次。

## 关键检查及对应旧问题

1. **旧问题：看T+1开盘价格/开盘权益决定股票与数量。** 现在每个实际成交都对应唯一T收盘 `order_planned`，成交代码、方向、股数、限价、原信号日完全对应。全部 `decision_date < execution_date` 且执行日为下个交易日，日志记录日=决策日，买入参考价=实际决策日收盘价。25项独立合成对抗进一步证明改变执行日开盘不改旧订单。
2. **旧问题：执行日全日量泄露到当日开盘可成交判断。** 新引擎撮合不读取该量；全部1,001个订单的数量不超过决策日已知成交量上限。独立合成路径将执行日量改为0/NaN/1/极大值，旧订单和当日成交均不变。
3. **旧问题：订单选择与同次未来卖款混在一起。** 各决策日买单最坏限价总预算（含最低费用）≤该日实际收盘现金，`cash_at_decision` 与账本差额全部为0；实际买入总成本不超过逐单已留预算。卖出资金到当天收盘才可能触发下一日的唯一补位订单，不存在借未来卖款支撑先前订单。另从已独立核验的登记日股份、原行动除权/支付日期和固定税费扣减重建545次真实现金分红与全部现金流水；现金最大误差2.18×10^-11元，应收款误差0。应收款未被直接用作预算。
4. **旧问题：未来公告生效状态可能倒填前晚订单/当天开盘。** 全部实际订单核对了ST及公司行动可知日期；缺公告或执行日才公告且无盘前时间证据的路径数量均为0。新增合成测试要求这种歧义停止认证，而不是按整日公告撤单后继续给收益。
5. **股份与头寸约束。** 不调用引擎的Position或旧核验器，重新维护整数股份、买入T+1锁、红股解锁、登记日权益及摘牌状态；逐日持有数与可卖数和封存账本一致，不超过5只。买单规划时退出待成交仓仍占名额。订单生命周期每单仅一个终态，成交数量没有在执行日重新计算。

## 每组真实结果核验

| 账户 | 锁定订单 | 成交 | 盘中穿价成交 | 订单现金与收盘现金最大差 |
|---|---:|---:|---:|---:|
| cold_baseline | 43 | 43 | 0 | 0.0 |
| continued_baseline | 62 | 62 | 0 | 0.0 |
| small_baseline | 45 | 45 | 0 | 0.0 |
| large_baseline | 47 | 47 | 0 | 0.0 |
| cost2_baseline | 44 | 44 | 0 | 0.0 |
| reverse_baseline | 43 | 43 | 0 | 0.0 |
| cold_baseline_buffer00 | 47 | 44 | 5 | 0.0 |
| continued_baseline_buffer00 | 63 | 63 | 9 | 0.0 |
| cold_baseline_buffer05 | 43 | 43 | 0 | 0.0 |
| continued_baseline_buffer05 | 63 | 63 | 0 | 0.0 |
| cold_hybrid | 41 | 41 | 0 | 0.0 |
| continued_hybrid | 67 | 67 | 0 | 0.0 |
| small_hybrid | 42 | 42 | 0 | 0.0 |
| large_hybrid | 48 | 48 | 0 | 0.0 |
| cost2_hybrid | 42 | 42 | 0 | 0.0 |
| reverse_hybrid | 41 | 41 | 0 | 0.0 |
| cold_hybrid_buffer00 | 47 | 45 | 6 | 0.0 |
| continued_hybrid_buffer00 | 70 | 68 | 9 | 0.0 |
| cold_hybrid_buffer05 | 38 | 38 | 0 | 0.0 |
| continued_hybrid_buffer05 | 65 | 65 | 0 | 0.0 |

所有实际成交均符合其冻结DAY限价与当日官方价格边界；盘中成交要求最低价严格低于买限价或最高价严格高于卖限价，未把仅触价当作成交。

## 真实触发与仅合成覆盖的分支

- 实际触发：965笔开盘成交、29笔盘中严格穿价假设、7笔未穿价取消、545次现金分红到账。
- 实际未触发：红股、摘牌、ST退出、已知次日ST/摘牌拒单、除权日撤单、卖出延迟、因量能或锁定而延后规划。不能声称这些已通过真实事件回放；目前由独立25项合成测试覆盖相关语义。
- 真实样本没有同日公告时间歧义；拒绝此类路径的代码只完成了合成反事实验证。

## 仍然不能认证的部分

日线OHLC不能证明委托排队或足额成交；29笔盘中限价成交是有明示条件的回测假设。前日量能上限也不是当天该价格档的流动性证据。供应商回溯修订、公告的真实盘前发布时间、实际滑点及真实税费仍需专门证据。所有年份已反复研究，因果执行修复不创造新的未见样本。

另已独立纠正“缺失旧代码导致漏32行训练”疑点：32/32已通过现canonical身份纳入原训练，隔离身份修复必须配合去重投影，不能把旧别名当新证券增加样本。2021常数与原51,883行训练保持一致，证据在 `../signals/identity_training_verification.json`。

## 复现与证据

```bash
python alpha_lab/outputs/causal_execution_audit/verify_real_orders.py --run alpha_lab/outputs/causal_execution/20260913T130810_374654Z
python alpha_lab/outputs/causal_execution_audit/summarize_real_orders.py
python alpha_lab/outputs/causal_execution_audit/verify_cash_entitlements.py
python -m unittest discover -s alpha_lab/tests -p test_causal_portfolio_adversarial.py -v
```

- `independent_real_order_verification.json`：20账户逐项结论、原文件hash、验证器hash。
- `independent_cash_entitlements.json`：20组真实现金和545次分红权益独立重建。
- `actual_branch_coverage.csv/.json`、`actual_cancellation_reasons.csv`：实际触发分支。
- `../signals/adversarial_final25.log`、`../signals/TEST_REPORT.md`：独立25项合成测试及修复过程中发现的问题。
- 冻结引擎SHA256：`5ebcc40650412afda56188f95bae30e61f46bb8af1202e1ce80b7cf79c8fe23c`。
