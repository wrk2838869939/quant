> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`alpha_lab/docs/MONTHLY_DAY15_START_20260914.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。

# 月度复核平移至15日

按用户希望缩短首次等待的要求，8万元空仓参考账户选择 `configs/monthly_risk_manual_day15.yaml`。仅将 `schedule.parameter` 从1改为15；持仓上限、风险模型、阈值、仓位预算、整手规则和费用参数相同。此后持续在每月15日或该日之后首个交易日复核，不在首次建仓后切回月初。

按已核验的交易日历：首次复核为2026-09-15收盘，21:00后可准备计划；对应2026-09-16 09:15—09:25人工执行。后续复核为10月15日、11月16日、12月15日。能否出现买单由当天完整数据、筛选、实际账户和资金约束决定。

选择15日是因为它是此前验证过的1/8/15/22偏移中最近的下一复核日。此前该偏移的124笔历史策略订单、其中12笔恢复计划已通过执行入口对照；这不是新的收益证明。日期平移会重新计算排名确认和风险状态，可能改变候选与后续持仓路径。

不能把9月14日数据改日期当作15日信号。截至14日，15日日历的最近复核仍为8月17日，没有因此获得立即买入授权。9月15日须获取新的收盘数据、指数与元数据，再重建输入。

构建和准备计划时均须显式使用同一新配置：

```bash
python scripts/build_monthly_manual_observation.py \
  --config configs/monthly_risk_manual_day15.yaml \
  --previous-observation PREVIOUS_OBSERVATION \
  --stage NEW_STAGE --index-stage NEW_INDEX \
  --snapshot snapshot.json --output NEW_OBSERVATION
python scripts/manual_execution.py prepare --db ACCOUNT.sqlite \
  --config configs/monthly_risk_manual_day15.yaml \
  --bundle NEW_OBSERVATION/bundle.json --calendar NEW_OBSERVATION/calendar.json \
  --output plan.json
```

跨日追加流程沿用[人工接入说明](MONTHLY_RISK_MANUAL_INTEGRATION_20260914.md)。变更复核日时，旧观察包仅包含旧复核日的ST截面，构建器会重新引用封存历史状态，避免把新复核日的缺行当作非ST；历史覆盖不足则停止。公司行动候选范围改变时须补查。真实账户已有风险记忆时不允许直接更换配置，本次参考账户尚未建立真实成交账本。

账户选择已记录于 `data/manual_accounts/retail_80k_20260914/strategy_selection.json`。没有启用定时任务、发送邮件或提交订单；实际开始仍需当日收盘核对及接收邮箱。验证记录位于 `outputs/monthly_manual_day15/verification.json`。

本轮实际重建9月14日输入后，15日日历的最近信号日为8月17日，候选与封存同日结果一致，当前空仓预览为0单；17项输入/元数据测试通过。新增候选601088.SH含2017年同日多条实施分红记录，日程全部早于当前日的两条记录仅作历史隔离，不自动合并金额；当前或未来日程未完成的歧义继续阻断，默认历史研究归一化仍采用严格口径。

## 2026-09-15补充：执行窗口改为9:30开盘后连续竞价

按用户当晚决定，参考账户执行窗口从9:15—9:25集合竞价（及后来选择的14:57收盘竞价）改为 **T+1日09:30—09:35连续竞价限价委托**，新配置 `configs/monthly_risk_open_day15.yaml`，账户选择记录见 `data/manual_accounts/retail_80k_20260914/strategy_selection.json`（历史选择文件保留）。账户当时无成交、无风险记忆，符合切换规则。复核日历不变：首次复核仍为2026-09-15收盘，首次执行窗口2026-09-16早间。详见 [OPENING_CONTINUOUS_EXECUTION_20260915](OPENING_CONTINUOUS_EXECUTION_20260915.md)。
