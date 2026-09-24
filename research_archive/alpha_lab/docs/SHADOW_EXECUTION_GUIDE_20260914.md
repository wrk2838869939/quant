> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`alpha_lab/docs/SHADOW_EXECUTION_GUIDE_20260914.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。

# 风险仓位版每日计划与账户对账

## 当前用途

这是 `Hybrid + full_risk` 的实盘准备层，先按**手工下单、导入券商流水**设计。股票选择、五股上限和风险折扣保持原研究规则；恢复旧仓、整手缓冲关闭。SQLite负责账户和订单状态，CLI只输出计划、校验和导入记录，没有券商下单或通知能力。

它解决回测末日无法规划、实际账户无法接续和重复运行状态丢失的问题。真实行情首发版本、券商委托适配、长期有效性和银行共同风险仍需验证。工程验收通过不等于策略获得实盘准入。

## 每日流程

1. 整理全部成交明细，再导入撤单、拒单、失效等最终状态；不要用“看起来没成交”推定终态。
2. 导入资金流水和已核对公司行动，随后导入券商收盘账户快照。现金或股数不一致时停止，不以新快照覆盖差异。
3. 用收盘后的完整全A数据、对应年度冻结模型及独立交易日历重建信号，生成下一交易日计划。新候选没有当日报价时明确记录跳过原因；旧持仓缺当日报价则停止，不能猜停牌估值。
4. 次日集合竞价窗口核对实际资金、持仓、可卖股数、涨跌停、停牌和公司行动，及券商是否接受预定限价。通过后仅标记 `ready`，仍由操作者执行。
5. 盘后重复第一步。未完成的风险减仓目标保留，只有真实卖出到账后才允许原规则的一次补位；不预支计划卖款。

不需要每天交易：每日核对账户，买卖仍由原月度复核和既有待执行订单触发。

## 账户及文件接口

从 `alpha_lab/` 运行，账户数据库路径由操作者指定；不要把历史示例当作当前账户。

```bash
python scripts/shadow_execution.py --db outputs/paper/account.sqlite snapshot --input snapshot.json
python scripts/shadow_execution.py --db outputs/paper/account.sqlite plan \
  --bundle bundle.json --calendar calendar.json --output plan.json
python scripts/shadow_execution.py --db outputs/paper/account.sqlite preflight --input preflight.json
python scripts/shadow_execution.py --db outputs/paper/account.sqlite fill --input fill.json
python scripts/shadow_execution.py --db outputs/paper/account.sqlite terminal \
  --order-id ORDER_ID --status cancelled --evidence BROKER_CANCEL_REFERENCE
python scripts/shadow_execution.py --db outputs/paper/account.sqlite adjustment --input adjustment.json
python scripts/shadow_execution.py --db outputs/paper/account.sqlite status
```

快照字段：`account_id, asof, source, cash, receivables, equity, frozen_cash, open_orders, positions`。
`positions` 以证券代码为键，每项包含 `shares, sellable_next_day, close`。净资产必须等于现金、分红应收和持仓估值之和。当前只接收沪深主板、最多五股的策略账户。

`sellable_next_day` 指下一交易日可卖量，不能直接照搬当日晚上的“当前可卖量”，也不能不顾红股等锁定期直接填总股数。次日盘前还会核对券商真实可卖量。初次接管已有账户须在无未结委托时导入；不能从快照猜回以前的策略待卖意图。

成交字段：`execution_id, order_id, date, shares, price, fee, evidence`。同一成交号、相同内容重复导入不重复记账；同号异内容、超量或突破限价会停止。委托买量为100股整数，但真实部分成交可能不是100股整数，账本允许这类回报。

调整字段：`event_id, kind, date, cash_delta, evidence`；公司行动另含 `ts_code, share_delta`。类型包括入金、出金、分红、税费、费用和公司行动。股数变化必须有明确凭据；当公司行动涉及非零待减仓目标时，本版会停止要求核对，不自动猜测目标换算。

盘前文件包含 `plan_id` 和 `market`。后者需要带时区的 `timestamp`、`source`、`session: opening_auction`、`account_id`、完整 `positions` 股数映射、空 `open_orders`、`available_cash`、`corporate_actions_checked`。每只订单证券在 `securities` 下提供 `down_limit, up_limit, suspended, corporate_action, broker_accepts_limit, buy_allowed, account_sellable`。

本版限定09:15—09:25之前的开盘集合竞价检查；不能在09:30后直接照搬昨晚卖出限价。数据和账户检查由导入文件声明，目前不是已经认证的券商自动查询。

## 当日信号构建

```bash
python scripts/shadow_execution.py build-bundle \
  --feature-panel FULL_A_FEATURES.parquet --market RAW_MARKET.parquet \
  --basics BASICS.parquet --st ST_HISTORY.parquet --actions ACTIONS.parquet \
  --calendar calendar.json --model model_2026.json \
  --snapshot snapshot.json --receipt data_receipt.json --output bundle.json
```

- 特征面板：`date, ts_code, hfq_close, total_mv_yuan, turnover_rate`，须包含足够完整的历史，以重建跨月失败计数。全A排名后再限制沪深主板，不能只拿历史281只执行股票计算排名。
- 原始报价：`date, ts_code, close, vol_share`。截止日期从账户快照读取；读取器过滤其后的价格。复权特征与实际成交价格分开使用。
- 交易日历：有来源的 `sessions` 唯一递增日期列表，必须包含下一交易日；不能用下一日行情充当日历。
- 模型：年度、训练标签截止日、标准化均值/尺度、Ridge系数、参考风险、原收益常数。没有对应年度模型会停止；不会临时重训选择更有利结果。
- 数据回执：`asof, complete, source, downloaded_at, expected_rows`；检查收盘时点、当日行数和交易日完整性。回执是运行约束，不是外部数据真实性认证。

独立构建器与冻结研究排名、持有许可及风险折扣进行了对照；实际持仓/候选风险特征未知时更严格地停止，避免把未知自动解释为安全。全市场源数据更新和年度模型发布还没有接入常驻调度。

## 幂等、恢复与限制

同日同输入返回同一个计划和订单编号；输入变化拒绝覆盖。导出文件通过临时文件和原子链接发布，已有不同内容不覆盖。所有账户修改在SQLite事务内完成，使用WAL和完整同步；进程重启后从原数据库读取。

请保留数据库及原始券商凭据。运行中不要只复制主SQLite文件而忽略WAL；备份应使用SQLite备份接口，或关闭所有连接后再复制。

本版接受按交易日整理好的人工回报，不是可直接接入乱序券商推送的适配器。先导入成交再导入终态；未匹配到计划的手工额外交易、晚到更正、复杂公司行动或资金差异会被拦截，需要完整核对。禁止为了让计划继续而编造调整流水。

历史演示中的盘前信息、数据回执和“券商”凭据均为测试夹具，不代表真实成交或已经取得当前行情。当前没有真实账户前向运行成绩，也没有新增策略收益承诺。
