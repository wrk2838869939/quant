# 行情增量与CSV对账操作指南

在 `alpha_lab/` 运行。全部命令仅用于数据与影子账本，不向券商提交委托。

## 隔离下载与核验

每次使用新的目录，`--after` 是已有数据的最后交易日：

```bash
python scripts/stage_execution_market.py --after 2026-09-08 --output outputs/live_ingestion/new_stage
python scripts/validate_execution_market.py --stage outputs/live_ingestion/new_stage
python scripts/check_execution_absences.py --stage outputs/live_ingestion/new_stage
```

示例锚点默认 `data/raw/by_date/daily/20260908.parquet`；后续日期应给验证脚本传入对应 `--anchor`，不能一直沿用旧日期。下载器沿用本地日历且18:00前不拉当日EOD，该时间限制不保证供应商已发布全部数据；日历覆盖和原始返回仍需检查。空缓存、重复键或跨接口不齐均停止验证。

`validated/package.json` 仅证明价格包检查结果。当前元数据未齐，不应拿增量替换旧面板后直接产出交易建议。公司行动、ST等覆盖声明必须有来源和原件哈希，不能为了通过检查填一个日期。

## 适配券商导出

复制 `configs/broker_csv_template.yaml` 为对应客户端配置，明确编码、日期格式、市场/方向/终态值和五类文件列名。不要模糊匹配。模板所列中文列名是交换格式，并非任何券商原生表头。

券商成交文件往往缺最终佣金，需以已结算回报补齐每笔总费用；不得将缺失费用填0。证券代码必须六位；次日可卖数量必须核实。空成交或空持仓文件也需保留表头。五个文件必须同账户、同日，委托绑定含计划订单号；存在其他手动交易时先停止对账并核对完整流水。

先沿用[第一阶段指南](SHADOW_EXECUTION_GUIDE_20260914.md)初始化真实账户快照，冻结计划并完成盘前检查。随后导入收盘凭据：

```bash
python scripts/import_broker_csv.py \
  --db outputs/my_shadow/account.sqlite \
  --profile configs/broker_csv_template.yaml \
  --bindings local_exports/bindings.csv \
  --fills local_exports/fills.csv \
  --orders local_exports/orders.csv \
  --account local_exports/account.csv \
  --positions local_exports/positions.csv \
  --archive outputs/my_shadow/broker_archive
```

`reconciled` 表示逐笔账本与该日快照核对通过；`duplicate` 表示已处理。失败时查看归档中的 `failure.json`，保留原件，修复缺失凭据再导入。校验失败不会提交半批；若进程中断或提交后写结果文件失败，使用相同文件重试并核对账本批次状态，不手工重加现金。归档包含账户隐私，应仅本地保存。

分红、资金出入等先导入对应有凭据事件，再执行CSV批次。应收款变化等未支持事件会停止，不能用新快照覆盖旧状态绕开核验。

## 复验

```bash
PYTHONPATH=src:tests python -m unittest test_broker_csv test_execution_market_feed test_shadow_execution test_shadow_inputs -v
python scripts/validate_live_ingestion.py
```

第二条命令默认复验本轮已下载数据，不重新请求远端；自动建立新的证据目录。`synthetic_csv_demo/` 内均为测试账户和合成成交，禁止作为真实账户初始状态使用。完整结果见[第二阶段验收报告](LIVE_INGESTION_VALIDATION_20260914.md)。
