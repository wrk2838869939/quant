# 执行元数据操作指南

所有命令在 `alpha_lab/` 运行。使用新目录保存每次数据版本，避免覆盖。

## 获取和重建

```bash
python scripts/stage_execution_metadata.py \
  --output outputs/execution_metadata/new_stage \
  --bundle outputs/shadow_execution/20260914T054517_770961Z/example_rebuilt_bundle.json \
  --price-date 2026-09-11

python scripts/build_execution_observation.py \
  --metadata outputs/execution_metadata/new_stage \
  --market-stage outputs/live_ingestion/20260914_market_stage \
  --output outputs/execution_metadata/new_observation
```

首条默认只查询给定信号的候选公司行动。以后接入真实账户时追加 `--snapshot 路径`，覆盖候选与实际持仓的并集；不要使用合成演练快照代替真实账户。信号变更导致候选超出查询范围时必须补查。

构建器当前使用固定历史特征与模型，定位为本轮可复现观察工具。它始终将输出标为回补，不会声称真实前向数据已齐。近期缺主表、行情日期/行数冲突、字段不齐等会停止。

`metadata_report.json` 列出实际可用状态、空接口、合并重复数、查询范围和首次获取时刻。深市日历为空时，按数据商日历说明显式共用已检查沪市日历。该规则不适用于ST、停牌或涨跌停价。

## 有时间检查的计划入口

```bash
python scripts/guarded_shadow_plan.py \
  --observation outputs/execution_metadata/new_observation \
  --snapshot local_account/eod_snapshot.json \
  --db outputs/my_shadow/account.sqlite
```

当前观察包应返回 `blocked`。晚于执行日9:25、数据缺失、真实账户范围未覆盖或回补标识存在，都不生成订单。请勿通过改日期、手填完成标记或换用旧研究入口绕过检查。

本轮工具尚不能自动收齐所有类型公司行动，也没有把券商订单实际提交出去。后续必须先实现补充事件核验，再接入有真实获取时刻的前向数据包。券商成交导入继续见[CSV对账指南](LIVE_INGESTION_GUIDE_20260914.md)。

## 复验

```bash
PYTHONPATH=src:tests python -m unittest test_execution_metadata test_broker_csv test_execution_market_feed test_shadow_execution test_shadow_inputs -v
python scripts/validate_execution_metadata.py \
  --stage outputs/execution_metadata/20260914_stage \
  --observation outputs/execution_metadata/20260914_observation_v4
```

第二条核验原件、截面观察包、除息参考价、过窗CLI拒绝及原封存，写入新目录；不会重新下载数据或创建实际账户。完整事实与剩余问题见[验收报告](EXECUTION_METADATA_VALIDATION_20260914.md)。
