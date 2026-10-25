# alpha_lab 工作台

从这里找当前策略、最新日报、研究文档和历史产物。索引只证明文件位置，不认证研究结论。

[可搜索网页](INDEX.html) · [文档目录](docs/INDEX.md) · [产物目录](outputs/INDEX.md) · [存储清单](catalog/STORAGE.md)

## 当前入口

- [当前策略规则](configs/strategy_2026_opportunity_scan.yaml)
- [年度板块过滤暂停说明](docs/current_strategy_20261013/ANNUAL_SECTOR_GATE_PAUSED_20261021.md)
- [日更与扫描操作入口](experiments/sector_live_v1/README.md)
- [存储与归档规则](docs/WORKSPACE_ORGANIZATION.md)
- [板块到个股技术方案（结合当前变更阅读）](docs/current_strategy_20261013/strategy_2026_plan/TECHNICAL_DESIGN.md)

## 明确指针所指的最新记录

| 入口 | 数据日期 | 记录性质 |
|---|---|---|
| [最新数据核验与日报](outputs/datahub_model_check/20261022_eod/REPORT.html) | 2026-10-22 | complete |
| [最新正式模型结果](outputs/strategy_2026_opportunity_scan/20261022_gate_paused_20261022T160739_495956Z/REPORT.html) | 2026-10-22 | current_policy_scan |
| [最新板块扫描](outputs/sector_live_scan/20261022T160204_763709Z/README.md) | 2026-10-22 | complete |
| [最新行情更新](outputs/sector_live_update/20261022T155629_062229Z/README.md) | 2026-10-22 | complete |
| [历史规则复核（不是正式新信号）](outputs/strategy_2026_opportunity_scan/20261020_gate_paused_20261021T070103_535863Z/REPORT.html) | 2026-10-20 | policy_change_review |

历史规则复核不会替代正式候选；不按目录名或修改时间猜测最新结果。

## 更新与检索

```bash
python alpha_lab/scripts/build_workspace_index.py build
python alpha_lab/scripts/build_workspace_index.py search "养殖"
python alpha_lab/scripts/build_workspace_index.py search "sector_live" --kind run
python alpha_lab/scripts/build_workspace_index.py build --inventory
```

元数据更新：2026-10-25T12:44:08+00:00；容量盘点：2026-10-25T12:44:08+00:00。普通build复用容量清单，新增目录容量显示“未盘点”。
