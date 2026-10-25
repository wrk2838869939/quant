# 多平台窗口的事件覆盖诊断

保持现有上涨15%、回落10%、10日内快跌7%、历史上行、参考区和5日顺序确认条件。只并行使用10／15／20／30／40日平台窗口，每个窗口独立维护已有状态机。未计算后续收益。

## 去重口径

- 每个窗口先取本事件首次信号，再按时间合并。
- 同板块同连续段，若新信号的峰值不晚于前一个已保留确认日，归为重复回落；只有前次确认后形成新峰值，才允许再计新事件。
- 同日冲突按配置顺序15、10、20、30、40日保留代表，不按未来走势或更晚信号挑选。观察池、触发、确认各层独立执行相同去重。
- 去重偏保守，可能合并部分相近回落；每板块各窗口仍各自只维护一个事件，不等于已经支持所有嵌套平台。

## 规模

- baseline_events: 230
- baseline_dates: 115
- merged_events: 330
- merged_dates: 162
- merged_sectors: 111
- raw_confirmation_signals: 956
- merged_duplicates: 626
- groups_without_baseline: 100
- new_groups_old_event_active: 82
- groups_with_baseline: 230

| 累计加入窗口 | 观察池 | 价格区触发 | 确认 | 确认日期 |
|---|---:|---:|---:|---:|
| 15日 | 532 | 340 | 230 | 115 |
| 10日 | 594 | 406 | 276 | 138 |
| 20日 | 613 | 433 | 307 | 151 |
| 30日 | 626 | 452 | 320 | 157 |
| 40日 | 633 | 478 | 330 | 162 |

## 如何解释

- 多窗口新增数量说明固定窗口存在覆盖限制；不代表这些案例都比旧版更符合主观形态，也不代表收益提高。
- groups_without_baseline 是最终归组后没有15日窗口信号的组数，只能用于事后覆盖诊断；它不等同于总数的简单差值。信号代表可能提前，也可能被另一窗口归并。
- new_groups_old_event_active 表示新增组确认当日，原窗口存在另一个活动事件。此状态是查找锚点占用或条件不匹配的线索，不单独证明早期事件阻挡了信号。
- 平台范围20%、首末变化10%的启发式未改，仍可能把V形或倾斜段当平台；图册标为平台候选，需继续复核。
- 原230事件收益报告仍只适用于原样本，不能直接套到多窗口集合。

## 文件

- confirmed_events.csv：按时间可产生的去重确认信号；trigger_events.csv和watch_events.csv为另两层。
- *_all_signals_with_dedup.parquet：每条来源信号及去重归属，保留审计痕迹。
- confirmed_coverage_diagnostics.parquet、group_coverage_RETROSPECTIVE.csv：含事后分组诊断，不能直接作为信号特征。
- additional_cases.pdf、review.html：按日期等距抽取新增覆盖案例，只展示确认时已知行情。
- single_window_coverage.csv、cumulative_union_coverage.csv、annual_confirmed.csv：单窗口、合并及年度覆盖。
- 10项测试通过；10日窗口真实截断重算一致；三层合并真实前缀一致；源文件哈希核验通过。
