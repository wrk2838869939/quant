# 暂停年度板块模型过滤：2026-10-21

依据用户“后面的模型中先暂停使用年度板块判断”的决定，自2026-10-21起，后续机会扫描暂停年度板块模型的否决作用，保留评分作参考。没有换用盈亏天数标签实验模型。

当前流程：**板块首次形态确认及多窗口去重 → 个股资格/ST检查 → 个股价格减速形态 → 原2026年度个股模型正分数前5名 → 执行条件核对。**

## 已落地配置与入口

- 当前配置：[strategy_2026_opportunity_scan.yaml](../../configs/strategy_2026_opportunity_scan.yaml)，`signal.annual_sector_gate_enabled: false`，`annual_sector_diagnostics: true`。暂停期间诊断不可用也不会因缺板块分数而否决个股入口。
- 当前实现：[scan_opportunities.py](../../experiments/sector_live_v1/scan_opportunities.py)。原[scan_stocks.py](../../experiments/sector_live_v1/scan_stocks.py)默认转入此入口；历史旧规则只有显式`--legacy-rules`才执行。
- 日期、板块数据集、行情快照来自本次形态扫描清单；不再复制日期固定的评估脚本。必须在对应冻结行情下运行。
- 旧技术设计、旧年度模型、旧扫描产物保留为历史记录。原设计配置`strategy_2026_sector_stock_plan.yaml`也保留，后续机会扫描使用上面的当前配置。

板块扫描原输出附带的旧随机模型分数也不作为资格条件。当前生效规则以新入口输出中的`stock_entry_eligible`为准，不能再用`model_pass_count`或年度模型参考分数解释是否进入个股筛选。

## 继续保留的约束

只接受当日去重后的首次确认；不把旧事件重新满足形态算成新的首次事件。个股正常名称/上市状态/有效报价、价格形态、正分数和最多5只等条件继续执行。个股模型的年度权重不改变，尚未到新年度不会自动重训。

暂停板块过滤不是取消全部模型，也不是延长历史信号执行窗口。候选生成后仍需核对下一交易日执行条件；本入口不下单。

## 20日行情的规则复核

[详细报告](../../outputs/strategy_2026_opportunity_scan/20261020_gate_paused_20261021T023746_584755Z/REPORT.html)

使用新规则复核2026-10-20收盘：2个板块形态确认中，只有养殖业是新的首次事件。养殖业22只成员中21只合格，产生5只候选：京基智农、神农集团、华英农业、巨星农牧、天域生物。个股分数与此前独立观察报告一致。

酒店餐饮仍为9月17日旧事件，不因为暂停年度板块判断而重开入场。此次复核标注`policy_change_review`，不改写20日原报告，不假定当前仍可按20日价格成交，也不恢复已过的执行窗口。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python datahub/scripts/frozen_research.py run --release datahub/releases/selection_live_20261020_eod alpha_lab/experiments/sector_live_v1/scan_stocks.py --scan alpha_lab/outputs/sector_live_scan/20261020T113957_683276Z --review-policy --name-history alpha_lab/outputs/breeding_stock_review/20261020_eod/fresh_namechange.parquet
```

此次只改策略应用层、当前配置和入口说明，数据更新及数据集构建程序未变。扩大到原板块模型拒绝的事件后，个股排序效果尚未完成全流程验证；本次执行用户指定的规则调整，不把旧回测表现套用到新事件池。
