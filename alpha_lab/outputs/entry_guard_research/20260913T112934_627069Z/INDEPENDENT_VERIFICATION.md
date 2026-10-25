# 新仓2×2独立核验

状态：complete。

24账户、1,311笔真实成交及86,456次有效原价收盘估值通过，stale=0。最大现金误差1.46e-11元、总资产误差2.91e-11元。

888个review、36,178对相关值及14,938个候选已独立重建；相关最大误差1.33e-15。488次首次hold_false退出逐原hold单元和确认生命周期通过，未将pending/ST重复计作首次趋势退出。

四组真实倒序各33次接收端证据通过；全部CSV对照80项（包括guard三表和recent两表），六组baseline对修复parent全部CSV逐值对照102项。只剔除name身份列，所有经济字段完整保留。

先前24组核心报告保存在execution/independent_entry_guard_core_verification.json。衍生表落盘后仅追加文件对照，并复核原核心CSV及输入哈希，没有重复账户回放。

完整结果：execution/independent_entry_guard_verification.json；真实成交：execution/independent_entry_guard_fill_checks.csv；日志：execution/independent_entry_guard_verification.log。

账户守恒与机制一致不代表策略有效；参数已见历史后提出，不能把历史重验当新样本外证据。
