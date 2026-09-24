# 研究过程档案

这里收录原研究过程中形成的实验设计、阶段报告、审计修正和实验脚本。原文保留，增补历史边界说明并去除敏感信息；没有按收益高低挑选记录。

当前收录 **178 份研究文档**、**56 份实验脚本**。另有 68 份内容完全相同的来源文件合并到同一记录，避免重复计数。

阅读建议：先看 [研究过程案例](../docs/research-journeys.md)，再按下方主题进入原始记录；阶段结果不等于当前有效结论。

## 其他资产与早期系统

| 日期标记 | 类型 | 原始记录 |
|---|---|---|
| 见原文 | 研究文档 | [a_share_v18 — 两层轮动 诚实口径版](a_share_v18/README.md) |
| 见原文 | 研究文档 | [cb_rotator — 可转债多因子轮动策略](cb_rotator/README.md) |
| 见原文 | 研究文档 | [ipo_trader — 新股日内做T交易系统](ipo_trader/README.md) |

## 因子挖掘与跨期复核

| 日期标记 | 类型 | 原始记录 |
|---|---|---|
| 见原文 | 实验脚本 | [replicate_factors](alpha_lab/scripts/replicate_factors.py) |
| 见原文 | 实验脚本 | [run_gp_mining](alpha_lab/scripts/run_gp_mining.py) |
| 见原文 | 实验脚本 | [synthesis_and_test](alpha_lab/scripts/synthesis_and_test.py) |
| 见原文 | 实验脚本 | [walkforward_ablation](alpha_lab/scripts/walkforward_ablation.py) |
| 见原文 | 实验脚本 | [walkforward_monthly_oos](alpha_lab/scripts/walkforward_monthly_oos.py) |
| 20260908 | 研究文档 | [因子挖掘调研与低频策略重构路线图（2026-09-08）](FACTOR_MINING_SURVEY_20260908.md) |
| 20260911 | 研究文档 | [IC 口径修复与固定候选重验（2026-09-11）](alpha_lab/docs/IC_REVALIDATION_20260911.md) |
| 20260911 | 研究文档 | [alpha_lab：中英文研究与工程方法调研](alpha_lab/docs/RESEARCH_METHODS_SURVEY_20260911.md) |
| 20260911 | 研究文档 | [统计口径修复与风格匹配归因](alpha_lab/docs/STATISTICS_STYLE_AUDIT_20260911.md) |
| 20260913 | 研究文档 | [三路定向重调研：精选、持有和现金决策](alpha_lab/docs/TARGETED_RESEARCH_SYNTHESIS_20260913.md) |
| 20260918 | 研究文档 | [Walk-forward 逐年样本外重验 — 预注册 (2026-09-18)](alpha_lab/docs/WALKFORWARD_OOS_PREREG_20260918.md) |
| 20260919 | 研究文档 | [Walk-forward 逐年样本外重验 — 结果报告 (2026-09-19)](alpha_lab/docs/WALKFORWARD_OOS_REPORT_20260919.md) |

## 持仓、风险与执行研究

| 日期标记 | 类型 | 原始记录 |
|---|---|---|
| 见原文 | 研究文档 | [alpha_lab — 低频量化研究与数据驱动因子挖掘 (重构版)](alpha_lab/README.md) |
| 见原文 | 研究文档 | [月频组合记账契约（2026-09-11）](alpha_lab/docs/PORTFOLIO_ACCOUNTING.md) |
| 见原文 | 研究文档 | [待测试集合的共同口径](alpha_lab/docs/strategy_inventory_20260920/TEST_CONTRACT.md) |
| 见原文 | 研究文档 | [独立调查 A：alpha_lab 低频选股与账户政策](alpha_lab/docs/strategy_inventory_20260920/investigator_alpha.md) |
| 见原文 | 研究文档 | [独立调查：成长与跨市场研究机制清单](alpha_lab/docs/strategy_inventory_20260920/investigator_growth.md) |
| 见原文 | 研究文档 | [独立调查：历史选股、择时、退出与特殊资产路线](alpha_lab/docs/strategy_inventory_20260920/investigator_legacy.md) |
| 见原文 | 研究文档 | [学习策略共同归因（2026-09-20）](alpha_lab/run_records/learning_diagnosis/20260920T122452_692950Z/REPORT.md) |
| 见原文 | 研究文档 | [无训练规则搜索执行报告](alpha_lab/run_records/rule_strategy_search/20260920T101619_734224Z/REPORT.md) |
| 见原文 | 实验脚本 | [build_sector_dataset](alpha_lab/scripts/build_sector_dataset.py) |
| 见原文 | 实验脚本 | [build_sector_shape_events](alpha_lab/scripts/build_sector_shape_events.py) |
| 见原文 | 实验脚本 | [recompute_deciles_oos](alpha_lab/scripts/recompute_deciles_oos.py) |
| 见原文 | 实验脚本 | [research_cash_economics](alpha_lab/scripts/research_cash_economics.py) |
| 见原文 | 实验脚本 | [research_forward_risk](alpha_lab/scripts/research_forward_risk.py) |
| 见原文 | 实验脚本 | [research_sector_cohesion](alpha_lab/scripts/research_sector_cohesion.py) |
| 见原文 | 实验脚本 | [research_sector_confirmation_returns](alpha_lab/scripts/research_sector_confirmation_returns.py) |
| 见原文 | 实验脚本 | [research_sector_multiscale_events](alpha_lab/scripts/research_sector_multiscale_events.py) |
| 见原文 | 实验脚本 | [research_sector_shape_expansion](alpha_lab/scripts/research_sector_shape_expansion.py) |
| 见原文 | 实验脚本 | [research_startup_timing](alpha_lab/scripts/research_startup_timing.py) |
| 见原文 | 实验脚本 | [run_selection_stability](alpha_lab/scripts/run_selection_stability.py) |
| 20260911 | 研究文档 | [纯多头目标与可投资股票池（2026-09-11）](alpha_lab/docs/LONG_ONLY_FOLLOWUP_20260911.md) |
| 20260911 | 研究文档 | [月频组合日频记账审计（2026-09-11）](alpha_lab/docs/MONTHLY_PORTFOLIO_AUDIT_20260911.md) |
| 20260911 | 研究文档 | [月频合成与换手改进（2026-09-11）](alpha_lab/docs/PORTFOLIO_IMPROVEMENT_20260911.md) |
| 20260912 | 研究文档 | [小资金账户的主动空仓：研究与旧代码审查](alpha_lab/docs/ACTIVE_CASH_RESEARCH_20260912.md) |
| 20260912 | 研究文档 | [主动空仓首轮验证：风险触发与恢复时点](alpha_lab/docs/ACTIVE_CASH_VALIDATION_20260912.md) |
| 20260912 | 研究文档 | [小资金、五只以内与长期趋势：研究定位及首轮检验](alpha_lab/docs/PATIENT_TREND_DIRECTION_20260912.md) |
| 20260912 | 研究文档 | [小资金组合与主动空仓：完整检验报告](alpha_lab/docs/RETAIL_COMPLETE_VALIDATION_20260912.md) |
| 20260912 | 研究文档 | [小账户定位修正与过程审计](alpha_lab/docs/RETAIL_PROCESS_AUDIT_20260912.md) |
| 20260912 | 研究文档 | [趋势入场状态分析：风险识别与长期收益空间](alpha_lab/docs/TREND_ENTRY_STATE_AUDIT_20260912.md) |
| 20260913 | 研究文档 | [小资金长期趋势：收益幅度、持有纪律与主动空仓迭代](alpha_lab/docs/AUTONOMOUS_VALUE_ITERATIONS_20260913.md) |
| 20260913 | 研究文档 | [订单时序修复、身份更正与20组完整复验](alpha_lab/docs/CAUSAL_EXECUTION_VALIDATION_20260913.md) |
| 20260913 | 研究文档 | [少量股票组合集中风险：原始证据与两个有界假设](alpha_lab/docs/CONCENTRATION_RISK_RESEARCH_20260913.md) |
| 20260913 | 研究文档 | [长期状态分治退出：30账户与固定持仓过程检验](alpha_lab/docs/CONDITIONAL_HOLD_VALIDATION_20260913.md) |
| 20260913 | 研究文档 | [换选择策略与分治选择：阶段判断、组内选择和实际配仓](alpha_lab/docs/DIVIDE_SELECTION_20260913.md) |
| 20260913 | 研究文档 | [集中风险与补仓暂停：调研、过程消融和完整结果](alpha_lab/docs/ENTRY_GUARD_VALIDATION_20260913.md) |
| 20260913 | 研究文档 | [退出与恢复事件的预测证据：完成报告](alpha_lab/docs/EXIT_EVIDENCE_RESEARCH_20260913.md) |
| 20260913 | 研究文档 | [历史退出事件扩展：收益方向与路径风险](alpha_lab/docs/EXIT_HISTORY_RESEARCH_20260913.md) |
| 20260913 | 研究文档 | [退出灵敏性与鲁棒性：36账户及过程检验](alpha_lab/docs/HOLD_SENSITIVITY_VALIDATION_20260913.md) |
| 20260913 | 研究文档 | [Hybrid策略详细说明与历史表现](alpha_lab/docs/HYBRID_STRATEGY_DETAIL_20260913.md) |
| 20260913 | 研究文档 | [Hybrid 独立未来信息与数据真实性审查](alpha_lab/docs/INDEPENDENT_LEAK_AUDIT_20260913.md) |
| 20260913 | 研究文档 | [日期敏感性后的路径机制研究：80账户完整检验](alpha_lab/docs/PATH_MECHANISM_RESEARCH_20260913.md) |
| 20260913 | 研究文档 | [持有目标与现金动作研究协议](alpha_lab/docs/POLICY_CASH_PROTOCOL_20260913.md) |
| 20260913 | 研究文档 | [持有目标与现金动作：过程检验结果](alpha_lab/docs/POLICY_CASH_VALIDATION_20260913.md) |
| 20260913 | 研究文档 | [Hybrid盈利来源与防守能力：14组机制复验](alpha_lab/docs/PROFIT_DEFENSE_RESEARCH_20260913.md) |
| 20260913 | 研究文档 | [低换手五股版本：2024—2026近期冻结回放](alpha_lab/docs/RECENT_LOW_TURNOVER_VALIDATION_20260913.md) |
| 20260913 | 研究文档 | [月频卖出后现金选择与组合新增风险：独立研究记录](alpha_lab/docs/REENTRY_CASH_RESEARCH_20260913.md) |
| 20260913 | 研究文档 | [日期偏移如何改变持仓与收益](alpha_lab/docs/REVIEW_DATE_MECHANISM_20260913.md) |
| 20260913 | 研究文档 | [月内起点与滚动复核：完整检验](alpha_lab/docs/ROLLING_REVIEW_RESEARCH_20260913.md) |
| 20260913 | 研究文档 | [月度趋势退出：灵敏性与鲁棒性的有界研究](alpha_lab/docs/SENSITIVITY_ROBUSTNESS_RESEARCH_20260913.md) |
| 20260913 | 研究文档 | [对称阶段分治、持有证据与入场过程检验](alpha_lab/docs/SYMMETRIC_PHASE_VALIDATION_20260913.md) |
| 20260913 | 研究文档 | [定向重调研：0—5股月频趋势账户的拒绝买入与主动空仓](alpha_lab/docs/TARGETED_RESEARCH_CASH_20260913.md) |
| 20260913 | 研究文档 | [持久趋势的入场、持有与退出：针对性重查](alpha_lab/docs/TARGETED_RESEARCH_HOLDING_20260913.md) |
| 20260913 | 研究文档 | [阶段分治后的精准选择：面向0—5只股票账户的定向调研](alpha_lab/docs/TARGETED_RESEARCH_SELECTION_20260913.md) |
| 20260914 | 研究文档 | [主动减仓的经济性：过程归因、机制修正与邻域验证](alpha_lab/docs/CASH_ECONOMICS_RESEARCH_20260914.md) |
| 20260914 | 研究文档 | [14:30盘中代理与收盘集合竞价](alpha_lab/docs/CLOSING_EXECUTION_20260914.md) |
| 20260914 | 研究文档 | [执行元数据操作指南](alpha_lab/docs/EXECUTION_METADATA_GUIDE_20260914.md) |
| 20260914 | 研究文档 | [执行元数据与最新观察包验收](alpha_lab/docs/EXECUTION_METADATA_VALIDATION_20260914.md) |
| 20260914 | 研究文档 | [前瞻风险证据与受限仓位预算实验](alpha_lab/docs/FORWARD_RISK_EVIDENCE_20260914.md) |
| 20260914 | 研究文档 | [定向恢复：独立因果执行审查](alpha_lab/docs/INDEPENDENT_TARGETED_RESTORE_CAUSAL_20260914.md) |
| 20260914 | 研究文档 | [定向恢复的独立实盘化审查](alpha_lab/docs/INDEPENDENT_TARGETED_RESTORE_LIVE_20260914.md) |
| 20260914 | 研究文档 | [定向恢复的独立实盘化审查](alpha_lab/docs/INDEPENDENT_TARGETED_RESTORE_READINESS_20260914.md) |
| 20260914 | 研究文档 | [行情增量与CSV对账操作指南](alpha_lab/docs/LIVE_INGESTION_GUIDE_20260914.md) |
| 20260914 | 研究文档 | [实盘准备第二阶段：行情增量与券商CSV对账](alpha_lab/docs/LIVE_INGESTION_VALIDATION_20260914.md) |
| 20260914 | 研究文档 | [风险仓位版实盘准入审查](alpha_lab/docs/LIVE_READINESS_AUDIT_20260914.md) |
| 20260914 | 研究文档 | [手动执行播报的持久化发件箱](alpha_lab/docs/MAIL_OUTBOX_20260914.md) |
| 20260914 | 研究文档 | [邮件播报—人工执行](alpha_lab/docs/MANUAL_EXECUTION_GUIDE_20260914.md) |
| 20260914 | 研究文档 | [月度复核平移至15日](alpha_lab/docs/MONTHLY_DAY15_START_20260914.md) |
| 20260914 | 研究文档 | [月度风险预算候选：人工执行接入与验证](alpha_lab/docs/MONTHLY_RISK_MANUAL_INTEGRATION_20260914.md) |
| 20260914 | 研究文档 | [组合降仓、空仓与恢复机制验证](alpha_lab/docs/PORTFOLIO_CASH_VALIDATION_20260914.md) |
| 20260914 | 研究文档 | [排名缓冲的实盘应用验证（2026-09-14）](alpha_lab/docs/RANK_LIVE_VALIDATION_20260914.md) |
| 20260914 | 研究文档 | [风险仓位折扣：76账户完整执行检验](alpha_lab/docs/RISK_BUDGET_RESEARCH_20260914.md) |
| 20260914 | 研究文档 | [风险恢复与整手缓冲：76账户完整检验](alpha_lab/docs/RISK_RECOVERY_RESEARCH_20260914.md) |
| 20260914 | 研究文档 | [选择持续性与排名缓冲：完整研究报告](alpha_lab/docs/SELECTION_STABILITY_RESEARCH_20260914.md) |
| 20260914 | 研究文档 | [风险仓位版每日计划与账户对账](alpha_lab/docs/SHADOW_EXECUTION_GUIDE_20260914.md) |
| 20260914 | 研究文档 | [实盘准备第一阶段：每日计划与账户闭环验收](alpha_lab/docs/SHADOW_EXECUTION_VALIDATION_20260914.md) |
| 20260914 | 研究文档 | [首次建仓与复核日期敏感性研究](alpha_lab/docs/STARTUP_TIMING_RESEARCH_20260914.md) |
| 20260914 | 研究文档 | [五席位旧仓定向恢复：修复、完整回放与独立实盘化审查](alpha_lab/docs/TARGETED_RESTORE_VALIDATION_20260914.md) |
| 20260915 | 研究文档 | [9:30开盘后连续竞价人工执行](alpha_lab/docs/OPENING_CONTINUOUS_EXECUTION_20260915.md) |
| 20260919 | 研究文档 | [高可行性策略复现 — 预注册 (2026-09-19)](alpha_lab/docs/REPRODUCE_FEASIBLE_PREREG_20260919.md) |
| 20260920 | 研究文档 | [经典策略工程验收与固定输入入口](alpha_lab/docs/CLASSIC_STRATEGY_SMOKE_20260920.md) |
| 20260920 | 研究文档 | [历史研究策略集合：按机制归并的待测试清单](alpha_lab/docs/STRATEGY_CANDIDATES_20260920.md) |

## 数据工程与研究契约

| 日期标记 | 类型 | 原始记录 |
|---|---|---|
| 见原文 | 研究文档 | [alpha_lab 数据 Schema 文档](alpha_lab/docs/DATA_SCHEMA.md) |
| 见原文 | 研究文档 | [算子一致性修复记录（2026-09-11）](alpha_lab/docs/OPERATOR_PARITY.md) |
| 见原文 | 研究文档 | [Phase 0 数据质量报告](alpha_lab/docs/QUALITY_REPORT.md) |
| 见原文 | 研究文档 | [固定数据、固定模拟口径、只迭代策略](datahub/FROZEN_RESEARCH.md) |
| 见原文 | 研究文档 | [datahub — A股 + 美股结构化数据集](datahub/README.md) |
| 20260913 | 研究文档 | [财务披露事件恢复与五股试点](alpha_lab/docs/FINANCIAL_EVENT_RECOVERY_20260913.md) |
| 20260913 | 研究文档 | [经营与现金流增量研究：历史版本准入检验](alpha_lab/docs/FINANCIAL_PIT_GATE_20260913.md) |
| 20260920 | 研究文档 | [datahub 研究数据验收（2026-09-20）](datahub/RESEARCH_READINESS_20260920.md) |

## 旧版演进与回测审计

| 日期标记 | 类型 | 原始记录 |
|---|---|---|
| 见原文 | 研究文档 | [量化选股模型 —— 版本演进总览](VERSIONS.md) |
| 见原文 | 研究文档 | [步骤 1 — 未来函数排查报告](audit/01_future_function.md) |
| 见原文 | 研究文档 | [步骤 2 — Alpha / Beta / Sharpe 评估报告](audit/02_alpha_beta_sharpe.md) |
| 见原文 | 研究文档 | [步骤 3 — 因子暴露分析报告](audit/03_factor_exposure.md) |
| 见原文 | 研究文档 | [步骤 4 — IC 与 Rank IC 计算报告](audit/04_ic_rank_ic.md) |
| 见原文 | 研究文档 | [步骤 5 — 信息比率 IR 计算报告](audit/05_ir.md) |
| 见原文 | 研究文档 | [步骤 6 — 修复执行报告](audit/06_fix_report.md) |
| 见原文 | 研究文档 | [步骤 7 — 修复后重新审计报告](audit/07_rea_audit_report.md) |
| 见原文 | 研究文档 | [步骤 8 — 跳空感知成交修复 + 重新评估](audit/08_gapaware_report.md) |
| 见原文 | 研究文档 | [量化选股模型审计与修复工作总结](audit/09_summary.md) |
| 见原文 | 研究文档 | [步骤 10 — 实盘预测脚本：当日实时特征修复](audit/10_realtime_features.md) |
| 见原文 | 研究文档 | [步骤 11 — 因子挖掘结论 + 差异化优化路线](audit/11_differentiated_strategy.md) |
| 见原文 | 研究文档 | [步骤 12 — regime 择时重构 + A股/KCB 组合对冲](audit/12_regime_analysis.md) |
| 见原文 | 研究文档 | [步骤 13 — 涨停板战法测试（抓涨停回调 / 打板）](audit/13_limitup_strategy.md) |
| 见原文 | 研究文档 | [步骤 14 — 止损日反T做T补充策略: 可行性、阈值优化与风险稳定性](audit/14_reverse_t_strategy.md) |
| 见原文 | 研究文档 | [步骤 15 — 模型调优研究: 板块选择 / 持仓期 / 移动止盈 / 分数过滤器](audit/15_model_tuning.md) |
| 见原文 | 研究文档 | [步骤 16 — 扩展因子库 Batch1 (换手/资金流/估值/日内/波动结构)](audit/16_factor_expansion.md) |
| 见原文 | 研究文档 | [步骤 17 — 月度滚动方案回测 (3个月训练窗 + 月度因子选择)](audit/17_rolling_backtest.md) |
| 见原文 | 研究文档 | [步骤 18 — 未来信息泄露专项审查 (扩展因子库 + 月度滚动集成后)](audit/18_lookahead_review.md) |
| 见原文 | 研究文档 | [最优可用策略报告 (截至 2026-08-22, 全口径审查后)](audit/19_best_strategy.md) |
| 见原文 | 研究文档 | [步骤 20 — 日频数据完整性检查](audit/20_data_integrity.md) |
| 见原文 | 研究文档 | [步骤 21 — 移动止盈的分钟级验证: 日线口径的退出层收益基本是水分](audit/21_minute_replay.md) |
| 见原文 | 研究文档 | [步骤 22 — 诚实口径终版: 固定 TP/SL + SL优先 + 分钟验证后的真实数字](audit/22_honest_fixed_tpsl.md) |
| 见原文 | 研究文档 | [步骤 23 — KCB 最终裁决: 个股短线全灭, ETF 躺平最优](audit/23_kcb_final_verdict.md) |
| 见原文 | 研究文档 | [步骤 24 — 因子组合方法与 LGBM 过拟合/稳定性研究](audit/24_model_stability_factors.md) |
| 见原文 | 研究文档 | [步骤 25 — 泄露审查(EOD因子) + 每日轮动/前3名评估](audit/25_leakage_fix_daily_rotation.md) |
| 见原文 | 研究文档 | [步骤 26 — 止损/退出机制研究系列 + 生产策略定稿](audit/26_exit_research.md) |
| 见原文 | 研究文档 | [步骤 27 — 主板诚实口径总清算 (a_share_v18)](audit/27_honest_v18.md) |
| 见原文 | 研究文档 | [步骤 28 — 研究思路总汇总 (2026-08-26 ~ 08-29)](audit/28_research_summary.md) |
| 见原文 | 研究文档 | [audit/29 — 进出场机制剥离实验 (mechanism ablation)](audit/29_mechanism_ablation.md) |
| 见原文 | 研究文档 | [量化选股模型 — 版本审计报告](audit/README.md) |
| 见原文 | 实验脚本 | [ashare_ablation](audit/scripts/ashare_ablation.py) |
| 见原文 | 实验脚本 | [ic_leak_ablation](audit/scripts/ic_leak_ablation.py) |
| 见原文 | 实验脚本 | [multi_label_audit](audit/scripts/multi_label_audit.py) |
| 见原文 | 实验脚本 | [trailing_minute_check](audit/scripts/trailing_minute_check.py) |

## 板块形态与模糊网络

| 日期标记 | 类型 | 原始记录 |
|---|---|---|
| 见原文 | 研究文档 | [板块数据集 v1：独立于策略的日线资产](alpha_lab/docs/SECTOR_DATASET_V1.md) |
| 见原文 | 研究文档 | [随机测试：板块之间与板块内部收益报告](alpha_lab/experiments/sector_fuzzy_v1/DISTRIBUTION_RESULTS.md) |
| 见原文 | 研究文档 | [具体板块收益：指定格式的直方图报告](alpha_lab/experiments/sector_fuzzy_v1/HISTOGRAM_REPORT.md) |
| 见原文 | 研究文档 | [模糊事件筛选实验 v1](alpha_lab/experiments/sector_fuzzy_v1/PREREG.md) |
| 见原文 | 研究文档 | [随机二等分、角色互换实验](alpha_lab/experiments/sector_fuzzy_v1/RANDOM_PREREG.md) |
| 见原文 | 研究文档 | [随机二等分互换：已复现实验](alpha_lab/experiments/sector_fuzzy_v1/RANDOM_RESULTS.md) |
| 见原文 | 研究文档 | [首轮结果与复现入口](alpha_lab/experiments/sector_fuzzy_v1/RESULTS.md) |
| 见原文 | 实验脚本 | [audit](alpha_lab/experiments/sector_fuzzy_v1/audit.py) |
| 见原文 | 实验脚本 | [audit_random](alpha_lab/experiments/sector_fuzzy_v1/audit_random.py) |
| 见原文 | 实验脚本 | [model](alpha_lab/experiments/sector_fuzzy_v1/model.py) |
| 见原文 | 实验脚本 | [random_split](alpha_lab/experiments/sector_fuzzy_v1/random_split.py) |
| 见原文 | 实验脚本 | [report_between](alpha_lab/experiments/sector_fuzzy_v1/report_between.py) |
| 见原文 | 实验脚本 | [report_distribution](alpha_lab/experiments/sector_fuzzy_v1/report_distribution.py) |
| 见原文 | 实验脚本 | [report_histograms](alpha_lab/experiments/sector_fuzzy_v1/report_histograms.py) |
| 见原文 | 实验脚本 | [run](alpha_lab/experiments/sector_fuzzy_v1/run.py) |
| 见原文 | 实验脚本 | [run_random](alpha_lab/experiments/sector_fuzzy_v1/run_random.py) |
| 见原文 | 实验脚本 | [test_model](alpha_lab/experiments/sector_fuzzy_v1/test_model.py) |
| 见原文 | 实验脚本 | [test_random_split](alpha_lab/experiments/sector_fuzzy_v1/test_random_split.py) |
| 见原文 | 研究文档 | [2026-09-23 更新及最新候选扫描](alpha_lab/experiments/sector_live_v1/README.md) |
| 见原文 | 研究文档 | [具体板块收益：直方图版](alpha_lab/run_records/sector_fuzzy_histogram_report/20260923T114347_981934Z/README.md) |
| 见原文 | 研究文档 | [随机两半互换实验](alpha_lab/run_records/sector_fuzzy_random_halves/20260923T111853_561924Z/README.md) |
| 见原文 | 研究文档 | [模糊事件筛选实验：历史滚动报告](alpha_lab/run_records/sector_fuzzy_v1/20260923T105203_887024Z/README.md) |
| 20260921 | 研究文档 | [左侧规则第一轮：案例检查与解释边界](alpha_lab/docs/LEFT_SIDE_REVIEW_20260921.md) |
| 20260921 | 研究文档 | [左侧板块—个股量价规则：第一版研究契约](alpha_lab/docs/LEFT_SIDE_RULES_DESIGN_20260921.md) |
| 20260921 | 研究文档 | [左侧板块—个股量价规则：首轮结果](alpha_lab/docs/LEFT_SIDE_RULES_RESULTS_20260921.md) |
| 20260921 | 研究文档 | [左侧规则 v2：板块与个股都必须先上行、再回落](alpha_lab/docs/LEFT_SIDE_TREND_PREREG_20260921.md) |
| 20260921 | 研究文档 | [左侧规则 v2：双层上行趋势及上涨回落修正](alpha_lab/docs/LEFT_SIDE_TREND_RESULTS_20260921.md) |
| 20260921 | 研究文档 | [板块整体性研究：结果与下一步](alpha_lab/docs/SECTOR_COHESION_RESULTS_20260921.md) |
| 20260921 | 研究文档 | [板块数据集构建报告](alpha_lab/docs/SECTOR_DATASET_BUILD_20260921.md) |

## 模型比较与直接策略学习

| 日期标记 | 类型 | 原始记录 |
|---|---|---|
| 见原文 | 研究文档 | [三条固定研究基线 v1](alpha_lab/docs/SELECTION_BASELINES_V1.md) |
| 见原文 | 实验脚本 | [ablate_baseline_combination](alpha_lab/scripts/ablate_baseline_combination.py) |
| 见原文 | 实验脚本 | [diagnose_learning_strategies](alpha_lab/scripts/diagnose_learning_strategies.py) |
| 见原文 | 实验脚本 | [report_learning_strategy_search](alpha_lab/scripts/report_learning_strategy_search.py) |
| 见原文 | 实验脚本 | [report_policy_balance](alpha_lab/scripts/report_policy_balance.py) |
| 见原文 | 实验脚本 | [report_rule_strategy_search](alpha_lab/scripts/report_rule_strategy_search.py) |
| 见原文 | 实验脚本 | [report_sequence_state_learning](alpha_lab/scripts/report_sequence_state_learning.py) |
| 见原文 | 实验脚本 | [run_baseline_attribution](alpha_lab/scripts/run_baseline_attribution.py) |
| 见原文 | 实验脚本 | [run_baseline_combination](alpha_lab/scripts/run_baseline_combination.py) |
| 见原文 | 实验脚本 | [run_baseline_increment](alpha_lab/scripts/run_baseline_increment.py) |
| 见原文 | 实验脚本 | [run_baseline_style_controls](alpha_lab/scripts/run_baseline_style_controls.py) |
| 见原文 | 实验脚本 | [run_initialized_policy](alpha_lab/scripts/run_initialized_policy.py) |
| 见原文 | 实验脚本 | [run_learning_extensions](alpha_lab/scripts/run_learning_extensions.py) |
| 见原文 | 实验脚本 | [run_learning_strategy_search](alpha_lab/scripts/run_learning_strategy_search.py) |
| 见原文 | 实验脚本 | [run_policy_balance](alpha_lab/scripts/run_policy_balance.py) |
| 见原文 | 实验脚本 | [run_policy_convergence](alpha_lab/scripts/run_policy_convergence.py) |
| 见原文 | 实验脚本 | [run_rule_strategy_search](alpha_lab/scripts/run_rule_strategy_search.py) |
| 见原文 | 实验脚本 | [run_sequence_state_learning](alpha_lab/scripts/run_sequence_state_learning.py) |
| 20260920 | 研究文档 | [等权结果的固定消融协议](alpha_lab/docs/BASELINE_COMBINATION_ABLATION_20260920.md) |
| 20260920 | 研究文档 | [基线包含性与组合目标诊断：执行前协议](alpha_lab/docs/BASELINE_COMBINATION_PREREG_20260920.md) |
| 20260920 | 研究文档 | [完整基线组合研究结果](alpha_lab/docs/BASELINE_COMBINATION_RESULTS_20260920.md) |
| 20260920 | 研究文档 | [基线＋监督学习增量：执行前登记](alpha_lab/docs/BASELINE_INCREMENT_PREREG_20260920.md) |
| 20260920 | 研究文档 | [基线＋监督学习增量：实际训练结果](alpha_lab/docs/BASELINE_INCREMENT_RESULTS_20260920.md) |
| 20260920 | 研究文档 | [按用户澄清重定研究目标：从基线初始化的完整策略学习](alpha_lab/docs/BASELINE_INITIALIZED_POLICY_LEARNING_20260920.md) |
| 20260920 | 研究文档 | [基线初始化完整策略学习：首轮执行登记](alpha_lab/docs/BASELINE_INITIALIZED_POLICY_PREREG_20260920.md) |
| 20260920 | 研究文档 | [基线初始化完整策略学习：首轮结果](alpha_lab/docs/BASELINE_INITIALIZED_POLICY_RESULTS_20260920.md) |
| 20260920 | 研究文档 | [为什么先前因子组合没有回到规则基线](alpha_lab/docs/BASELINE_LEARNING_GAP_20260920.md) |
| 20260920 | 研究文档 | [学习策略共同归因（2026-09-20）](alpha_lab/docs/LEARNING_DIAGNOSIS_RESULTS_20260920.md) |
| 20260920 | 研究文档 | [共同归因及CNN/HMM补测约定](alpha_lab/docs/LEARNING_DIAGNOSIS_SEQUENCE_PREREG_20260920.md) |
| 20260920 | 研究文档 | [第二批学习策略实验约定](alpha_lab/docs/LEARNING_EXTENSIONS_PREREG_20260920.md) |
| 20260920 | 研究文档 | [第二批学习策略结果（2026-09-20）](alpha_lab/docs/LEARNING_EXTENSIONS_RESULTS_20260920.md) |
| 20260920 | 研究文档 | [学习策略首批实验设计](alpha_lab/docs/LEARNING_STRATEGY_SEARCH_20260920.md) |
| 20260920 | 研究文档 | [首批学习策略执行结果](alpha_lab/docs/LEARNING_STRATEGY_SEARCH_RESULTS_20260920.md) |
| 20260920 | 研究文档 | [完整策略学习延长训练：预登记](alpha_lab/docs/POLICY_CONVERGENCE_PREREG_20260920.md) |
| 20260920 | 研究文档 | [完整策略学习延长训练结果](alpha_lab/docs/POLICY_CONVERGENCE_RESULTS_20260920.md) |
| 20260920 | 研究文档 | [无需训练策略：范围与非单调参数搜索](alpha_lab/docs/RULE_STRATEGY_SEARCH_20260920.md) |
| 20260920 | 研究文档 | [首批无需训练策略：执行结果](alpha_lab/docs/RULE_STRATEGY_SEARCH_RESULTS_20260920.md) |
| 20260920 | 研究文档 | [CNN / HMM 补测结果](alpha_lab/docs/SEQUENCE_STATE_LEARNING_RESULTS_20260920.md) |
| 20260921 | 研究文档 | [基线收益来源与外推边界](alpha_lab/docs/BASELINE_ATTRIBUTION_RESULTS_20260921.md) |
| 20260921 | 研究文档 | [基线风格匹配与时间稳定性：运行前方案](alpha_lab/docs/BASELINE_STYLE_CONTROLS_PREREG_20260921.md) |
| 20260921 | 研究文档 | [基线风格对照与跨年稳定性](alpha_lab/docs/BASELINE_STYLE_CONTROLS_RESULTS_20260921.md) |
| 20260921 | 研究文档 | [衰减、迭代次数与训练窗口联合实验](alpha_lab/docs/POLICY_BALANCE_PREREG_20260921.md) |
| 20260921 | 研究文档 | [模型衰减—迭代—窗口联合实验结果](alpha_lab/docs/POLICY_BALANCE_RESULTS_20260921.md) |

## 跨市场与成长线

| 日期标记 | 类型 | 原始记录 |
|---|---|---|
| 见原文 | 研究文档 | [alpha_lab_growth — USX 重构 (2026-09-18 起)](alpha_lab_growth/README.md) |
| 见原文 | 研究文档 | [信号方向契约 (SIGNAL_DIRECTION_CONTRACT)](alpha_lab_growth/docs/SIGNAL_DIRECTION_CONTRACT.md) |
| 见原文 | 研究文档 | [alpha_lab_growth — 高波成长独立策略线研究副本](alpha_lab_growth/docs/legacy_v1_README_untrusted.md) |
| 见原文 | 实验脚本 | [usx_composite](alpha_lab_growth/scripts/usx_composite.py) |
| 见原文 | 实验脚本 | [usx_fetch_us](alpha_lab_growth/scripts/usx_fetch_us.py) |
| 见原文 | 实验脚本 | [usx_final_vs_null](alpha_lab_growth/scripts/usx_final_vs_null.py) |
| 见原文 | 实验脚本 | [usx_p0_1_index](alpha_lab_growth/scripts/usx_p0_1_index.py) |
| 见原文 | 实验脚本 | [usx_p0_2_sector](alpha_lab_growth/scripts/usx_p0_2_sector.py) |
| 见原文 | 实验脚本 | [usx_p0_2b_dynamics](alpha_lab_growth/scripts/usx_p0_2b_dynamics.py) |
| 见原文 | 实验脚本 | [usx_p0_3_stock](alpha_lab_growth/scripts/usx_p0_3_stock.py) |
| 见原文 | 实验脚本 | [usx_sensitivity](alpha_lab_growth/scripts/usx_sensitivity.py) |
| 20260916 | 研究文档 | [P0 副本初始化与隔离边界 (FORK_INIT)](alpha_lab_growth/docs/FORK_INIT_20260916.md) |
| 20260916 | 研究文档 | [growth P1: 全市场管线适配 — 约束地图、可行性画像与因子缓存](alpha_lab_growth/docs/GROWTH_P1_PIPELINE_20260916.md) |
| 20260916 | 研究文档 | [growth P2: 全市场信号研究 — 预注册扫描、双维验收与 VALID 确认](alpha_lab_growth/docs/GROWTH_P2_SIGNALS_20260916.md) |
| 20260917 | 研究文档 | [growth P2': 重设计信号研究 — 7 条机械候选扫描、ideal_amp_v2 账本回放与方向事故](alpha_lab_growth/docs/GROWTH_P2B_REDESIGN_20260917.md) |
| 20260917 | 研究文档 | [growth P3: 独立账户回放 — 管线建成、完整性验证与 8 情形矩阵](alpha_lab_growth/docs/GROWTH_P3_ACCOUNT_20260917.md) |
| 20260917 | 研究文档 | [growth P4: VALID 窗评估 — 数值化预注册验收](alpha_lab_growth/docs/GROWTH_P4_EVAL_20260917.md) |
| 20260917 | 研究文档 | [growth P5: FAIL 后研究续 — 探针、双路调研综合与重设计预注册提案](alpha_lab_growth/docs/GROWTH_P5_REDESIGN_20260917.md) |
| 20260918 | 研究文档 | [growth 路线 A/D/F-US 执行报告 — 广度前提被证伪、域结构确认高波、美股首期](alpha_lab_growth/docs/GROWTH_ROUTE_AD_US_20260918.md) |
| 20260918 | 研究文档 | [USX 架构设计与预注册 (2026-09-18)](alpha_lab_growth/docs/USX_DESIGN_20260918.md) |
| 20260918 | 研究文档 | [USX 终报 — 美股隔夜→A股板块轮动→个股中短期持有: 重构、证伪与遗产 (2026-09-18)](alpha_lab_growth/docs/USX_FINAL_REPORT_20260918.md) |
| 20260918 | 研究文档 | [P0.2b 美股"历史动态"扩展 — 预注册 (2026-09-18, 用户方向)](alpha_lab_growth/docs/USX_P0_2B_PREREG_20260918.md) |
| 20260918 | 研究文档 | [USX P0 实验发现报告 (2026-09-18)](alpha_lab_growth/docs/USX_P0_FINDINGS_20260918.md) |
| 20260918 | 研究文档 | [USX 独立调研报告 (2026-09-18)](alpha_lab_growth/docs/USX_RESEARCH_SURVEY_20260918.md) |
