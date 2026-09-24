# 从问题到修正：六条研究过程

这里串联原始文档与实验脚本。每条路径都保留起点、对照、遇到的问题和后续修正，便于理解研究实际推进了什么。数字和结论仍以各次实验的条件为限，本次整理不重新认证历史成绩。

## 1. 机器学习选股：从高回测表现到标签与执行审计

早期模型把截面预测、行业轮动与退出规则组合起来，但优秀的历史曲线无法回答信息是否可得、订单是否能成交，以及是否使用了不合理的资金口径。

研究依次检查前向标签的成熟边界、涨停买入、现金约束与日内高低点顺序。随后使用分钟回放和机制剥离，将预测、过滤、择时和退出的作用拆开。过程说明了为什么需要撤回旧结论，以及单独的过滤规则为何不能自动变成选股能力。

原始记录：

1. [前视检查](../research_archive/audit/01_future_function.md)
2. [分钟级回放](../research_archive/audit/21_minute_replay.md)
3. [诚实口径总清算](../research_archive/audit/27_honest_v18.md)
4. [机制剥离实验](../research_archive/audit/29_mechanism_ablation.md)
5. [标签泄露消融脚本](../research_archive/audit/scripts/ic_leak_ablation.py)

## 2. 自动因子挖掘：计算一致性与因子发现期污染

低频主线从文献与数据建设开始，引入表达式树、遗传规划、GPU 计算和候选筛选。计算速度提高之后，算子缺失值和并列秩的一致性成为必要检查项。

因子进入组合后，再用风格匹配与年度滚动检验检查表面超额。后续消融指出，训练权重按时间前推，并不能消除因子发现阶段已经观察验证年份带来的污染。该过程把“算法实现完成”“统计结构存在”和“未见样本净技能”分开讨论。

原始记录：

1. [初始调研](../research_archive/FACTOR_MINING_SURVEY_20260908.md)
2. [CPU/GPU 算子一致性](../research_archive/alpha_lab/docs/OPERATOR_PARITY.md)
3. [年度滚动预注册](../research_archive/alpha_lab/docs/WALKFORWARD_OOS_PREREG_20260918.md)
4. [年度滚动报告及后续消融](../research_archive/alpha_lab/docs/WALKFORWARD_OOS_REPORT_20260919.md)
5. [GP 挖掘入口](../research_archive/alpha_lab/scripts/run_gp_mining.py) · [消融脚本](../research_archive/alpha_lab/scripts/walkforward_ablation.py)

## 3. 现金与持仓：状态修复不等于策略增益

小资金组合研究关注排名变化、首次建仓日期、主动减仓和风险解除后的恢复。过程中发现，持股席位已占满但仓位不足时，仅允许新股票入场无法恢复被削减的旧仓。

修正恢复额度和实际成交状态后，继续检查卖出与买回之间的经济性、额外换手和错过反弹的代价。账户机制修复可以成立，而策略收益改进仍不成立。这些记录保留了从异常账户到机制解释、修复和再次验证的完整路径。

原始记录：

1. [首次建仓日期敏感性](../research_archive/alpha_lab/docs/STARTUP_TIMING_RESEARCH_20260914.md)
2. [选择持续性与排名缓冲](../research_archive/alpha_lab/docs/SELECTION_STABILITY_RESEARCH_20260914.md)
3. [组合现金政策](../research_archive/alpha_lab/docs/PORTFOLIO_CASH_VALIDATION_20260914.md)
4. [定向恢复验证](../research_archive/alpha_lab/docs/TARGETED_RESTORE_VALIDATION_20260914.md)
5. [主动减仓经济性](../research_archive/alpha_lab/docs/CASH_ECONOMICS_RESEARCH_20260914.md)

## 4. 从预测到策略：目标函数和账户反馈的变化

先比较因子加权、线性与树模型、序列和状态模型，再追问预测误差降低是否意味着扣费后账户收益改善。随后将研究对象改为直接读取市场和账户状态、输出动作的可训练策略。

以基线初始化后，开放排序、持股、调仓与现金动作；使用演化搜索评价真实约束下的交易轨迹，并联合考察训练窗口、训练代数及重训间隔。保留训练改善但后续表现下降的情况，不把训练收敛作为策略有效的证据。

原始记录：

1. [学习策略实验设计](../research_archive/alpha_lab/docs/LEARNING_STRATEGY_SEARCH_20260920.md)
2. [CNN/HMM 对照](../research_archive/alpha_lab/docs/SEQUENCE_STATE_LEARNING_RESULTS_20260920.md)
3. [完整策略学习目标](../research_archive/alpha_lab/docs/BASELINE_INITIALIZED_POLICY_LEARNING_20260920.md)
4. [收敛诊断](../research_archive/alpha_lab/docs/POLICY_CONVERGENCE_RESULTS_20260920.md)
5. [窗口—迭代—重训联合实验](../research_archive/alpha_lab/docs/POLICY_BALANCE_RESULTS_20260921.md)
6. [联合实验运行器](../research_archive/alpha_lab/scripts/run_policy_balance.py)

## 5. 美股到 A 股：解释跳空与可交易预测之间的距离

研究美股指数、行业 ETF 和中概股是否可以用于 A 股行业轮动。实验分别观察隔夜跳空、开盘后的日内及多日收益，并扩展到多周期、市场状态和长假累积等条件。

过程中的关键检查是信号完成时间与入场价格匹配：如果信号使用当日收盘数据，标签不能从当日开盘开始。随机对照和毛净收益分解进一步用于检查短持有期下的成本负担。

原始记录：

1. [研究调查](../research_archive/alpha_lab_growth/docs/USX_RESEARCH_SURVEY_20260918.md)
2. [方案设计](../research_archive/alpha_lab_growth/docs/USX_DESIGN_20260918.md)
3. [扩展假设预注册](../research_archive/alpha_lab_growth/docs/USX_P0_2B_PREREG_20260918.md)
4. [阶段发现](../research_archive/alpha_lab_growth/docs/USX_P0_FINDINGS_20260918.md)
5. [最终报告](../research_archive/alpha_lab_growth/docs/USX_FINAL_REPORT_20260918.md)

## 6. 板块形态：从主观描述到事件与模糊模型

先解决历史成员、板块指数连续性与覆盖问题，再将上涨、回落到参照区域和下跌减速写成明确事件规则。不同窗口识别出的事件需要去重，个股成员的变化也需要单独诊断。

筛选层比较固定模糊规则、岭回归与可训练模糊网络。时间滚动和随机两半互换回答不同问题；后者需要保留同日与重叠收益窗口的关联诊断。随后用冻结模型进行新日期扫描，避免根据当天候选数量临时放松规则。

原始记录：

1. [板块数据规格](../research_archive/alpha_lab/docs/SECTOR_DATASET_V1.md)
2. [内部联动研究](../research_archive/alpha_lab/docs/SECTOR_COHESION_RESULTS_20260921.md)
3. [模糊事件筛选预注册](../research_archive/alpha_lab/experiments/sector_fuzzy_v1/PREREG.md)
4. [随机互换协议](../research_archive/alpha_lab/experiments/sector_fuzzy_v1/RANDOM_PREREG.md)
5. [随机互换结果与重叠诊断](../research_archive/alpha_lab/run_records/sector_fuzzy_random_halves/20260923T111853_561924Z/README.md)
6. [最新扫描开发记录](../research_archive/alpha_lab/experiments/sector_live_v1/README.md)

更多账户研究、财务时点检查、执行接口和失败分支见 [完整档案索引](../research_archive/README.md)。
