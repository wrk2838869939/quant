# 源码阅读路径

## 1. 数据查询与特征

- [hub.py](../datahub/src/datahub/hub.py)：A 股、美股、指数和元数据接口。
- [io_utils.py](../datahub/src/datahub/io_utils.py)：分区读取、字段选择和缓存。
- [indicators.py](../datahub/src/datahub/indicators.py)：技术指标计算与所需字段声明。
- [frozen.py](../datahub/src/datahub/frozen.py)：快照、文件清单与完整性校验。
- [test_research_contracts.py](../datahub/tests/test_research_contracts.py)：独立数据根、查询起点一致性、未来前缀与行业冲突测试。

## 2. 因子表达式与搜索

- [expr_engine.py](../alpha_lab/src/expr_engine.py)：表达式树、截面与时序算子。
- [gp_miner.py](../alpha_lab/src/gp_miner.py)：种群、交叉、变异、适应度和候选收割。
- [gpu_ops.py](../alpha_lab/src/gpu_ops.py)、[gpu_eval.py](../alpha_lab/src/gpu_eval.py)：PyTorch 后端。
- [rank_ic.py](../alpha_lab/src/rank_ic.py)、[neutralize.py](../alpha_lab/src/neutralize.py)：成对有效样本的秩相关和风险暴露处理。
- [ledger.py](../alpha_lab/src/ledger.py)：实验记录及有条件的 DSR 计算。
- [test_operator_parity.py](../alpha_lab/tests/test_operator_parity.py)：并列秩、NaN、窗口边界与后端一致性。

## 3. 账户与学习

- [portfolio.py](../alpha_lab/src/portfolio.py)：订单、现金、持股和公司行动。
- [audit_monthly_portfolio.py](../alpha_lab/scripts/audit_monthly_portfolio.py)：本归档仅保留独立 reconcile 函数。
- [learning_search.py](../alpha_lab/src/learning_search.py)：成熟训练样本与预测/排序模型。
- [sequence_state_learning.py](../alpha_lab/src/sequence_state_learning.py)：序列网络及在线状态推断。
- [initialized_policy.py](../alpha_lab/src/initialized_policy.py)：神经策略、初始参数与动作表达。
- [policy_portfolio.py](../alpha_lab/src/policy_portfolio.py)：使用已结算账户状态的策略回放。
- [policy_evolution.py](../alpha_lab/src/policy_evolution.py)：候选扰动、更新、停滞判断与随机流恢复。
- [test_initialized_policy.py](../alpha_lab/tests/test_initialized_policy.py)：账户反馈、动作边界和基线等价检查。

## 4. 板块与事件筛选

- [sector_dataset.py](../alpha_lab/src/sector_dataset.py)：历史成员、前日权重、限权和有效段。
- [sector_shape_events.py](../alpha_lab/src/sector_shape_events.py)：形态状态及事后结果计算，二者应分开使用。
- [sector_multiscale_events.py](../alpha_lab/src/sector_multiscale_events.py)：多窗口信号合并。
- [model.py](../alpha_lab/experiments/sector_fuzzy_v1/model.py)：训练期缩放、成熟标签隔离及模糊网络。
- [test_model.py](../alpha_lab/experiments/sector_fuzzy_v1/test_model.py)：缩放泄露、标签边界、非单调表达及训练可重复性。

## 历史依赖

核心依赖涉及 NumPy、pandas、SciPy、PyArrow、PyYAML、joblib、scikit-learn、PyTorch、LightGBM 和 hmmlearn；部分算子可使用 bottleneck。不同实验的输入和依赖并不相同，本归档不提供统一部署环境。
