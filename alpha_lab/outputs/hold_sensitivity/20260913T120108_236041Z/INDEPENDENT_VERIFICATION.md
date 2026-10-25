# 退出确认敏感性独立核验

状态：complete。

36账户、2450笔真实成交、131520次原价收盘估值检查；102项真实倒序CSV、102项原基线CSV对照通过。

持有矩阵由冻结原价用numpy重建，172原Top20并集代码之外保持原hold；所有原月度review前缀逐一核验，候选/预测/进入条件未改变。

退出请求逐hold许可和订单生命周期检查；买入限定原月度、原Top20与原趋势恢复门，pending继续每日重试不等于每日选股。

矩阵：independent_hold_matrix_verification.json；账户：execution/independent_hold_sensitivity_verification.json；成交：execution/independent_hold_sensitivity_fill_checks.csv。

核验的是账本与因果实现，不认证政策有效；persist10为已登记主假设，不能择优用persist5/15替换。

许可顺序已在全部9,616个已知月度review单元逐格验证：`fast ≤ persist5 ≤ persist10 ≤ persist15 ≤ baseline ≤ slow`，五组相邻比较均零违反，未知掩码完全相同。见[顺序证据](independent_hold_order_verification.json)。persist系列保留原两次确认兜底，在同一股、同一外生历史状态下只会加速或同日确认，不会晚于baseline；各自账户后续重入路径不同，不能据此直接排列实际成交日或收益。

数值边界已单列：[敏感性证据](matrix_numerical_sensitivity.json)。605296.SH在2025-10-15的复权收盘为41.65625，独立numpy120日均值为41.65625，冻结pandas均值为41.65624999999999，因此严格大于比较相差一格。核验保留冻结pandas数值语义，并保留首次失败日志`independent_hold_matrix_numpy_boundary_attempt.log`。将该日趋势单独翻为false，只使2025-11-03的20日失败计数18→19；六种hold矩阵均零单元变化，持有政策的实际账户决策输入完全不变。未改原入场信号、数据或策略，未追加反事实账户回放。

矩阵核验共3,084日×172代码、1,060,896个趋势/可观察单元、58,824个六政策许可单元；57个原review逐一截断，六政策共342项前缀对照通过。基线有效完整hold与原全A矩阵精确相同，其余变体只覆盖172列，其他列和值保持原样。

11项独立回归测试已通过，已纳入主agent完成的300项全套测试；本核验之后没有修改策略或测试源码，无需因补写说明重复测试。账户核验通过只确认实现与账本，风险/收益折中及跨窗不稳需按预登记主假设另行解释。
