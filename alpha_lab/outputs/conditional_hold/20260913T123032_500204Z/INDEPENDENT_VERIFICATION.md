# 条件退出独立核验

状态：complete。

30账户、1591笔真实成交，85项倒序CSV、306项原对照CSV逐值通过。

独立重建172列、57review的五种许可和长期状态；285项前缀验证；baseline/persist10原对照保持精确。

允许度：persist10≤accelerated≤baseline≤patient，accelerated≤hybrid≤patient；hybrid与baseline无统一顺序。未知掩码和全A接口保持原样，确认计数不因分类切换重置。

矩阵及数值边界见independent_conditional_matrix.json、independent_conditional_numerical.json；账户明细见execution/independent_conditional_verification.json。

账户核验通过不代表政策有效；本轮是观察既有历史后的适应性研究，主假设hybrid不能事后改为最高收益分支。

原对照计数拆分：baseline与persist10共12账户，对上轮hold_sensitivity逐值比较204项CSV；baseline六账户再对最初修复recent比较102项，合计306项。不是新增了18个策略账户。五组真实倒序各核验全部17项CSV，并检查实际selection_first接收端逐次反序证据。

月度长期分类的numpy与冻结pandas符号比较无边界差异。继承的每日原趋势仍存在上轮已披露的一格均线机器精度边界；[追加证据](inherited_daily_boundary_effect.json)确认将其翻转只使弱计数18→19，不改变本轮>=10谓词、月度计数或长期状态，五种hold许可均不受影响。没有更改冻结策略或重跑额外反事实账户。

10项新独立测试已通过，并纳入主agent完成的310项全套测试。此后仅追加报告与独立证据，策略和测试源码未再修改，无需重复完整suite。

逐日原价估值共109,600次，stale=0；首次退出生命周期576项、原monthly Top20入场726项。最大现金误差2.92e-11元、总资产误差2.91e-11元。
