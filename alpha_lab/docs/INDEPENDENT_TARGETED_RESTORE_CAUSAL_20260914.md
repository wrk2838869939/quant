# 定向恢复：独立因果执行审查

审查对象为 `targeted_restore_engine.py` 与 `restore_quota.py`，对照 `portfolio_cash_reentry_engine.py` / `portfolio_cash_state.py`。只读源码审查与独立合成场景；没有引用研究收益结论，没有操作实盘账户、邮件或密钥。

## 审查裁定

修复后，所测试的恢复额度、现金预约、五席位、整手、T+1、正式退出优先与前缀因果性通过。当前证据支持研究引擎的局部实现正确性，不能替代实盘成交链路接入验收或策略盈利能力证明。

## 发现并复验的问题

1. **总组合预算遗漏（已修复）**：初版恢复只检查单股上限和现金，半仓状态可从 5,000 元股票资产恢复至 9,000 元，超过 10,000 元净值的一半。当前 `targeted_restore_engine.py:626` 使用 `min(cash, equity*exposure_cap-stock_value)-restore_reserved`；入口还拒绝不兼容的 fixed_half 状态。反例改为通过。
2. **缺失恢复收益被误判为合格（已修复）**：`NaN <= 0` 为假，初版 known=True、ret20=NaN 时仍恢复 400 股。当前 `:613`—`:616` 要求 known/weak 非缺失且 ret20 有限并大于零。反例改为通过。
3. **无成交证据额度可从快照加载（已修复）**：初版 `epoch=1, shares={000001.SZ:500}, receipts={}` 可加载。当前 `restore_quota.py:13`—`:23` 校验成交证据结构，并要求剩余额度不超过本轮真实风控卖出减恢复买入；反例改为通过。该验证不是防篡改签名，实际接入仍需与账户成交账本核对。
4. **月度退出被待成交风险减仓遮蔽（另一审查者发现，本审查独立复验通过）**：当前 `targeted_restore_engine.py:515`—`:519` 使 restore_enabled=True 时正式 hold_false 优先于风险减仓 pending。独立场景连续两日卖不成、第三次月度复核明确退出，仍生成正式退出而不恢复。

## 独立检验

新增独立脚本 `/tmp/audit_targeted_restore.py`，15/15 通过：

- 关闭 restore_enabled 时，与旧引擎 daily / trades / holdings / events 精确一致。
- 真实减掉 400 股才可恢复 400 股；恢复日新增 400 股不可卖。
- 五只股票同时恢复仍保持五席位，买单整手且全部预约不超过前日现金。
- 清仓不能通过旧恢复额度重新入场；未知 hold 或缺失恢复证据不能恢复。
- 漏过恢复限价单后不在后续每一天自动追买。
- 部分成交分笔累计、重复回报幂等、快照重载、轮次失效、非法快照拒绝。
- 明日开盘突变不改变今日恢复订单；未来风险特征突变及截断未来账户区间不改变此前账本。
- 组合总上限和正式退出优先的回归反例。

另把原 25 项因果执行对抗测试整体替换为新引擎和其 AccountingError，25/25 通过。总计 40 项；不是新增 40 个互不相关的经济学实验。

运行命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /tmp/audit_targeted_restore.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /tmp/audit_targeted_restore_existing.py
```

日志：`/tmp/audit_targeted_restore_final.log`、`/tmp/audit_targeted_restore_existing_final.log`。

## 实盘范围限制

审查时 `shadow_inputs.py`、`shadow_planner.py`、`shadow_account.py` 和 `manual_execution.py` 均未引用新恢复引擎或 RestoreQuota。因而本次通过不表示现有邮件／人工成交路径已经支持恢复。上线前至少需要真实成交回报驱动额度、额度状态持久化及重启一致性、持仓和现金对账，以及与冻结研究规则的订单一致性。日线全量成交假设也不能保证人工订单能成交。

源文件 SHA256（源码继续修改后需重新验证）：

- targeted_restore_engine.py: `e00ab181992825574761630272224b8f683cb6adabaae117c7f2f5b6873bf4fc`
- restore_quota.py: `b2ca6fc4025ead4fc9c42ccec475457b52895e5e0ef90e3c14d8d40a44d5e2c1`
- 根 AGENTS.md 保持 `82928cd87d095193da9adc18307823f6db6f4a425455146f8f4b45ba0c6e8ad6`。
