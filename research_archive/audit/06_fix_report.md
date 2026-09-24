> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`audit/06_fix_report.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。
> 早期结论请结合后续 27/29 号审计与低频主线复核阅读，不能按单篇旧报告判断当前有效性。

# 步骤 6 — 修复执行报告

**修复日期：** 2026-08-15
**修复原则：** 保留旧版本，创建新版本文件路径；逐项修复 → 测试 → 评估。

---

## 修复清单与结果

| # | 优先级 | 原问题（审计步骤1） | 旧版本 | 新版本 | 修复内容 | 状态 |
|:-:|:--:|------|------|------|------|:--:|
| 1 | P0 | a_share_v16 资金分配循环作用域 bug | `a_share_v16/` | **`a_share_v17/`** | TP/SL 结算移入内层循环，逐只股票结算 | ✅ |
| 2 | P0 | a_share_v15 股票池用全样本末日筛选 + online_expand | `a_share_v15/scripts/adaptive_backtest.py` | `a_share_v15/scripts/adaptive_backtest_fixed.py` | 股票池/板块动量按 `test_start` 截断；删除 online_expand | ✅ |
| 3 | P0 | kcb_v15 online_expand 用测试集训练 | `kcb_v15/scripts/v15_unified_backtest.py` | `kcb_v15/scripts/v15_unified_backtest_fixed.py` | 删除 online_expand | ✅ |
| 4 | P1 | kcb_v16 情绪用当日全日指数收益 + 开盘 ATR 前视 | `kcb_v16/src/backtest/backtest.py` | `kcb_v16/src/backtest/backtest_fixed.py` + `scripts/run_fixed.py` | 情绪改用前一日市场数据；开盘 ATR 改用前一日分钟数据 | ✅ |
| 5 | P2 | kcb_v17 全局模式特征滞后 + 配置重复键 | `kcb_v17/` | **`kcb_v18/`** | 全局模式改用当日特征；清理重复配置键；文档化标签持有期 | ✅ |

---

## 逐项验证

### Fix 1：a_share_v16 → a_share_v17（作用域 bug）

**回测（3-8月，同参数）**：

| 指标 | 旧版 a_share_v16 | 修复版 a_share_v17 |
|------|:--:|:--:|
| 累计收益 | +181.45% | **+439.81%** |
| 总交易笔数 | 175 | **198** |
| 单板块单日最多股票 | 1（bug 只结算最后一只） | **2+** |
| MDD | −6.17% | −6.17% |

**结论**：bug 导致「只结算板块内最后一只股票」但「投入累计全部股票」，日收益被系统性低估。修复后收益 +181%→+440%，说明旧版是**低估**而非高估。

### Fix 2：a_share_v15（股票池前视 + online_expand）

**回测（5月）**：

| 指标 | 旧版 | 修复版 |
|------|:--:|:--:|
| 累计收益 | +32.91% | +29.25% |
| 回撤 | — | −1.68% |

**结论**：股票池改用 `test_start` 截断后收益小幅下降（+32.9%→+29.3%），符合「消除前视后收益回落」的预期。

### Fix 3：kcb_v15（online_expand）

- 代码已删除 `add_state_model(pd.concat([train_df, test_df]))` 路径。
- 5月回测（train 03-01~04-30）中 HMM 两个状态均在训练期出现，online_expand 未触发，故该窗口收益不变（−4.19%）。但**潜在前视风险已消除**（未来市场出现训练期未见的 HMM 状态时不再泄漏）。

### Fix 4：kcb_v16（情绪 + ATR 前视）

**回测（688041，2026-06）**：

| 指标 | 旧版 | 修复版 |
|------|:--:|:--:|
| 总收益 | +9.79% | +9.79% |
| 平均情绪 | −0.04 | **−0.10** |

**结论**：情绪值改变（证明已改用前一日数据），但该区间无 T0 做T交易，故总收益不变。情绪/ATR 前视影响的是**有活跃做T的场景**（如 07-13 被套解套场景），需在带 T0 交易的场景下进一步验证。

### Fix 5：kcb_v17 → kcb_v18（全局特征滞后 + 配置）

**回测（3-7月）**：

| 模式 | 旧版 kcb_v17（当前配置） | 修复版 kcb_v18 | 变化 |
|------|:--:|:--:|:--:|
| sector | +570.97% | +570.97% | 无（sector 模式未改动） |
| global | **+1205.50%**（t-1 滞后特征） | **+285.65%**（当日特征） | 修复滞后后回落 |

**结论**：
- **配置重复键**：`bull_idx20_threshold` 等重复定义，PyYAML 后者覆盖前者，有效值未变，清理后行为一致。
- **全局特征滞后**：原 global 模式用 `cached_last_f`（t-1 特征）预测，与 sector 模式（当日特征）不一致，且训练/预测错位。修复后统一用当日特征，global 收益从 +1205% 回落到 +286%，**说明原滞后特征路径本身是 bug 放大源**。

---

## 修复后残留问题（未修复，如实披露）

1. **同日收盘价假设**（P2，跨版本）：所有日频版本仍假设「t 日收盘特征 → t 日收盘价入场」。这是行业常见 close-to-close 约定，但严格意义上需要「t 日特征 → t+1 日开盘入场」才能完全消除。本次未改，因其是设计约定而非代码 bug，且改动会牵动所有版本。
2. **幸存者偏差**（跨版本）：股票池来自当前 parquet，未验证是否含已退市股。
3. **a_share_v16/v17 全局模式 `n_trades` 计数**：`n_trades` 用分配循环后的 `remaining` 计算，与实际交易笔数不符（仅影响统计打印，不影响收益）。
4. **kcb_v16 分钟缺失时的日线模拟**：`simulate_minute_from_daily` 仍用当日全日 OHLC 反推（仅回退路径）。

---

## 复现命令

```bash
# Fix 1
cd a_share_v17 && python scripts/sector_rotation_backtest.py --test-start 2026-03-01 --test-end 2026-08-11 --output-name v17_fixed_full
# Fix 2
cd a_share_v15 && python scripts/adaptive_backtest_fixed.py --train-start 2026-02-01 --train-end 2026-04-30 --test-start 2026-05-01 --test-end 2026-05-31 --output-name a_share_adapt_may_fixed
# Fix 3
cd kcb_v15 && python scripts/v15_unified_backtest_fixed.py --train-start 2026-03-01 --train-end 2026-04-30 --test-start 2026-05-01 --test-end 2026-05-31 --output-name v15u_may_fixed2
# Fix 4
cd kcb_v16 && python scripts/run_fixed.py --stock 688041 --start 2026-06-01 --end 2026-06-30
# Fix 5
cd kcb_v18 && python scripts/kcb_v17_backtest.py --mode global --train-start 2025-07-01 --train-end 2026-02-28 --test-start 2026-03-01 --test-end 2026-07-31 --output-name v18_global_full
```
