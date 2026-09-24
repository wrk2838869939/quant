> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`audit/18_lookahead_review.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。
> 早期结论请结合后续 27/29 号审计与低频主线复核阅读，不能按单篇旧报告判断当前有效性。

# 步骤 18 — 未来信息泄露专项审查 (扩展因子库 + 月度滚动集成后)

**日期：** 2026-08-22
**范围：** Batch1 扩展因子、月度滚动回测、实盘管线 (realtime_features / patch_board_features / predict)、板块轮动回测

## 一、扫描结果 (无泄露项)

| 检查项 | 结果 |
|---|---|
| shift(−k) 负位移 | ✅ 无 (标签 fwd 计算除外，仅训练用) |
| rolling center=True 双向窗 | ✅ 无 |
| 训练窗边界 | ✅ 全部严格 `< date` / `< TODAY` |
| future_ret 标签混入特征 | ✅ 无 |
| patch_board_features 大类特征 | ✅ ret_mat.loc[:d] 截断正确，指数 vol 同日截断 |
| predict 脚本 | ✅ 历史 parquet 截断 + 今日快照 (14:30 决策用当日信息合法) |
| 月度滚动 IC 因子选择 | ✅ 逐月仅用前 3 个月 (walk-forward) |
| regime 判定 | ✅ 滚动窗因果；实盘含今日快照合法 |

## 二、修复项: breadth 未来函数 (本次唯一实锤)

**问题**: `a_share_v17/src/rotation/sector_backtest.py:405` 和 `kcb_v18/scripts/kcb_v17_backtest.py:326` 调用
`SectorScorer(sector_ret_mat.loc[:date], df, ...)` 传全量 df → `_breadth()` 取 `df["日期"].max()` 起算，
历史上每个调仓日都用了**数据集最后 10 天**的板块广度。

**修复**: 传 `df[df[date_col] <= date]`。predict 脚本天然正确 (数据截止当日)，无需改。

**影响量化** (A股 2026-01~08-19 重跑):

| 方案 | 修复前 (buggy) | 修复后 (fixed) |
|---|---|---|
| A股 基线 (周频26因子) | +1267% / MDD −4.1% | **+1247% / MDD −6.5%** |
| A股 月度+扩展因子 | +1054% / MDD −4.4% | **+497% / MDD −5.9%** ⚠️ 大幅缩水 |

- 基线影响小 (−20pp)；**月度+扩展因子方案收益腰斩** — 该方案的 A股回测之前明显被 breadth 泄露虚增。
- KCB 全局模式不用 SectorScorer，不受影响；KCB sector 模式未用于生产。
- 结论修正: A股 维持周频旧池的决策更加稳固；`sector_rotation_extraf.yaml` (50因子) 修复后增益存疑，暂不切换。

## 三、发布滞后 (EOD 数据当日收盘后才发布)

| 数据 | 字段 | 处理 |
|---|---|---|
| daily_basic | turnover_rate_f / pe_ttm / pb | 回测未滞后版偏乐观; **KCB lag1 已验证 +1572%**(vs 未滞后 +2076%, 保留 ~3/4 超额); A股扩展配置未测 lag1 (未上实盘) |
| moneyflow | 分级资金流 | 同上 |
| 实时管线 | EOD 类因子复用昨日值 | ✅ 与 lag1 口径一致 |

## 四、已知残留口径水分 (非泄露, 但使回测偏乐观)

1. **日内高低顺序假设**: TP/SL/移动止盈用日线高低点判断触发顺序，实盘可能更差;
2. **amihud 口径**: 回测用全日成交额，实盘 14:30 只有部分量 (轻度);
3. **边界 bug (已修)**: 回测窗口最后一日无 T+1 卖出窗口时仍交易，导致「同日买收盘、卖当日高点」时间倒置 (曾虚增 KCB 8/19 +10%); 已在 kcb_v17_backtest.py / sector_backtest.py 修复 (末日不交易);
4. **7 月成交额单位断档** (07-01~07-10 全 0 + 千元/元混用): amount_ma20/vol_ma20 在 7 月窗口受污染，未修;
5. **因子收录样本内**: Batch1 按 |IC|≥0.01 收录的窗口与部分回测窗口重叠 (A股全重叠); 月度滚动方案内部选择已 walk-forward，但**因子池构成**仍是全局选出的 34 个——真正的纯 OOS 需要池子也按时间窗滚动筛选，暂未完成。

## 五、结论

除 breadth（已修）外未发现新的未来信息泄露。当前最可信的实盘预期口径：
**KCB 月度+扩展因子 lag1: +1572%/8个月 (回测), MDD −9.4%**；A股 周频基线: +1247%/8个月, MDD −6.5%。
所有回测数字仍含日内成交假设水分，实盘预期应再打 3~5 折。

## 复现

- 修复: `sector_backtest.py:405` / `kcb_v17_backtest.py:326` (传截断 df)
- 修复后回测: `a_share_v17/outputs/roll_baseline_breadthfix` / `roll_monthly_extraf_breadthfix`
