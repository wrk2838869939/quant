> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`audit/01_future_function.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。
> 早期结论请结合后续 27/29 号审计与低频主线复核阅读，不能按单篇旧报告判断当前有效性。

# 步骤 1 — 未来函数排查报告

**审计日期：** 2026-08-15
**审计范围：** kcb_v15 / kcb_v16 / kcb_v17 / a_share_v15 / a_share_v16
**审计方法：** 逐行阅读数据加载、特征工程、标签构造、训练/测试切分、回测循环代码，追踪每个特征/标签的时点归属。

---

## 结论速览

| 版本 | 未来函数风险等级 | 关键问题 |
|------|:---:|----------|
| kcb_v17 | 🟡 中 | 全局模式特征滞后不一致；同日收盘价假设 |
| a_share_v16 | 🔴 高 | **资金分配循环作用域 bug（结果不可信）**；同日收盘价假设 |
| a_share_v15 | 🔴 高 | **股票池用全样本末日筛选（前视）**；**online_expand 用测试集训练** |
| kcb_v15 | 🔴 高 | **online_expand 用测试集训练** |
| kcb_v16 | 🟠 中高 | 情绪用当日全日指数收益（盘中前视）；开盘 ATR 用前 30 根 K 线 |

> 图例：🟢 无 / 🟡 轻微（约定问题）/ 🟠 中等 / 🔴 严重（结果不可信）

---

## 一、共享数据管线（所有日频版本）

### 1.1 基础日频特征（`kcb_sniper/src/features/daily_features.py`）

| 检查项 | 结论 |
|--------|------|
| 技术指标（MA/MACD/RSI/KDJ/BOLL/ATR） | ✅ 全部 `rolling`/`ewm`/`pct_change`，严格后视 |
| `ret_1d`/`ret_5d`/`ret_10d`/`ret_20d` | ✅ `pct_change(window)` 后视 |
| `amplitude = (high-low)/close.shift(1)` | ⚠️ 用到**当日**最高/最低价（收盘后才能知道） |
| `add_relative_features` 的 z-score/rank | ✅ 按「日期」分组截面计算，无跨期泄漏 |
| `mkt_up_ratio`（市场广度） | ⚠️ 当日截面，收盘后可知 |

**结论**：基础特征无跨期未来函数，但**广泛使用当日 OHLC**（`ret_1d`、`ma20_dev`、`amplitude`、`mkt_up_ratio` 等）。这隐含「收盘时已知全量当日数据」的假设——是行业标准做法（close-to-close 回测），但若实盘在收盘前下单则不成立。

### 1.2 大类相关性特征（`kcb_v15/scripts/build_v15_dataset.py`）

- **聚类映射**：逐月末用过去 60 日收益重聚类，次月使用该映射 → ✅ **设计上已消除前视**。
- **`cat_rank_5d_clu`**：`cat_ret_5d.rank(axis=1)` 截面排名 → ✅ 截面，无未来。
- **缺失填充**：`groupby("日期").median()` 同日截面中位数 → ✅ 无未来。
- **`future_ret` 标签**：`closes[end]/closes[i]-1`，`end = min(i+hold, n-1)` → ✅ 正确的前向标签。

### 1.3 标签 `future_ret` 与特征池隔离

- 所有版本的 `stock_selection.features` 列表均**不含** `future_ret`（已核对 kcb_v17.yaml / sector_rotation.yaml）→ ✅ 标签不会作为特征泄漏进模型。

### 1.4 幸存者偏差（跨版本风险）

- 股票池来自**当前** `daily_features_v15.parquet`，疑似不含已退市股票。**未验证**，需数据源（Tushare 是否拉取退市股）确认。⚠️

---

## 二、kcb_v17（v17.14）

### 逐项核查

| 位置 | 结论 |
|------|------|
| `_prepare_data` 的 `future_ret` 标签 | ✅ 前向标签，末尾 `hold` 行置 NaN |
| `index_vol_20d` | ✅ `rolling(20).std()` 后视 |
| `classify_regime_v17` 冷却期 `i20.iloc[start:i]` | ✅ 只看过去 |
| 训练池 `train_pool = (date >= cutoff) & (date < date)` | ✅ 严格不含当日及未来 |
| 特征缺失填充 `df[c].median()` | ✅ 仅训练窗口内中位数 |
| 行业收益 `build_sector_returns` | ✅ 用 `ret_1d` 按日截面 |

### 发现的问题

1. **🟡 全局模式特征滞后不一致**（`run_global`）：
   ```python
   cached_last_f = tp.groupby(code_col).last().reset_index()
   for fc in cached_sel_f:
       fm = dict(zip(cached_last_f[code_col], cached_last_f[fc]))
       ts[fc] = ts[code_col].map(fm).fillna(0)
   ```
   模型用「t 日特征 → t 日未来收益」训练，但预测时喂入的是 **t-1 日特征**（`cached_last_f` = 训练池最后一行）。这不是未来函数（用了更旧的数据，偏保守），但造成**训练/预测特征错位**。且与 sector 模式（用当日特征）不一致。疑为防御「当日特征收盘后才可得」而有意滞后，但缺乏说明。

2. **🟡 训练标签持有期与实盘持有期不一致**：
   `future_ret` 用 `data.hold_days=5`（5 日前向收益）训练模型，而回测实际持有 `regime_params.hold_days`（bull=3 / neutral=1 天）+ 止盈止损。模型在优化 5 日收益，交易却按 1-3 日执行。

3. **🟡 同日收盘价假设**：`today_sector` 用当日 `ret_1d/ma20_dev/amplitude` 特征，`price = 当日收盘`，入场价=当日收盘。隐含「收盘后决策并收盘价成交」。

4. **🟡 配置重复键**：`kcb_v17.yaml` 中 `bull_idx20_threshold` 出现两次（0.05 → 0.015，后值生效）、`bull_breakdown_bm`/`panic_*` 亦重复。YAML 后者覆盖前者，易误读。

**判定：kcb_v17 未发现硬性未来函数，但存在特征滞后不一致与标签持有期错位，需修正后才能严谨采信 +435% 回测。**

---

## 三、a_share_v16（v16.4）

### 逐项核查

与 kcb_v17 相同的标签/训练切分/中位数填充逻辑均 ✅。但发现**严重代码缺陷**：

### 🔴 严重：资金分配循环作用域 bug（`sector_backtest.py`）

```python
for sector, picks in stock_picks.items():
    picks = picks.sort_values("pred_ret", ascending=False)
    remaining = capital_per_sector
    for _, row in picks.iterrows():      # ← 内层循环
        code = row[self.code_col]
        price = row[self.close_col]
        ...
        remaining -= cost
        total_invested += cost
    # ← 内层循环结束
    code_hold = hold_data[... == code]   # ← 用的是内层循环最后一行 code/price
    ... # TP/SL 判定、pnl 计算、all_trades.append
```

TP/SL 判定与 `all_trades.append` 位于**内层 `for row` 循环之外**，因此：
- 每个板块**只有最后一只股票**（排序后 pred_ret 最低者）参与收益结算与交易记录；
- `total_invested` 累计了全部股票成本，但 `total_pnl` 只反映最后一只股票 → **日收益被系统性扭曲**。

这不是未来函数，但**直接导致 +181% 回测结果不可信**，必须先修复再进入后续 alpha/IC/IR 分析。

### 其他发现

- `compute_lgbm_scores`/`_build_sector_features` 存在但**未被回测调用**（死代码），其「用 t 日特征预测 t+1 日板块收益」的设计本身无前视，但影响不了当前回测结果。
- 🟡 同日收盘价假设（同 kcb_v17）。

---

## 四、a_share_v15（adaptive v3）

### 🔴 严重 1：股票池用全样本末日筛选（前视）

`dynamic_stock_pool(all_df, ...)` 与 `identify_sector_momentum(all_df, ...)` 中：

```python
recent = pool[pool["日期"] >= pool["日期"].max() - pd.Timedelta(days=60)]
#                           ^^^^^^^^^^^^^^^^^^ 全样本最大日期（含未来测试期）
```

`pool["日期"].max()` 是**全量数据集（含未来）的最后一天**。因此：
- 「近 60 日成交额前 N 名」股票池的构建用到了**测试期之后**的数据；
- 「动量板块识别」同样用了全样本末日。

这是经典的**组合构建前视偏差**：未来的成交额/动量信息泄漏进了测试期选股池。

### 🔴 严重 2：`online_expand` 用测试集训练模型

```python
if cfg["market_state"].get("online_expand", True):
    for st in test_df["state"].dropna().unique():
        if st not in selector.models:
            selector.add_state_model(pd.concat([train_df, test_df]), st)
```

当测试期出现训练期未见的 HMM 状态时，用 **train+test 合并数据**训练新模型 → **测试集标签直接参与训练**，硬性前视。

### 其他

- 🟡 同日收盘价假设（同前）。
- V4 预筛选（`apply_v4_filter`）传入 `all_df` 但用后视 rolling → ✅ 无前视。

---

## 五、kcb_v15（v15.1 unified）

### 🔴 严重：`online_expand` 用测试集训练

与 a_share_v15 完全相同：
```python
selector.add_state_model(pd.concat([train_df, test_df]), st)
```
→ 测试集数据泄漏进训练。

### 其他

- `apply_v4_filter` 传入 `all_df` 但 `rolling` 后视 → ✅ 无前视。
- `build_market_factors`/`classify_regime` 全为 `rolling` 后视 → ✅。
- 🟡 同日收盘价假设。

---

## 六、kcb_v16（日内做T）

规则型分钟级策略，未来函数形态不同：

| 位置 | 结论 |
|------|------|
| 分钟主循环 `current_slice = minute_df.iloc[:idx+1]` | ✅ 逐分钟时点正确 |
| `prev_close` | ✅ 用前一日收盘 |

### 发现的问题

1. **🟠 情绪分用当日全日指数收益/广度**（`_get_market_sentiment`）：
   ```python
   idx1 = row.get("index_ret_1d")   # 今日指数收益（收盘才知）
   mkt_up = row.get("mkt_up_ratio") # 今日全市场上涨占比（收盘才知）
   ```
   该情绪分在**盘中全程**用于做T方向决策，但 `index_ret_1d`/`mkt_up_ratio` 是当日收盘后才确定的量 → **盘中前视**。

2. **🟠 开盘 ATR 用前 30 根 K 线**（`_compute_initial_atr`）：
   ```python
   highs = minute_df["最高"].values[:30]  # 前 30 根 = 2.5 小时
   ```
   开盘方向决策（`decide_opening_direction`）在开盘即调用，却用了未来 2.5 小时的分钟数据计算 ATR → **前视**。

3. **🟠 分钟数据缺失时用日线模拟分钟**（`simulate_minute_from_daily`）：模拟分钟路径由**当日全日 OHLC** 反推，天然含未来信息（仅回退路径）。

---

## 七、修复建议（按优先级）

| 优先级 | 问题 | 修复 |
|:--:|------|------|
| P0 | a_share_v16 资金分配循环作用域 bug | 将 TP/SL 结算与 `all_trades.append` 移入内层 `for row` 循环 |
| P0 | a_share_v15 股票池用全样本末日 | 股票池/板块动量改为按 `test_start` 截断（只用测试期前数据） |
| P0 | a_share_v15 / kcb_v15 `online_expand` | 删除 `add_state_model(pd.concat([train_df, test_df]))`，改用「最近训练窗口」或直接返回 0 |
| P1 | kcb_v16 情绪用当日全日指数收益 | 改用「昨日」的 index_ret_1d / mkt_up_ratio（shift(1)） |
| P1 | kcb_v16 开盘 ATR 前视 | ATR 只用「截至当前 bar」的分钟数据，或用前一日 ATR |
| P2 | kcb_v17 全局模式特征滞后不一致 | 统一 sector/global 的特征时点（明确定义用 t 日还是 t-1 日特征） |
| P2 | kcb_v17 标签持有期错位 | 训练标签 `future_ret` 的持有期与交易持有期对齐（如都用 5 日或都用 1 日） |
| P2 | 同日收盘价假设 | 若要求严格，改为「t 日特征 → t+1 日开盘入场」并实测滑点 |
| P3 | 配置重复键 | 清理 kcb_v17.yaml 重复键 |

---

## 附：核查方法复现

```bash
# 已核查文件
kcb_sniper/src/features/daily_features.py      # 基础特征
kcb_sniper/scripts/build_features.py           # 特征流水线
kcb_v15/scripts/build_v15_dataset.py           # 大类特征 + 聚类
kcb_v15/src/features/category_features.py      # 相关性特征
kcb_v15/src/market/regime.py                   # regime 判定
kcb_v15/scripts/v15_unified_backtest.py        # v15 回测
kcb_v15/src/selection/__init__.py              # V4 过滤
kcb_v16/src/backtest/backtest.py               # v16 日内回测
kcb_v17/scripts/kcb_v17_backtest.py            # v17 回测
kcb_v17/src/rotation/sector_scorer.py          # 板块评分
a_share_v15/scripts/adaptive_backtest.py       # A股自适应
a_share_v16/src/rotation/sector_backtest.py    # A股两层轮动
a_share_v16/src/rotation/sector_scorer.py      # A股板块评分
kcb_sniper_v14/src/data/loader.py              # DailyDataLoader
kcb_sniper_v14/src/models/selector.py          # StateAwareSelector
```
