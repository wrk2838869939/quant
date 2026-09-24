> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`audit/10_realtime_features.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。
> 早期结论请结合后续 27/29 号审计与低频主线复核阅读，不能按单篇旧报告判断当前有效性。

# 步骤 10 — 实盘预测脚本：当日实时特征修复

**日期：** 2026-08-16
**背景：** 用户实盘检查发现——预测脚本原用「昨日特征」选股，会导致**追高被套**（昨日特征描述的是「昨天已涨」的股票，今天追进去正好接盘）。这与审计阶段发现的「科创板动量反转」结论一致。

---

## 一、问题与修复

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| 特征时点 | 用**昨日**（parquet 最后一天）特征 | 用**当日 14:30 实时快照**（现价≈收盘）重算技术特征 |
| KCB 训练标签 | 1 日前向收益（与回测 5 日不一致） | 改为 5 日（对齐回测 + 去噪更优） |
| 大类相关性特征 | 昨日值 | 昨日值（20~60 日滚动慢变量，单日误差可忽略） |

## 二、特征处理策略

`compute_today_features(df_hist, realtime, cfg, today, idx_ret_today)`：

| 特征类型 | 处理 |
|---------|------|
| 技术特征：ma20/60_dev、ret_10d/20d、macd_dif/dea、atr14_ratio、amplitude、amount_ma20、vol_ma20 | **用当日实时快照重算**（现价=收盘，拼接 120 天历史后滚动计算） |
| 市场特征：index_ret_1d/5d/20d、mkt_up_ratio | 当日实时指数 + 实时涨跌比 + 历史滚动 |
| 大类相关性：corr_sw/clu_*、rel_str_*、resid_sw_5d、cat_ret/rank/disp | **复用昨日值**（慢变量） |
| index_vol_ratio、total_mv | 复用昨日值（total_mv 仅 A股配置有，但数据中实际缺失，自动跳过） |

## 三、修改文件

| 文件 | 改动 |
|------|------|
| `kcb_v18/scripts/realtime_features.py` | 新增：当日特征计算函数 |
| `kcb_v18/scripts/kcb_v17_predict.py` | ① 标签 1日→5日 ② global/sector 两处改用当日特征 |
| `a_share_v17/src/realtime_features.py` | 新增：同 kcb_v18 |
| `a_share_v17/scripts/sector_rotation_predict.py` | 改用当日特征 |

## 四、验证

- KCB：610 只股票，24 个特征全部正确算出 ✅
- A股：5199 只股票，25/26 特征正确（total_mv 数据缺失，与回测行为一致自动跳过）✅

## 五、实盘使用（14:30 前）

```bash
# KCB（global 模式）
cd kcb_v18 && python scripts/kcb_v17_predict.py --mode global

# A股
cd a_share_v17 && python scripts/sector_rotation_predict.py
```

> ⚠️ **不要用 `run_daily.py`**：其第一步用 Tushare 下载当日 EOD 数据，14:30 时尚未发布。直接跑预测脚本即可（用新浪实时行情）。

## 六、前提条件

- 历史 parquet 需更新到**前一交易日**（当前到 2026-08-14，周一 08-17 可用）。
- 若跨多日未更新，需先更新 parquet 到最近交易日（`run_daily.py` 收盘后跑，或手动更新）。
