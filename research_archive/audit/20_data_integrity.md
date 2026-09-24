> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`audit/20_data_integrity.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。
> 早期结论请结合后续 27/29 号审计与低频主线复核阅读，不能按单篇旧报告判断当前有效性。

# 步骤 20 — 日频数据完整性检查

**日期：** 2026-08-23
**脚本：** `audit/scripts/data_integrity_check.py`

## 健康项 ✅

- **KCB parquet**: 交易日 397/397 完整 (2025-01 起)、无重复行、关键列 (OHLCV/ret/ma20_dev/atr/index) 0% NaN
- **A股 parquet**: 无重复行、关键列 0% NaN、成交额=0 占比 0%
- **raw 配套**: 两板 daily_basic/moneyflow 均已到 2026-08-20

## 本次修复

1. **KCB turnover 缺失段 (07-20~07-31)**: 源数据 `turnover_rate_f`(自由流通) 该段 92% NaN →
   `build_extra_features.py` 改为 `turnover_rate_f.fillna(turnover_rate)` 回退总股本口径, 重建+重合并后 NaN 41%→0.2%
2. **mkt_up_ratio 口径修正**: 08-17~08-21 补丁日原来只按主板股票算广度 → 改为全市场 (Tushare daily 全量),
   已重算回填 (如 08-19 全市场上涨占比 8.1%)

## 已知残留 (记录在案, 暂不修)

| 问题 | 影响 | 说明 |
|---|---|---|
| ~~KCB 成交额单位断档~~ | ✅ **已修复** (fix_amount_units.py, Tushare权威值替换+千元段×1000) | 两板成交额现已全统一为元 |
| A股 parquet 起点 2025-08 | 缺 2025-01~07 (140 交易日) | 历史构建时点限制 |
| A股 parquet 宇宙断档 | 无害(模型只用400只SW映射池) | 08-14 前 5199 只(含300/688), 08-17 起 3064 只(主板); 非池股票 08-03~13 行为 NaN 特征 |
| KCB daily_basic pe_ttm | 07-15~31 段 32% NaN | 数据源缺失, ep 因子该段受影响 |

## 每日维护流程 (最终版)

```
14:00 前:
  python audit/scripts/daily_update.py                    # 市场 regime 特征
  python audit/scripts/patch_board_features.py --board kcb     # 个股特征+扩展因子+raw增量
  python audit/scripts/patch_board_features.py --board ashare
```
