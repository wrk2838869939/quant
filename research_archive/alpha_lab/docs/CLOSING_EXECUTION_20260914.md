> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`alpha_lab/docs/CLOSING_EXECUTION_20260914.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。

# 14:30盘中代理与收盘集合竞价

2026-09-14按用户要求新增执行分支。月中15日复核、最多5股、100股买入单位、风险预算与真实减仓恢复额度保留。新配置为 `configs/monthly_risk_close_day15.yaml`。**工程检查已通过，真实盘中运行和新口径收益尚待验证；未启动调度、发送邮件或提交委托。**

## 当天流程

1. 14:30后取得全A股票和沪深300的实时快照，保留原始响应、报价时刻和散列。全批报价跨度不得超过120秒；缺报价不能静默删股票，停牌必须有当天状态依据。
2. 用此时价格模拟收盘价。账户数量、当日可卖量和现金以券商实际记录为准，估值统一到同一份冻结行情，不要求人工计算市值。
3. 在快照取得后10分钟内、且14:55前完成计算与计划冻结，生成邮件清单。没有信号就没有买单；不通过反复重算挑选更喜欢的盘中结果。
4. **14:57—15:00参与收盘集合竞价**。核对股票、数量、限价、可卖量和可用现金后，人工提交。买价上限/卖价下限是价格边界，最终撮合价以券商成交回报为准。该时段不能撤单；限价不满足或排队不足仍可能不成交。[上交所现行交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)、[深交所交易规则](https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf)。
5. 导入实际成交、部分成交及未成交终态，再核对同日收盘账户。盘中和收盘快照均保留，不能覆盖、重复入账或把今日买入计入当日可卖量。卖出款不提前用于买单；符合原补位条件时，下一交易日仅使用实际到账资金补位一次。

## 数据口径

- 市值代理＝前日供应商市值×实时价÷前日价；换手率代理＝当时累计成交股数÷前日已知流通股本×100。没有把最后半小时尚未发生的成交量外推为事实。
- 复权价格使用前日已知复权基准。实时昨收与基准不一致时停止构建，先核对除权或源数据差异，不自动套用旧股本。罕见公司行动的自动处理仍未补齐。
- 上线前使用已封存历史；上线后每日保存盘中代理序列供此分支延续，**正式收盘数据单独保存，不回写盘中策略历史**。即使非复核日也需要持续采集，否则下次复核会缺交易日。这会使上线后的技术指标、换手率窗口与旧EOD分支有所不同，需要前向比较。
- 新鲜的证券主表、ST和停牌状态由既有元数据采集脚本提供；具体订单仍须逐股人工核对。模型继续使用冻结系数，不在本次改动中优化收益参数。

## 运行入口

从 `alpha_lab/` 运行。`RUN` 为当天新建目录，不能复用旧快照。首次历史包为 `outputs/monthly_manual_day15/observation_final`；后续 `PREVIOUS` 必须指向本分支上一交易日观察包，指数也使用其中的 `index_close.parquet`。每日真实EOD更新仅用于另行准备最新股本基准。

```bash
# 先用既有入口准备当天元数据，price-date 必须是当天。
python scripts/stage_execution_metadata.py --bundle "$PREVIOUS/bundle.json" \
  --price-date "$TODAY" --output "$RUN/metadata"
python scripts/closing_observation.py status \
  --metadata-stage "$RUN/metadata" --output "$RUN/status"

# 14:30后采集。状态全集与盘前股本必须一致。
python scripts/closing_observation.py collect \
  --status "$RUN/status/status.json" --output "$RUN/quotes"
python scripts/closing_observation.py account-snapshot \
  --broker "$BROKER_JSON" --quotes "$RUN/quotes" --output "$RUN/account.json"
python scripts/closing_observation.py build --previous "$PREVIOUS" \
  --quotes "$RUN/quotes" --basis "$BASIS" --index-history "$INDEX_HISTORY" \
  --model "$STOCK_MODEL" --snapshot "$RUN/account.json" --output "$RUN/observation"
python scripts/manual_execution.py snapshot --db "$ACCOUNT_DB" --input "$RUN/account.json"
python scripts/manual_execution.py prepare --db "$ACCOUNT_DB" \
  --bundle "$RUN/observation/bundle.json" --calendar "$RUN/observation/calendar.json" \
  --config configs/monthly_risk_close_day15.yaml --output "$RUN/plan.json"
python scripts/manual_execution.py draft --plan "$RUN/plan.json" \
  --metadata "$RUN/observation/metadata_report.json" --output "$RUN/draft"
```

`BROKER_JSON` 必须包含实际 `account_id/asof/timestamp/source/evidence/cash/receivables/frozen_cash/open_orders/positions`；每仓包含 `shares/sellable_today`。示例清单不等于券商凭据。现有 `review/fill/terminal` 入口继续使用，收盘成交需提供实际 `timestamp`，不能填写回报接收时刻。收盘核对使用 `phase: eod`、收盘后时刻与 `sellable_next_day`。

已准备的首次基准：`outputs/closing_manual_20260914/basis_v2/basis.parquet`，覆盖5,562只证券；12只前日无报价保留历史基准，下一日是否继续停牌须重新获取。首次指数为 `outputs/monthly_manual_current/index_verified/index_close.parquet`；股票模型为 `outputs/shadow_execution/20260914T054517_770961Z/model_2026.json`。`build_closing_basis.py --help` 提供后续基准更新入口。模板 `status_template.json` 故意标记不完整，不能直接执行。

## 已检验及边界

- 单元/回归覆盖同日冻结、重启、部分成交、收盘核对、T+1、实际到账补位、邮件到期、陈旧/未来/缺失行情及合成数据禁止生成计划；结果见 `outputs/closing_manual_20260914/tests.log`。
- 全规模工程样本使用真实历史至9月14日，**9月15日行情完全合成**：12,302,463行历史，5,550只合成交易股票，12只合成停牌；总耗时45.84秒。不能据此计算策略收益。市值重定价保持前日口径，换手率重建最大误差0.00005个百分点。
- 首次检查发现688235.SH的“总股本×价格”比供应商市值高约4.97%；未猜测差异原因，改用已知市值同比例重定价。旧基准目录保留供追溯，使用 `basis_v2`。
- 盘后6个接口样本覆盖沪深主板、创业板、科创板、北交所及沪深300，解析成功；时间为盘后，明确不能当14:30快照。真实全A盘中覆盖率、延迟和收盘成交情况仍待下一交易日验证。
- 旧版次日开盘回测不构成本分支的业绩证据。下一步记录14:30信号与最终EOD信号的候选重合率、持有/风险开关分歧、委托成交比例、实际价格偏离及空闲资金，而不是只比较最终收益。

如果9月15日数据与账户核对通过、且策略产生信号，首次目标执行时间为**9月15日收盘**。这不是保证当日有买单或全部成交。原¥80,000空仓登记尚未替代当天券商账户核对。
