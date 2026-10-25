# alpha_lab 数据 Schema 文档

> Phase 0 建立, 2026-09-09 更新。所有面板/因子研究以本文件口径为准, 修改须在此留痕。

## 1. data/panel/daily_panel.parquet (主面板)

| 列 | 类型 | 单位/口径 | 说明 |
|---|---|---|---|
| ts_code | str | — | 带交易所后缀 (000001.SZ) |
| code | str | — | 6位代码 |
| date | datetime64 | — | 交易日 |
| trade_date | str | YYYYMMDD | 原始交易日字符串 |
| open/high/low/close | float | 元 | **不复权**原始价 |
| pre_close | float | 元 | 前收盘 (除权日=调整后前收) |
| hfq_open/high/low/close | float | 元 | 后复权价 = 原始×adj_factor |
| adj_factor | float | — | 后复权因子 (Tushare) |
| vol / vol_share | float | 手 / **股** | 成交量 |
| amount / amount_yuan | float | 千元 / **元** | 成交额 |
| circ_mv / circ_mv_yuan | float | 万元 / **元** | 流通市值 |
| total_mv / total_mv_yuan | float | 万元 / **元** | 总市值 |
| turnover_rate / _f | float | % | 换手率 (自由流通) |
| volume_ratio | float | — | 量比 |
| pe / pe_ttm / pb / ps / dv_ratio | float | — | 估值 (daily_basic) |
| buy/sell_{sm,md,lg,elg}_vol(+_share) | float | 手 / **股** | 资金流分单量 |
| buy/sell_{sm,md,lg,elg}_amount(+_yuan) | float | 万元 / **元** | 资金流分单额 |
| net_mf_vol / net_mf_amount | float | 手 / 万元 | 主力净流入(原始) |

**规则**: 计算一律用 `*_yuan`/`*_share`/`hfq_*` 列, 禁用原始单位列。
**覆盖**: 1227万行 × 5819 只 (2014 年后退市股 100% 含) × 2014-01-02 ~ 2026-09-08。

**2026-09-14完整性修正**：上述日期跨度不等于交易日完整。与指数日历核对发现原主面板缺2021-12-13整日，原始daily缓存为空；本次补取4706条，4646个重合收盘价与daily_basic一致、复权因子覆盖完整。原面板和缓存保留，修复投影及来源在 `outputs/forward_risk/repair_20211213/`。新风险研究已显式应用该投影并要求整日日历无缺口；旧封存选股账户未追溯重建。详情见[前瞻风险证据报告](FORWARD_RISK_EVIDENCE_20260914.md)。

## 2. data/raw/ 其他表

| 文件 | 内容 | 关键列 / 说明 |
|---|---|---|
| stock_basic_all.parquet | 股票主档 L+D (339只退市含 delist_date) | ts_code, list_status, list_date, delist_date |
| calendar.parquet | 交易日历 2014-今 | cal_date, is_open (**注意 is_open 可能是字符串, 读时 astype(int)**) |
| sw_l1_classify.parquet | 31个申万L1 | index_code, industry_name |
| sw_l1_member_hist.parquet | **PIT 行业成分** | con_code, in_date, out_date (20991231=在册) |
| index_daily.parquet | 8指数日线 | 000001/300/905/852/399001/399006/000688/399303 |
| index_dailybasic.parquet | 300指数估值 | pe/pb 等 |
| namechange_all.parquet | ST/改名历史 | ts_code, name, start/end_date (ST过滤用它) |
| fina_indicator_all.parquet | **带公告日的财务指标快照；未通过整表PIT认证** 25.3万行×108列 | ann_date不能保证每个字段当时可见；已确认部分字段混合修订版本，见下方2026-09-13审计 |
| share_float_parts/ | 解禁 (每股一文件) | float_date, float_ratio |
| by_date/{daily,adj_factor,daily_basic,moneyflow,top_list,top_inst}/ | 按日全市场快照 | 原始 Tushare 列 (amount=千元, vol=手) |

**2026-09-13财务时点修正**：原表缺少实际公告日与版本历史。康美2018一行的`ocfps`匹配2020更正值、`eps`保留初报，但整行`ann_date`仍为2019-04-30；不能直接按该日期作历史训练。其他原公告核对也有匹配样本，因此不是整表所有值均错误。详见[财务版本准入报告](FINANCIAL_PIT_GATE_20260913.md)。季度候选只作未认证覆盖审计，原快照保留，生产未改。

**2026-09-13披露事件试点**：新增`src/financial_events.py`，按字段保留原件、期末、实际公开日、单位、合并范围、会计口径及新旧版本。已恢复107个试点事件；历史完整性仍为未认证，不能自动替代原表用于训练。原初报差分与决策日最新已知值分别保留，详见[事件恢复报告](FINANCIAL_EVENT_RECOVERY_20260913.md)。

## 3. 标签 (labels.py / synthesis 口径, 2026-09-08 定稿)

- `fwd5_excess_rank` (周频) / **`fwd20_excess_rank` (月频, 生产口径)**: 前向 h 日收益 − 全市场等权 → 截面 rank(pct) → 反正态
- 停牌日标签 NaN; 退市以最后成交价结算; 截面<100 只当日置 NaN
- 训练窗末端必须 purge(h) + embargo(5)

## 4. 因子与产出 (outputs/gp_mining/)

| 文件 | 说明 |
|---|---|
| synth_factor_h20.parquet | **历史月频合成产物** (date×code 宽表，尚未按新账本验收) |
| synth_factor.parquet | 周频合成因子 (已证伪执行层, 仅存档) |
| synth_weights_h20.csv | 月频合成权重 (ICIR, 定于 TRAIN+VALID) |
| harvest_*.csv | 各轮 GP 收割清单 (train/valid IC, decay, 库相关, passed) |
| TEST_REPORT_h20.md / TEST_factor_report_h20.csv | 月频 TEST 终评 (一次性) |
| pool_check_*.csv | 因子池体检 (年度 IC + 相关矩阵) |
| phase4_backtest*.csv | 组合回测净值与敏感性 |
| ../experiment_ledger.csv | **实验台账 (每次尝试必记录, DSR 基数)** |

## 5. 双后端引擎注意事项

- `expr_engine.py` (CPU) 与 `gpu_ops.py/gpu_eval.py` (GPU) 算子语义**大部分一致**, 但:
  - ⚠️ **decay 类在 sqrt/log 产生的大面积 NaN 输入下语义分裂** (CPU 严格 min_periods → 覆盖 0%, GPU NaN→0 加权 → 44%): `decay(sqrt(zscore(gap)))` 因此剔除 (2026-09-09)
  - 规则: **挖掘入库的因子必须用 CPU 后端复算一遍 IC 确认一致后才能进因子库**
  - x_rank 已精确对齐 pandas 并列秩语义 (99.99% 完全相等, float32 精度内)

## 6. 历史教训登记 (单位/泄露/引擎)

- 生产 v15 特征 `成交额` 2026-06-30前=千元 / 07-20后=元 —— 本面板以 amount_yuan 唯一为准
- IC 衰减: 标签窗口与训练窗重叠即泄露 (audit/27: IC +0.356→−0.004)
- fina_indicator 传无效字段名会被代理拒绝且**每次重试 4 次** (ts_client 已识别 BADARG 永久失败)
- cgroup 内存上限 48GB (free 显示 503GB 是宿主机假象), 缓存 DataFrame 前先算内存

## 7. 月频组合执行审计资产（2026-09-11）

- `data/raw/execution_audit/20220101_20231231/{dividend,stk_limit}/<ts_code>.parquet`：固定549只候选的分红事件及官方价格上下限；API分红按代码返回全史，入本缓存前已过滤除权日到2022–2023。
- `configs/corporate_action_components.yaml`：已核验的多项分红组成、配股情景及证据；不静默修改原始缓存。
- `outputs/monthly_portfolio_audit/<run>/`：固定 TRAIN 权重、VALID 信号、执行候选并集及配置快照。
- 其 `ledger_<time>/` 子目录：每日现金/股票/分红应收、成交、权益事件、持仓、指标、输入及代码哈希。
- 每日价格用原始价配实际股数，分红送转单独记账；复权价用于因子研究，禁止把复权系数直接当作可卖股数。
- 当前验收范围及限制见 [组合记账审计](MONTHLY_PORTFOLIO_AUDIT_20260911.md)；旧TEST数字继续仅作错误口径存档。
