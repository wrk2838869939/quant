> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`alpha_lab_growth/docs/USX_RESEARCH_SURVEY_20260918.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。

# USX 独立调研报告 (2026-09-18)

> 目的: 为「美股隔夜信息 → A股板块轮动 → 个股中短期持有(开盘买入)」策略重构提供独立的机制与证据基础。
> 纪律: 本项目 workspace 内所有历史 md 结论(AGENTS.md / VERSIONS.md / audit/* / alpha_lab_growth/docs/legacy)一律视为**未验证线索**, 本文只引用外部文献 + 本仓库原始数据可复算的事实。
> 外部文献同样以中性情感对待: 文献结论有样本期/市场结构依赖, 用于生成假设, 不作为证据。

## 一、策略涉及的因果时序 (第一性原理约束)

```
北京时间           美股                        A股
D-1 09:30~15:00                                A股 D-1 交易 (收盘 15:00)
D-1 16:00          ADR/ETF 等正常交易
D-1 21:30          美股 D-1 开盘 (夏令时)
D 日 04:00/05:00   美股 D-1 收盘  ← 隔夜信息源
D 日 09:15~09:25   (集合竞价开始, A股 D 开盘定价吸收隔夜信息)
D 日 09:30         A股 D 开盘  ← 策略买点 (此时点可用的信息: 美股 D-1 全部 + A股 ≤D-1 全部)
```

**硬约束**:
1. A股 D 日开盘前可用的美股数据 = 美股日历日 ≤ D-1 的全部 session(夏令时/冬令时/半日市均收盘于北京时间 D 日清晨 04:00-05:00 之前)。
2. A股 D 日开盘前可用的 A股数据 = 面板 trade_date ≤ D-1 的行 + PIT 行业成分 (D-1 时点在册) + ST/上市史 (D-1 时点状态)。
3. 开盘买入 => 策略收益口径必须是 **open(D) → open(D+h)** (后复权), 不得用 close-to-close 冒充。买不到的: 开盘涨停(一字/秒板无法成交)按不可成交处理。
4. T+1: D 日买入的仓位最早 D+1 日可卖。持仓 h≥1 时卖出日 = D+h 开盘, 合规。

## 二、机制文献结论 (假设来源, 非证据)

### 2.1 隔夜-日内分解: A股与美股结构相反
- A股隔夜收益系统性为**负**、日内为**正**, 与美股相反; 与 T+1 规则、卖空约束、散户结构有关。
  - An (2022) The Asymmetric Overnight Return Anomaly in the Chinese Stock Market: https://www.mdpi.com/ (journal version MDPI)
  - 综述性来源 (overnight puzzle, T+1): https://www.researchgate.net/
  - Zhou (2022) 能源股隔夜负/日内正: https://www.frontiersin.org/
  - 经典分解框架: Cliff-Cooper-Gulen; Lou-Polk-Skurass (overnight vs intraday return premium)
- **对本策略的含义**: 平均而言, 隔夜段是亏损段、日内段是盈利段。开盘买入者恰好跳过隔夜段(好), 但高开后日内均值回归(坏)。**美股夜盘利好 → A股高开 → 日内走弱** 有直接文献支持:
  - Guo (2025) The nexus of overnight trend and asset prices in China: 强隔夜趋势股"隔夜收益更高、日内收益更低"。 https://www.sciencedirect.com
- **推论(需本地数据验证)**: 开盘买入策略若想赚钱, 赚的**不是隔夜 gap**(gap 已在开盘价里), 只能是 gap 之后的持续(轮动扩散/资金流入惯性)或选股 alpha。任何用 close-to-close 回测高估的"美股信号策略"都应怀疑。

### 2.2 美股 → 国际市场的日频领先性
- Monteiro, Silva & Sebastião (2023) Industry return lead-lag relationships between the US and other major countries, Financial Innovation 9:40: 美股**行业**指数对他国**同行业**指数存在日频 lead-lag。 https://link.springer.com/article/10.1186/s40854-022-00439-1
- Wen et al. (2023) 滞后美股日收益对 17 个发达市场收益的预测力。 https://www.tandfonline.com
- Hong, Torous & Valkanov (2007) Do Industries Lead Stock Markets? JFE: 行业组合收益领先市场, 信息跨行业扩散有延迟。 (cited ~950+)
- ADR 价格发现: 重叠交易时段母国市场主导; **母国休市时段(我们的隔夜窗口) ADR 承担更大价格发现角色**。 Duppati (2017) https://www.tandfonline.com ; Loureiro (2022)。
- 前次尝试的**自报未验证**数据点(仅作线索): SPX 隔夜对 A股指数当日 corr≈+0.10, 滞后一日衰减到 ≈+0.015 —— 指数层信息开盘即被吸收, 残差预测力主要疑存于**板块/个股层**。

### 2.3 A股截面因子 (哪些先验方向站得住)
- **换手率负效应**(高换手→低收益): Liu, Stambaugh, Yuan (2019) Size and Value in China, JFE 134(1):48-69, CH-3 模型, turnover 是 A股最稳健异常之一; Zhang (2021) Turnover premia in China's stock markets。 https://www.nber.org/papers/w26376
- **壳价值污染**: CH-3 剔除最小 30% 市值股(壳价值扭曲)。=> 本策略市值过滤的第一性依据。
- **个股短期动量不存在**(反转主导): Yue (2023) Does short-term momentum exist in China? https://www.sciencedirect.com ; Gao (2024) AEA Daily Momentum and New Investors。
- **行业层面动量存在**(个股反转 vs 行业动量并存的"分解结构"): Gong (2017) Does the momentum strategy work at the industry level? https://www.tandfonline.com ; Gao (2026) The Anatomy of Industry Momentum (SSRN): 行业/特征组合有短期动量而个股一个月反转。
- **52周高点锚定**(价格分位): George & Hwang (2004) JF; A股证据: Zhou (2022, EPU 论文确认 A股 52周高点动量); Wang (2023) MAX/锚定; Chen (2024) Anchoring effect。方向: **接近 52周高点的股票后续收益更好**(未反应→漂移), 远离高点(深度回撤股)反转策略在 long-only 下有"毒尾"风险(前次尝试自报未验证, 但与文献的困境股机制一致)。
- **资金流/订单失衡**: Narayan et al. (2015) Do order imbalances predict Chinese stock returns? https://www.sciencedirect.com ; Lu (2024) order imbalance 桥接 trade flow 与收益, 有显著预测力。注意: 大单净流入的预测力集中在**短周期**, 且 Tushare moneyflow 的分单口径(样本股/估计)有限制, 需要按自家面板重测。

### 2.4 综合推论 (策略假设, 待 P0 实验证实/证伪)
H1: 美股隔夜信息在 A股**开盘价**中被快速吸收(指数层), 但在**板块层**吸收不完全——同行业映射下, 美股行业隔夜收益对 A股板块 T+1~T+5 的相对收益有残余解释力(轮动扩散)。
H2: 板块层动量(自体 5-20日)与美股隔夜信号组合, 优于单独任一(信息来源正交: 全球隔夜 vs 国内延续)。
H3: 个股层(板块内): 资金流强度(大单+特大单净流入/流通市值)与 52周高点近度为正先验; 换手率水平为负先验、换手率变化(attention 启动)方向存疑(双向预注册); 个股短期反转对 h≥3 持有期不作为选股信号(long-only 毒尾)。
H4: 若 P0 显示美股信号对 gap 后收益无任何残余解释力(即开盘完全吸收且无轮动扩散), 则**该策略主线不成立**, 降级为"隔夜风控门"(如美股暴跌次日降仓), 不得强行上调收益预期。

## 三、本地数据资产盘点 (可复算事实, 2026-09-18 实测)

| 资产 | 路径 | 关键事实 |
|---|---|---|
| A股主面板 | alpha_lab/data/panel/daily_panel.parquet | 1227万行×5819只(含退市)×2014-01-02~2026-09-08; open/hfq 100%覆盖; turnover 98-100%; moneyflow(大单/特大单) 2014-2026 各年 95-99% |
| PIT 行业成分 | alpha_lab/data/raw/sw_l1_member_hist.parquet | 31个申万L1, in_date/out_date 完整(含2021-12分类改版) |
| ST/改名史 | alpha_lab/data/raw/namechange_all.parquet | ST 过滤 |
| 上市/退市 | alpha_lab/data/raw/stock_basic_all.parquet | 339只退市含 delist_date |
| 美股指数 (Tushare index_global) | growth_cache + 在线 | SPX/IXIC/DJI/N225/HSI/FTSE 可用, 新鲜至 2026-09-17 |
| 美股 ETF/ADR (AKShare 新浪 stock_us_daily) | 需缓存 | **11个 SPDR 行业 ETF(XLK/XLE/XLF/XLV/XLY/XLP/XLI/XLB/XLU + XLC/XLRE) 2001/2005起; KWEB 2017-11起; ASHR 2015-03起; FXI; GLD 2004起; USO 2006起; BABA/PDD/JD/LI/NIO/XPEV 等 ADR** 全部新鲜至 2026-09-17 |
| 美债/中债收益率 | AKShare bond_zh_us_rate | 美债10Y 可用(新鲜) |
| Tushare us_daily | — | **无权限**(美股个股不可用), ETF/ADR 必须走 AKShare 新浪通道(qfq) |

**单位纪律**: 一律用面板 `*_yuan`/`*_share`/`hfq_*` 列; 收益率用 hfq 价计算。

## 四、遗留资产的处理

- alpha_lab_growth 旧 docs (~76份) 与 outputs: **全部标记为未验证线索**, 不进入新架构的证据链; 新代码放 `src/usx/`、`scripts/usx_*.py`, 与旧物物理隔离。
- 旧 growth_cache/us_index/{SPX,IXIC}.parquet (2013-06~2023-12): 仅缓存格式参考, 重新下载全窗口。
