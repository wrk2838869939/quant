> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`datahub/README.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。

# datahub — A股 + 美股结构化数据集

> **2026-09-20 研究验收：有条件可用于日频价格/成交量研究，尚未通过完整可交易回测验收。**
> 32 项测试通过；已发现历史行业缺失/冲突、主档漏历史代码、北交所上市前行情、
> 两个指数缺 2018 年 5 月、美股日历混入利率日期等问题。
> 详见 [研究验收报告](RESEARCH_READINESS_20260920.md)；以下基础覆盖不能替代该报告的使用边界。

> 独立于任何策略工程的**纯数据资产层**：复用 `alpha_lab` / `alpha_lab_growth` 已下载的本地数据，
> 统一 schema、修复已知缺陷、按年分区，提供带进阶指标（均线/换手率/MACD/ATR…）和进程级并行的查询接口。

## 数据覆盖

| 数据集 | 内容 | 范围 | 体积 |
|---|---|---|---|
| `datasets/cn_daily/` | A股日线（沪主板/深主板/创业板/科创板/北交所，**含退市股**） | 5819 只 × 2014-01-02 ~ 2026-09-08，1228 万行 | 2.1 GB |
| `datasets/us_daily/` | 美股 + 全球指数（SPX/DJI/IXIC/HSI/N225/FTSE）+ 美债10Y利率 | 33 只 × 2000 ~ 2026-09-17，13.5 万行 | ~10 MB |
| `datasets/index/` | A股指数日线（上证/沪深300/中证500/1000/深成/创业板指/科创50/国证2000） | 8 指数 | ~3 MB |
| `datasets/meta/` | 股票主档(5897, 含339退市)/改名ST史/申万L1分类/**PIT行业成分**/美股清单 | — | — |
| `datasets/calendar/` | 中/美交易日历 | 2014~ | — |

## 数据来源与清洗（构建时做了什么）

| 来源（只读，未改动源工程） | 处理 |
|---|---|
| `alpha_lab/data/panel/daily_panel.parquet` (3.3GB) | 剔除原始单位列（千元/手/万元），只保留统一单位列；新增 `board` 板块标签；按年 hive 分区 + zstd |
| `alpha_lab/outputs/forward_risk/repair_20211213/` + `by_date/20211213` | **修复面板缺失的 2021-12-13 整日**（原 by_date/daily 当日 0 行坏下载），用重下的 daily + 其余 3 接口按 `build_panel.py` 同构变换重建 4706 行 |
| `alpha_lab_growth/data/us_cache/*.parquet` | 统一为 `ticker/date/asset_type/OHLCV` 长表；US10Y 利率序列放 `close`（`asset_type=rate`） |
| `alpha_lab/data/raw/{stock_basic_all,namechange_all,sw_*}` | 元数据直通（加 board 列） |

**单位纪律**（继承 alpha_lab DATA_SCHEMA，历史上多次因单位混乱作废结论）：
`amount_yuan/circ_mv_yuan/total_mv_yuan/*_amount_yuan` 一律**元**；`vol_share/*_vol_share` 一律**股**；
`total_share/float_share/free_share` 仍为源字段的**万股**（例外，不能按 `_share` 后缀当作股）；
`hfq_*` 为后复权价；`turnover_rate` 为 **%**；价格 OHLC 为不复权原始价（`adjust='hfq'` 可切换）。

## 快速上手

```python
import sys; sys.path.insert(0, "datahub/src")   # 或 pip install -e datahub
from datahub import DataHub

hub = DataHub()   # 自动发现数据集; 或 DataHub(root=...) / 环境变量 DATAHUB_ROOT

# 1) A股日线长表（codes: 代码/列表/None=全市场; board 可按板块过滤）
df = hub.cn.daily("000001.SZ", "2024-01-01", "2024-06-30",
                  fields=["close", "amount_yuan", "turnover_rate"])
df = hub.cn.daily(["600519.SH", "300750.SZ"], adjust="hfq")      # OHLC=后复权, 列名不变
df = hub.cn.daily(board="chinext", start="2024-06-03", end="2024-06-03")

# 2) 宽表快捷方式 (index=date, columns=代码)
close = hub.cn.close(["000001.SZ", "600519.SH"], "2024-01-01", "2024-06-30")
to    = hub.cn.turnover(codes, "2024-01-01", "2024-06-30")       # 换手率(%); free=True 自由流通
amt   = hub.cn.amount(codes, ...)                                # 成交额(元)
snap  = hub.cn.snapshot("2024-12-31")                            # 全市场截面

# 3) ★ 进阶指标（默认 hfq_close 无前视计算; 看盘口径传 price='close'）
f = hub.cn.features(["600519.SH", "300750.SZ"], "2024-06-01", "2024-12-31",
                    specs=["ma5", "ma20", "ma60", "turnover", "turnover_ma5",
                           "macd", "atr14", "rv20", "overnight", "lg_net_ratio"],
                    n_jobs=8)                                    # 进程级并行
ma20 = hub.cn.ma(codes, 20)                                      # 单指标宽表
macd = hub.cn.macd(codes, ...)                                   # {'dif','dea','hist'} 三个宽表

# 4) 美股 / 全球指数 / 利率
us  = hub.us.daily(["BABA", "SPX", "US10Y"], "2026-09-01", "2026-09-17")
uma = hub.us.ma("SPX", 50)                                       # 美股仅价格/量类指标

# 5) 指数 / 元数据 / PIT行业 / 日历
hs300 = hub.index.daily("000300.SH", "2026-09-01", "2026-09-08")
hub.meta.sector_of("600519.SH", "2020-06-01")                    # → 食品饮料 (point-in-time)
hub.meta.sector_members(industry_name="食品饮料", date="2024-12-31")
hub.meta.stock("300750.SZ"); hub.meta.name_at("600519.SH", "2015-06-01"); hub.meta.is_st(code, date)
hub.calendar.cn_days("2024-01-01", "2024-12-31"); hub.calendar.shift_sessions_cn("2024-06-01", -20)
```

## 指标库（specs 名称）

| 类别 | 名称（数字为参数，可换） |
|---|---|
| 均线/偏离 | `ma20` `ema12` `bias20`（乖离率） |
| 动量/波动 | `ret5` `rv20`（年化波动） `rsi14` `macd`（dif/dea/hist） `boll20_2` `atr14` `dd20`（距20日高点回撤） `high20` `low20` |
| 量/换手 | `turnover` `turnover_f` `turnover_ma5` `turnover_std10` `amount_ma20` `vol_ma20` `vratio5`（量比，分母 shift(1) 防当日稀释） `amihud20`（非流动性） |
| 日内结构 | `overnight`（隔夜） `intraday`（日内） `upper_shadow` `lower_shadow`（上下影，与排雷口径一致） |
| 资金流 | `mf_net_ratio` `lg_net_ratio` `elg_net_ratio` `sm_net_ratio`（净流入/成交额） |
| 基础列直通 | `pe_ttm` `pb` `circ_mv_yuan` 等任意 `datasets/cn_daily` 列名 |

纪律：指标只读取 ≤t 的行情行，不等于当日开盘可知；日线收盘和量额等不能用于同日开盘决策。
窗口按个股观测行计算，读取完整历史前缀后截取输出，避免停牌/递归预热引起查询起点依赖。
普通窗口统计 `min_periods=窗口长度`，MACD 沿用首值初始化的 EWM；RSI/ATR 为 pandas EWM 初始化口径，
不承诺与其他软件的 SMA 种子初始化逐值相同。
`hub.us.features` 对无对应基础字段的指标（如换手率/资金流）直接报错而非给错数。

## 并行与缓存

- **IO**：按年分区 parquet + pyarrow 谓词下推/列裁剪，单股查 12 年只扫该股数据。
- **CPU**：`n_jobs>1` 按代码分块走 joblib 进程池（每个 worker 自行读数，无大对象传输）。
  为保证查询起点一致性，现读取完整历史前缀，旧速度基准已失效。
  本次 5 只股票 × 9 日输出：串行 0.49s、4 进程 1.44s，结果一致；小任务宜串行。
  全市场单日实测见 `outputs/research_readiness/full_market_smoke.json`。美股特征当前仍串行，`n_jobs` 未实现加速。
- **磁盘缓存**：`hub.cn.features(..., cache=True)` 按参数哈希落盘 `datasets/cache/`；
  键含数据集版本（重建即失效）。`hub.clear_cache()` 手动清理。

## 构建与更新

```bash
cd datahub
python scripts/build_dataset.py        # 全量构建 (源数据只读; ~2分钟)
python scripts/audit_dataset.py        # 审计: 与源面板对账/日历连续性/重复/空值 → outputs/audit_report.md
python scripts/update_data.py          # 增量: 联网补拉(alpha_lab fetch + usx update_cache) → 重建近两年分区
python scripts/update_data.py --rebuild-only   # 离线: 只用本地快照重建
python -m pytest tests -q             # 32 项测试（含研究边界回归）
python scripts/audit_research_readiness.py # 全量研究质量诊断；成功退出不等于质量全通过
python scripts/demo.py                 # 可运行示例
```

最近一次审计（2026-09-20，全文见 `outputs/audit_report.md`）：与源面板抽样对账全部一致、
(ts_code,date) 零重复、A股 3085 个交易日**缺日 0**（含修复的 2021-12-13）、美股 close 零空值。

## 已知边界（诚实口径）

1. **A股止于 2026-09-08**：源面板与按日快照均止于该日；`update_data.py` 联网可续。
2. **美股为 qfq 前复权**（源缓存如此）；SPX/DJI/IXIC 等指数无 high/low/volume；US10Y 是利率水平值。
3. 全行数加权空值率：`pe_ttm` 18.97%、`turnover_rate` 0.915%、资金流 2.96%；缺失原因需按字段核验，不能一概当正常值。
4. 未纳入（源工程另有存档，需要时可扩）：5min 分钟线(`audit/outputs/min5_tushare`)、
   财务指标(`fina_indicator` **未通过 PIT 认证**，alpha_lab 有事件级试点)、解禁/龙虎榜(`by_date/top_*`)、可转债(`cb_rotator`)。
5. stock_basic 中 1 条脏数据 `T600018.SH`(上港集箱退) board=other，2014 后无行情，无实际影响。
6. 依赖：pandas / pyarrow / numpy / joblib / PyYAML（无数据库、无网络依赖，全离线可用）。

7. `cn.codes()` 默认仅返回当前在市主档，不适合直接构造历史股票池；全市场 `features()` 已改从实际行情发现代码。
8. 历史行业冲突时 `sector_of/sector_members` 报错；历史名称缺失时 `is_st` 返回 `None`（未知），只有 `is False` 才确认非 ST。
9. 交易日平移超出日历范围时明确报错；美股日历尚为含利率的观测日期并集，存在 6 个额外日期，须先核验再用于跨市场研究。
10. 上证指数和国证2000均缺 2018 年 5 月的 22 个交易日；`total_share` 等股本列仍为万股。其他异常及明细以研究验收报告为准。

## 固定数据研究入口

日频至月频选股研究使用独立快照 `releases/selection_v1_20260920`。
数据、查询源码及数值依赖版本一起记录，SHA256 校验后运行策略；活动数据更新不影响旧快照。
固定数据的使用方式和数据/模拟/策略三层职责见 [固定研究契约](FROZEN_RESEARCH.md)。
快照不是完整回测框架，现有模拟引擎的执行资料与账户参数仍需一次性接入并固定。

```bash
python scripts/frozen_research.py verify --release releases/selection_v1_20260920
```
