> **最新运行：行情已更新至2026-10-22收盘，对应下一交易日10月23日。** 103个有效板块、7个观察板块、当日形态确认0个、新首次确认0个、新个股候选0只。[完整报告与图表](../../outputs/datahub_model_check/20261022_eod/REPORT.html) · [Markdown](../../outputs/datahub_model_check/20261022_eod/REPORT.md)。年度板块过滤继续暂停，旧事件不重开。

> **2026-10-21当前执行规则：年度板块模型暂停过滤。** 去重后的当日首次形态确认直接进入既有个股资格、价格形态和年度个股排序。年度板块分数只作参考，旧事件不重开。
>
> 当前配置：[strategy_2026_opportunity_scan.yaml](../../configs/strategy_2026_opportunity_scan.yaml)；[变更说明与20日复核](../../docs/current_strategy_20261013/ANNUAL_SECTOR_GATE_PAUSED_20261021.md)。`scan_stocks.py`默认调用`scan_opportunities.py`，不再使用旧随机模型过滤。
>
> 后续流程继续使用原`update_market.py`、`build_sector_dataset.py`、`scan_sectors.py`完成数据与形态扫描，再在同一冻结行情下执行：
>
> `python datahub/scripts/frozen_research.py run --release <本次快照> alpha_lab/experiments/sector_live_v1/scan_stocks.py --scan <本次板块扫描目录>`
>
> 入口从扫描清单读取行情日期与快照，不复制旧日期脚本。当前策略生效前的数据需显式`--review-policy`，结果仅为规则复核；复核不更新`LATEST_CANDIDATES`。默认重新查询名称历史，复核可用`--name-history <已核对文件>`。旧脚本仅通过`--legacy-rules`显式复现，不应用于当前候选判断。

> 以下为历史记录，涉及旧板块模型过滤的结论不代表当前规则。

> 历史策略技术报告：[PDF](../../docs/current_strategy_20261013/TECHNICAL_REPORT.pdf) · [Markdown](../../docs/current_strategy_20261013/TECHNICAL_REPORT.md)，含模型重放、收益重算与实现边界。

> 最新进展：[2026-10-13 更新与模型扫描](REPORT_20261013.md)，完整行情截至10月12日。

> 最新进展：[2026-10-11 更新与机会检查](REPORT_20261011.md)，行情截至10月9日。以下保留先前记录。

> 最新进展：[2026-09-28 更新与机会检查](REPORT_20260928.md)。28日盘中执行，行情截止最近完整交易日24日；新增确认0个、联合个股候选0只。以下保留23日记录。

# 2026-09-23 更新及最新候选扫描

已补齐 2026-09-09 至 09-23 的 11 个交易日，新增 61,066 条有效个股日线。
今天 5,556 条日线的复权因子、市值、流通股本及换手率均齐全；43 条零成交且开高低全零的
代理停牌占位记录单独保存，未作为有效行情拼接。
旧 `selection_v1_20260920` 快照全量哈希复核通过，没有修改旧研究输入。

## 当前结论

103 个具备有效指数的板块中，4 个进入观察池，2 个进入价格区触发层，
**今天满足全部确认条件的板块为 0；符合联合观察标准的个股为 0。**

| 板块 | 编号 | 同轮前期涨幅 | 从峰值回落 | 今日未确认原因 |
|---|---|---:|---:|---|
| 农产品加工 | 801012.SI | 32.54% | -11.65% | 未到价格参照区，未减速 |
| 渔业 | 801015.SI | 37.05% | -12.59% | 未到价格参照区，未减速 |
| 贵金属 | 801053.SI | 30.42% | -14.95% | 已接近参照区并减速，但近5日未跌破峰前支撑 |
| 酒店餐饮 | 801219.SI | 16.99% | -10.67% | 在参照区且支撑已破，但当前跌速未放缓 |

9 月 8 日之后新增 1 次确认事件：酒店餐饮，9 月 17 日。
两个冻结网络评分分别 -0.022768 / -0.020273，低于各自训练期固定阈值
0.009238 / -0.006769，两个模型均未选中。这些是模型排序分数，不是校准后的收益预测。

## 个股

检查上述板块 48 只冻结成员及 4 只当前成员补充，共 52 只。
当前 `index_member_all(is_new=Y)` 仍返回退市证券，已与最新在市主档及进出日期交叉核查，
排除这些记录后再识别成员差异。新增诊断股票为东方海洋及三只北交所股票；
没有将当前成员表回填历史指数。

| 板块 | 检查个股 | 趋势/先涨后落通过 | 五日量价通过 | 两者都通过 |
|---|---:|---:|---:|---:|
| 农产品加工 | 26 | 5 | 0 | 0 |
| 渔业 | 6 | 2 | 0 | 0 |
| 贵金属 | 11 | 0 | 0 | 0 |
| 酒店餐饮 | 9 | 1 | 0 | 0 |

个股使用旧 `left_side_trend_v2.yaml` 的未训练规则：峰值前上行、先涨后落、
最近5日先放量下跌后缩量减速；历史龙头仅排序。它们尚未在当前板块模型下做选股回测。
实时联合候选还要求板块今天确认、近5个交易日有新确认且至少一个旧网络通过；
此组合用法没有单独回测，不能直接套用历史事件收益。本次既无板块确认，也无个股量价通过，
所以结论不依赖两个模型究竟采用“至少一个”还是“两者都通过”。
没有新增训练、放松门槛、下单或建立10万元组合。

## 接口测试

用户提供的令牌与本地凭据一致；没有在新脚本或报告中复制令牌。
本机 Tushare 的 `query` 实现会把接口名追加到 URL，设置 `/api` 后实际访问 `/api/daily`，
服务端返回 404，SDK 随后静默返回空表。
直接 POST `/api` 的按交易日查询可用，9 月 23 日 daily 返回 5,556 行；
用户的 2018 年单股区间示例直接 POST 仍返回“接口用法错误”。
本次下载优先使用标准 `/api` POST，失败时按源码明确尝试仍可用的兼容分路；
HTTP 失败或空表不会当作完整交易日保存。`update_market.py::query` 为本次请求适配。

## 验证

- 核心趋势、形态、确认及多窗口去重共 15 项既有测试通过。
- 新板块数据 9 月 8 日以前的全部历史与旧版本一致，330 个旧确认事件也逐项一致。
- 所有新增日线唯一键、复权覆盖、有效 OHLC、单位换算、旧价格前缀检查通过。
- 板块层原构建器的权重上限、连续指数、报价覆盖、物理前缀等验证通过。
- 最新评分仅使用截至确认日特征；复用既有两轮三种子模型和训练阈值，不读取未来标签。
- 4 页观察池图已查看，时间轴明确限制在当前回落事件对应区间。

## 产物

- 更新记录：`alpha_lab/outputs/sector_live_update/20260923T124411_929745Z/`
- 新市场快照：`datahub/releases/selection_live_20260923/`
- 新板块数据：`alpha_lab/data/sector_live_datasets/sw_l2_v1_20260923T125853_083011Z/`
- 板块扫描：`alpha_lab/outputs/sector_live_scan/20260923T131248_195204Z/`
- **最终候选报告与图：`alpha_lab/outputs/sector_live_candidates/20260923T140102_328247Z/`**
- 个股逐项原因：最终目录 `stock_diagnostics_all.csv`；观察图 `sector_watch_charts.pdf`。

本目录是固定 9 月 23 日的一次更新和扫描，非自动交易任务。新市场快照仅更新 A 股日线和当前在市主档，
美股、基准指数及历史行业补充资料未更新；当前研究仍使用有覆盖限制的申万二级自算板块。
当前行业历史来源未获完整 PIT 认证；最新 103 个有效板块不代表整个市场全部板块。

复现扫描（已有冻结新快照，不必再次下载）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python datahub/scripts/frozen_research.py run \
  --release datahub/releases/selection_live_20260923 \
  alpha_lab/experiments/sector_live_v1/scan_sectors.py \
  --dataset alpha_lab/data/sector_live_datasets/sw_l2_v1_20260923T125853_083011Z

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python datahub/scripts/frozen_research.py run \
  --release datahub/releases/selection_live_20260923 \
  alpha_lab/experiments/sector_live_v1/scan_stocks.py \
  --scan alpha_lab/outputs/sector_live_scan/20260923T131248_195204Z \
  --membership-cache alpha_lab/outputs/sector_live_candidates/20260923T140102_328247Z
```
