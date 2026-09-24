> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`alpha_lab/docs/SECTOR_DATASET_V1.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。

# 板块数据集 v1：独立于策略的日线资产

本数据集只完成板块分类、历史成员和自算日线面板，不包含板块选择信号、价值判断阈值或回测结论。行情绑定 `selection_v1_20260920`，原数据和已有研究产物均不修改。每次构建新建独立目录，保存配置、构建源码、输入哈希、产物哈希与检查结果。

## 版本与范围

- 全市场冻结行情：2014-01-02 至 2026-09-08；使用全部行情股票，包括历史退市股票，不使用此前的 93 股工程样本。
- 首个可计算分类：申万二级，合并 SW2014 / SW2021 目录；2021-12-13 为版本切换约定。历史成员来自独立补充表的进出区间，不能据此宣称完整的历史发布时间 PIT 认证。
- 东方财富目录：保存已取得快照的行业一级、二级、三级编号和名称；当前只提供分类目录，不提供其完整历史成员面板。不同分类体系不能混用一个代码空间。
- 只按主档的上市、退市日期限制存续，不按当前上市状态筛选；没有主档的历史代码仍保留并在汇总中提示。主档日期本身可能有修订。
- `coverage_yearly.csv`、`coverage_sectors.csv` 和 `coverage_daily.parquet` 是必要配套，不能用“存在行情”替代“归属完整”。

## 文件

| 文件 | 内容 |
|---|---|
| `daily.parquet` | 每个有效分类版本内的板块逐交易日面板；包含无成员和无法算指数的记录 |
| `catalog_sw_l2.parquet` | 原始申万二级名称、编号、父级编号、分类版本 |
| `catalog_dc.parquet` | 东方财富快照目录，包含快照日期、层级，不是历史有效区间 |
| `membership_intervals.parquet` | 原始历史成员进出区间，供追溯 |
| `membership_daily.parquet` | 交易日 × 股票的实际计算归属，值为 sector_codes 的序号；-1 未知/非存续，-2 冲突 |
| `sector_codes.json` | 归属矩阵的板块编号字典 |
| `membership_requests.json` | 原始成员查询状态，包括请求失败或空结果 |
| `coverage_*.csv/parquet` | 全市场归属覆盖、各板块指数可用天数与段数 |
| `summary.json`、`validation.json` | 构建规模和自动核验结果 |
| `manifest.json`、`sources/` | 固定配置、来源绑定、文件哈希和构建源码 |

## 日线字段与计算

主键为 `(date, sector_code)`，`classification_version` 标记当天分类版本。完整收盘字段只能用于收盘后研究或下一交易日操作，不能用于当日开盘决策。

### 板块点位

- `index_close/index_open`：前一交易日成员、前日总市值加权，单股权重上限 20%；每日重新限权，不包含交易成本。这是用于观察板块的代理指数，不是投资组合收益承诺。
- 价格变化用后复权价格的相邻交易日比值；`index_return` 是当日有效成员的加权收盘收益，`equal_return/equal_index_close` 为同一成员样本的等权参照。
- 用于指数的每只股票必须同时具有前日收盘、前日正市值、当日完整正 OHLC。至少 5 只，且占前日已知成员数至少 80%，才令 `index_valid=true`。缺失成员剔除后归一化，不向前填充停牌估值，因此指数会受到缺行情样本变化影响。这里的 80% 仅是已知成员报价覆盖，不是官方真实成员完整度。
- `index_high_bound/index_low_bound` 是个股各自最高/最低价组合的外边界；个股极值发生时刻不同，不能当真实板块最高/最低点，不能据此模拟盘中成交或识别真实板块上下影线。
- `segment_id` 标记连续有效段。无报价或覆盖不合格时点位为空，恢复后收盘重新从 1000 开始；分类切换也断开。首次有效日 `chain_return` 为空，后续等于同段点位涨跌。**均线、平台、历史涨幅和回撤必须在同一 segment_id 内计算，不能跨段连接。** `index_return` 在新段首日仍记录可计算的单日成分收益，不表示与上一段相连。
- `index_status/index_members/index_member_coverage/max_weight/weight_hhi` 记录失效原因、样本数、覆盖和集中度。

### 广度与活跃度

- 当日成员的相邻交易日复权收益区分 `up_count/down_count/flat_count`；没有前日或当日报价归入 `breadth_unknown`，不作为平盘。
- `up_share` = 上涨 /（上涨 + 下跌 + 平盘）；`advancing_share` = 上涨 /（上涨 + 下跌）；`up_down_ratio` = 上涨 / 下跌，无下跌时为空，配合原始计数解释。
- `vol_share/amount_yuan` 为当日有效观测成员的成交股数/成交金额合计；全缺失返回空值，真实全零保留零。每个总量有对应 `_members` 数量。
- `turnover_pct` = 同时有成交量和正流通股本的成员成交股数合计 / 流通股数合计 × 100。源 `float_share` 是万股，先乘 10000；该值不是个股换手率简单求和。`turnover_members` 标记参与数量。

### 市值与变动拆分

- `total_mv_yuan/circ_mv_yuan` 为当日已知成员观测到的总/流通市值，单位元。缺失市值不当零估值；合计不等于官方完整板块市值，须配合 `_members` 与 `known_members`。
- `matched_cap_return/matched_cap_change_yuan` 仅比较昨天、今天归属相同且两日市值都有数据的成员。该变化含价格变化、增发/回购等股本变化，不是纯估值变化。
- `member_add_cap_yuan/member_remove_cap_yuan` 为有观测市值的成员加入/退出效果；`missing_cap_effect_yuan` 为持续成员市值缺失/恢复产生的观测差；`added_members/removed_members` 为分类归属变化数量。
- 数据首日没有前日归属，加入数量及加入市值只表示初始化，不代表真实调入；首日市值日变为空。
- 当两日都有可观察总市值时：`observed_cap_change_yuan = matched_cap_change_yuan(无匹配取0) + member_add_cap_yuan - member_remove_cap_yuan + missing_cap_effect_yuan`。
- 以后比较上涨前平台和当前市值时，必须检查成员、报价及股本变化，不能直接把板块总市值下降都解释成低估。

## 使用

```python
import sys
sys.path.insert(0, 'alpha_lab/src')
from sector_dataset import SectorDataset

# 填入本次报告给出的明确版本目录；不隐式加载 latest。
ds = SectorDataset('alpha_lab/data/sector_datasets/<版本>')
board = ds.history('801012.SI', '2023-01-01', '2023-12-31')
members = ds.members_at('801012.SI', '2023-06-01')
# 先看 index_valid、segment_id、member_request_failed 及覆盖表，再计算形态。
```

构建命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python datahub/scripts/frozen_research.py run \
  --release datahub/releases/selection_v1_20260920 \
  alpha_lab/scripts/build_sector_dataset.py
```

本版不创建“主力意图”“筹码成本”等未经观测的字段。后续历史龙头及 5 日换手/K 线状态应从此处的历史成员和冻结个股行情计算，作为独立特征版本保存；板块筛选规则也应独立保存。
