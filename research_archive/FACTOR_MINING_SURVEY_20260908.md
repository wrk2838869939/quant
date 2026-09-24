> **历史过程记录**｜以下为当时的研究判断，含可能已被修正的结论。本次归档未重跑实验。
> 原始来源：`FACTOR_MINING_SURVEY_20260908.md`。保留实验设计、失败与修正过程；收益及指标不能脱离窗口和后续审查单独引用。

# 因子挖掘调研与低频策略重构路线图（2026-09-08）
> **执行状态 (2026-09-09 更新)**: 本路线图已执行至 Phase 4 完成 — Phase 0~2 全过, GP 挖掘以自研引擎替代 gplearn(截面IC语义), 月频(fwd20)为生产口径。周频管线在 Phase 4 被证伪(alpha半衰期3-4日×换手成本不可调和), 一处偏离: AlphaAgent 面板未用(自建面板先行完成)。结果: 月频组合 TEST 年化 +15.3%/MDD -15%/Sharpe 1.55。详见 VERSIONS.md §24。


> 定位：配合"现架构冻结进入维护态"（VERSIONS.md §23）的决策，为下一阶段
> 【低频（周/月）精准买入 + 数据驱动复杂因子挖掘】提供调研依据与路线图。
> 调研范围：英文学术界（OpenAlex 核验）、中文券商研报与实践社区、GitHub 开源项目，
> 并盘点了本机 `.` 现有数据资产。

---

## 0. 先说结论（TL;DR）

1. **你们已经用实验复现了文献里最著名的三个坑**：标签泄露（v18 诚实口径：+1189% → -14.2%）、
   多次实验导致的过拟合（v12→v18 多版本迭代）、日频交易成本+信噪比拖累（个股日σ≈2% vs alpha≈0.2σ）。
   这不是执行问题，是研究方法论问题——文献有现成解法。
2. **最大的硬约束不是模型，是数据**：主板面板只有 2025-08 ~ 2026-09（约 250 个交易日）。
   13 个月单一样本段内，任何"挖出来的因子"都无法与运气区分（audit/29 的 LGBM p≈0.09 已经说明了这一点）。
   复杂因子挖掘（GP/RL/NN/LLM）的前提是 **≥8~10 年、含退市股、PIT 正确的日线面板**，
   而你的 Tushare Premium token 完全可以回补——**这是 Phase 0，优先级高于一切**。
3. **降频方向有直接文献背书**：华泰《机器学习选股模型的调仓频率实证》（周频 XGBoost 最优）、
   GKX 2020（月频信号稳健）、Detzel/Novy-Marx/Velikov JF 2023（费后排名重排）。
   周/月频 + 排序型标签 + 换手惩罚，是学界和私募的共同默认实践。
4. **"更精准买入"在文献里的对应物是 meta-labeling**（AFML）：主模型管选股，
   第二层模型管"这次信号要不要做/做多大"。你们的 5 条件排雷本质上就是一个人工版 meta-filter，
   文献建议把它**数据化**（事件研究你们已做，方向正确），并回到更长数据上重新验证剂量反应。
5. **板块轮动降级为特征，不作为独立择时层**：你们自己的消融（择时贡献 -397%、板块历史超额为负、
   合成 ETF 条件性反转）与社区共识一致。板块/概念信息以"行业动量、行业中性化、图先验（HIST/MASTER）"
   的形式进入因子层。
6. **工具选型**：qlib 做研究底座（数据自转 bin）+ alphalens-reloaded 评估 + gplearn 起步挖掘 +
   AlphaGen/AlphaForge 抄算法 + RD-Agent(Q)/AlphaAgent 抄"LLM 假设→代码→验证"闭环。
   AlphaAgent 仓库自带 2015-2026 中证1000并集面板（parquet 网盘），可在自建数据回补期间先用它起步。

---

## 1. 现状诊断：你的失败模式 × 文献对照

| 你们的历史事实（VERSIONS/audit） | 文献中的对应坑 | 解法出处 |
|---|---|---|
| v17 官方 +1189% vs 诚实口径 -14.2%；训练池标签前向泄露；IC 随剔除天数 +0.356→-0.004 | 标签重叠 + 单一时间切分；purged CV with embargo | AFML (López de Prado 2018)；quant67 实践文 |
| 多版本多参数迭代后回测虚高 | 多重检验：实验次数本身制造假发现；千次实验夏普虚高约 3.7σ | Harvey et al. RFS 2016 (t>3)；DSR；PBO |
| 日频 T+2：个股日σ≈2%，选股 alpha≈0.2σ，TP+4%≈2σ | 高换手×成本的费后侵蚀；纸面最优≠费后最优 | Detzel/Novy-Marx/Velikov JF 2023；Hanauer 2023 |
| 择时/板块轮动贡献 -397%（消融）；板块选择历史超额为负；合成 ETF 条件性反转 | 轮动=低信噪比择时；"年化 511% ETF 轮动"多为过拟合 | 知乎/次方量化复盘；华泰消融方法 |
| LGBM 排序 +7.8pp 但 p≈0.09（与运气不可区分） | 13 个月样本段不足以识别截面 alpha；月频 OOS R² 本就只有 ~0.4% | GKX 2020；Replicating Anomalies |
| 排雷是承重墙但不能独立选股（≥3/5 子集 NW-t=-3.0 显著为负） | 手工规则过滤=人工 meta-labeling；应数据化+回长历史验证 | AFML meta-labeling；上财质押-造假链研究 |
| 事件研究有效（封板 +2.67%/2d, t=9.5）且"统计因子→事件触发"方法论转向 | 事件驱动/条件型因子在 A 股micro结构上有真实文献与研报支持 | 开源证券微观结构系列（理想振幅等） |

**一句话诊断**：方向（降频、精准买入、数据驱动挖掘）是对的，但成立前提是把数据面板从 13 个月
扩到 10 年级，并把"每次实验"变成可统计校正的受控实验。

## 2. 数据资产盘点（本机实查，2026-09-08）

| 资产 | 位置 | 规模 | 覆盖 |
|---|---|---|---|
| 主板特征面板 v15（108 特征） | `a_share_v15/data/processed/daily_features_v15.parquet` | 20.1 万行 × 108，5199 票 | **2025-08-01 ~ 2026-09-08** |
| 主板全量备份（31 特征） | `a_share_sniper/data/processed_rebuilt/daily_features_full.parquet` | 75.4 万行 × 31 | 2025-08 ~ 2026-07 |
| Tushare daily_basic（PE/PB/换手/市值） | `a_share_sniper/data/raw/daily_basic/ab_daily_basic_all.parquet` | 127.7 万行 × 18 | 2025-01 ~ 2026-09 |
| 资金流 moneyflow | `a_share_sniper/data/raw/money_flow/ab_moneyflow_all.parquet` | 129.7 万行 × 20 | 2025-01 ~ 2026-09 |
| 龙虎榜 top_inst | `a_share_sniper/data/raw/lhb/ab_top_inst_all.parquet` | 21.8 万行 × 10 | 2025-01 ~ 2026-08 |
| KCB 面板（89 特征） | `kcb_sniper/data/processed_rebuilt/daily_features.parquet` | 49.4 万行 × 89，610 票 | 2022-07 ~ 2026-08 |
| 5 分钟线存档 | `audit/outputs/min5_tushare/`（400 票） | 400 文件 | 2026-08 |
| 申万 L1 成分 | `a_share_v15/data/raw/sw_l1_member.parquet` | 3446 行 | 静态 |
| 数据接口 | Tushare Premium（含 stk_mins 权限） | — | 可回补历史 |

**缺口**：全 A 日线（含退市、复权因子）、指数日线、财务指标（PIT）、质押、解禁、ST/名称变更历史、
更长 LHB/资金流——全部可由 Tushare 回补；缺**10 年级面板**是当前唯一硬瓶颈。

---

## 3. 英文学术界：主线脉络（全部经 OpenAlex 核验）

### 3.1 方法论底线（必读四篇）
- **Gu, Kelly & Xiu 2020, RFS**《Empirical Asset Pricing via Machine Learning》：月频面板上 NN/GBRT
  最强，但 OOS R² 仅 ~0.4%；最大增量来自"特征×宏观状态"交互。→ 对预期收益打折；交互特征值得做。
  https://doi.org/10.1093/rfs/hhaa009
- **Harvey, Liu & Zhu 2016, RFS**《…and the Cross-Section of Expected Returns》：因子是多重检验产物，
  新因子 t 应 >3。→ 给挖掘定统计门槛。
  https://doi.org/10.1093/rfs/hhv059
- **López de Prado 2018**《Advances in Financial Machine Learning》：triple-barrier/meta-labeling、
  purged K-fold + embargo、CPCV、样本唯一性加权。→ 新管线硬性规范。
  https://doi.org/10.2139/ssrn.3266136 ；CPCV 最抗过拟合的受控验证：
  https://doi.org/10.1016/j.knosys.2024.112477
- **Bailey & López de Prado 2014**《Deflated Sharpe Ratio》 + **PBO** 2017：按尝试次数折减夏普、
  量化回测过拟合概率。https://doi.org/10.3905/jpm.2014.40.5.094 ；https://doi.org/10.21314/jcf.2016.322

### 3.2 因子有效性边界（预期管理）
- **Hou, Xue & Zhang 2020, RFS**《Replicating Anomalies》：452 个异象 65% 复制失败。
  https://doi.org/10.1093/rfs/hhy131
- **McLean & Pontiff 2016, JF**：异象发表后收益衰减 ~58%。→ 持续挖掘机制 + 对回测打折。
  https://doi.org/10.1111/jofi.12365
- **Liu, Stambaugh & Yuan 2019, JFE**《Size and Value in China》：A 股 CH-3，剔壳价值后重估规模/价值。
  https://doi.org/10.1016/j.jfineco.2019.03.008

### 3.3 公式化因子自动挖掘（与你目标最对口）
- **Kakushadze 2016**《101 Formulaic Alphas》：算子库范式起点。https://doi.org/10.1002/wilm.10525
- **AlphaGen, CIKM 2023**：RL 生成因子，奖励=对因子**组合**的增量 IC → 天然去冗余；开源、默认 A 股管线。
  https://doi.org/10.1145/3580305.3599831 ；https://github.com/RL-MLDM/alphagen
- **AlphaForge, AAAI 2025**：生成-预测双网络挖掘 + 动态组合权重。
  https://arxiv.org/abs/2406.18394 （代码工程化弱，抄思路）
- **QuantFactor REINFORCE, IEEE TSP 2025**：方差有界 REINFORCE 挖"稳"因子。
  https://doi.org/10.1109/TSP.2025.3576781

### 3.4 LLM/Agent 挖因子（2024-2026 前沿）
- **Alpha-GPT 2023**：人给先验思想 → LLM 译成因子表达式。https://arxiv.org/abs/2308.00016
- **RD-Agent(Q) 2025（微软）**：多 Agent"假设→代码→回测→迭代"闭环，与 qlib 无缝；论文称 <$10 成本
  挖出可用因子集。https://arxiv.org/abs/2505.15155 ；https://github.com/microsoft/RD-Agent
- **AlphaAgent, KDD 2025**：AST 原创性正则 + 语义一致性 + 复杂度控制抗 alpha 衰减；**A 股 CSI500 验证**；
  仓库自带 2015-2026 中证1000并集面板。https://doi.org/10.1145/3711896.3736838 ；https://github.com/RndmVariableQ/AlphaAgent
- **Alpha Jungle（MCTS+LLM）, AAAI 2026**：LLM 当价值网络导航算子空间，比 GP 盲目变异样本高效。
  https://doi.org/10.1609/aaai.v40i2.37069
- 趋势：K 线基础模型 Kronos（https://github.com/shiyu-coder/Kronos ）、非平稳适应 DoubleAdapt（KDD 2023）。

### 3.5 神经网络截面模型（二期再说）
AlphaNet（华泰，CNN 式端到端因子挖掘，https://github.com/ryanluoli1/AlphaNet ）、
HIST（概念图先验，https://arxiv.org/abs/2110.13716 ）、MASTER（市场引导注意力, AAAI 2024，
https://github.com/SJTU-DMTai/MASTER ）、FactorVAE（α/β 解耦, AAAI 2022）、
SFM/ATFNet（多频率/时频分解，与降频天然契合）。**注意：GKX 量级下，这些模型的边际收益远小于
数据与验证纪律；放到 Phase 3 之后。**

### 3.6 成本与频率
- **Detzel, Novy-Marx & Velikov 2023, JF**《Model Comparison with Transaction Costs》：
  计入费用表后模型排名大幅重排。https://doi.org/10.1111/jofi.13225
- **Hanauer 2023**：41 国 ML 异象，月频调仓换手口径。https://doi.org/10.1057/s41260-023-00318-z
- **Chen, Hanauer & Kalsbach 2024, SSRN**：1000+ 模型受控实验，**标签定义的影响 ≈ 算法选择**。
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5031755

## 4. 中文研报与实践社区：最相关清单

### 4.1 必读研报（按优先级）
1. **华泰人工智能系列 17《数据标注方法实证》、18《调仓频率实证》（周频 XGBoost 最优的直接出处）、
   19《重采样技术检验过拟合》、22《基于 CSCV 框架的回测过拟合概率》、39《周频量价选股组合优化》、
   40《Qlib 体验》**——四篇正好回答"标签怎么定、频率怎么选、过拟合怎么测、工具怎么搭"。
   目录：https://quant-wiki.com/advanced/研报精选/华泰-人工智能系列/
   调仓频率 PDF：https://crm.htsc.com.cn/doc/2019/10750101/446ce42b-d197-4c5e-aeda-5fa2fa3707dd.pdf
2. **华泰《AlphaNet：因子挖掘神经网络》+《再探》**（端到端因子挖掘）；**《基于遗传规划的选股因子挖掘》
   +《遗传规划因子挖掘的 GPU 加速》（2024.2）**（个人可负担的自动挖掘最低门槛；GPU 版说明路线仍在演进）。
   https://bigquant.com/wiki/doc/gZhImiZjLC ；https://bigquant.com/wiki/doc/LdRKJvtssz
3. **东方证券 DFQ 系列**：90《遗传规划价量因子挖掘系统》、91《循环神经网络多频率因子挖掘》、
   93《集成模型在量价特征中的应用》、DFQ-HIST、DFQ-diversify（分布外泛化——防"风格一切换就失效"）。
   https://www.fxbaogao.com/detail/3773133 ；https://www.fxbaogao.com/detail/4139372
4. **开源证券微观结构系列**（理想振幅等）：在经典因子上做结构切割是个人挖因子性价比最高的路径；
   有知乎复现。https://zhuanlan.zhihu.com/p/156851121
5. **国盛《量价指纹（一）（二）》**：自监督量价特征；其二证明"标签混入收益方向会降低风格稳健性"
   → 支持用排序/相对强弱标签。https://www.fxbaogao.com/detail/5228818 ；https://www.fxbaogao.com/detail/5605349
6. 2024-2026 LLM 挖因子研报群（华泰《大模型+强化学习因子挖掘》2025.12、QuantaAlpha 等）：
   LLM 生成表达式/假设 + 进化/RL 筛选是主流形态。https://www.fxbaogao.com/detail/5339368

### 4.2 实践方法论文（强烈推荐精读）
- **quant67《机器学习选股：标签构造、防过拟合、SHAP 归因》**（已精读全文）：
  Rank+反正态标签折中方案；A 股 10 年面板 1200 万样本的"等价独立样本仅 20-50 万"（→ 13 个月面板
  等价样本 ≈ 几万，做什么复杂模型都不够）；Purged K-Fold+Embargo；1000 次实验夏普虚高 3.7σ；
  上线后"20 日 IC 均值跌破训练值 50% 即告警"；**调好的 LightGBM 比线性 IC 只提升 0.01-0.03——
  工程纪律远比模型重要**。https://quant67.com/post/quant/12-ml-alpha/12-ml-alpha.html
- 因子检验标准流程（多来源互证）：去极值 → 行业市值中性化（截面回归取残差）→ IC/ICIR/分层 →
  ICIR 加权或最大化 ICIR 合成；华泰《因子合成方法实证分析》测 6 种合成法。
  https://crm.htsc.com.cn/doc/2019/10750101/ae933a8c-5a94-4d92-8f33-d92b70c36119.pdf
- 回测必须写死的 A 股制度约束：涨停禁买/跌停禁卖、T+1、停牌冻结、含退市股票池。
  https://zhuanlan.zhihu.com/p/359675184 ；https://github.com/henrylin99/quantitative_analysis

### 4.3 与你直接相关的社区结论
- 排雷数据源（做成硬过滤层而非预测因子）：质押比例、商誉/净资产、Altman Z-Score、审计意见、
  减持公告频次、ST/退市风险。https://zhuanlan.zhihu.com/p/137141973 ；https://bigquant.com/wiki/doc/RUOERVTdda
- 小市值/微盘 2024-2026 三次过山车（DMA 踩踏 -50%、退市新规 -23%、2026 拥挤回撤 -28%）→
  市值暴露要么中性化、要么显式披露，不能默认当 alpha。
- A 股月频动量效应弱甚至反转（T+1 隔夜折价）→ 动量类因子按"短期+反转化处理"并做滚动有效性监控。

## 5. 开源项目选型

### 直接用
| 项目 | 用途 | 地址 |
|---|---|---|
| **microsoft/qlib** (≈48k★) | 研究底座：数据 bin 格式、Alpha158/360、qrun、IC/回测评估；周/月频改配置即可 | https://github.com/microsoft/qlib |
| **alphalens-reloaded** | 单因子 IC/分层/换手 tear sheet 标准件 | https://github.com/stefan-jansen/alphalens-reloaded |
| **gplearn** | GP 挖因子引擎（自定义算子 + IC 适应度） | https://github.com/trevorstephens/gplearn |
| **tsfresh / featuretools** | 滚动窗口批量自动特征 / 多表（行情×财务）交叉特征 | https://github.com/blue-yonder/tsfresh |
| **vectorbt** | 低频参数扫描、walk-forward（注意 Commons Clause） | https://github.com/polakowo/vectorbt |

### 抄思路/抄代码
- **RL-MLDM/alphagen**（AlphaGen 官方，默认 A 股+qlib 数据管线；抄 AlphaCalculator 与算子库、
  组合增量 IC 奖励）https://github.com/RL-MLDM/alphagen
- **microsoft/RD-Agent**（RD-Agent(Q)：`rdagent fin_factor` 全闭环，仅 Linux+Docker；抄"假设→代码→验证"架构）
- **RndmVariableQ/AlphaAgent**（KDD 2025；**自带 2015-2026 中证1000并集面板 parquet 网盘**，可在自建
  数据回补期间先行起步；因子 DSL + FactorZoo 评估 + AgentScope LLM 循环）https://github.com/RndmVariableQ/AlphaAgent
- **hudson-and-thames/mlfinlab**（当前商业闭源；抄旧版历史 commit 的 PurgedCV/CPCV/标签实现）
- **popbo/alphas**（Alpha101/191 向量化实现，基线因子库）https://github.com/popbo/alphas
- **SJTU-DMTai/MASTER**、**Wentao-Xu/HIST**（qlib 接口下的深度模型，二期）
- 不推荐整体使用：AlphaForge（8 commits 论文代码）、finhack（重构中跑不起来）、FinAgent（toy 级）

---

## 6. 重构路线图

### Phase 0 —— 数据回补（1-2 周，最高优先级，决定一切）
- **目标**：2014-01-01 至今全 A 日线面板（**含退市股**，`stock_basic` list_status 全集合并），字段：
  OHLCV（hfq 复权 + adj_factor）、daily_basic、moneyflow、LHB、sw 行业成分（历史）、指数日线
  （000300/000905/000852 等）。
- **新增低频风险数据（排雷数据化）**：fina_indicator（PIT 用 ann_date）、质押 pledge、解禁、
  namechange（ST 历史）、审计意见。
- **质量门（写进 audit）**：① 退市股数量核对（幸存者偏差检查）；② 财报 PIT 检查（ann_date 滞后）；
  ③ 复权因子连续性抽检；④ 停牌日不填充收益；⑤ 每张表落一份 schema 文档。
- **里程碑**：一张 ~500 万行 × 全 A × 10 年的长表 + PIT 财务表。5min 存档与本阶段无关，留着。

### Phase 1 —— 研究基础设施（1 周）
- qlib 为底座：把 Phase 0 长表转 qlib bin；自己保留一个带 A 股约束的向量化周频回测器
  （涨停禁买/跌停禁卖/T+1/双边成本 0.15-0.3%）做最终裁决——qlib 管研究速度，自研回测管真实性。
- **标签**（先定死，不再反复改）：未来 5 日（周频主）/20 日（月频稳健性）**相对全市场等权的超额收益**
  的**截面 rank**（可选 rank+反正态变换）；严格 asof 对齐，训练集剔除标签窗口（purge 5-20 日 + embargo）。
- **评估件**：RankIC 均值/ICIR、分位数多空、换手、费后；DSR 计算器 + **实验台账（每次尝试计数）**——
  这是把 audit/ 文化制度化的关键，Harvey t>3 与 DSR 都需要"尝试次数"。
- **里程碑**：任一因子值DataFrame 一条命令出全套诊断图。

### Phase 2 —— 复现已知因子，验证管线（1-2 周）
- 先手工复现 3-5 个有公开预期符号的 A 股因子：理想振幅（开源证券）、短期反转、Amihud 非流动性、
  换手率、20-60 日动量。中性化：申万 L1 + log 市值截面回归取残差，去极值 3σ，zscore。
- **里程碑**：符号与研报一致、IC 量级合理（|RankIC| 0.02-0.08）。**管线复现不出文献结果之前，
  不开始任何自动挖掘**——这一步就是给管线本身做 audit。

### Phase 3 —— 自动因子挖掘（2-4 周，核心）
1. **起步：gplearn GP**，适应度 = 与现有因子集去相关后的组合增量 RankIC（- λ·max|corr|），
   算子库抄 101 Formulaic Alphas + tsfresh 统计量；GPU 加速参考华泰 2024 GP 篇。
2. **进阶：AlphaGen（RL）**——把它的数据层换成 Phase 0 面单，奖励已经是组合增量 IC；单卡可跑。
3. **LLM 辅助（并行小成本试）**：RD-Agent(Q) 或 AlphaAgent 的挖掘循环 + 你的 LLM API；
   LLM 的正确用法是"研报逻辑/事件研究结论 → 候选因子表达式"的放大器，不当收益来源。
4. **抗衰减三件套**（AlphaAgent）：AST 原创性正则（与因子库同质化即拒）、复杂度惩罚、
   假设-因子语义一致性。每条因子入库需过：t>3（含尝试次数校正）、跨子期稳定性（2015/2018/2024
   危机窗口）、与现有库最大相关 <0.7。
- **里程碑**：10-20 条通过全检的新因子 + 因子库管理（表达式、覆盖度、IC 衰减曲线）。

### Phase 4 —— 合成、组合与精准买入（2 周）
- **合成**：先 ridge 线性堆叠 rank-z 因子（文献：线性与 LGBM 差距仅 0.01-0.03 IC），
  再对照 LGBM（特征只用入库因子，purged CV）；选简单者。
- **组合**：周频调仓、持有 10-20 只分散、no-trade band（新票得分不显著优于持仓不换）、
  费后评估；出具"频率×成本"敏感性表（周/双周/月 × 0.1%/0.2%/0.3%）。
- **精准买入 = meta-labeling**：主模型出买入清单 → 第二层模型 P(这笔交易盈利) 用你们的事件特征
  （封板强度、隔夜、早盘急拉 hold、波动状态 + Phase 0 排雷数据）训练；阈值可调"宁缺毋滥"。
  5 条件排雷作为硬过滤层保留，但用 10 年数据重新做剂量反应检验。
- **板块信息降级为特征**：行业动量/拥挤度/集中度特征 + （二期）HIST/MASTER 图先验。

### Phase 5 —— 验证协议与上线（持续）
- CPCV 出 Sharpe 分布而非单一路径；PBO < 50% 才可信；多种子 LGBM 稳定性。
- **前向纸面跟踪 4-8 周**（IC 与分层收益 vs 训练期基线），再小资金实盘。
- 在线监控：20 日滚动 RankIC 跌破训练值 50% → 告警降仓（quant67 阈值）。
- **预期锚**：月频 OOS R² ~0.1-0.4%、组合 RankIC 0.03-0.06、费后年化超额个位数到 15% 是优秀水平；
  v17 的 +1189% 是泄露产物，永远不要拿它当基准。

### 明确不做（与冻结决策一致）
- 不再演进日频 T+2 排序器、不再调 TP/SL 退出参数、KCB 单票维持"仅择时参考"。
- 深度模型（MASTER/HIST/Kronos）放二期：等 Phase 3 的因子库和 Phase 5 的协议就绪后再对比，
  否则又回到"换模型"的老路——你们 v13 已证明此路到头。

---

## 7. 本报告的未核实项
- QuantaAlpha、华泰《大模型+强化学习因子挖掘》(2025.12) 的券商归属以慧博/发现报告页面为准，未逐一确证。
- AlphaAgent 仓库 README 自述的数据面板规模（2757 票 / 620 万行）未实测。
- mlfinlab 旧版 commit 中 PurgedCV/CPCV 实现完整性需自行核对。
- quant67 博客中"1000 次实验夏普虚高 3.7σ"为该作者转述，原始出处即 Bailey & López de Prado DSR 系列。

（调研执行：三路并行检索——英文学术 21 条经 OpenAlex 核验、中文研报/社区 20 条、GitHub 仓库 20 个逐个访问核实；数据盘点脚本 inventory_data.py / inventory_dates.py 已跑完，可删除。）
