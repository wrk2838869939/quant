# 首轮结果与复现入口

实验分支：`experiment/sector-fuzzy-selection-20260923`。实现与初始方案提交：`783c93a`。

本轮可运行，但尚未证明模糊筛选能使收益分布整体稳定右移。2019—2026年共255个历史滚动测试事件，20日完整窗口226个；其余75个早期事件用于建立训练历史。因此本轮不筛选基准为+0.77%，不能直接拿全部330事件报告的+1.05%作对照。

| 20日主口径 | 完整事件数 | 均值 | 中位数 | 10%分位 |
|---|---:|---:|---:|---:|
| 不筛选 | 226 | +0.77% | -0.33% | -8.30% |
| 固定模糊规则 | 114 | +0.37% | -0.57% | -7.19% |
| 岭回归对照 | 102 | +1.14% | -0.42% | -7.82% |
| 模糊神经网络 | 91 | +1.49% | -0.05% | -8.57% |

模糊网络均值改善0.71个百分点，95%行情簇重采样区间为[-0.94,+1.94]个百分点；左尾略恶化。逐年均值相对基准4年改善、4年退化。剔除最大行情簇后均值增量仍为+0.54个百分点，10%分位增量为-1.13个百分点，仍未通过整体右移判据。训练保留参数70%的敏感性也未通过判据。固定模糊规则更像规避左尾，但均值和中位数下降。

## 学习过程

13项确认时特征。每年训练61—239个完整事件；内层按日期截出验证段，训练标签成熟后额外隔离5个交易日。每个规则数×学习率组合运行2种子，验证选组合与轮数，再按3种子集成。最终重训轮数100—330，最大训练预算1500轮；验证曲线多次在训练损失继续下降时上升，实际由验证早停确定轮数。记录包括全部候选组合，不只保留赢家。

当前目标是Huber收益回归，对左尾没有单独建模。若继续研究，可另开预设方案测试分位数后件或收益与左尾双目标；不能因为看到了此次测试结果就宣称下一版是全新未见数据验证。

## 产物

- 主报告：`alpha_lab/outputs/sector_fuzzy_v1/20260923T105203_887024Z/README.md`
- 图册：同目录 `fuzzy_selection_report.pdf`
- 训练过程：同目录 `inner_search.csv`、`training_curves.csv`、`folds.csv`、`models/`
- 独立重放：`alpha_lab/outputs/sector_fuzzy_audit/20260923T105932_720843Z/manifest.json`

4项测试通过；真实特征截断重算一致；1530条模型评分、阈值与选中标记重放通过，训练期填充/标准化参数逐值一致，修改测试年及之后标签不改变当年的训练样本。主报告63个文件哈希通过；另存公共依赖源码与运行库版本。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python datahub/scripts/frozen_research.py run --release datahub/releases/selection_v1_20260920 alpha_lab/experiments/sector_fuzzy_v1/run.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python datahub/scripts/frozen_research.py run --release datahub/releases/selection_v1_20260920 alpha_lab/experiments/sector_fuzzy_v1/audit.py alpha_lab/outputs/sector_fuzzy_v1/20260923T105203_887024Z
```

运行依赖本工作区已有的冻结数据和板块研究产物，未把数据文件加入Git。本实验不涉及实盘部署或账户收益验证。
