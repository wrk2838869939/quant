# 五日入场资格：历史回放结果

本轮保持全部个股阈值，只改变板块事件资格时间语义。数据截至2026-10-12；实时配置、模型及候选指针未修改。

## 机会数量

| 事件范围 | 版本 | 板块资格日 | 股票候选日 | 去重事件×股票 | 有股票的板块事件 | 候选日期 |
|---|---|---:|---:|---:|---:|---:|
| all_events | legacy_today | 897 | 72 | 69 | 51 | 40 |
| all_events | same_event_today | 875 | 71 | 68 | 50 | 39 |
| all_events | persistent_5d | 1282 | 82 | 75 | 56 | 46 |
| random_test_selected | legacy_today | 228 | 9 | 9 | 8 | 7 |
| random_test_selected | same_event_today | 226 | 9 | 9 | 8 | 7 |
| random_test_selected | persistent_5d | 324 | 11 | 11 | 10 | 9 |
| chronological_selected | legacy_today | 304 | 27 | 26 | 19 | 20 |
| chronological_selected | same_event_today | 298 | 27 | 26 | 19 | 20 |
| chronological_selected | persistent_5d | 443 | 32 | 27 | 19 | 24 |

all_events是不经过模型的结构对照；random_test_selected是原随机留出且FNN通过事件，含跨时间训练问题；chronological_selected按年初已成熟标签重训，固定4/8规则两组模型取或。三个范围不能混作同一总体。

legacy_today保留当日代表窗口确认；same_event_today要求与原确认同窗口、同事件、同峰值；persistent_5d保留确认日及后4个交易日资格，结构失效后永久取消。

## 五日保留相对严格同事件当天确认的变化

scope                   status   
all_events              added         7
                        earlier       1
                        unchanged    67
chronological_selected  added         1
                        earlier       1
                        unchanged    25
random_test_selected    added         2
                        unchanged     9

## 次日开盘价格观察

以下仅是复权开盘价格观察及固定成本压力测试：买入10bp、卖出20bp、双边各5bp滑点；没有整手最小佣金、正式涨跌停与停牌匹配、资金占用、公司行动现金账本。不能当作可实现收益或组合最大回撤。缺失路径与截尾数量单独披露。

```
                 scope          variant  horizon subset  opportunities  complete  incomplete  sector_events      mean    median       q10  positive   minimum  mean_minimum_close
            all_events     legacy_today        3    all             69        69           0             51 -0.013264 -0.015377 -0.083161  0.405797 -0.165653           -0.029250
            all_events     legacy_today       10    all             69        69           0             51 -0.007344  0.002251 -0.106213  0.507246 -0.263359           -0.064021
            all_events     legacy_today       20    all             69        69           0             51  0.021895  0.004328 -0.135997  0.536232 -0.297574           -0.087311
            all_events    persistent_5d        3    all             75        75           0             56 -0.009819 -0.015377 -0.081943  0.413333 -0.165653           -0.026930
            all_events    persistent_5d        3  added              7         7           0              7  0.024000  0.010924 -0.023605  0.571429 -0.027515           -0.003762
            all_events    persistent_5d       10    all             75        75           0             56 -0.009920 -0.004604 -0.114512  0.480000 -0.263359           -0.062837
            all_events    persistent_5d       10  added              7         7           0              7 -0.048808 -0.043608 -0.134898  0.142857 -0.157563           -0.055110
            all_events    persistent_5d       20    all             75        74           1             55  0.027560  0.010275 -0.123016  0.567568 -0.297574           -0.086631
            all_events    persistent_5d       20  added              7         6           1              6  0.069361  0.064679 -0.034904  0.833333 -0.091145           -0.083586
            all_events same_event_today        3    all             68        68           0             50 -0.013504 -0.016248 -0.083680  0.397059 -0.165653           -0.029501
            all_events same_event_today       10    all             68        68           0             50 -0.006068  0.002989 -0.106261  0.514706 -0.263359           -0.063692
            all_events same_event_today       20    all             68        68           0             50  0.023496  0.004782 -0.138063  0.544118 -0.297574           -0.087130
chronological_selected     legacy_today        3    all             26        26           0             19 -0.019270 -0.027370 -0.114506  0.384615 -0.165653           -0.033820
chronological_selected     legacy_today       10    all             26        26           0             19  0.012107  0.013948 -0.084969  0.615385 -0.209289           -0.073362
chronological_selected     legacy_today       20    all             26        26           0             19  0.048750  0.034218 -0.117116  0.653846 -0.170621           -0.099697
chronological_selected    persistent_5d        3    all             27        27           0             19 -0.013418 -0.027219 -0.109736  0.407407 -0.165653           -0.028192
chronological_selected    persistent_5d        3  added              1         1           0              1  0.124855  0.124855  0.124855  1.000000  0.124855            0.105473
chronological_selected    persistent_5d       10    all             27        27           0             19  0.009872  0.011397 -0.080098  0.592593 -0.209289           -0.071988
chronological_selected    persistent_5d       10  added              1         1           0              1 -0.058507 -0.058507 -0.058507  0.000000 -0.058507           -0.040347
chronological_selected    persistent_5d       20    all             27        27           0             19  0.050236  0.047791 -0.114166  0.666667 -0.169369           -0.098973
chronological_selected    persistent_5d       20  added              1         1           0              1  0.063277  0.063277  0.063277  1.000000  0.063277           -0.095818
chronological_selected same_event_today        3    all             26        26           0             19 -0.019270 -0.027370 -0.114506  0.384615 -0.165653           -0.033820
chronological_selected same_event_today       10    all             26        26           0             19  0.012107  0.013948 -0.084969  0.615385 -0.209289           -0.073362
chronological_selected same_event_today       20    all             26        26           0             19  0.048750  0.034218 -0.117116  0.653846 -0.170621           -0.099697
  random_test_selected     legacy_today        3    all              9         9           0              8 -0.025612 -0.015377 -0.082207  0.333333 -0.162606           -0.041593
  random_test_selected     legacy_today       10    all              9         9           0              8  0.002961  0.025738 -0.059957  0.555556 -0.207924           -0.101448
  random_test_selected     legacy_today       20    all              9         9           0              8  0.019139  0.072754 -0.130936  0.666667 -0.297574           -0.123776
  random_test_selected    persistent_5d        3    all             11        11           0             10 -0.021871 -0.015377 -0.062107  0.363636 -0.162606           -0.041708
  random_test_selected    persistent_5d        3  added              2         2           0              2 -0.005037 -0.005037 -0.017807  0.500000 -0.020999           -0.042221
  random_test_selected    persistent_5d       10    all             11        11           0             10 -0.012706 -0.004604 -0.157563  0.454545 -0.207924           -0.097728
  random_test_selected    persistent_5d       10  added              2         2           0              2 -0.083208 -0.083208 -0.142692  0.000000 -0.157563           -0.080987
  random_test_selected    persistent_5d       20    all             11        11           0             10  0.048079  0.098680 -0.089277  0.727273 -0.297574           -0.119100
  random_test_selected    persistent_5d       20  added              2         2           0              2  0.178309  0.178309  0.137412  1.000000  0.127187           -0.098061
  random_test_selected same_event_today        3    all              9         9           0              8 -0.025612 -0.015377 -0.082207  0.333333 -0.162606           -0.041593
  random_test_selected same_event_today       10    all              9         9           0              8  0.002961  0.025738 -0.059957  0.555556 -0.207924           -0.101448
  random_test_selected same_event_today       20    all              9         9           0              8  0.019139  0.072754 -0.130936  0.666667 -0.297574           -0.123776
```

## 有效性边界

- 时间推进模型仅使用成熟且年初前20交易日已结束的标签；训练样本不足60的年份不评分。全部训练ID、截止日、门槛和权重保存。固定超参数来自已观察历史，仍是历史开发复核，不是新的独立测试。
- 历史行业成员和名称区间使用当时生效区间，没有用当前在市状态筛选历史；名称缺失或多条冲突拒绝。供应商历史修订与发布时间PIT尚未完整认证。
- 当前在线补充成员无法完全历史复原；本轮使用冻结历史成员，不能称原实时扫描逐股票完全一致。板块331个原事件已逐条精确复原。
- 不能用同一板块多股票、多日重复信号扩大独立样本数；新增/消失/提前入场分别记录。
- execution_coverage.csv单独核对计划买卖日的原始开盘、成交量及已有官方涨跌停缓存；即使这些通过，也没有完成整手现金账户与公司行动结算验证。价格观察汇总不会把缺少限价资料的样本冒充可交易样本。
- 生产采用还须新增机会成交与账户风险验证，本次没有自动部署。

## 复现

运行目录：`alpha_lab/outputs/sector_entry_window_v1/20261013T064932_361598Z`。先经frozen_research.py分别运行research_sector_entry_window.py的events、models、stocks阶段，再运行本汇总脚本。
