# TEST 窗终评报告 (2026-09-09, 一次性)

- 面板: 3084 日 | TEST 截面 33 | 合成权重定于 TRAIN+VALID
- 实验台账尝试数 58 → 合成因子 DSR p = 0.0000

## 单因子 TEST IC

                factor   tv_ic   te_ic  te_icir  te_t  te_t_nw  te_n  sign_stable
      lit:low_turnover  0.1008  0.0776     0.38   2.1      3.3    32         True
           lit:low_amp  0.1066  0.0742     0.31   1.7      3.0    32         True
           lit:low_vol  0.0997  0.0822     0.37   2.1      3.2    32         True
             lit:rev20  0.0716  0.0868     0.50   2.8      2.9    32         True
              lit:rev5  0.0361  0.0800     0.76   4.3      3.8    32         True
            lit:amihud  0.0813  0.0436     0.22   1.2      1.2    32         True
           lit:mom60s5 -0.0543 -0.0581    -0.31  -1.8     -1.9    32         True
 gp:ts_min(vwap_dev,5)  0.0868  0.0914     0.67   3.8      5.1    32         True
gp:ts_std(lg_share,10)  0.0726  0.0423     0.44   2.5      3.0    32         True
                lit:ep  0.0466  0.0423     0.22   1.3      1.7    32         True
                lit:bp  0.0622  0.0638     0.36   2.0      3.7    32         True
   SYNTH:icir_weighted  0.1250  0.1054     0.49   2.8      4.1    32         True

## 合成因子十分组 (TEST, 等权持有5日/期)

Q1:+0.0012 | Q2:+0.0132 | Q3:+0.0141 | Q4:+0.0145 | Q5:+0.0154 | Q6:+0.0159 | Q7:+0.0153 | Q8:+0.0147 | Q9:+0.0148 | Q10:+0.0159

多空(Q10-Q1) 每期: +0.0146 | 年化≈+17.8%

生成: factor_test_report.csv / pool CSV