# TEST 窗终评报告 (2026-09-09, 一次性)

- 面板: 3084 日 | TEST 截面 131 | 合成权重定于 TRAIN+VALID
- 实验台账尝试数 58 → 合成因子 DSR p = 0.0000

## 单因子 TEST IC

                     factor   tv_ic   te_ic  te_icir  te_t  te_t_nw  te_n  sign_stable
           lit:low_turnover  0.0631  0.0603     0.29   3.3      3.0   130         True
                lit:low_amp  0.0726  0.0654     0.28   3.2      2.8   130         True
                lit:low_vol  0.0695  0.0667     0.29   3.3      3.0   130         True
                  lit:rev20  0.0565  0.0711     0.43   4.9      5.1   130         True
                   lit:rev5  0.0411  0.0263     0.17   2.0      2.3   130         True
                 lit:amihud  0.0447  0.0351     0.22   2.5      2.3   130         True
                lit:mom60s5 -0.0350 -0.0583    -0.33  -3.7     -3.3   130         True
        gp:sqrt(zscore(to)) -0.0957 -0.1181    -1.33 -15.2    -14.4   130         True
   gp:sqrt(ts_delta(vol,5)) -0.0905 -0.0799    -0.78  -8.9     -8.3   130         True
gp:sqrt(ts_zscore(amt,60),2 -0.0707 -0.0692    -0.86  -9.8    -10.2   130         True
        gp:ts_delta(amt,60) -0.0572 -0.0710    -0.62  -7.0     -6.1   130         True
      gp:ts_min(vwap_dev,5)  0.0664  0.0560     0.37   4.3      4.3   130         True
     gp:corr(hclose,rng,10) -0.0664 -0.0626    -0.63  -7.1     -6.9   130         True
                     lit:ep  0.0287  0.0294     0.15   1.8      1.7   130         True
                     lit:bp  0.0394  0.0484     0.28   3.2      3.0   130         True
        SYNTH:icir_weighted  0.1177  0.1321     0.91  10.4     10.6   130         True

## 合成因子十分组 (TEST, 等权持有5日/期)

Q1:-0.0210 | Q2:-0.0082 | Q3:-0.0059 | Q4:-0.0032 | Q5:-0.0004 | Q6:+0.0017 | Q7:+0.0001 | Q8:+0.0014 | Q9:+0.0039 | Q10:+0.0047

多空(Q10-Q1) 每期: +0.0257 | 年化≈+125.5%

生成: factor_test_report.csv / pool CSV