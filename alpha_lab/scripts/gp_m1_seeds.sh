#!/bin/bash
# 月频管线 (fwd20 标签, 20日采样): 5 种子
cd /root/autodl-tmp/alpha_lab
for SEED in 7 11 22 33 44; do
  echo "[$(date)] m1 seed $SEED"
  python -u scripts/run_gp_mining.py --gpu --pop 400 --gen 60 --tag m1_s$SEED --seed $SEED \
    --train-years 2014-2021 --horizon 20 > logs/gp_m1_s$SEED.log 2>&1
done
echo "[$(date)] 月频 m1 全部完成"
