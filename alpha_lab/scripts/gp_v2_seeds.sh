#!/bin/bash
# GPU 多种子扫描: 3 种子 × pop400 × gen60 (防单一种子的遗传漂移)
cd /root/autodl-tmp/alpha_lab
for SEED in 101 202 303; do
  echo "[$(date)] seed $SEED 启动"
  python -u scripts/run_gp_mining.py --gpu --pop 400 --gen 60 --tag v2_s$SEED --seed $SEED \
    --train-years 2014-2021 > logs/gp_v2_s$SEED.log 2>&1
done
echo "[$(date)] v2 全部完成"
