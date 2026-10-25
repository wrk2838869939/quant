#!/bin/bash
# v3: 扩充序列(20个) + 历届PASS入库惩罚, 5 种子
cd /root/autodl-tmp/alpha_lab
LIBS="outputs/gp_mining/harvest_v2_s101.csv,outputs/gp_mining/harvest_v2_s202.csv,outputs/gp_mining/harvest_v2_s303.csv"
for SEED in 11 22 33 44 55; do
  echo "[$(date)] v3 seed $SEED"
  python -u scripts/run_gp_mining.py --gpu --pop 400 --gen 60 --tag v3_s$SEED --seed $SEED \
    --train-years 2014-2021 --extra-libs $LIBS > logs/gp_v3_s$SEED.log 2>&1
done
echo "[$(date)] v3 全部完成"
