#!/bin/bash
# 等 Phase 0 全部下载完成 → 自动: 拼面板 → 质量门 → 文献因子复现 (Phase 2 正式跑)
cd /root/autodl-tmp/alpha_lab
echo "[$(date)] 等待 phase0_queue 完成..."
while screen -ls 2>/dev/null | grep -q phase0_queue; do sleep 300; done
echo "[$(date)] 拼接统一面板..."
python -u scripts/build_panel.py > logs/build_panel.log 2>&1 || { echo "build_panel 失败"; exit 1; }
echo "[$(date)] 数据质量门..."
python -u scripts/data_quality_gate.py > logs/quality_gate.log 2>&1
echo "[$(date)] Phase 2 文献因子复现..."
python -u scripts/replicate_factors.py > logs/replicate_factors.log 2>&1
echo "[$(date)] 全链完成: docs/QUALITY_REPORT.md + outputs/phase2_replication/replication_summary.csv"
