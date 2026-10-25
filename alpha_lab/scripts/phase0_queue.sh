#!/bin/bash
# Phase 0 排队执行 (串行, 严守代理2并发): 主力日线 → 主档/指数 → SW成分 → LHB → 财务/解禁
cd /root/autodl-tmp/alpha_lab
echo "[$(date)] 等待 fetch_panel 完成..."
while screen -ls 2>/dev/null | grep -q fetch_panel; do sleep 120; done
echo "[$(date)] 主力完成 -> lowfreq basic+index"
python -u scripts/fetch_lowfreq.py basic index >> logs/fetch_lowfreq.log 2>&1
echo "[$(date)] -> SW 行业历史成分"
python -u scripts/fetch_sw_member.py >> logs/fetch_sw_member.log 2>&1
echo "[$(date)] -> LHB 龙虎榜"
python -u scripts/fetch_lhb.py >> logs/fetch_lhb.log 2>&1
echo "[$(date)] -> namechange+fina+float"
python -u scripts/fetch_lowfreq.py namechange fina float >> logs/fetch_lowfreq.log 2>&1
echo "[$(date)] Phase 0 下载队列全部完成"
