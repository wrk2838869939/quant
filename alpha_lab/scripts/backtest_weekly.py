#!/usr/bin/env python3
"""月频组合审计入口（旧脚本名兼容）。

使用 --prepared 指向 prepare_monthly_audit.py 固定的信号与执行数据目录。
旧版按调仓日估值的实现已移入 legacy/backtest_weekly_v3.py，仅供审计复现。
"""
from audit_monthly_portfolio import main

if __name__ == "__main__":
    main()
