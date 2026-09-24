# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
# -*- coding: utf-8 -*-
"""下载全部美股隔夜数据到本地缓存 (US ETF/ADR/指数/美债)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from usx.us_data import update_cache  # noqa: E402

if __name__ == "__main__":
    res = update_cache()
    fail = [k for k, v in res.items() if v == 0]
    print("\n=== SUMMARY ===")
    print(f"ok={sum(1 for v in res.values() if v > 0)} fail={len(fail)} {fail if fail else ''}")
