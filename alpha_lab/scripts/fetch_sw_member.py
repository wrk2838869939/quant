#!/usr/bin/env python3
"""Phase 0: 申万 L1 行业分类 + 历史成分 (含 in_date/out_date, PIT 正确)。

输出:
  data/raw/sw_l1_classify.parquet   31 个 L1 行业
  data/raw/sw_l1_member_hist.parquet 成分进出记录 (可用于任意时点重建行业归属)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ts_client import fetch, RAW  # noqa: E402


def main():
    cls = fetch("index_classify", src="SW2021", level="L1",
                fields="index_code,industry_name")
    cls.to_parquet(RAW / "sw_l1_classify.parquet", index=False)
    print(f"L1 行业 {len(cls)} 个")

    frames = []
    for code in cls["index_code"]:
        d = fetch("index_member", index_code=code,
                  fields="index_code,con_code,in_date,out_date,is_new")
        if d is not None and len(d):
            frames.append(d)
            print(f"  {code}: {len(d)} 条")
    m = pd.concat(frames, ignore_index=True)
    m["in_date"] = m["in_date"].astype(str)
    m["out_date"] = m["out_date"].fillna("20991231").astype(str)
    m.to_parquet(RAW / "sw_l1_member_hist.parquet", index=False)
    print(f"成分记录 {len(m):,} 条, 覆盖股票 {m['con_code'].nunique()} 只 -> sw_l1_member_hist.parquet")


if __name__ == "__main__":
    main()
