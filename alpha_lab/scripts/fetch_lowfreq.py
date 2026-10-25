#!/usr/bin/env python3
"""Phase 0 第二批: 低频风险数据 + 股票主档 + 指数日线。

- stock_basic: L+D 全集 (含退市股 delist_date) → data/raw/stock_basic_all.parquet
- namechange:  ST/名称变更历史 (尝试全量, 失败则按 ts_code)
- fina_indicator: PIT 财务指标 (按 ts_code 拉全历史, 保留 ann_date)
- share_float:  解禁 (按 ts_code 拉全历史)
- index_daily:  主要指数 2014-至今
- index_dailybasic: 指数估值(可选)

用法: python fetch_lowfreq.py [all|basic|namechange|fina|float|index]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ts_client import fetch, RAW  # noqa: E402
import pandas as pd  # noqa: E402

INDEXES = ["000001.SH", "000300.SH", "000905.SH", "000852.SH", "399001.SZ", "399006.SZ", "000688.SH", "399303.SZ"]


def job_stock_basic():
    out = RAW / "stock_basic_all.parquet"
    frames = []
    for st in ["L", "D", "P"]:
        d = fetch("stock_basic", exchange="", list_status=st,
                  fields="ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date")
        if d is not None and len(d):
            frames.append(d)
            print(f"  stock_basic {st}: {len(d)}")
    if frames:
        allb = pd.concat(frames).drop_duplicates("ts_code").reset_index(drop=True)
        allb.to_parquet(out, index=False)
        print(f"stock_basic 全集 {len(allb)} 只 -> {out.name} (退市 {int((allb.list_status=='D').sum())})")


def job_namechange():
    out = RAW / "namechange_all.parquet"
    # 先试全量 (该接口部分代理支持不带参数)
    d = fetch("namechange", fields="ts_code,name,start_date,end_date,ann_date,change_reason")
    if d is not None and len(d) > 10000:
        d.to_parquet(out, index=False)
        print(f"namechange 全量 {len(d)} 行 -> {out.name}")
        return
    # 兜底: 按 ts_code
    codes = pd.read_parquet(RAW / "stock_basic_all.parquet")["ts_code"].tolist()
    frames = []
    for i, c in enumerate(codes):
        dd = fetch("namechange", ts_code=c)
        if dd is not None and len(dd):
            frames.append(dd)
        if (i + 1) % 500 == 0:
            print(f"  namechange {i+1}/{len(codes)}", flush=True)
    if frames:
        pd.concat(frames).to_parquet(out, index=False)
        print(f"namechange 按股 {sum(len(f) for f in frames)} 行 -> {out.name}")


def job_index():
    out = RAW / "index_daily.parquet"
    frames = []
    for code in INDEXES:
        d = fetch("index_daily", ts_code=code, start_date="20140101", end_date="20261231")
        if d is not None and len(d):
            frames.append(d)
            print(f"  index {code}: {len(d)}")
    if frames:
        pd.concat(frames).to_parquet(out, index=False)
        print(f"index_daily {sum(len(f) for f in frames)} 行 -> {out.name}")
    # 指数估值(可失败)
    d = fetch("index_dailybasic", ts_code="000300.SH", start_date="20140101", end_date="20261231")
    if d is not None and len(d):
        d.to_parquet(RAW / "index_dailybasic.parquet", index=False)
        print(f"index_dailybasic 300: {len(d)} 行")


def _iter_codes():
    return pd.read_parquet(RAW / "stock_basic_all.parquet")["ts_code"].tolist()


def job_fina():
    """fina_indicator PIT: 按 ts_code 拉全历史, 分片落盘断点续传。"""
    out_dir = RAW / "fina_indicator_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = _iter_codes()
    done = {p.stem for p in out_dir.glob("*.parquet")}
    todo = [c for c in codes if c not in done]
    print(f"fina_indicator: {len(codes)} 只, 已 {len(done)}, 待 {len(todo)}")
    # 不传 fields 拉全列 (代理支持的列集与官方文档不完全一致, 传错字段名=永久失败)
    n_ok = 0
    for i, c in enumerate(todo):
        d = fetch("fina_indicator", ts_code=c, start_date="20130101", end_date="20261231")
        if d is None:
            d = pd.DataFrame()
        d.to_parquet(out_dir / f"{c}.parquet", index=False)
        if len(d):
            n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"  fina {i+1}/{len(todo)} (非空 {n_ok})", flush=True)
    print(f"fina_indicator 完成: 非空 {n_ok}/{len(todo)}")

def job_float():
    """share_float 解禁: 按 ts_code 拉全历史。"""
    out_dir = RAW / "share_float_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = _iter_codes()
    done = {p.stem for p in out_dir.glob("*.parquet")}
    todo = [c for c in codes if c not in done]
    print(f"share_float: {len(codes)} 只, 已 {len(done)}, 待 {len(todo)}")
    flds = "ts_code,ann_date,float_date,float_share,float_ratio,holder_name,share_type"
    n_ok = 0
    for i, c in enumerate(todo):
        d = fetch("share_float", ts_code=c, start_date="20130101", end_date="20261231", fields=flds)
        if d is None:
            d = pd.DataFrame()
        d.to_parquet(out_dir / f"{c}.parquet", index=False)
        if len(d):
            n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"  float {i+1}/{len(todo)} (非空 {n_ok})", flush=True)
    print(f"share_float 完成: 非空 {n_ok}/{len(todo)}")


if __name__ == "__main__":
    jobs = sys.argv[1:] or ["all"]
    if "all" in jobs or "basic" in jobs:
        job_stock_basic()
    if "all" in jobs or "index" in jobs:
        job_index()
    if "all" in jobs or "namechange" in jobs:
        job_namechange()
    if "all" in jobs or "fina" in jobs:
        job_fina()
    if "all" in jobs or "float" in jobs:
        job_float()
