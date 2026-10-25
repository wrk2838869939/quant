#!/usr/bin/env python3
"""Phase 0 第三批: 龙虎榜 top_list / top_inst 按交易日 (2014-至今)。

存储: data/raw/by_date/{top_list,top_inst}/{yyyymmdd}.parquet
复用 fetch_daily_panel 的日历与断点续传逻辑。
"""
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread, Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ts_client import fetch, save_day, done_days, RAW  # noqa: E402
import pandas as pd  # noqa: E402

START = "20140102"
END = pd.Timestamp.now().strftime("%Y%m%d")


def calendar():
    c = pd.read_parquet(RAW / "calendar.parquet")
    days = c[c["is_open"] == 1]["cal_date"].tolist()
    return [d for d in days if START <= d <= END]


def main():
    days = calendar()
    for api in ["top_list", "top_inst"]:
        done = done_days(api)
        miss = [d for d in days if d not in done]
        print(f"{api}: 待 {len(miss)} 天", flush=True)
        q = Queue()
        for d in miss:
            q.put(d)
        lock = Lock()
        cnt = {"ok": 0, "empty": 0, "fail": 0}
        t0 = time.time()

        def worker():
            while True:
                try:
                    day = q.get_nowait()
                except Exception:
                    return
                df = fetch(api, trade_date=day)
                with lock:
                    if df is None:
                        cnt["fail"] += 1
                    else:
                        cnt["ok" if len(df) else "empty"] += 1
                        save_day(df, api, day)
                    n = sum(cnt.values())
                    if n % 500 == 0:
                        el = time.time() - t0
                        print(f"  {api} {n}/{len(miss)} {cnt} | {el/60:.0f}min", flush=True)
                q.task_done()

        ts = [Thread(target=worker, daemon=True) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        print(f"{api} 完成 {cnt}", flush=True)


if __name__ == "__main__":
    main()
