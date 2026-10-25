#!/usr/bin/env python3
"""Phase 0 主力下载器: 4 核心接口按交易日拉全市场快照。

接口: daily / adj_factor / daily_basic / moneyflow
区间: 2014-01-01 ~ 今天 (自动检测已有交易日历)
存储: data/raw/by_date/{api}/{yyyymmdd}.parquet  (断点续传按天粒度)
并发: 2 worker (代理限 2 并发), 队列按交易日分配。

用法: nohup python fetch_daily_panel.py > logs/fetch_daily.log 2>&1 &
"""
import sys
import time
from queue import Queue
from threading import Thread, Lock

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
from ts_client import fetch, save_day, done_days, RAW  # noqa: E402

APIS = ["daily", "adj_factor", "daily_basic", "moneyflow"]
START = "20140102"
END = __import__("datetime").date.today().strftime("%Y%m%d")  # 只拉到今天

# ---- 交易日历 ----
CAL_FILE = RAW / "calendar.parquet"


def build_calendar():
    """交易日历 2014-至今, 落盘缓存。"""
    import pandas as pd
    if CAL_FILE.exists():
        c = pd.read_parquet(CAL_FILE)
        days = c[c["is_open"] == 1]["cal_date"].tolist()
        return [d for d in days if START <= d <= END]
    from ts_client import get_pro
    pro = get_pro()
    frames = []
    for y in range(2014, 2027):
        d = pro.trade_cal(exchange="SSE", start_date=f"{y}0101", end_date=f"{y}1231",
                          is_open="1", fields="exchange,cal_date,is_open,pretrade_date")
        if d is not None and len(d):
            frames.append(d)
    cal = pd.concat(frames).sort_values("cal_date").reset_index(drop=True)
    cal["is_open"] = cal["is_open"].astype(int)  # 代理返回字符串, 统一转 int
    CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(CAL_FILE, index=False)
    days = cal[cal["is_open"] == 1]["cal_date"].tolist()
    return [d for d in days if START <= d <= END]


def main():
    days = build_calendar()
    print(f"交易日 {len(days)} 天: {days[0]} ~ {days[-1]}", flush=True)

    # 任务队列: (api, day), 跳过已完成
    todo = []
    for api in APIS:
        done = done_days(api)
        miss = [d for d in days if d not in done]
        print(f"  {api}: 已 {len(done)} / {len(days)}, 待 {len(miss)}", flush=True)
        todo += [(api, d) for d in miss]
    if not todo:
        print("全部已完成")
        return

    q = Queue()
    for t in todo:
        q.put(t)
    lock = Lock()
    stats = {"ok": 0, "empty": 0, "fail": 0}
    fails = []
    t0 = time.time()

    def worker():
        while True:
            try:
                api, day = q.get_nowait()
            except Exception:
                return
            df = fetch(api, trade_date=day)
            with lock:
                if df is None:
                    stats["fail"] += 1
                    fails.append((api, day))
                elif len(df) == 0:
                    stats["empty"] += 1
                    save_day(df, api, day)  # 空也落盘, 防反复拉
                else:
                    stats["ok"] += 1
                    save_day(df, api, day)
                n = stats["ok"] + stats["empty"] + stats["fail"]
                if n % 200 == 0:
                    el = time.time() - t0
                    eta = el / n * (len(todo) - n) / 60
                    print(f"  {n}/{len(todo)} ok={stats['ok']} empty={stats['empty']} "
                          f"fail={stats['fail']} | {el/60:.0f}min ETA {eta:.0f}min", flush=True)
            q.task_done()

    threads = [Thread(target=worker, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"完成: {stats} | 用时 {(time.time()-t0)/60:.0f}min", flush=True)
    if fails:
        import pandas as pd
        pd.DataFrame(fails, columns=["api", "day"]).to_csv(RAW / "fetch_fails.csv", index=False)
        print(f"失败 {len(fails)} 条已存 fetch_fails.csv (重跑本脚本即可补拉)")


if __name__ == "__main__":
    main()
