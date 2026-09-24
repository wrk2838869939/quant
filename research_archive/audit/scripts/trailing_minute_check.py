# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
"""移动止盈的分钟级验证: 消除日内高低顺序假设。

对 KCB 生产口径 (月度模型, trailing 2.5%/1.25%) 的交易,
用 1 分钟 K 线精确重放退出路径, 与日线口径对比:
  - 日线口径: 同日 high/low 顺序未知, 假设「先创新高后触线」(有利)
  - 分钟口径: 逐 bar 推进, 多次扫线/先跌后涨都被正确处理

样本: roll_monthly_lag1_bndfix 中 buy_date ≤ 2026-06-10 的交易 (分钟数据到 06-12)。
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

TRIG, DIST = 0.025, 0.0125
MIN_PATH = "kcb_sniper/data/raw/min1_minishare/kcb_min1_all.parquet"
TRADES = "kcb_v18/outputs/roll_monthly_lag1_bndfix/trades.csv"

trades = pd.read_csv(TRADES, parse_dates=["buy_date", "sell_date"])
trades["code"] = trades["code"].astype(str).str.zfill(6)
trades = trades[trades["buy_date"] <= "2026-06-10"].copy()
codes = sorted(trades["code"].unique())
print(f"样本: {len(trades)} 笔, {len(codes)} 只股票 (buy ≤ 06-10)")

# 只读需要的股票列 (pyarrow 过滤)
import pyarrow.parquet as pq
mins = pq.read_table(MIN_PATH, columns=["时间", "股票代码", "开盘", "收盘", "最高", "最低"],
                     filters=[("股票代码", "in", codes)]).to_pandas()
mins["时间"] = pd.to_datetime(mins["时间"])
mins = mins.sort_values(["股票代码", "时间"]).reset_index(drop=True)
mins["date"] = mins["时间"].dt.normalize()
by_code = {c: g for c, g in mins.groupby("股票代码")}
print(f"分钟数据: {len(mins):,} 行")


def replay_minute(code, buy_date, buy_price, tp, sl, hold_days):
    """1分钟逐bar重放。返回 (exit_price, hit, n_crossings_extra)"""
    g = by_code.get(code)
    if g is None:
        return None
    hd = g[g["date"] > buy_date]
    if hd.empty:
        return None
    days = sorted(hd["date"].unique())[:int(hold_days) + 1]
    hd = hd[hd["date"].isin(days)]

    highest = buy_price
    trailing = False
    trail_sl = buy_price * (1 - sl)
    for day, dg in hd.groupby("date"):
        # 跳空检查 (当日第一根 bar 的开盘价)
        day_open = dg["开盘"].iloc[0]
        if trailing and day_open <= trail_sl:
            return day_open, "TSL_gap"
        if not trailing:
            tp_p, sl_p = buy_price * (1 + tp), buy_price * (1 - sl)
            if day_open >= tp_p:
                return day_open, "TP_gap"
            if day_open <= sl_p:
                return day_open, "SL_gap"
        for _, b in dg.iterrows():
            hi, lo = b["最高"], b["最低"]
            if pd.notna(hi):
                highest = max(highest, hi)
            if not trailing and highest / buy_price - 1 >= TRIG:
                trailing = True
                trail_sl = highest * (1 - DIST)
            if trailing:
                if pd.notna(lo) and lo <= trail_sl:
                    return trail_sl, "TSL"
            else:
                if pd.notna(hi) and hi >= buy_price * (1 + tp):
                    return buy_price * (1 + tp), "TP"
                if pd.notna(lo) and lo <= buy_price * (1 - sl):
                    return buy_price * (1 - sl), "SL"
    return hd["收盘"].iloc[-1], "HOLD"


rows = []
for _, t in trades.iterrows():
    r = replay_minute(t["code"], t["buy_date"], t["buy_price"], t["tp"], t["sl"], t["hold_days"])
    if r is None:
        continue
    px_min, hit_min = r
    ret_min = px_min / t["buy_price"] - 1
    rows.append({
        "code": t["code"], "buy_date": t["buy_date"], "buy_price": t["buy_price"],
        "ret_daily": t["ret"], "hit_daily": t["hit"],
        "ret_min": ret_min, "hit_min": hit_min,
        "diff": ret_min - t["ret"],
    })
    # 同一交易日内高低顺序导致的差异标记
m = pd.DataFrame(rows)
print(f"\n成功重放: {len(m)} 笔\n")

print("=" * 76)
print("日线口径 vs 分钟口径")
print("=" * 76)
print(f"  笔均收益:  日线 {m['ret_daily'].mean():+.2%}   分钟 {m['ret_min'].mean():+.2%}   "
      f"差 {m['diff'].mean():+.2%}")
print(f"  胜率:      日线 {(m['ret_daily']>0).mean():.0%}   分钟 {(m['ret_min']>0).mean():.0%}")
print(f"  收益差分布: p10 {m['diff'].quantile(.1):+.2%}  中位 {m['diff'].median():+.2%}  "
      f"p90 {m['diff'].quantile(.9):+.2%}")
print(f"  分钟口径更差的占比: {(m['diff']<-0.002).mean():.0%}   更好的: {(m['diff']>0.002).mean():.0%}   "
      f"基本一致: {(m['diff'].abs()<=0.002).mean():.0%}")

print("\n按日线退出类型分:")
for hit, g in m.groupby("hit_daily"):
    print(f"  {hit:5s} n={len(g):3d}  日线 {g['ret_daily'].mean():+.2%} → 分钟 {g['ret_min'].mean():+.2%}  "
          f"(差 {g['diff'].mean():+.2%})")

print("\n分钟口径下最差10笔 (vs 日线):")
worst = m.nsmallest(10, "diff")
print(worst[["buy_date", "code", "ret_daily", "ret_min", "diff", "hit_daily", "hit_min"]].to_string(
    index=False, formatters={"ret_daily": "{:+.2%}".format, "ret_min": "{:+.2%}".format,
                             "diff": "{:+.2%}".format}))

m.to_csv("audit/outputs/trailing_minute_check.csv", index=False)
print("\n明细: audit/outputs/trailing_minute_check.csv")
