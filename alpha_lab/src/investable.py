"""信号日可知的股票池约束；次日是否成交仍交给现金账本判断。"""
import numpy as np
import pandas as pd

from portfolio import board_of


def investable_signals(signals, close, volume, basics, st_intervals, cfg, price_multiplier):
    """按固定每席预算与信号日价格预留跳空空间，不使用下一交易日行情。"""
    if price_multiplier < 1 or not np.isfinite(price_multiplier) or cfg["top_n"] <= 0:
        raise ValueError("价格预算倍数必须有限且不小于1，目标数量必须为正")
    for panel in (close, volume):
        if not panel.index.equals(signals.index) or not panel.columns.equals(signals.columns):
            raise ValueError("预算行情与信号轴必须完全对齐")
    if basics.ts_code.duplicated().any():
        raise ValueError("上市资料代码重复")
    listings = basics.set_index("ts_code").list_date.reindex(signals.columns)
    delistings = basics.set_index("ts_code").delist_date.reindex(signals.columns)
    rules = [cfg["boards"][board_of(code)] for code in signals.columns]
    minimum = pd.Series([rule["min_buy"] for rule in rules], index=signals.columns)
    starts = pd.Series(pd.to_datetime([rule["start"] for rule in rules]), index=signals.columns)
    prices = close.to_numpy(dtype=float)
    volumes = volume.to_numpy(dtype=float)
    minimum = minimum.to_numpy()
    gross = prices * minimum * price_multiplier
    fee = np.maximum(gross * cfg["buy_cost"], cfg["min_fee"])
    allowed = ((gross + fee <= cfg["capital"] / cfg["top_n"])
               & (volumes * cfg["max_volume_participation"] >= minimum)
               & np.isfinite(prices) & (prices > 0) & np.isfinite(volumes))
    for i, date in enumerate(signals.index):
        st = set(st_intervals.loc[(st_intervals.start <= date)
                     & (st_intervals.end.isna() | (st_intervals.end >= date)), "ts_code"])
        # 宽表逐列碎片化时 .loc 回写极慢；掩码始终保存在连续 NumPy 数组。
        allowed[i] &= ((date - listings).dt.days >= cfg["min_listing_calendar_days"]).to_numpy()
        allowed[i] &= ((starts <= date) & (delistings.isna() | (date < delistings))).to_numpy()
        allowed[i, signals.columns.isin(st)] = False
    allowed = pd.DataFrame(allowed, index=signals.index, columns=signals.columns)
    return signals.where(allowed), allowed
