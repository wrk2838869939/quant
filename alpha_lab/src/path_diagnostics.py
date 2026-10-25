"""从实际现金/股数流水重建逐股每日利润与持仓生命周期。"""
import numpy as np
import pandas as pd


def asset_daily_states(daily, trades, holdings, events, initial_capital):
    """利润=持仓市值变化+买卖净现金流+新确认股息+退市回收。

    红股通过实际股数和市值体现；股息到账不再次计利润。
    返回实际持仓日状态，逐日与总权益变化独立核对。
    """
    codes = sorted(set(holdings.ts_code) | set(trades.ts_code))
    values = holdings.pivot(index="date", columns="ts_code", values="value").reindex(
        index=daily.index, columns=codes).fillna(0.)
    previous = values.shift(1, fill_value=0.)
    t = trades.copy()
    t["flow"] = np.where(t.side == "sell", t.gross, -t.gross) - t.fee
    flows = t.pivot_table(index="date", columns="ts_code", values="flow", aggfunc="sum").reindex_like(values).fillna(0.)
    fees = t.pivot_table(index="date", columns="ts_code", values="fee", aggfunc="sum").reindex_like(values).fillna(0.)
    income = values * 0.
    for kind, field in [("dividend_receivable", "amount"), ("delist_writeoff", "cash_recovery")]:
        rows = events[events.kind == kind]
        if len(rows):
            income += rows.pivot_table(index="date", columns="ts_code", values=field,
                aggfunc="sum").reindex_like(values).fillna(0.)
    pnl = values - previous + flows + income
    total_pnl = daily.equity.diff()
    total_pnl.iloc[0] = daily.equity.iloc[0] - initial_capital
    if not np.allclose(pnl.sum(axis=1), total_pnl, rtol=0., atol=1e-6):
        raise ValueError("逐股每日利润无法与总权益变化对账")
    long = pd.concat({"value": values.stack(), "previous_value": previous.stack(),
        "pnl": pnl.stack(), "fees": fees.stack(), "trade_flow": flows.stack(),
        "income": income.stack()}, axis=1).reset_index()
    long = long[(long.value != 0) | (long.previous_value != 0) | (long.trade_flow != 0) | (long.income != 0)]
    return long


def holding_episodes(states, trades, calendar):
    """按真实买入开新持仓段；每日收盘浮盈/浮亏不是可实现的盘中价格。"""
    buys = trades[trades.side == "buy"].copy()
    if buys.duplicated(["ts_code", "date"]).any():
        raise ValueError("需要先合并同一股票同日多笔买入")
    buys["entry_cost"] = buys.gross + buys.fee
    buys["episode_id"] = np.arange(len(buys))
    buys = buys.rename(columns={"date": "entry_date"})
    x = pd.merge_asof(states.sort_values("date"),
        buys[["entry_date", "ts_code", "entry_cost", "episode_id"]].sort_values("entry_date"),
        left_on="date", right_on="entry_date", by="ts_code", direction="backward")
    if x.episode_id.isna().any():
        raise ValueError("存在买入之前的持仓利润")
    loc = pd.Series(np.arange(len(calendar)), index=calendar)
    x["age"] = x.date.map(loc) - x.entry_date.map(loc)
    x = x.sort_values(["episode_id", "date"])
    x["cumulative_pnl"] = x.groupby("episode_id").pnl.cumsum()
    x["episode_return"] = x.cumulative_pnl / x.entry_cost
    x["age_bucket"] = pd.cut(x.age, bins=[-1, 0, 4, 9, 19, 39, np.inf],
        labels=["entry_day", "days2_5", "days6_10", "days11_20", "days21_40", "days41_plus"])
    episodes = []
    for episode, g in x.groupby("episode_id"):
        last = g.iloc[-1]
        peak = max(0., g.episode_return.max())
        episodes.append(dict(episode_id=episode, ts_code=g.ts_code.iloc[0], entry_date=g.entry_date.iloc[0],
            last_date=last.date, trading_days=int(last.age+1), closed=bool(last.value == 0),
            entry_cost=g.entry_cost.iloc[0], pnl=g.pnl.sum(), return_on_entry_cost=last.episode_return,
            max_marked_gain=peak, max_marked_loss=min(0., g.episode_return.min()),
            gain_given_back=peak-last.episode_return, fees=g.fees.sum()))
    return x, pd.DataFrame(episodes)
