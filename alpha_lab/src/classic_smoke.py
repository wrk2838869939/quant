"""经典策略工程验收：因果信号、交易账本独立核对与诊断。

仅生成可替换的策略信号，不修改通用 portfolio 引擎。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decision_grid(dates, frequency):
    """首个交易日收盘初始化，其后每日或每月最后一个交易日决策。"""
    dates = pd.DatetimeIndex(dates).sort_values()
    if frequency == "daily":
        return dates[:-1]
    if frequency != "monthly":
        raise ValueError(f"未知调仓频率: {frequency}")
    month_ends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).last()
    return pd.DatetimeIndex(sorted(set([dates[0], *month_ends]))).intersection(dates[:-1])


def make_signals(features, eligible, cfg, algorithm, frequency):
    """只用当日及之前特征；返回每个决策日所有合格候选的分数。"""
    if algorithm == "momentum":
        score = features[f"ret{cfg['momentum_window']}"]
        positive = score.gt(0)
    elif algorithm == "macd":
        score = (features["macd_dif"] - features["macd_dea"]) / features["hfq_close"]
        positive = score.gt(0) & features["macd_dif"].gt(0)
    elif algorithm == "moving_average":
        fast, slow = features[f"ma{cfg['ma_fast']}"], features[f"ma{cfg['ma_slow']}"]
        score = fast / slow - 1
        positive = score.gt(0) & features["hfq_close"].gt(fast)
    elif algorithm == "random":
        # 固定列顺序、按日生成，截断末尾日期不改变过去的随机排名。
        rng = np.random.default_rng(cfg["random_seed"])
        score = pd.DataFrame(rng.random(eligible.shape), index=eligible.index, columns=eligible.columns)
        positive = pd.DataFrame(True, index=eligible.index, columns=eligible.columns)
    else:
        raise ValueError(f"未知算法: {algorithm}")
    score = score.reindex_like(eligible).where(eligible & positive)
    return score.loc[decision_grid(eligible.index, frequency)].replace([np.inf, -np.inf], np.nan)


def verify_ledger(result, capital, market, limits, actions, cfg, slippage_rate, tolerance):
    """独立重放现金/股数、T+1和公司行动锁定，并检查费用、成交价格与净值。"""
    from audit_monthly_portfolio import reconcile
    report = reconcile(result, capital)
    daily, trades, holdings = result.daily, result.trades, result.holdings
    assert np.isfinite(daily[["cash", "equity", "stock_value", "receivables"]]).all().all()
    assert daily.cash.min() >= -tolerance
    assert daily.equity.min() >= 0
    residual = daily.equity - daily.cash - daily.stock_value - daily.receivables
    assert residual.abs().max() <= tolerance
    marked = (holdings.groupby("date").value.sum().reindex(daily.index, fill_value=0)
              if not holdings.empty else pd.Series(0., index=daily.index))
    assert (marked - daily.stock_value).abs().max() <= tolerance
    raw = market.set_index(["date", "ts_code"])
    bounds = limits.set_index(["date", "ts_code"])
    action_map = {(r.ts_code, r.ex_date): r for r in actions.itertuples()}
    event_map = (dict(tuple(result.events.groupby("date"))) if not result.events.empty else {})
    trade_map = dict(tuple(trades.groupby("date", sort=False))) if not trades.empty else {}
    shares, locks = {}, {}
    cash = float(capital)
    fees_error = cash_error = 0.
    checked = 0
    for i, date in enumerate(daily.index):
        for event in event_map.get(date, pd.DataFrame()).to_dict("records"):
            code = event["ts_code"]
            if event["kind"] == "dividend_paid":
                cash += event["amount"]
            elif event["kind"] == "bonus_shares":
                quantity = int(event["shares"])
                shares[code] = shares.get(code, 0) + quantity
                release = action_map[(code, date)].div_listdate
                locks.setdefault(code, []).append((quantity, release))
            elif event["kind"] == "delist_writeoff":
                cash += event["cash_recovery"]
                shares.pop(code, None)
                locks.pop(code, None)
        for row in trade_map.get(date, pd.DataFrame()).itertuples():
            quote = raw.loc[(date, row.ts_code)]
            limit = bounds.loc[(date, row.ts_code)]
            assert abs(row.price - quote.open) <= tolerance
            assert abs(row.gross - row.shares * row.price) <= tolerance
            assert row.shares > 0 and float(row.shares).is_integer()
            rate = cfg["buy_cost"] if row.side == "buy" else cfg["sell_cost"]
            fee = max(cfg["min_fee"], row.gross * rate)
            if row.side == "sell":
                fee = min(row.gross, fee)
            fee += row.gross * slippage_rate
            fees_error = max(fees_error, abs(fee - row.fee))
            assert abs(fee - row.fee) <= tolerance
            old = shares.get(row.ts_code, 0)
            if row.side == "buy":
                assert row.signal_date < date
                assert row.price < limit.up_limit - cfg["price_tick"] / 2
                assert row.shares % cfg["boards"]["main"]["buy_step"] == 0
                shares[row.ts_code] = old + row.shares
                release = daily.index[i + 1] if i + 1 < len(daily.index) else date + pd.Timedelta(days=1)
                locks.setdefault(row.ts_code, []).append((row.shares, release))
                cash -= row.gross + row.fee
            else:
                assert row.price > limit.down_limit + cfg["price_tick"] / 2
                available = old - sum(q for q, release in locks.get(row.ts_code, []) if date < release)
                assert row.shares <= available, (date, row.ts_code, "T+1/红股锁定不符")
                shares[row.ts_code] = old - row.shares
                cash += row.gross - row.fee
            assert cash >= -tolerance
            cash_error = max(cash_error, abs(cash - row.cash_after))
            assert abs(cash - row.cash_after) <= tolerance
            checked += 1
        assert abs(cash - daily.loc[date, "cash"]) <= tolerance
    report.update(equity_identity_max_error=float(residual.abs().max()),
                  trade_cash_max_error=cash_error, fees_max_error=fees_error,
                  trades_checked=checked, t_plus_one_pass=True,
                  official_limits_pass=True, no_negative_cash=True)
    return report
