#!/usr/bin/env python3
"""独立核对六组风控/固定现金账本；不运行策略、不请求接口、不读 2024+ 行情。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]


def assert_close(actual, expected, message, atol=1e-6):
    """对账采用人民币绝对容差；拒绝 NaN 或无限金额。"""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(message + ": 非有限值")
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=atol, err_msg=message)


def read_table(path):
    """读取原始输出，保留空表兼容。"""
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["date", "ts_code", "kind"])
    for col in ["date", "signal_date"]:
        if col in table:
            table[col] = pd.to_datetime(table[col])
    return table


def grouped_event(events, kind, field, dates):
    """直接汇总事件流水；不存在的事件贡献为零。"""
    selected = events[events.kind.eq(kind)]
    if selected.empty:
        return pd.Series(0.0, index=dates)
    return selected.groupby("date")[field].sum().reindex(dates, fill_value=0.0)


def verify_episodes(folder, daily, trades, baseline):
    """逐个请求独立计算退出、重入、等待及下一请求截断，不调用原报告函数。"""
    episodes = pd.read_csv(folder / "risk_episodes.csv")
    for col in ["request_date", "signal_date", "exit_complete_date", "recovery_execution_date",
                "first_reentry_date", "comparison_end"]:
        episodes[col] = pd.to_datetime(episodes[col])
    pd.testing.assert_index_equal(daily.index, baseline.index)
    requests = daily.index[daily.risk_target.eq(0) & daily.risk_target.shift(1).fillna(1).ne(0)]
    assert len(episodes) == len(requests)
    records = []
    for i, start in enumerate(requests):
        row = episodes[episodes.request_date.eq(start)].iloc[0]
        next_request = requests[i + 1] if i + 1 < len(requests) else pd.NaT
        window = daily.loc[start:]
        if pd.notna(next_request):
            window = window[window.index < next_request]
        sold_flat = window.index[window.n_hold.eq(0)]
        restored = window.index[window.risk_target.eq(1)]
        buys = trades[trades.side.eq("buy") & trades.date.ge(start) & trades.date.le(window.index[-1])]
        entry = buys.date.min() if len(buys) else pd.NaT
        complete = sold_flat[0] if len(sold_flat) else pd.NaT
        recovery = restored[0] if len(restored) else pd.NaT
        end = entry if pd.notna(entry) else window.index[-1]
        for actual, expected in [(row.exit_complete_date, complete), (row.first_reentry_date, entry),
                                 (row.recovery_execution_date, recovery), (row.comparison_end, end)]:
            assert (pd.isna(actual) and pd.isna(expected)) or actual == expected
        assert bool(row.unfinished) == pd.isna(entry)
        position = daily.index.get_loc(start)
        baseline_before = baseline.equity.iloc[max(0, position - 1)]
        risk_before = daily.equity.iloc[max(0, position - 1)]
        base_return = baseline.loc[end, "equity"] / baseline_before - 1
        risk_return = daily.loc[end, "equity"] / risk_before - 1
        assert_close([row.baseline_return_through_reentry, row.risk_return_through_reentry,
                      row.paired_return_difference], [base_return, risk_return, risk_return - base_return],
                     "风险事件同期账本收益", atol=1e-12)
        status = ("reentered" if pd.notna(entry) else "rolled_into_next_request"
                  if pd.notna(next_request) else "sample_end_without_reentry")
        records.append({"request_date": str(start.date()), "next_request": str(next_request),
                        "status": status, "exit_completed": pd.notna(complete),
                        "cash_at_request": bool(daily.loc[start, "n_hold"] == 0)})
    return records


def verify_case(folder, run, cfg, portfolio_cfg, risk_plan, require_episodes):
    """逐行重新累加现金与股数，核对模型效用、风险时点和总账户指标。"""
    summary = json.loads((folder / "summary.json").read_text())
    reserve = 25000.0 if summary["kind"] == "fixed_half_cash" else 0.0
    capital, active = 50000.0, 50000.0 - reserve
    assert summary["capital"] == capital
    if reserve:
        assert summary["active_capital"] == active and summary["reserve_cash"] == reserve
    daily = pd.read_csv(folder / "daily.csv", index_col=0, parse_dates=True)
    trades, events, holdings = [read_table(folder / f"{name}.csv") for name in ["trades", "events", "holdings"]]
    dates = daily.index
    assert dates.is_monotonic_increasing and dates.is_unique
    assert dates.min() >= pd.Timestamp("2022-01-01") and dates.max() < pd.Timestamp("2024-01-01")
    assert (daily.n_hold <= 5).all() and (daily.cash >= reserve - 1e-6).all()
    assert_close(trades.gross, trades.shares * trades.price, "成交额")
    assert (trades.shares > 0).all() and np.equal(trades.shares, np.floor(trades.shares)).all()
    assert (trades.signal_date < trades.date).all()
    # 公司行动在当日买卖前支付；股息权利和付款分列，不重复计算。
    paid = grouped_event(events, "dividend_paid", "amount", dates)
    recoveries = grouped_event(events, "delist_writeoff", "cash_recovery", dates)
    created = grouped_event(events, "dividend_receivable", "amount", dates)
    trade_flow = trades.assign(flow=np.where(trades.side.eq("sell"), trades.gross, -trades.gross) - trades.fee)
    flows = trade_flow.groupby("date").flow.sum().reindex(dates, fill_value=0.0)
    reconstructed_cash = capital + (flows + paid + recoveries).cumsum()
    assert_close(daily.cash, reconstructed_cash, "总账户现金流水")
    assert_close(daily.receivables, (created - paid).cumsum(), "未付分红权益")
    cash = active
    for date in dates:
        cash += paid.loc[date] + recoveries.loc[date]
        for trade in trades[trades.date.eq(date)].itertuples():
            cash += (trade.gross if trade.side == "sell" else -trade.gross) - trade.fee
            assert_close(trade.cash_after, cash, "交易子账户cash_after")
        assert_close(cash + reserve, daily.loc[date, "cash"], "固定现金合并")
    assert_close(daily.equity, daily.cash + daily.stock_value + daily.receivables, "资产恒等式")
    assert_close(holdings.value, holdings.shares * holdings.price, "持仓市值")
    assert_close(daily.stock_value, holdings.groupby("date").value.sum().reindex(dates, fill_value=0.0), "逐股合计市值")
    assert_close(daily.n_hold, holdings.groupby("date").size().reindex(dates, fill_value=0), "持仓数")
    assert (holdings.available >= 0).all() and (holdings.available <= holdings.shares).all()
    # 逐日股票流量与期末股数；送股和退市均单独入账。
    share_records = trades[["date", "ts_code"]].assign(shares=np.where(trades.side.eq("buy"), trades.shares, -trades.shares))
    for kind, sign in [("bonus_shares", 1), ("delist_writeoff", -1)]:
        selected = events[events.kind.eq(kind)]
        if len(selected):
            share_records = pd.concat([share_records, selected[["date", "ts_code"]].assign(shares=sign * selected.shares)])
    expected_shares = share_records.pivot_table(index="date", columns="ts_code", values="shares", aggfunc="sum").reindex(dates).fillna(0).cumsum()
    actual_shares = holdings.pivot(index="date", columns="ts_code", values="shares").reindex(index=dates, columns=expected_shares.columns).fillna(0)
    assert_close(actual_shares, expected_shares, "逐股股份流水", atol=0)
    # 从现金流、已确立股息和最终持仓独立得到每只股票利润。
    pnl = trade_flow.groupby("ts_code").flow.sum()
    for kind, field in [("dividend_receivable", "amount"), ("delist_writeoff", "cash_recovery")]:
        selected = events[events.kind.eq(kind)]
        if len(selected):
            pnl = pnl.add(selected.groupby("ts_code")[field].sum(), fill_value=0)
    pnl = pnl.add(holdings[holdings.date.eq(dates[-1])].set_index("ts_code").value, fill_value=0)
    saved_pnl = pd.read_csv(folder / "asset_pnl.csv").set_index("ts_code").pnl_yuan
    assert_close(saved_pnl.sort_index(), pnl.sort_index(), "逐股净利润")
    assert_close(pnl.sum(), daily.equity.iloc[-1] - capital, "总利润")
    # 原始行情仅按明确的2022—2023日期过滤读取，验证成交与每日标价。
    codes = sorted(set(trades.ts_code) | set(holdings.ts_code))
    market = pd.read_parquet(ROOT / "data/panel/daily_panel.parquet",
        columns=["date", "ts_code", "open", "close"],
        filters=[("date", ">=", dates.min()), ("date", "<=", dates.max()), ("ts_code", "in", codes)])
    quotes = market.set_index(["date", "ts_code"])
    assert_close(trades.price, quotes.loc[pd.MultiIndex.from_frame(trades[["date", "ts_code"]]), "open"], "原始开盘成交价")
    holding_keys = pd.MultiIndex.from_frame(holdings[["date", "ts_code"]])
    quoted_close = quotes.close.reindex(holding_keys).to_numpy()
    quoted = np.isfinite(quoted_close) & (quoted_close > 0)
    assert_close(holdings.loc[quoted, "price"], quoted_close[quoted], "原始收盘标价")
    carried_marks = []
    for row in holdings.loc[~quoted].itertuples():
        history = quotes.close.xs(row.ts_code, level="ts_code").reindex(dates).loc[:row.date]
        history = history[np.isfinite(history) & history.gt(0)]
        assert len(history), "沿用标价必须存在此前有效报价"
        last_date, last_price = history.index[-1], history.iloc[-1]
        assert_close(row.price, last_price, "缺价日沿用此前报价")
        assert row.stale_days == dates.get_loc(row.date) - dates.get_loc(last_date)
        carried_marks.append({"date": str(row.date.date()), "ts_code": row.ts_code,
                              "last_quote_date": str(last_date.date()), "price": float(row.price),
                              "stale_days": int(row.stale_days)})
        assert trades[trades.ts_code.eq(row.ts_code) & trades.date.eq(row.date)].empty
    assert int(holdings.stale_days.gt(0).sum()) == len(carried_marks)
    if carried_marks:
        assert {mark["ts_code"] for mark in carried_marks} == {"601298.SH"}
        assert {mark["date"] for mark in carried_marks} == {"2023-06-28", "2023-06-29", "2023-06-30"}
        resumed = holdings[holdings.ts_code.eq("601298.SH") & holdings.date.eq(pd.Timestamp("2023-07-03"))]
        assert len(resumed) == 1 and resumed.stale_days.iloc[0] == 0
        assert_close(resumed.price.iloc[0], quotes.loc[(pd.Timestamp("2023-07-03"), "601298.SH"), "close"], "复牌恢复报价")
    expected_return = daily.equity / daily.equity.shift(1).fillna(capital) - 1
    assert_close(daily.ret, expected_return, "总资产收益率", atol=1e-12)
    assert_close(daily.nav, daily.equity / capital, "总账户净值", atol=1e-12)
    peaks = np.maximum.accumulate(np.r_[capital, daily.equity.to_numpy()])[1:]
    total = daily.equity.iloc[-1] / capital - 1
    metrics = [total, (1 + total) ** (portfolio_cfg["annual_days"] / len(daily)) - 1,
               (daily.equity.to_numpy() / peaks - 1).min(),
               expected_return.mean() / expected_return.std(ddof=1) * np.sqrt(portfolio_cfg["annual_days"])]
    assert_close([summary[key] for key in ["total", "ann", "mdd", "sharpe"]], metrics, "收益风险指标", atol=1e-11)
    signals = pd.read_parquet(run / "execution" / f"monthly_{summary['policy']}_signals.parquet")
    forecasts = pd.read_parquet(run / "execution" / f"monthly_{summary['policy']}_point_forecasts.parquet")
    scheduled = {dates[dates.get_loc(d) + 1] for d in signals.index if d in dates and dates.get_loc(d) + 1 < len(dates)}
    buys = trades[trades.side.eq("buy")]
    assert set(buys.date).issubset(scheduled)
    assert all(pd.notna(signals.loc[row.signal_date, row.ts_code]) for row in buys.itertuples())
    allocation = events[events.kind.eq("allocation_decision")]
    unknown_retained = []
    for row in allocation.itertuples():
        forecast = forecasts.loc[row.signal_date, row.ts_code]
        if np.isfinite(forecast):
            assert_close(row.expected_return, forecast, "分配器使用点预测", atol=1e-12)
        else:
            # 停牌后的滚动特征不足可导致旧仓预测未知；零只作中性占位，不能驱动新买入。
            assert row.expected_return == 0 and row.old_shares > 0 and row.buy_shares == 0
            assert row.target_shares <= row.old_shares
            retained = events[events.date.eq(row.date) & events.ts_code.eq(row.ts_code)
                              & events.kind.eq("hold_retained") & events.reason.eq("unknown")]
            assert len(retained) == 1
            unknown_retained.append({"date": str(row.date.date()), "ts_code": row.ts_code,
                                     "old_shares": float(row.old_shares), "target_shares": float(row.target_shares)})
    assert_close(allocation.target_shares, allocation.old_shares + allocation.buy_shares - allocation.sell_shares, "整手计划恒等式", atol=0)
    assert (allocation.target_shares >= allocation.min_target_shares).all()
    episodes = []
    if summary["kind"] == "market_gate":
        expected_gate = risk_plan.reindex(dates).shift(1).fillna(1).astype(int)
        assert_close(daily.risk_target, expected_gate, "日频风险信号次日生效", atol=0)
        assert (daily.loc[buys.date, "risk_target"] == 1).all()
        transitions = daily.index[daily.risk_target.ne(daily.risk_target.shift(1).fillna(1))]
        transition_events = events[events.kind.eq("risk_transition")]
        assert list(transition_events.date) == list(transitions)
        requests = events[events.kind.eq("hold_exit_requested") & events.reason.eq("portfolio_risk")]
        for row in requests.itertuples():
            assert row.signal_date < row.date
            assert (daily.loc[row.date, "risk_target"] == 0 or daily.loc[row.date, "risk_state"] in ["exit_pending", "flat", "reentry_wait"])
        # 风险要求空仓却仍有旧仓时，必须留有不能成交/锁定的延迟证据。
        for date in daily.index[daily.risk_target.eq(0) & daily.n_hold.gt(0)]:
            remaining = set(holdings.loc[holdings.date.eq(date), "ts_code"])
            deferred = set(events.loc[events.date.eq(date) & events.kind.eq("sell_deferred"), "ts_code"])
            assert remaining.issubset(deferred)
        if (folder / "risk_episodes.csv").exists():
            baseline = pd.read_csv(run / "execution" / f"base_50000_monthly_{summary['policy']}" / "daily.csv", index_col=0, parse_dates=True)
            episodes = verify_episodes(folder, daily, trades, baseline)
        elif require_episodes:
            raise AssertionError(f"缺少尚待父账本完成的风险事件表: {folder.name}")
    return {"name": folder.name, "days": len(daily), "trades": len(trades),
            "max_holdings": int(daily.n_hold.max()), "reserve_cash": reserve,
            "cash_max_error": float(abs(daily.cash - reconstructed_cash).max()),
            "profit": float(pnl.sum()), "monthly_buys_verified": True,
            "point_forecast_utility_verified": True, "risk_episodes": episodes,
            "carried_quote_marks": carried_marks,
            "unknown_forecast_retained_rows": unknown_retained,
            "source_execution_data_sufficient_flag": summary["execution_data_sufficient"]}


def main():
    """完整六组必须存在；事件表未齐时仅可显式运行暂行账本审核。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--allow-missing-episodes", action="store_true")
    args = parser.parse_args()
    run = args.run or Path(Path("/tmp/autonomous_value_run_path").read_text().strip())
    output = run / "risk_overlay"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert hashlib.sha256((output / "parent_execution_manifest_at_start.json").read_bytes()).hexdigest() == manifest["parent_execution_manifest_sha256"]
    for path, digest in manifest["source_hashes"].items():
        assert hashlib.sha256((output / "sources" / path).read_bytes()).hexdigest() == digest
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    pc = yaml.safe_load((run / "execution/execution_config.yaml").read_text())["portfolio"]
    rc = yaml.safe_load((output / "sources/configs/retail_validation.yaml").read_text())
    schema = pq.read_schema(ROOT / rc["risk_features"])
    assert "date" in schema.names
    features = pd.read_parquet(ROOT / rc["risk_features"], filters=[("date", "<", pd.Timestamp("2024-01-01"))])
    assert features.index.name == "date" and features.index.max() < pd.Timestamp("2024-01-01")
    assert hashlib.sha256(pd.util.hash_pandas_object(features, index=True).values.tobytes()).hexdigest() == manifest["risk_features_sha256"]
    plan = pd.read_csv(output / "risk_allowed.csv", index_col=0, parse_dates=True).iloc[:, 0]
    assert hashlib.sha256(pd.util.hash_pandas_object(plan, index=True).values.tobytes()).hexdigest() == manifest["risk_plan_sha256"]
    old = pd.read_csv(ROOT / manifest["registration"]["source"], index_col=0, parse_dates=True).extreme_recovery
    pd.testing.assert_series_equal(plan, old, check_names=False, check_dtype=False)
    expected = {kind + "_" + policy for kind in ["market_gate", "fixed_half_cash"] for policy in cfg["execution_policies"]}
    assert set(manifest["results"]) == expected and len(manifest["results"]) == 6
    records = []
    for name in sorted(expected):
        records.append(verify_case(output / name, run, cfg, pc, plan, not args.allow_missing_episodes))
        print("verified " + name, flush=True)
    marking_limits = sum(len(record["carried_quote_marks"]) for record in records)
    result = {"status": "ledger_passed_episodes_pending" if args.allow_missing_episodes else
                         "passed_with_marking_limitations" if marking_limits else "passed",
              "case_count": 6, "records": records, "market_data_2024_plus_read": False,
              "source_snapshots_verified": True, "risk_date_column_filter_verified": True,
              "fixed_cash_semantics": "25000 integer active account plus fixed 25000 idle cash; trades.cash_after excludes reserve, daily.cash includes reserve",
              "risk_episode_semantics": "unfinished can mean rollover before monthly reentry; paired evolving portfolios are descriptive and not isolated timing alpha",
              "carried_quote_mark_rows": marking_limits,
              "marking_limitation": "No stale warning is not proof of complete quotes; short missing-quote spans use verified prior marks and do not represent executable prices.",
              "confirmed_suspension": {"ts_code": "601298.SH", "from": "2023-06-28", "resumption": "2023-07-03",
                  "evidence": ["https://paper.cnstock.com/html/2023-06/28/content_1787348.htm",
                               "https://pdf.dfcfw.com/pdf/H2_AN202307031591988002_1.pdf"],
                  "checks": "No trades on three suspension days; prior 6.97 mark carried; resumption uses actual close and resets stale_days."},
              "fresh_holdout": False}
    target = "verification_pending_episodes.json" if args.allow_missing_episodes else "verification.json"
    (output / target).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
