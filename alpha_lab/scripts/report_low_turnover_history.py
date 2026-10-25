"""冻结低换手账户的历史表现复核；不运行选股、不读取2024年及以后行情。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "outputs/policy_cash/20260913T063048_495344Z/cash_execution/base_50000_none"


def metrics(equity, capital):
    returns = equity.pct_change()
    returns.iloc[0] = equity.iloc[0] / capital - 1
    return dict(total=equity.iloc[-1] / capital - 1,
                ann=(equity.iloc[-1] / capital) ** (244 / len(equity)) - 1,
                mdd=(equity / equity.cummax().clip(lower=capital) - 1).min(),
                sharpe=returns.mean() / returns.std(ddof=1) * np.sqrt(244))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def read(name):
        path = CASE / name
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)

    daily = read("daily.csv").set_index("date")
    holds, trades, events = read("holdings.csv"), read("trades.csv"), read("events.csv")
    assets, episodes = read("asset_pnl.csv"), read("entry_phase_episodes.csv")
    periods = read("holding_periods.csv")
    summary = pd.read_csv(CASE.parent / "execution_summary.csv")
    eq = daily.equity
    assert eq.index.is_unique and eq.index.is_monotonic_increasing
    assert eq.index.min() == "2022-01-04" and eq.index.max() == "2023-12-29"
    assert daily.receivables.eq(0).all(), "本归因器要求无跨日应收余额"
    values = holds.pivot(index="date", columns="ts_code", values="value").reindex(eq.index).fillna(0)
    np.testing.assert_allclose(values.sum(axis=1), daily.stock_value, atol=1e-7)
    np.testing.assert_allclose(daily.cash + daily.stock_value, eq, atol=1e-7)
    np.testing.assert_allclose(eq.pct_change().fillna(0), daily.ret, atol=1e-12)
    np.testing.assert_allclose(eq / 50000, daily.nav, atol=1e-12)
    assert daily.n_hold.max() <= 5 and trades.shares.mod(100).eq(0).all()
    assert (trades.signal_date < trades.date).all()
    # 股份市值变化 + 卖出现金 - 买入现金 - 费用 + 实际到账分红。
    flow = trades.assign(flow=np.where(trades.side.eq("sell"), trades.gross, -trades.gross) - trades.fee)
    flow = flow.pivot_table(index="date", columns="ts_code", values="flow", aggfunc="sum")
    div = events[events.kind.eq("dividend_paid")].pivot_table(index="date", columns="ts_code", values="amount", aggfunc="sum")
    pnl = values.diff().fillna(values) + flow.reindex(index=eq.index, columns=values.columns).fillna(0)
    pnl += div.reindex(index=eq.index, columns=values.columns).fillna(0)
    np.testing.assert_allclose(pnl.sum(axis=1), eq.diff().fillna(eq.iloc[0] - 50000), atol=1e-7)
    np.testing.assert_allclose(pnl.sum().sort_index(), assets.set_index("ts_code").pnl_yuan.sort_index(), atol=1e-7)
    np.testing.assert_allclose(episodes.pnl_yuan.sum(), eq.iloc[-1] - 50000, atol=1e-7)
    pnl.to_csv(out / "daily_asset_pnl.csv")
    monthly_eq = eq.groupby(eq.index.str[:7]).last()
    monthly_ret = monthly_eq.pct_change()
    monthly_ret.iloc[0] = monthly_eq.iloc[0] / 50000 - 1
    monthly = pd.DataFrame(dict(equity=monthly_eq, ret=monthly_ret, pnl_yuan=monthly_eq.diff()))
    monthly.iloc[0, 2] = monthly_eq.iloc[0] - 50000
    np.testing.assert_allclose((1 + monthly_ret).prod(), eq.iloc[-1] / 50000)
    monthly.to_csv(out / "monthly.csv")
    pnl.groupby(pnl.index.str[:7]).sum().to_csv(out / "monthly_asset_pnl.csv")
    overall = metrics(eq, 50000)
    frozen = summary[summary.name.eq("base_50000_none")].iloc[0]
    for key, value in overall.items():
        np.testing.assert_allclose(value, frozen[key], atol=1e-10)
    # 行过滤在读取时执行；这些原价列只用于账本价格复核。
    prices = pd.read_parquet(ROOT / "data/panel/daily_panel.parquet",
        columns=["ts_code", "trade_date", "open", "close"],
        filters=[("trade_date", ">=", "20220104"), ("trade_date", "<=", "20231229"),
                 ("ts_code", "in", list(values.columns))])
    prices["date"] = pd.to_datetime(prices.trade_date).dt.strftime("%Y-%m-%d")
    assert not prices.duplicated(["date", "ts_code"]).any()
    mark_check = holds.merge(prices, on=["date", "ts_code"], how="left", validate="many_to_one")
    trade_check = trades.merge(prices, on=["date", "ts_code"], how="left", validate="many_to_one")
    np.testing.assert_allclose(mark_check.price, mark_check.close, atol=1e-8)
    np.testing.assert_allclose(trade_check.price, trade_check.open, atol=1e-8)
    prices.to_csv(out / "held_stock_price_evidence.csv", index=False)
    benchmark = pd.read_parquet(ROOT / "data/raw/index_daily.parquet",
        columns=["ts_code", "trade_date", "close"],
        filters=[("trade_date", ">=", "20220104"), ("trade_date", "<=", "20231229"),
                 ("ts_code", "in", ["000300.SH", "000905.SH"])])
    benchmark["date"] = pd.to_datetime(benchmark.trade_date).dt.strftime("%Y-%m-%d")
    bench = benchmark.pivot(index="date", columns="ts_code", values="close").reindex(eq.index)
    assert bench.notna().all().all() and (bench > 0).all().all()
    bench = bench / bench.iloc[0]
    bench.to_csv(out / "benchmark_nav.csv")
    years = []
    for year, curve in eq.groupby(eq.index.str[:4]):
        prior = 50000 if year == "2022" else eq[eq.index < curve.index[0]].iloc[-1]
        row = dict(year=year, start_equity=prior, end_equity=curve.iloc[-1], **metrics(curve, prior))
        for code in bench:
            previous = 1 if year == "2022" else bench.loc[bench.index < curve.index[0], code].iloc[-1]
            row[code] = bench.loc[curve.index[-1], code] / previous - 1
        years.append(row)
    annual = pd.DataFrame(years)
    annual.to_csv(out / "annual.csv", index=False)
    dd = eq / eq.cummax() - 1
    drawdowns, peak, trough, active = [], eq.index[0], eq.index[0], False
    for date in eq.index[1:]:
        if eq.loc[date] >= eq.loc[peak]:
            if active:
                drawdowns.append(dict(peak=peak, trough=trough, recovery=date,
                    mdd=eq.loc[trough] / eq.loc[peak] - 1,
                    elapsed_trading_days=eq.index.get_loc(date) - eq.index.get_loc(peak)))
            peak, trough, active = date, date, False
        else:
            active = True
            if eq.loc[date] < eq.loc[trough]:
                trough = date
    if active:
        drawdowns.append(dict(peak=peak, trough=trough, recovery="未恢复（截至2023-12-29）",
            mdd=eq.loc[trough] / eq.loc[peak] - 1,
            elapsed_trading_days=eq.index.get_loc(eq.index[-1]) - eq.index.get_loc(peak)))
    drawdowns = pd.DataFrame(drawdowns).sort_values("mdd")
    drawdowns.to_csv(out / "drawdowns.csv", index=False)
    maxdd = drawdowns.iloc[0]
    april = pnl.loc["2023-04-01":"2023-04-30"].sum().sort_values(ascending=False)
    giveback = pnl.loc[pnl.index > maxdd.peak].loc[:maxdd.trough].sum().sort_values()
    pd.DataFrame(dict(april_pnl=april, peak_to_trough_pnl=giveback)).to_csv(out / "key_period_attribution.csv")
    money = summary[summary.name.isin(["base_20000_none", "base_50000_none", "base_100000_none", "cost2_50000_none"])]
    money.to_csv(out / "capital_cost_sensitivity.csv", index=False)
    closed = episodes[episodes.close_date.notna()]
    stats = dict(**overall, final_equity=float(eq.iloc[-1]), net_profit=float(eq.iloc[-1] - 50000),
        peak_date=maxdd.peak, peak_equity=float(eq.loc[maxdd.peak]), trough_date=maxdd.trough,
        trough_equity=float(eq.loc[maxdd.trough]), ending_drawdown=float(dd.iloc[-1]),
        unrecovered_trading_days=int(maxdd.elapsed_trading_days),
        winning_months=int(monthly_ret.gt(0).sum()), total_months=len(monthly_ret),
        april_pnl=float(april.sum()), april_share_net_profit=float(april.sum() / (eq.iloc[-1] - 50000)),
        top3_share_net_profit=float(assets.nlargest(3, "pnl_yuan").pnl_yuan.sum() / (eq.iloc[-1] - 50000)),
        dividends=float(events.loc[events.kind.eq("dividend_paid"), "amount"].sum()),
        fees=float(trades.fee.sum()), closed_episodes=len(closed), closed_wins=int(closed.pnl_yuan.gt(0).sum()),
        median_closed_holding_days=float(periods[~periods.open_at_end].holding_trading_days.median()),
        mean_exposure=float((daily.stock_value / eq).mean()), max_stock_weight=float(values.div(eq, axis=0).max().max()),
        trading_dates=int(trades.date.nunique()), best_day=daily.ret.idxmax(), best_daily_return=float(daily.ret.max()),
        worst_day=daily.ret.idxmin(), worst_daily_return=float(daily.ret.min()),
        checks="每日现金/市值/权益/收益，逐股逐日损益，月度复利，汇总指标，2415条持仓收盘价及29笔成交开盘价全部一致")
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1, 1]})
    dates = pd.to_datetime(eq.index)
    axes[0].plot(dates, eq / 50000, label="Low turnover, 5 stocks")
    for code, name in [("000300.SH", "CSI 300 price index"), ("000905.SH", "CSI 500 price index")]:
        axes[0].plot(dates, bench[code], label=name, alpha=.8)
    axes[0].legend(loc="upper left"); axes[0].set_ylabel("NAV (start = 1)")
    axes[0].set_title("2022-2023 frozen development replay; not a fresh out-of-sample test")
    axes[1].fill_between(dates, dd * 100, 0, alpha=.5, color="firebrick")
    axes[1].set_ylabel("Drawdown (%)")
    axes[2].bar(pd.to_datetime(monthly.index), monthly.ret * 100, width=20,
                color=np.where(monthly.ret > 0, "#26856a", "#bb4545"))
    axes[2].set_ylabel("Monthly return (%)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "history.png", dpi=160); plt.close(fig)
    source = Path(__file__).read_bytes()
    (out / "report_low_turnover_history.py").write_bytes(source)
    manifest = dict(purpose="posthoc_reporting_only_not_strategy_selection", inputs=inputs,
        market_data_max_date="2023-12-29", source_sha256=hashlib.sha256(source).hexdigest(),
        evidence_sha256={name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                         for name in ["held_stock_price_evidence.csv", "benchmark_nav.csv"]})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("ANNUAL\n", annual.to_string(index=False))
    print("DRAWDOWNS\n", drawdowns.head(5).to_string(index=False))
    print("APRIL\n", april.to_string())
    print("GIVEBACK\n", giveback.to_string())
    print("BENCHMARK\n", (bench.iloc[-1] - 1).to_string())


if __name__ == "__main__":
    main()
