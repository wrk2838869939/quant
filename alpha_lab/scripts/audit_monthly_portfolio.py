"""从原审计脚本提取的独立现金与股数流水核验。"""
import pandas as pd
from portfolio import AccountingError


def reconcile(result, capital):
    """仅从成交/事件流水重建现金和股数，与引擎状态独立对账。"""
    cashflow = pd.Series(0., index=result.daily.index)
    shareflow = []
    for row in result.trades.itertuples():
        sign = -1 if row.side == "buy" else 1
        cashflow.loc[row.date] += sign * row.gross - row.fee
        shareflow.append(dict(date=row.date, ts_code=row.ts_code, shares=-sign * row.shares))
    if not result.events.empty:
        for row in result.events.to_dict("records"):
            if row["kind"] == "dividend_paid":
                cashflow.loc[row["date"]] += row["amount"]
            if row["kind"] == "delist_writeoff":
                cashflow.loc[row["date"]] += row["cash_recovery"]
                shareflow.append(dict(date=row["date"], ts_code=row["ts_code"], shares=-row["shares"]))
            if row["kind"] == "bonus_shares":
                shareflow.append(dict(date=row["date"], ts_code=row["ts_code"], shares=row["shares"]))
    cash_error = float((capital + cashflow.cumsum() - result.daily.cash).abs().max())
    share_error = 0.
    if shareflow:
        flow = pd.DataFrame(shareflow).pivot_table(index="date", columns="ts_code", values="shares", aggfunc="sum")
        expected = flow.reindex(result.daily.index).fillna(0).cumsum()
        if result.holdings.empty:
            actual = expected * 0
        else:
            actual = result.holdings.pivot(index="date", columns="ts_code", values="shares").reindex(
                index=expected.index, columns=expected.columns).fillna(0)
        share_error = float((expected - actual).abs().max().max())
    if cash_error > 1e-6 or share_error != 0:
        raise AccountingError(f"独立流水对账失败：cash={cash_error}, shares={share_error}")
    return dict(cash_max_error=cash_error, shares_max_error=share_error)
