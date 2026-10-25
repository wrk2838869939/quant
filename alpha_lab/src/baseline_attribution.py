"""按已执行账本分解收益；不将价格盈亏直接解释为选股 alpha。"""
import numpy as np
import pandas as pd


def attribute_account(daily, trades, events, holdings, capital):
    """逐日逐股净利润，含应计分红、退市回收和交易费用。

    分红在应收确立时归因，到账时不重复计算；配股弃权与送转造成的
    市值变化已经进入估值差，不重复扣减事件中的参考损失。
    """
    dates = daily.index
    codes = sorted(set(holdings.ts_code if len(holdings) else [])
                   | set(trades.ts_code if len(trades) else [])
                   | (set(events.ts_code) - {'__portfolio__'} if len(events) else set()))
    empty = pd.DataFrame(0., index=dates, columns=codes)
    values = (holdings.pivot(index='date', columns='ts_code', values='value').reindex(index=dates, columns=codes).fillna(0.)
              if len(holdings) else empty.copy())
    pd.testing.assert_series_equal(values.sum(axis=1), daily.stock_value, check_names=False, atol=1e-7, rtol=1e-12)
    flows, fees, dividends = empty.copy(), empty.copy(), empty.copy()
    for row in trades.itertuples():
        flows.loc[row.date, row.ts_code] += row.gross * (1 if row.side == 'sell' else -1)
        fees.loc[row.date, row.ts_code] += row.fee
    for row in events.itertuples():
        if row.kind == 'dividend_receivable':
            dividends.loc[row.date, row.ts_code] += row.amount
        elif row.kind == 'delist_writeoff':
            flows.loc[row.date, row.ts_code] += row.cash_recovery
    price = values.diff().fillna(values.iloc[0]) + flows
    pnl = price + dividends - fees
    expected = daily.equity.diff()
    expected.iloc[0] = daily.equity.iloc[0] - capital
    error = float((pnl.sum(axis=1) - expected).abs().max())
    if error > 1e-7:
        raise AssertionError(f'归因与账本不符: {error}')
    stock = pd.DataFrame(dict(price_pnl=price.sum(), dividends=dividends.sum(), fees=fees.sum(), net_pnl=pnl.sum()))
    stock.index.name = 'ts_code'
    stock['contribution_pp'] = stock.net_pnl / capital * 100
    stock = stock.sort_values('net_pnl', ascending=False)
    return stock, pnl, values, error
