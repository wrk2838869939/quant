"""汇总已登记近期账户；两种初始状态及全部资金/费用情景一起报告。"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from portfolio import daily_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    summaries, yearly, monthly, attribution, paths, checks = [], [], [], [], {}, []
    for case in manifest['cases']:
        folder = run / 'execution' / case['name']
        summary = json.loads((folder / 'recent_summary.json').read_text())
        d = pd.read_csv(folder / 'daily.csv', index_col='date')
        h = pd.read_csv(folder / 'holdings.csv')
        t = pd.read_csv(folder / 'trades.csv')
        e = pd.read_csv(folder / 'events.csv')
        recent = d.loc['2024-01-01':].copy()
        initial = summary['initial_equity']
        summary.update(cash_days=int(recent.n_hold.eq(0).sum()), max_holdings=int(recent.n_hold.max()),
            worst_day=recent.ret.idxmin(), worst_daily_return=float(recent.ret.min()))
        dd = recent.equity / recent.equity.cummax().clip(lower=initial) - 1
        trough = dd.idxmin()
        prior = recent.loc[:trough].equity
        peak = prior.idxmax() if prior.max() >= initial else 'initial'
        peak_value = initial if peak == 'initial' else recent.loc[peak, 'equity']
        recovered = recent.loc[recent.index > trough].equity.ge(peak_value)
        summary.update(drawdown_peak=peak, drawdown_trough=trough,
            drawdown_recovery=recovered[recovered].index[0] if recovered.any() else 'not_recovered',
            ending_drawdown=float(dd.iloc[-1]),
            unknown_hold_reviews=int(((e.kind == 'hold_retained') & (e.reason == 'unknown') & (e.date >= '2024-01-01')).sum()))
        summaries.append(summary)
        years = pd.read_csv(folder / 'yearly.csv')
        yearly.append(years[years.year >= 2024])
        me = recent.equity.groupby(recent.index.str[:7]).last()
        mr = me.pct_change(); mr.iloc[0] = me.iloc[0] / initial - 1
        monthly.append(pd.DataFrame(dict(name=case['name'], month=me.index, equity=me.values, ret=mr.values)))
        values = h.pivot(index='date', columns='ts_code', values='value').reindex(d.index).fillna(0)
        flows = t.assign(flow=np.where(t.side.eq('sell'), t.gross, -t.gross) - t.fee)
        cash = flows.pivot_table(index='date', columns='ts_code', values='flow', aggfunc='sum')
        pnl = values.diff().fillna(values) + cash.reindex(index=d.index, columns=values.columns).fillna(0)
        for kind, col in [('dividend_receivable', 'amount'), ('delist_writeoff', 'cash_recovery')]:
            selected = e[e.kind.eq(kind)]
            if len(selected):
                extra = selected.pivot_table(index='date', columns='ts_code', values=col, aggfunc='sum')
                pnl += extra.reindex(index=d.index, columns=values.columns).fillna(0)
        np.testing.assert_allclose(pnl.sum(axis=1), d.equity.diff().fillna(d.equity.iloc[0] - case['capital']), atol=1e-6)
        rp = pnl.loc['2024-01-01':]
        np.testing.assert_allclose(rp.sum().sum(), recent.equity.iloc[-1] - initial, atol=1e-6)
        asset = rp.sum().sort_values(ascending=False).rename('pnl_yuan').rename_axis('ts_code').reset_index()
        asset['name'] = case['name']; attribution.append(asset)
        rp.to_csv(folder / 'recent_daily_asset_pnl.csv')
        recent['rebased_nav'] = recent.equity / initial
        recent['drawdown'] = dd
        recent.to_csv(folder / 'recent_daily.csv')
        paths[case['name']] = recent
        checks.append(dict(name=case['name'], daily_and_recent_attribution_verified=True))
    summary = pd.DataFrame(summaries)
    annual = pd.concat(yearly, ignore_index=True)
    months = pd.concat(monthly, ignore_index=True)
    assets = pd.concat(attribution, ignore_index=True)
    for name, frame in [('recent_summary', summary), ('annual', annual), ('monthly', months), ('asset_attribution', assets)]:
        frame.to_csv(run / f'{name}.csv', index=False)
    (run / 'report_checks.json').write_text(json.dumps(checks, indent=2))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
    for name, label in [('cold_50000', '50k cash start in 2024'), ('continued_50000', 'Continue 2023 holdings')]:
        d = paths[name]; dates = pd.to_datetime(d.index)
        axes[0].plot(dates, d.rebased_nav, label=label)
        axes[1].plot(dates, d.drawdown * 100, label=label)
        axes[2].plot(dates, d.stock_value / d.equity * 100, label=label)
    axes[0].set_title('Frozen low-turnover strategy: 2024 onward; no recent-period refit')
    axes[0].set_ylabel('NAV (2024 opening equity = 1)'); axes[0].legend()
    axes[1].set_ylabel('Drawdown (%)'); axes[2].set_ylabel('Stock exposure (%)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run / 'recent_paths.png', dpi=160); plt.close(fig)
    print(summary.to_string(index=False))
    print(annual.to_string(index=False))


if __name__ == '__main__':
    main()
