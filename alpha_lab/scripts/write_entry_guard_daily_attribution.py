"""从各账户CSV独立还原每日逐股损益，并生成与原基线同口径的近期衍生表。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    checks = []
    for case in manifest['cases']:
        folder = run / 'execution' / case['name']
        d = pd.read_csv(folder / 'daily.csv', index_col='date')
        h = pd.read_csv(folder / 'holdings.csv')
        t = pd.read_csv(folder / 'trades.csv')
        e = pd.read_csv(folder / 'events.csv')
        summary = json.loads((folder / 'recent_summary.json').read_text())
        values = h.pivot(index='date', columns='ts_code', values='value').reindex(d.index).fillna(0)
        flows = t.assign(flow=np.where(t.side.eq('sell'), t.gross, -t.gross) - t.fee)
        cash = flows.pivot_table(index='date', columns='ts_code', values='flow', aggfunc='sum')
        pnl = values.diff().fillna(values) + cash.reindex(index=d.index, columns=values.columns).fillna(0)
        for kind, col in [('dividend_receivable', 'amount'), ('delist_writeoff', 'cash_recovery')]:
            selected = e[e.kind.eq(kind)]
            if len(selected):
                extra = selected.pivot_table(index='date', columns='ts_code', values=col, aggfunc='sum')
                pnl += extra.reindex(index=d.index, columns=values.columns).fillna(0)
        expected = d.equity.diff().fillna(d.equity.iloc[0] - case['capital'])
        np.testing.assert_allclose(pnl.sum(axis=1), expected, atol=1e-6)
        recent = d.loc['2024-01-01':].copy()
        initial = summary['initial_equity']
        rp = pnl.loc['2024-01-01':]
        np.testing.assert_allclose(rp.sum().sum(), recent.equity.iloc[-1] - initial, atol=1e-6)
        rp.to_csv(folder / 'recent_daily_asset_pnl.csv')
        recent['rebased_nav'] = recent.equity / initial
        recent['drawdown'] = recent.equity / recent.equity.cummax().clip(lower=initial) - 1
        recent.to_csv(folder / 'recent_daily.csv')
        checks.append(dict(name=case['name'], daily_pnl_max_error=float((pnl.sum(axis=1)-expected).abs().max()),
                           recent_pnl_error=float(abs(rp.sum().sum()-(recent.equity.iloc[-1]-initial)))))
    (run / 'daily_attribution_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    print('Daily attribution checks passed:', len(checks))


if __name__ == '__main__':
    main()
