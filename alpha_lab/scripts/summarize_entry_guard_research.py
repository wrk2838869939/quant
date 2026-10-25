"""完整汇总预登记的24组新增仓位消融，不选择最有利本金或年度。"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from portfolio import daily_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    rows, years, activations, old_rows = [], [], [], []
    curves = {}
    for case in manifest['cases']:
        folder = run / 'execution' / case['name']
        summary = json.loads((folder / 'recent_summary.json').read_text())
        daily = pd.read_csv(folder / 'daily.csv', index_col='date')
        events = pd.read_csv(folder / 'events.csv')
        reviews = pd.read_csv(folder / 'guard_reviews.csv')
        recent = daily.loc['2024-01-01':]
        summary.update(variant=case['variant'], capital=case['capital'], cost_multiplier=case['cost_multiplier'],
            start=case['start'], reverse_input=case['reverse_input'], cash_days=int(recent.n_hold.eq(0).sum()),
            mean_holdings=float(recent.n_hold.mean()), max_holdings=int(recent.n_hold.max()))
        rows.append(summary)
        y = pd.read_csv(folder / 'yearly.csv'); y['variant'] = case['variant']; years.append(y)
        if case['name'].startswith('continued_'):
            old = daily.loc[:'2023-12-31']
            metrics, _ = daily_metrics(old.equity, case['capital'], 244)
            old_rows.append(dict(name=case['name'], variant=case['variant'], **metrics))
        for label, lo, hi in [('old', '2022-01-01', '2023-12-31'), ('recent', '2024-01-01', '2026-09-08')]:
            e = events[events.date.between(lo, hi)]
            rv = reviews[reviews.date.between(lo, hi)]
            decisions = e[e.kind.eq('selection_decision')]
            activations.append(dict(name=case['name'], variant=case['variant'], window=label,
                reviews=len(rv), collective_trigger_reviews=int(rv.collective_trigger.sum()),
                paused_reviews=int(rv.paused.sum()),
                correlation_rejections=int(decisions.reason.eq('correlation_limit').sum()),
                unknown_correlation_rejections=int(decisions.reason.eq('unknown_correlation').sum()),
                pause_rejections=int(decisions.reason.eq('collective_exit_pause').sum())))
        if case['name'].startswith('cold_'):
            curves[case['variant']] = recent.equity / case['capital']
    summary = pd.DataFrame(rows)
    old = pd.DataFrame(old_rows)
    for name, frame in [('recent_summary', summary), ('annual', pd.concat(years)),
                        ('old_window', old), ('activation_summary', pd.DataFrame(activations))]:
        frame.to_csv(run / f'{name}.csv', index=False)
    gates = []
    for variant in ['correlation', 'pause', 'combined']:
        for window, frame in [('old', old), ('recent_cold', summary[summary.name.str.startswith('cold_')]),
                              ('recent_continued', summary[summary.name.str.startswith('continued_')])]:
            baseline = frame[frame.variant.eq('baseline')].iloc[0]
            candidate = frame[frame.variant.eq(variant)].iloc[0]
            gates.append(dict(variant=variant, window=window,
                return_delta=candidate.total-baseline.total, mdd_delta=candidate.mdd-baseline.mdd,
                sharpe_delta=candidate.sharpe-baseline.sharpe,
                no_return_sacrifice=bool(candidate.total >= baseline.total-1e-10),
                no_deeper_drawdown=bool(candidate.mdd >= baseline.mdd-1e-10),
                no_lower_sharpe=bool(candidate.sharpe >= baseline.sharpe-1e-10)))
    pd.DataFrame(gates).to_csv(run / 'comparison_gate.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for variant, curve in curves.items():
        dates = pd.to_datetime(curve.index)
        axes[0].plot(dates, curve, label=variant)
        axes[1].plot(dates, (curve/curve.cummax().clip(lower=1)-1)*100, label=variant)
    axes[0].set_title('Monthly new-entry guards: previously observed development data, 2024 onward')
    axes[0].set_ylabel('NAV (50k cash start = 1)'); axes[0].legend()
    axes[1].set_ylabel('Drawdown (%)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run / 'entry_guard_paths.png', dpi=150); plt.close(fig)
    print(summary[summary.name.str.startswith(('cold_', 'continued_'))].to_string(index=False))
    print('OLD\n', old.to_string(index=False))
    print('GATES\n', pd.DataFrame(gates).to_string(index=False))


if __name__ == '__main__':
    main()
