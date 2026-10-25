"""汇总预登记的全部因果执行情景，原账本差异只作模型变化归因。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_causal_execution import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    parent = Path(manifest['parent'])
    assert digest(parent / 'completion.json') == manifest['parent_completion_sha256']
    full, recent, annual, original, differences, events, fills = [], [], [], [], [], [], []
    daily_differences, asset_differences = [], []
    market = pd.read_parquet(run / 'inputs/market.parquet')
    for case in manifest['cases']:
        folder = run / 'execution' / case['name']
        completed = json.loads((folder / 'completion.json').read_text())
        assert completed['status'] == 'complete'
        for rel, expected in completed['artifacts'].items():
            assert digest(folder / rel) == expected, (case['name'], rel)
        row = json.loads((folder / 'summary.json').read_text())
        full.append(row)
        metrics = json.loads((folder / 'recent_summary.json').read_text())
        recent.append({**case, **metrics})
        year = pd.read_csv(folder / 'yearly.csv')
        year['name'] = case['name']
        annual.append(year)
        old_name = case['scenario'] + '_' + case['variant']
        old_folder = parent / 'execution' / old_name
        old = json.loads((old_folder / 'summary.json').read_text())
        original.append(dict(name=case['name'], original_name=old_name,
            **{key:old[key] for key in ['total', 'ann', 'mdd', 'sharpe', 'n_trades']}))
        differences.append(dict(name=case['name'], original_name=old_name,
            **{key+'_delta':row[key]-old[key] for key in ['total', 'ann', 'mdd', 'sharpe', 'n_trades']}))
        current_daily = pd.read_csv(folder / 'daily.csv', parse_dates=['date']).set_index('date')
        old_daily = pd.read_csv(old_folder / 'daily.csv', parse_dates=['date']).set_index('date')
        assert current_daily.index.equals(old_daily.index)
        delta = pd.DataFrame(dict(name=case['name'], new_equity=current_daily.equity,
            original_equity=old_daily.equity, equity_difference=current_daily.equity-old_daily.equity,
            new_stock_fraction=current_daily.stock_value/current_daily.equity,
            original_stock_fraction=old_daily.stock_value/old_daily.equity))
        daily_differences.append(delta.reset_index())
        current_pnl = pd.read_csv(folder / 'asset_pnl.csv').set_index('ts_code').pnl_yuan
        old_pnl = pd.read_csv(old_folder / 'asset_pnl.csv').set_index('ts_code').pnl_yuan
        delta_pnl = current_pnl.subtract(old_pnl, fill_value=0).rename('pnl_delta').to_frame()
        delta_pnl['name'] = case['name']
        asset_differences.append(delta_pnl.reset_index())
        event = pd.read_csv(folder / 'events.csv')
        counts = event.groupby('kind', dropna=False).size().rename('count').reset_index()
        counts['name'] = case['name']
        events.append(counts)
        trades = pd.read_csv(folder / 'trades.csv', parse_dates=['date', 'signal_date'])
        quotes = trades.merge(market[['date','ts_code','open','high','low']], on=['date','ts_code'],
                               how='left', validate='many_to_one')
        assert quotes[['open','high','low']].notna().all().all()
        assert (quotes.signal_date < quotes.date).all()
        assert (quotes.price >= quotes.low-1e-8).all() and (quotes.price <= quotes.high+1e-8).all()
        fills.append(dict(name=case['name'], trades=len(quotes),
            open_fills=int(np.isclose(quotes.price, quotes.open, atol=1e-8, rtol=0).sum()),
            non_open_fills=int((~np.isclose(quotes.price, quotes.open, atol=1e-8, rtol=0)).sum()),
            strict_signal_before_fill=True, within_daily_range=True))
    frames = {'full_summary':pd.DataFrame(full),'recent_summary':pd.DataFrame(recent),
        'yearly':pd.concat(annual, ignore_index=True),'original_summary':pd.DataFrame(original),
        'execution_model_differences':pd.DataFrame(differences),
        'event_counts':pd.concat(events,ignore_index=True),'fill_ranges':pd.DataFrame(fills),
        'daily_execution_differences':pd.concat(daily_differences,ignore_index=True),
        'asset_execution_differences':pd.concat(asset_differences,ignore_index=True)}
    comparisons = []
    table = frames['full_summary'].set_index('name')
    for base_name in table.index:
        if '_baseline' not in base_name:
            continue
        hybrid_name = base_name.replace('_baseline','_hybrid')
        a, b = table.loc[base_name], table.loc[hybrid_name]
        comparisons.append(dict(baseline=base_name, hybrid=hybrid_name,
            **{key+'_hybrid_minus_baseline':b[key]-a[key] for key in ['total','ann','mdd','sharpe','n_trades']}))
    frames['paired_strategy_differences'] = pd.DataFrame(comparisons)
    reverse = []
    for variant in ['baseline','hybrid']:
        for filename in ['daily','trades','holdings','events','process_daily','asset_pnl','holding_periods','exit_reasons']:
            a = run/'execution'/f'cold_{variant}'/f'{filename}.csv'
            b = run/'execution'/f'reverse_{variant}'/f'{filename}.csv'
            identical = digest(a) == digest(b)
            reverse.append(dict(variant=variant, file=filename, byte_identical=identical))
            assert identical, ('reverse input altered result',variant,filename)
    for key, frame in frames.items():
        frame.to_csv(run / f'{key}.csv', index=False)
    save(run / 'aggregation_verification.json', dict(case_count=len(full),
        reverse_comparisons=reverse, all_cases_complete=True, source_artifacts_unchanged=True,
        fill_range_checks=fills, limitation='Range and ledger checks do not certify order chronology or actual market fills.'))
    print(frames['full_summary'][['name','total','mdd','sharpe','n_trades']].to_string(index=False))


if __name__ == '__main__':
    main()
