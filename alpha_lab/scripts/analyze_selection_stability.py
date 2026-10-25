"""汇总全体预定对照、日期离散度、持仓路径与排名退出后的价格诊断。"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_selection_stability import save


def pnl_by_asset(item):
    daily, hold, trades, events = (item[k] for k in ['daily', 'holdings', 'trades', 'events'])
    terminal = hold.loc[hold.date.eq(daily.index[-1])].set_index('ts_code').value
    flow = trades.assign(flow=np.where(trades.side.eq('sell'), trades.gross, -trades.gross) - trades.fee).groupby('ts_code').flow.sum()
    dividends = events.loc[events.kind.eq('dividend_receivable')]
    dividend = dividends.groupby('ts_code').amount.sum() if len(dividends) else pd.Series(dtype=float)
    recovery = events.loc[events.kind.eq('delist_writeoff')]
    recovered = recovery.groupby('ts_code').cash_recovery.sum() if len(recovery) else pd.Series(dtype=float)
    return terminal.add(flow, fill_value=0).add(dividend, fill_value=0).add(recovered, fill_value=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run; cfg = yaml.safe_load((run / 'config.yaml').read_text())
    out = run / 'analysis'; out.mkdir(exist_ok=False)
    manifest = json.loads((run / 'execution_manifest.json').read_text())
    records = pd.read_csv(run / 'summary.csv').to_dict('records'); folders = {r['name']: run / 'accounts' / r['name'] for r in records}
    for old in manifest['reused_startup_baselines']:
        folder = ROOT / cfg['startup_parent'] / 'accounts' / old['source']
        record = json.loads((folder / 'summary.json').read_text())
        name = f"startup_{old['cohort']}_{old['offset']:02d}_baseline"
        record.update(name=name, kind='startup', policy='baseline', rank_exit_requests=0)
        records.append(record); folders[name] = folder
    summary = pd.DataFrame(records).set_index('name', drop=False).sort_index()
    summary.to_csv(out / 'all_accounts.csv', index=False)
    accounts = {}; assets = []; paths = []; after_exit = []
    price = pd.read_parquet(run / 'signals/close.parquet')
    for name, row in summary.iterrows():
        item = {k: pd.read_parquet(folders[name] / f'{k}.parquet') for k in ['daily', 'trades', 'holdings', 'events']}
        accounts[name] = item; pnl = pnl_by_asset(item); item['pnl'] = pnl
        assert abs(pnl.sum() - (item['daily'].equity.iloc[-1] - cfg['capital'])) < 1e-6, name
        assets.extend(dict(account=name, policy=row.policy, kind=row.kind, ts_code=code, net_pnl_yuan=value) for code, value in pnl.items())
        if row.policy in ['rank_buffer', 'combined']:
            original = pd.read_parquet(run / f"schedules/calendar_{1 if row.kind == 'startup' else row.offset:02d}/baseline/hold.parquet")
            exits = item['events'].loc[item['events'].kind.eq('hold_exit_requested') & item['events'].reason.eq('hold_false')]
            for e in exits.itertuples():
                if e.date not in original.index or original.at[e.date, e.ts_code] != 1: continue
                sales = item['trades'].loc[item['trades'].side.eq('sell') & item['trades'].ts_code.eq(e.ts_code) & item['trades'].signal_date.eq(e.date)]
                if sales.empty: continue
                date = sales.date.iloc[0]; index = price.index.get_loc(date)
                r = dict(account=name, policy=row.policy, kind=row.kind, ts_code=e.ts_code, signal_date=e.date, sell_date=date)
                for horizon in [20, 60]:
                    if index + horizon < len(price) and pd.notna(price.at[date, e.ts_code]):
                        r[f'after_{horizon}'] = price.iloc[index + horizon][e.ts_code] / price.at[date, e.ts_code] - 1
                    else: r[f'after_{horizon}'] = np.nan
                after_exit.append(r)
    pd.DataFrame(assets).to_csv(out / 'asset_pnl.csv', index=False)
    pd.DataFrame(after_exit).to_csv(out / 'rank_exit_forward_marks.csv', index=False)
    pairs = []; contributions = []
    for name, row in summary.loc[summary.policy.ne('baseline')].iterrows():
        base = f"startup_{row.cohort}_{row.offset:02d}_baseline" if row.kind == 'startup' else f'{row.cohort}_{row.offset:02d}_baseline'
        b = summary.loc[base]; a_daily = accounts[name]['daily']; b_daily = accounts[base]['daily']
        assert a_daily.index.equals(b_daily.index)
        delta = (a_daily.equity - b_daily.equity) / cfg['capital']
        pairs.append(dict(account=name, base=base, kind=row.kind, cohort=row.cohort, offset=row.offset, policy=row.policy,
            total_delta=row.total-b.total, ann_delta=row.ann-b.ann, mdd_delta=row.mdd-b.mdd,
            sharpe_delta=row.sharpe-b.sharpe, trade_delta=row.n_trades-b.n_trades,
            trade_fraction=row.n_trades/b.n_trades-1, fee_delta=row.total_fees-b.total_fees,
            invested_delta=row.mean_invested-b.mean_invested, rank_exit_requests=row.rank_exit_requests))
        for n in [20, 60, 126, 244]:
            if n <= len(delta): paths.append(dict(account=name, policy=row.policy, kind=row.kind, sessions=n, difference=float(delta.iloc[n-1])))
        diff = accounts[name]['pnl'].subtract(accounts[base]['pnl'], fill_value=0)
        assert abs(diff.sum() - delta.iloc[-1]*cfg['capital']) < 1e-6
        contributions.extend(dict(account=name, policy=row.policy, kind=row.kind, ts_code=code, difference_yuan=value) for code, value in diff.items())
    pairs = pd.DataFrame(pairs); pairs.to_csv(out / 'paired_results.csv', index=False)
    pd.DataFrame(paths).to_csv(out / 'path_checkpoints.csv', index=False)
    pd.DataFrame(contributions).to_csv(out / 'paired_asset_contributions.csv', index=False)
    spread = []; initial = []
    for (cohort, policy), group in summary.loc[summary.kind.eq('startup')].groupby(['cohort', 'policy']):
        assert len(group) == 3
        spread.append(dict(cohort=cohort, policy=policy, return_range=group.total.max()-group.total.min(),
            mdd_range=group.mdd.max()-group.mdd.min(), sharpe_range=group.sharpe.max()-group.sharpe.min()))
        first_sets = {}; final_sets = {}
        for name, row in group.iterrows():
            item = accounts[name]; buys = item['trades'].loc[item['trades'].side.eq('buy') & item['trades'].decision_date.eq(pd.Timestamp(row.boot_date))]
            first_sets[row.offset] = set(buys.ts_code)
            final_sets[row.offset] = set(item['holdings'].loc[item['holdings'].date.eq(item['daily'].index[-1]), 'ts_code'])
        for a, b in [(8, 15), (15, 22), (8, 22)]:
            initial.append(dict(cohort=cohort, policy=policy, first=a, second=b,
                initial_jaccard=len(first_sets[a]&first_sets[b])/len(first_sets[a]|first_sets[b]) if first_sets[a]|first_sets[b] else 1.,
                final_jaccard=len(final_sets[a]&final_sets[b])/len(final_sets[a]|final_sets[b]) if final_sets[a]|final_sets[b] else 1.))
    spreads = pd.DataFrame(spread); spreads.to_csv(out / 'startup_ranges.csv', index=False)
    pd.DataFrame(initial).to_csv(out / 'startup_stock_overlap.csv', index=False)
    ranges = spreads.groupby('policy').agg(median_range=('return_range','median'), max_range=('return_range','max'),
        median_mdd_range=('mdd_range','median'), max_mdd_range=('mdd_range','max'))
    ranges.to_csv(out / 'startup_range_summary.csv')
    long = pairs.loc[pairs.kind.eq('long')]
    med = long.groupby('policy')[['ann_delta', 'mdd_delta', 'sharpe_delta', 'trade_fraction', 'fee_delta', 'invested_delta']].median()
    med.to_csv(out / 'long_paired_medians.csv')
    summary.loc[summary.kind.eq('long')].groupby('policy')[['ann','mdd','sharpe','n_trades','mean_invested']].median().to_csv(out / 'long_metric_medians.csv')
    gate = cfg['acceptance']; decisions = []
    for policy in cfg['policies'][1:]:
        ratio = ranges.loc[policy, 'median_range'] / ranges.loc['baseline', 'median_range']
        tests = dict(date_dispersion=bool(ratio <= gate['startup_median_range_ratio_max']),
                     annual_return=bool(med.loc[policy, 'ann_delta'] >= -gate['median_annual_return_loss_max']),
                     drawdown=bool(med.loc[policy, 'mdd_delta'] >= -gate['median_mdd_worsening_max']),
                     trading_frequency=bool(med.loc[policy, 'trade_fraction'] <= gate['median_trade_increase_fraction_max']))
        decisions.append(dict(policy=policy, startup_range_ratio=ratio, tests=tests, passes_all=all(tests.values())))
    save(out / 'acceptance.json', decisions)
    save(out / 'verification.json', dict(accounts=len(summary), new_accounts=len(manifest['cases']),
        reused_accounts=len(manifest['reused_startup_baselines']), all_asset_pnl_reconciled=True,
        all_paired_contributions_reconciled=True, independent_samples=False))
    print(ranges.to_string()); print(med.to_string()); print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
