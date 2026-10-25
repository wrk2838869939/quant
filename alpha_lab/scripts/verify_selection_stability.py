"""全体来源、账户前缀和退出后同月禁止补买的完整回放核验。"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from build_selection_stability import digest, save
from run_selection_stability import initialize, simulate, STATE


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, value in manifest['source_sha256'].items(): assert digest(ROOT / name) == value, name
    for name, value in manifest['input_sha256'].items(): assert digest(name) == value, name
    assert digest(run / 'PROTOCOL.md') == manifest['protocol_sha256']
    assert digest(ROOT.parent / 'AGENTS.md') == manifest['root_agents_sha256']
    for name, value in json.loads((run / 'signal_hashes.json').read_text()).items(): assert digest(run / name) == value, name
    inputs = json.loads((run / 'execution_input_completion.json').read_text())
    for name, value in inputs['input_sha256'].items(): assert digest(run / name) == value, name
    for name, value in inputs['new_raw_sha256'].items(): assert digest(run / 'new_execution_raw' / name) == value, name
    execution = json.loads((run / 'execution_manifest.json').read_text())
    initialize(run); rows = []
    scope_checks = 0
    available = set(STATE['frames']['market'].ts_code)
    for calendar in STATE['cfg']['calendar_days']:
        for policy in STATE['cfg']['policies'] + STATE['cfg']['neighbors']:
            scores = STATE['schedules'][calendar][policy]['signals']
            needed = set().union(*(set(x.dropna().nlargest(20).index) for _, x in scores.iterrows()))
            assert needed <= available
            assert STATE['risk'].loc[scores.index, sorted(needed)].notna().all().all()
            if policy in ['rank_buffer', 'combined']:
                raw = pd.read_parquet(run / 'attempts/before_entry_exit_alignment' / f'{calendar:02d}_{policy}_signals.parquet')
                for date, row in scores.iterrows():
                    assert set(row.dropna().index) <= set(raw.loc[date].dropna().nlargest(20).index)
                    scope_checks += 1
    for name in ['startup_2022-04_15_smooth20', 'startup_2023-01_15_rank_buffer', 'cold_22_combined']:
        case = next(c for c in execution['cases'] if c['name'] == name)
        folder = run / 'accounts' / name
        old = {key: pd.read_parquet(folder / f'{key}.parquet') for key in ['daily', 'trades', 'holdings', 'events']}
        cutoff = old['daily'].index[125 if case['kind'] == 'startup' else 243]
        result, _ = simulate(case, cutoff)
        for key in old:
            expected = old[key]; actual = getattr(result, key)
            if key == 'daily': expected = expected.loc[:cutoff]
            else:
                if key == 'events':
                    expected = expected.loc[expected.date.lt(cutoff)]
                    actual = actual.loc[actual.date.lt(cutoff)]
                    assert expected[expected.columns.difference(actual.columns)].isna().all().all()
                    actual = actual.reindex(columns=expected.columns)
                else: expected = expected.loc[expected.date.le(cutoff)]
                expected, actual = expected.reset_index(drop=True), actual.reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=False)
        rows.append(dict(account=name, cutoff=str(cutoff.date()), prefix_exact=True))
        print(name, 'prefix passed', flush=True)
    denied_checks = 0; checked_buys = 0; initial_parities = 0
    for case in execution['cases']:
        folder = run / 'accounts' / case['name']
        assert (folder / 'summary.json').exists() and not (folder / 'failure.json').exists()
        if case['policy'] not in ['rank_buffer', 'combined']: continue
        ev = pd.read_parquet(folder / 'events.parquet'); trades = pd.read_parquet(folder / 'trades.parquet')
        if case['kind'] == 'startup' and case['policy'] == 'rank_buffer':
            previous = ROOT / STATE['cfg']['startup_parent'] / 'accounts' / f"{case['cohort']}_{case['offset']:02d}_fresh_once"
            old = pd.read_parquet(previous / 'trades.parquet'); boot = pd.Timestamp(case['boot_date'])
            columns = ['ts_code', 'shares', 'price', 'fee', 'date', 'decision_date']
            left = trades.loc[trades.side.eq('buy') & trades.decision_date.eq(boot), columns].sort_values('ts_code').reset_index(drop=True)
            right = old.loc[old.side.eq('buy') & old.decision_date.eq(boot), columns].sort_values('ts_code').reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right, check_exact=True)
            initial_parities += 1
        calendar = 1 if case['kind'] == 'startup' else case['offset']
        original = STATE['schedules'][calendar]['baseline']['hold']
        exits = ev.loc[ev.kind.eq('hold_exit_requested') & ev.reason.eq('hold_false')]
        for e in exits.itertuples():
            if e.date not in original.index or original.at[e.date, e.ts_code] != 1: continue
            denied_checks += 1
            assert not (trades.side.eq('buy') & trades.ts_code.eq(e.ts_code) & trades.signal_date.eq(e.date)).any()
        # 所有已成交买单都必须能追溯至当期未被屏蔽的正当排名来源。
        for t in trades.loc[trades.side.eq('buy')].itertuples():
            source_calendar = case['offset'] if case['kind'] == 'startup' and t.signal_date == pd.Timestamp(case['boot_date']) else calendar
            signals = STATE['schedules'][source_calendar][case['policy']]['signals']
            assert pd.notna(signals.at[t.signal_date, t.ts_code])
            checked_buys += 1
    save(run / 'verification.json', dict(prefixes=rows, no_same_review_rebuy_checks=denied_checks,
        buffered_buy_signal_lineages=checked_buys, rank_buffer_first_fill_parities=initial_parities,
        corrected_top20_scope_checks=scope_checks, all_candidate_execution_and_risk_coverage=True,
        source_and_input_hashes_unchanged=True,
        root_agents_unchanged=True, scope='新增信号与执行层；未把历史窗口当独立样本'))


if __name__ == '__main__': main()
