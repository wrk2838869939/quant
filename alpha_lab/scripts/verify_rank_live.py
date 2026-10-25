"""验证邻域规则前缀、真实账户截断回放与全部成交来源。"""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import validate_rank_live as research
from selection_stability import buffered_holds
from build_selection_stability import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    protocol = json.loads((run / 'protocol.json').read_text())
    research.initialize(run)
    prefix_rules = 0
    for day in protocol['days']:
        base = research.prior.STATE['schedules'][day]['baseline']
        for buffer in protocol['buffers']:
            for confirmations in protocol['confirmations']:
                full = buffered_holds(base['signals'], base['hold'], buffer, confirmations)
                for cut in [12, 24, 36, 48]:
                    truncated = buffered_holds(base['signals'].iloc[:cut],base['hold'].iloc[:cut],buffer,confirmations)
                    for actual, expected in zip(truncated, full):
                        pd.testing.assert_frame_equal(actual,expected.iloc[:cut],check_exact=True)
                    prefix_rules += 1
    prefixes = []
    for name in ['continued_08_rank8_confirm1_cost0', 'cold_22_rank12_confirm3_cost0',
                 'cold_01_rank10_confirm2_cost30']:
        case = next(c for c in protocol['cases'] if c['name']==name)
        research.COST = case['extra_cost_bps']/10000
        folder = run / 'accounts' / name
        old = {k:pd.read_parquet(folder/f'{k}.parquet') for k in ['daily','trades','holdings','events']}
        cutoff = old['daily'].index[243]
        result, _ = research.prior.simulate(case, cutoff)
        for key, expected in old.items():
            actual = getattr(result,key)
            if key == 'daily': expected = expected.loc[:cutoff]
            else:
                expected = expected.loc[expected.date.lt(cutoff) if key=='events' else expected.date.le(cutoff)]
                if key == 'events':
                    actual = actual.loc[actual.date.lt(cutoff)]
                    assert expected[expected.columns.difference(actual.columns)].isna().all().all()
                    actual = actual.reindex(columns=expected.columns)
                actual, expected = actual.reset_index(drop=True), expected.reset_index(drop=True)
            pd.testing.assert_frame_equal(actual,expected,check_exact=True,check_dtype=False)
        prefixes.append(dict(account=name, cutoff=str(cutoff.date()), all_four_ledgers_exact=True))
    buys = 0; exits = 0
    for case in protocol['cases']:
        folder = run / 'accounts' / case['name']
        assert not (folder / 'failure.json').exists()
        trade = pd.read_parquet(folder/'trades.parquet')
        events = pd.read_parquet(folder/'events.parquet')
        schedules = research.prior.STATE['schedules'][case['offset']]
        scores = schedules[case['policy']]['signals']
        for t in trade.loc[trade.side.eq('buy')].itertuples():
            assert pd.notna(scores.at[t.signal_date,t.ts_code])
            buys += 1
        old_hold = schedules['baseline']['hold']
        for event in events.loc[events.kind.eq('hold_exit_requested') & events.reason.eq('hold_false')].itertuples():
            if old_hold.at[event.date,event.ts_code] != 1: continue
            assert not (trade.side.eq('buy') & trade.ts_code.eq(event.ts_code) & trade.signal_date.eq(event.date)).any()
            exits += 1
    for name, expected in protocol['source_sha256'].items(): assert digest(ROOT/name)==expected
    assert digest(ROOT.parent/'AGENTS.md')==protocol['root_agents_sha256']
    parities = list((run/'accounts').glob('*/parent_parity.json'))
    assert len(parities)==16
    save(run/'verification.json',dict(rule_prefixes=prefix_rules, account_prefixes=prefixes,
        parent_account_parities=len(parities), buy_source_checks=buys, no_same_review_rebuy_checks=exits,
        all_candidate_coverage_guarded=True, sources_unchanged=True, root_agents_unchanged=True))
    print('Verified',prefix_rules,'rule prefixes,',len(prefixes),'account prefixes,',buys,'buys,',exits,'exits')


if __name__ == '__main__': main()
