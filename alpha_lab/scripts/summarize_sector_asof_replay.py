#!/usr/bin/env python3
"""补充同事件同日对照、新增事件和去除2022年的敏感性，不重选策略。"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'alpha_lab/src'))
from sector_dataset import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    out = args.run
    frame = pd.read_parquet(out / 'stock_decisions_BEFORE_EVALUATION.parquet')
    choices = pd.read_parquet(out / 'first_choices_BEFORE_EVALUATION.parquet')
    obs = pd.read_parquet(out / 'price_observations_NOT_ACCOUNT.parquet')
    obs = obs[obs.horizon.eq(20)]
    events = pd.read_csv(out / 'event_price_comparison_NOT_ACCOUNT.csv', parse_dates=['date'])
    events = events[events.horizon.eq(20)]
    originals = choices[choices.variant.eq('original')]
    anchors = originals[['event_id', 'date']].drop_duplicates()
    common = frame.merge(anchors, on=['event_id', 'date'], validate='many_to_one')
    rows = []
    for variant in sorted(choices.variant.unique()):
        selected = common[common[variant]].merge(obs, on=['event_id', 'date', 'ts_code'], validate='one_to_one')
        basket = selected.groupby(['event_id', 'date']).agg(complete=('complete', 'all'), value=('price_return', 'mean'))
        basket.loc[~basket.complete, 'value'] = float('nan')
        basket['variant'] = variant
        rows.append(basket.reset_index())
    baskets = pd.concat(rows, ignore_index=True)
    wide = baskets.pivot(index=['event_id', 'date'], columns='variant', values='value')
    controlled = []
    for variant in wide.columns:
        pair = wide[['original']] if variant == 'original' else wide[['original', variant]].dropna()
        delta = pair[variant] - pair.original
        controlled.append(dict(variant=variant, common_date_events=len(pair), mean_difference=delta.mean(), median_difference=delta.median()))
    pd.DataFrame(controlled).to_csv(out / 'same_event_same_date_vs_original.csv', index=False)
    baskets.to_csv(out / 'same_event_same_date_baskets.csv', index=False)
    sensitivity = []
    for variant, group in events.groupby('variant'):
        for scope, subset in [('all', group), ('added_event', group[~group.event_id.isin(originals.event_id)]),
                              ('excluding_2022', group[group.date.dt.year.ne(2022)])]:
            valid = subset[subset.complete]
            paired = subset[subset.complete & subset.base_complete]
            sensitivity.append(dict(variant=variant, scope=scope, events=len(subset), complete_events=len(valid),
                                    paired_events=len(paired), mean=valid.price_return.mean(), median=valid.price_return.median(),
                                    mean_excess=paired.excess.mean()))
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(out / 'added_and_year_sensitivity.csv', index=False)
    print(pd.DataFrame(controlled).to_string(index=False))
    print(sensitivity[sensitivity.scope.eq('excluding_2022')].to_string(index=False))
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['supplement_source'] = {str(Path(__file__).relative_to(ROOT)): digest(__file__)}
    manifest['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
