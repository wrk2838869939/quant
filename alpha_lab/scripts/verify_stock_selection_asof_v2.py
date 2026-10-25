#!/usr/bin/env python3
"""核验个股历史训练边界、保存权重、候选约束和文件哈希。"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'alpha_lab/src'))
from sector_dataset import digest
from stock_selection_asof import select_positive_top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    out = args.run
    manifest = json.loads((out / 'manifest.json').read_text())
    cfg = yaml.safe_load((out / 'config.yaml').read_text())
    assert manifest['status'] == 'complete'
    checks = 0
    for group, root in [('sources', ROOT), ('reporter_source', ROOT), ('inputs', Path(manifest['parent'])), ('artifacts', out)]:
        for filename, expected in manifest[group].items():
            assert digest(root / filename) == expected, filename
            checks += 1
    data = pd.read_parquet(out / 'research_labels_WITH_FUTURE.parquet')
    frame = pd.read_parquet(out / 'all_decisions_BEFORE_EVALUATION.parquet')
    assert frame.date.eq(frame.confirmation_date).all()
    assert frame.qualified.all()
    assert not frame.duplicated(['event_id', 'ts_code']).any()
    fits = json.loads((out / 'fits.json').read_text())
    years = []
    for fit in fits:
        if fit['status'] != 'complete':
            assert not frame.loc[frame.date.dt.year.eq(fit['year']), 'ridge_price5'].any()
            continue
        train = data[data.event_id.isin(fit['train_events'])]
        inner = data[data.event_id.isin(fit['fit_events'])]
        valid = data[data.event_id.isin(fit['valid_events'])]
        start = pd.Timestamp(fit['start'])
        assert train.label_complete.all() and train.target.notna().all()
        assert train.date.lt(start).all() and train.exit_date.lt(pd.Timestamp(fit['cutoff'])).all()
        assert inner.exit_date.lt(pd.Timestamp(fit['inner_cutoff'])).all()
        assert set(inner.date).isdisjoint(valid.date)
        target = frame[frame.model_start.eq(start)].copy()
        assert set(train.event_id).isdisjoint(target.event_id)
        weights = np.load(out / 'models' / f"{fit['year']}.npz")
        np.testing.assert_allclose(target[fit['features']].to_numpy() @ weights['coef'], target.model_score, rtol=1e-12, atol=1e-14)
        selected = select_positive_top(target, 'model_score', cfg)
        assert set(selected) == set(target.index[target.ridge_price5])
        years.append(fit['year'])
    first = frame[frame.ridge_price5]
    assert first.groupby('event_id').size().le(cfg['top_n']).all()
    assert first.price.all() and first.model_score.gt(cfg['model']['minimum_predicted_excess']).all()
    result = dict(status='passed', checked_hashes=checks, checked_model_years=years,
                  sector_events=frame.event_id.nunique(), stocks=len(frame), selected_event_stocks=len(first),
                  selected_events=first.event_id.nunique(), data_vintage_certified=False, account_backtest=False)
    (out / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
