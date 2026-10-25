#!/usr/bin/env python3
"""复核年度标签边界、预测阈值及保存文件哈希。"""
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
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    count = 0
    for group, root in [('sources', ROOT), ('inputs', ROOT), ('artifacts', out)]:
        for name, value in manifest[group].items():
            assert digest(root / name) == value, f'{group}: {name}'
            count += 1
    data = pd.read_parquet(out / 'research_inputs_WITH_LABELS.parquet').set_index('event_id')
    pred = pd.read_parquet(out / 'predictions_BEFORE_EVALUATION.parquet')
    assert pred.event_id.is_unique
    assert pred.selected.eq(pred.score.ge(pred.threshold)).all()
    years = []
    for fit in json.loads((out / 'fits.json').read_text()):
        if fit['status'] != 'complete':
            continue
        train = data.loc[fit['train_ids']]
        inner = data.loc[fit['fit_ids']]
        valid = data.loc[fit['valid_ids']]
        start = pd.Timestamp(fit['start'])
        test = pred[pred.model_start.eq(start)]
        assert train.date.lt(start).all() and train.end_date.lt(pd.Timestamp(fit['cutoff'])).all()
        assert inner.end_date.lt(pd.Timestamp(fit['inner_cutoff'])).all()
        assert inner.date.max() < valid.date.min()
        assert set(inner.index).isdisjoint(valid.index)
        assert set(train.index).isdisjoint(test.event_id)
        assert test.threshold.eq(fit['threshold']).all()
        years.append(fit['year'])
    assert all(r['exact'] for r in json.loads((out / 'event_prefix_audit.json').read_text()))
    assert all(r['exact'] for r in json.loads((out / 'stock_prefix_audit.json').read_text()))
    decisions = pd.read_parquet(out / 'stock_decisions_BEFORE_EVALUATION.parquet')
    assert set(decisions.event_id).issubset(pred.loc[pred.selected, 'event_id'])
    assert not decisions.duplicated(['event_id', 'date', 'ts_code']).any()
    result = dict(status='passed', checked_hashes=count, annual_timing_years=years,
                  prediction_rows=len(pred), selected_events=int(pred.selected.sum()),
                  stock_decision_rows=len(decisions), data_vintage_certified=False)
    (out / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
