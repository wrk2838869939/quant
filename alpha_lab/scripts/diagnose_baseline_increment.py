#!/usr/bin/env python3
"""重载已选模型，比较训练与紧邻预测段的排名质量；不训练、不调参。"""
from pathlib import Path
import argparse
import json
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from datahub import DataHub
import run_baseline_increment as runner
from baseline_increment import training_mask
from learning_search import daily_ic
from run_classic_strategy_smoke import digest, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert os.environ.get('DATAHUB_CONTENT_ID') == manifest['dataset']['content_id']
    assert all(digest(PROJECT / p) == h for p, h in manifest['source_hashes'].items())
    plan = yaml.safe_load((run / 'sources/baseline_increment_v1.yaml').read_text())
    registry = yaml.safe_load((run / 'sources/selection_baselines_v1.yaml').read_text())
    parent_choices = json.loads((PROJECT / plan['parent_run'] / 'all_locked_choices.json').read_text())
    base = PROJECT / manifest['source_run']
    hashes = json.loads((run / 'input_hashes.json').read_text())
    assert all(digest(base / 'inputs' / p) == h for p, h in hashes.items())
    eligible = pd.read_parquet(base / 'inputs/eligible.parquet')
    history = pd.read_parquet(base / 'inputs/rule_history.parquet')
    calendar = DataHub().calendar.cn_days()
    grid = calendar[(calendar >= pd.Timestamp('2016-01-01')) & (calendar <= eligible.index.max())]
    wide = {k: history.pivot(index='date', columns='ts_code', values=k).reindex(index=grid, columns=eligible.columns)
            for k in history if k not in ['date', 'ts_code']}
    runner.G.update(plan=plan, registry=registry, parent_choices=parent_choices, eligible=eligible, wide=wide, calendar=calendar)
    cache, rows = {}, []
    for fit_file in sorted(run.glob('outer_models/*/*/*/fits.json')):
        records = json.loads(fit_file.read_text())
        record = next((r for r in records if r['status'] == 'ok'), None)
        if record is None:
            continue
        year, host = int(fit_file.parents[2].name), fit_file.parents[1].name
        if year not in cache:
            cache[year] = runner.packages(year, f'{year}-12-31')
        data = cache[year][host]['data']
        spec, fit_date = record['params'], pd.Timestamp(record['fit_date'])
        later = [pd.Timestamp(r['fit_date']) for r in records if pd.Timestamp(r['fit_date']) > fit_date]
        next_fit = min(later) if later else pd.Timestamp(f'{year+1}-01-01')
        mask = training_mask(data, spec, fit_date, plan['training']['train_window'])
        import hashlib
        assert hashlib.sha256(mask.tobytes()).hexdigest() == record['mask_sha256']
        artifact = fit_file.parent / record['model_file']
        assert digest(artifact) == record['model_sha256']
        models = joblib.load(artifact)
        label = data['labels'][spec['horizon']]
        dates, codes = data['dates'], data['codes']
        end_known = ~pd.isna(label['end']) & (label['end'] < np.datetime64(next_fit))
        future = data['allowed'] & (((dates >= fit_date) & (dates < next_fit)) & end_known)[:, None]
        future &= np.isfinite(label['direct'])
        predictions = {}
        for mode, model in models.items():
            prediction = np.full(data['allowed'].shape, np.nan)
            prediction_rows = mask | future
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names')
                prediction[prediction_rows] = model.predict(data['x'][prediction_rows])
            saved = pd.read_parquet(fit_file.parent / f'{mode}_predictions.parquet').reindex(index=dates, columns=codes).to_numpy()
            np.testing.assert_allclose(prediction[future], saved[future], rtol=0, atol=1e-12)
            predictions[mode] = prediction
        # MSE另列常数0.5对照，避免把缩向截面均值误写成选股能力。
        scores = dict(host=data['prior'], direct=predictions['direct'],
                      residual_full=data['prior'] + np.clip(predictions['residual'], -plan['increment_clip'], plan['increment_clip']),
                      constant_half=np.full(data['allowed'].shape, .5))
        for segment, selected in [('train', mask), ('next_segment', future)]:
            target = pd.DataFrame(label['direct'], index=dates, columns=codes).where(selected)
            frames = {k: pd.DataFrame(v, index=dates, columns=codes).where(selected) for k, v in scores.items()}
            ics = daily_ic(frames, target, plan['training']['min_label_stocks'])
            for mode, frame in frames.items():
                errors = (frame - target) ** 2
                rows.append(dict(year=year, host=host, model_spec=spec, fit_date=str(fit_date.date()),
                                 next_fit=str(next_fit.date()), segment=segment, score=mode,
                                 rows=int(selected.sum()), dates=int(selected.any(axis=1).sum()),
                                 equal_date_mse=float(errors.mean(axis=1).mean()), mean_ic=float(ics[mode].mean()),
                                 ic_dates=int(ics[mode].notna().sum()),
                                 max_label_end=str(pd.Timestamp(label['end'][selected.any(axis=1)].max()).date()) if selected.any() else None))
    pd.DataFrame(rows).to_csv(run / 'prediction_objective_diagnosis.csv', index=False)
    save_json(run / 'prediction_objective_diagnosis.json', dict(status='complete', readonly=True,
              source_sha256=digest(Path(__file__)), fit_streams=len(rows) // 8, metric_rows=len(rows),
              training_masks_reconstructed=True, saved_first_segment_predictions_match=True,
              limits='每条已选规格仅首个实际拟合和紧邻预测段；并非全年预测质量或统计显著性；residual_full固定幅度1仅用于目标诊断，不作交易选择'))
    print(f'DIAGNOSED {len(rows)//8} paired fit streams; no model refit or parameter selection')


if __name__ == '__main__':
    main()
