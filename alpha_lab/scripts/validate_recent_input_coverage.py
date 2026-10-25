#!/usr/bin/env python3
"""冻结输入独立覆盖门：未知数据月份不能自动解释成现金机会。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_coverage(features, predictions, hold, signals, forecasts):
    """逐月要求机会与预测身份存在；允许有数据但趋势门全部拒绝。"""
    expected = pd.DatetimeIndex(hold.index)
    if expected.has_duplicates or not expected.is_monotonic_increasing:
        raise ValueError('持有日期重复或未排序')
    counts = {}
    for name, table in [('features', features), ('predictions', predictions)]:
        if table.duplicated(['date', 'ts_code']).any():
            raise ValueError(name + '身份重复')
        observed = pd.DatetimeIndex(sorted(pd.to_datetime(table.date).unique()))
        missing, extra = expected.difference(observed), observed.difference(expected)
        if len(missing) or len(extra):
            raise ValueError(f'{name}日期覆盖失败: missing={missing.strftime("%Y-%m-%d").tolist()}, extra={extra.strftime("%Y-%m-%d").tolist()}')
        counts[name] = table.groupby('date').size().to_dict()
    for name, matrix, column in [('signals', signals, 'low_turnover__score'),
                                  ('forecasts', forecasts, 'low_turnover__expected_return')]:
        if not matrix.index.equals(hold.index) or not matrix.columns.equals(hold.columns):
            raise ValueError(name + '持有矩阵身份不对齐')
        reference = predictions.pivot(index='date', columns='ts_code', values=column)
        if not reference.columns.isin(hold.columns).all():
            raise ValueError(name + '预测代码不在持有矩阵')
        pd.testing.assert_frame_equal(matrix, reference.reindex(index=hold.index, columns=hold.columns), check_exact=True)
    if not np.isfinite(forecasts.to_numpy(float)).any(axis=1).all():
        raise ValueError('至少一个月份没有任何有限预测，不能解释成现金机会')
    return {'status': 'passed', 'review_dates': len(expected),
            'minimum_feature_rows_per_review': int(min(counts['features'].values())),
            'minimum_prediction_rows_per_review': int(min(counts['predictions'].values())),
            'all_nan_signal_dates_with_observed_candidates': signals.index[signals.isna().all(axis=1)].strftime('%Y-%m-%d').tolist()}


def check_run(run):
    run = Path(run)
    paths = [run / 'pit_review_features.parquet', run / 'execution_predictions.parquet'] + [
        run / 'execution' / (name + '.parquet') for name in
        ['monthly_hold_allowed', 'low_turnover_signals', 'low_turnover_point_forecasts']]
    report = {'status': 'blocked', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_hashes': {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    try:
        report.update(validate_coverage(*(pd.read_parquet(p) for p in paths)))
    except (ValueError, AssertionError) as error:
        report['error'] = str(error)
    output = run / 'input_coverage_audit'
    output.mkdir(exist_ok=True)
    (output / 'coverage_gate.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'passed':
        raise ValueError(report['error'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(check_run(parser.parse_args().run), ensure_ascii=False, indent=2))
