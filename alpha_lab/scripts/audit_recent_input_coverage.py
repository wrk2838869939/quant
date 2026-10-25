#!/usr/bin/env python3
"""只读定位近期冻结输入缺月；不读收益标签，不改策略或原产物。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if p.name == 'alpha_lab')
sys.path.insert(0, str(ROOT / 'src'))
from trend_entry import entry_features
from validate_recent_input_coverage import check_run


def intervals(mask):
    """按现有交易日顺序合并全市场缺失区间。"""
    result, start, last, count = [], None, None, 0
    for date, missing in mask.items():
        if missing:
            start = date if start is None else start
            last, count = date, count + 1
        elif start is not None:
            result.append({'start': str(start.date()), 'end': str(last.date()), 'trading_days': count})
            start, count = None, 0
    if start is not None:
        result.append({'start': str(start.date()), 'end': str(last.date()), 'trading_days': count})
    return result


def audit(run, panel):
    run, panel = Path(run), Path(panel)
    output = run / 'input_coverage_audit'
    output.mkdir(exist_ok=True)
    hold = pd.read_parquet(run / 'execution/monthly_hold_allowed.parquet')
    end = hold.index.max() + pd.offsets.MonthEnd(0)
    required = ['date', 'ts_code', 'hfq_close', 'total_mv_yuan', 'turnover_rate']
    extras = [c for c in ['open', 'close', 'total_mv'] if c in pq.ParquetFile(panel).schema.names]
    # 2023整年仅提供120日回看；审计评估范围严格从2024起。
    raw = pd.read_parquet(panel, columns=required + extras,
        filters=[('date', '>=', pd.Timestamp('2023-01-01')), ('date', '<=', end)])
    fields = required[2:] + extras
    daily = pd.DataFrame({'date': raw.date})
    for name in fields:
        daily[name + '_finite'] = np.isfinite(raw[name])
        daily[name + '_positive'] = raw[name].gt(0)
    counts = daily.groupby('date').sum()
    counts['panel_rows'] = raw.groupby('date').size()
    counts = counts.loc['2024-01-01':]
    counts.to_csv(output / 'raw_daily_coverage.csv')
    wide = {name: raw.pivot(index='date', columns='ts_code', values=name).sort_index()
            for name in required[2:]}
    cfg_path = ROOT / 'outputs/autonomous_value_research/20260912T160734_658714Z/config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())['trend']
    features = entry_features(wide['hfq_close'], wide['total_mv_yuan'], wide['turnover_rate'],
                             medium=cfg['medium_window'], long=cfg['long_window'])
    reviews = hold.index[hold.index >= '2024-01-01']
    monthly = counts.reindex(reviews).copy()
    finite_all = None
    feature_intervals = {}
    for name, frame in features.items():
        finite = np.isfinite(frame)
        monthly[name + '_feature_finite'] = finite.loc[reviews].sum(axis=1)
        finite_all = finite if finite_all is None else finite_all & finite
        feature_intervals[name] = intervals(finite.loc['2024-01-01':].sum(axis=1).eq(0))
    monthly['all9_finite_before_PIT'] = finite_all.loc[reviews].sum(axis=1)
    for name, path in [('features', run / 'pit_review_features.parquet'), ('predictions', run / 'execution_predictions.parquet')]:
        table = pd.read_parquet(path, columns=['date', 'ts_code'])
        monthly[name + '_rows'] = table.groupby('date').size().reindex(reviews, fill_value=0)
    signals = pd.read_parquet(run / 'execution/low_turnover_signals.parquet')
    monthly['finite_signals'] = signals.loc[reviews].notna().sum(axis=1)
    monthly['observable_hold_codes'] = hold.loc[reviews].notna().sum(axis=1)
    monthly.to_csv(output / 'monthly_feature_coverage.csv')
    absent = monthly.index[monthly.predictions_rows.eq(0)].strftime('%Y-%m-%d').tolist()
    try:
        gate = check_run(run)
    except (ValueError, AssertionError):
        gate = json.loads((output / 'coverage_gate.json').read_text())
    result = {'status': 'blocked_missing_input_dates' if absent else gate['status'],
        'audit_range': [str(counts.index.min().date()), str(counts.index.max().date())],
        'absent_prediction_review_count': len(absent), 'absent_prediction_reviews': absent,
        'raw_global_missing_dates': {name: counts.index[counts[name + '_finite'].eq(0)].strftime('%Y-%m-%d').tolist() for name in fields},
        'derived_feature_global_missing_intervals': feature_intervals,
        'coverage_gate': gate['status'], 'source_panel': str(panel.resolve()),
        'source_projection_sha256': hashlib.sha256(pd.util.hash_pandas_object(raw, index=False).values.tobytes()).hexdigest(),
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / 'src/trend_entry.py', cfg_path]},
        'historical_failure_cause': '原total_mv_yuan及turnover_rate全缺日；turnover的60日满窗口均值使其后多个评估日全市场特征不可用。',
        'adapter_defect': '旧执行器要求prediction日期与hold日期一致；新冻结adapter遗漏检查，reindex把缺预测月份填成全NaN信号。2022–23前缀精确检查不能证明新增窗口覆盖。',
        'interpretation': ('旧近期执行受人工无候选/现金路径污染，不能解释成策略主动成功择时或作为近期策略业绩结论。' if absent else '本run覆盖完整。旧run人工无候选现金路径仍作废；此审计只认证输入覆盖，不评价账户收益。'),
        'repair_requirements': ['独立恢复四个缺失日的正式daily_basic源数据，验证身份、字段单位和范围，仅修复登记列。',
            '新run重建同规则全A rank/预测；旧输入和结果保留。',
            '所有hold月必须有特征及预测身份，逐月矩阵复现；缺月即拦截账户，未知不能默认为无机会。',
            '重新核对2022–23精确前缀，再重跑受影响近期账户；不得填零换手或调滚动min_periods。']}
    (output / 'coverage_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    guard = Path(__file__).with_name('validate_recent_input_coverage.py')
    (output / guard.name).write_bytes(guard.read_bytes())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-panel', type=Path, default=ROOT / 'data/panel/daily_panel.parquet')
    args = parser.parse_args()
    audit(args.run, args.source_panel)
