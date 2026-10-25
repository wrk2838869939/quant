#!/usr/bin/env python3
"""用户授权近期评估的冻结低换手输入；只延长日期，不训练或运行账户。"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from patient_trend import review_dates, trend_state
from trend_entry import entry_features
from divide_selection import phase_features

FEATURE_PARENT = ROOT / 'outputs/autonomous_value_research/20260912T160734_658714Z'
FROZEN = ROOT / 'outputs/symmetric_phase/20260913T061312_007478Z'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def monthly_hold_states(trend, observable, confirmations):
    """逐字沿用原确认递推：未知打断失败计数，并保留NaN。"""
    if not trend.index.equals(observable.index) or not trend.columns.equals(observable.columns):
        raise ValueError('持有状态矩阵不对齐')
    failures = np.zeros(len(trend.columns), dtype=int)
    states = np.full(trend.shape, np.nan)
    for i in range(len(trend)):
        known = observable.iloc[i].to_numpy(bool)
        good = trend.iloc[i].to_numpy(bool)
        failures = np.where(known & ~good, failures + 1, 0)
        states[i, known] = (failures[known] < confirmations).astype(float)
    return pd.DataFrame(states, index=trend.index, columns=trend.columns)


def build_inputs(raw, basics, names, cfg, start, end):
    """复用原价量函数与全A资格，输出全部合格月度机会和原始持有许可。"""
    if raw.duplicated(['date', 'ts_code']).any() or raw.date.max() > pd.Timestamp(end):
        raise ValueError('行情身份重复或超出登记终点')
    columns = ['hfq_close', 'total_mv_yuan', 'turnover_rate']
    wide = {c: raw.pivot(index='date', columns='ts_code', values=c).sort_index() for c in columns}
    close = wide['hfq_close']
    spec = cfg['trend']
    features = entry_features(close, wide['total_mv_yuan'], wide['turnover_rate'],
                              medium=spec['medium_window'], long=spec['long_window'])
    score, trend = trend_state(close, spec)
    observable = (close.gt(0) & close.shift(spec['medium_window']).gt(0)
                  & close.shift(spec['long_window']).gt(0))
    if spec['require_above_long_ma']:
        observable &= close.where(close.gt(0)).rolling(spec['long_window'],
            min_periods=spec['long_window']).mean().notna()
    dates = close.index[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
    calendars = {frequency: review_dates(dates, frequency) for frequency in cfg['review_frequencies']}
    monthly = calendars['monthly']
    hold = monthly_hold_states(trend.loc[monthly], observable.loc[monthly], cfg['exit_confirmations'])
    basics = basics.set_index('ts_code')
    listing = pd.to_datetime(basics.list_date).reindex(close.columns)
    delisting = pd.to_datetime(basics.delist_date).reindex(close.columns)
    st = names[names.name.str.contains('ST', case=False, na=False)].copy()
    st['start'] = pd.to_datetime(st.start_date)
    st['end'] = pd.to_datetime(st.end_date)
    blocks = []
    for date in monthly:
        frame = pd.DataFrame({key: value.loc[date] for key, value in features.items()})
        eligible = ((date-listing).dt.days.ge(cfg['min_listing_days'])
            & (delisting.isna() | delisting.gt(date)) & np.isfinite(frame).all(axis=1))
        active_st = st.loc[st.start.le(date) & (st.end.isna() | st.end.ge(date)), 'ts_code']
        eligible.loc[eligible.index.isin(active_st)] = False
        frame = frame.loc[eligible].copy()
        # 全A合格截面先rank；主板/trend选择在后，不能改变原来的低换手名次。
        ranks = frame.rank(pct=True) - .5
        for key in features:
            frame['x_' + key] = ranks[key]
        frame['date'] = date
        frame['trend'] = trend.loc[date].reindex(frame.index)
        frame['momentum'] = score.loc[date].reindex(frame.index)
        for frequency, calendar in calendars.items():
            frame[frequency] = date in calendar
        blocks.append(frame.reset_index(names='ts_code'))
    return pd.concat(blocks, ignore_index=True), hold


def make_predictions(table, model, cfg):
    """只加载2021训练常数；phase只作原账本归因，不参与排序或门槛。"""
    if model.feature_cols != () or model.metadata['asof'] != cfg['final_fit_end']:
        raise ValueError('必须使用冻结2021年底主板常数模型')
    query = table[table.monthly & table.ts_code.str.match(cfg['mainboard_pattern'])].copy()
    expanded, _ = phase_features(query, [])
    result = expanded[['date', 'ts_code', 'phase', 'trend']].copy()
    pred = model.predict(query)
    for column in pred:
        result['low_turnover__' + column] = pred[column]
    gate = query.trend & pred.expected_return.gt(cfg['entry']['min_expected_return'])
    gate &= pred.expected_adverse.le(cfg['entry']['max_expected_adverse'])
    result['low_turnover__score'] = (-query.x_turnover).where(gate)
    return result.reset_index(drop=True)


def comparison_details(actual, expected, label):
    """严格逐值比较，报告浮点差异而不自动调宽阈值。"""
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        return dict(check=label, passed=True, rows=len(actual), columns=len(actual.columns))
    except AssertionError as error:
        detail = dict(check=label, passed=False, error=str(error)[:5000],
                      actual_shape=list(actual.shape), expected_shape=list(expected.shape))
        if actual.index.equals(expected.index) and actual.columns.equals(expected.columns):
            changed = ~((actual == expected) | (actual.isna() & expected.isna()))
            detail['changed_cells'] = int(changed.to_numpy().sum())
            numeric = actual.select_dtypes(include=np.number).columns.intersection(expected.select_dtypes(include=np.number).columns)
            if len(numeric):
                diff = np.abs(actual[numeric].to_numpy(float) - expected[numeric].to_numpy(float))
                detail['maximum_absolute_numeric_difference'] = float(np.nanmax(diff)) if np.isfinite(diff).any() else None
                detail['changed_numeric_columns'] = [c for c in numeric if changed[c].any()]
        return detail


def check_prefix(table, hold, predictions, signals, forecasts):
    """原2022—2023月度数据精确复现；新增股票旧日前必须无可观察状态。"""
    end = pd.Timestamp('2023-12-31')
    old_features = pd.read_parquet(FEATURE_PARENT / 'execution/pit_review_features.parquet')
    old_features = old_features[old_features.monthly].reset_index(drop=True)
    new_features = table[table.date.le(end)].reset_index(drop=True)
    checks = [comparison_details(new_features, old_features, 'all_A_monthly_features_exact')]
    old_predictions = pd.read_parquet(FROZEN / 'execution_predictions.parquet')
    wanted = ['date', 'ts_code', 'phase', 'trend'] + [c for c in old_predictions if c.startswith('low_turnover__')]
    checks.append(comparison_details(predictions[predictions.date.le(end)].reset_index(drop=True),
                                     old_predictions[wanted], 'mainboard_low_turnover_predictions_exact'))
    for name, matrix in [('monthly_hold_allowed', hold), ('low_turnover_signals', signals),
                          ('low_turnover_point_forecasts', forecasts)]:
        original = pd.read_parquet(FROZEN / 'execution' / (name + '.parquet'))
        if not original.columns.isin(matrix.columns).all():
            checks.append(dict(check=name + '_old_codes_preserved', passed=False))
            continue
        prefix = matrix.loc[matrix.index <= end]
        checks.append(comparison_details(prefix.reindex(columns=original.columns), original, name + '_exact'))
        added = prefix.columns.difference(original.columns)
        checks.append(dict(check=name + '_new_code_prefix_unavailable', passed=bool(prefix[added].isna().all().all()),
                           added_codes=len(added), nonmissing_prefix_values=int(prefix[added].notna().sum().sum())))
    return checks


def run_preparation(run, panel):
    """新输出目录先核对前缀再交付；不改root登记、不下载、不训练、不跑账户。"""
    run, panel = Path(run).resolve(), Path(panel).resolve()
    registration = json.loads((run / 'manifest.json').read_text())
    output = run / 'execution'
    output.mkdir(exist_ok=True)
    if (run / 'inputs_manifest.json').exists():
        raise ValueError('输入审计记录已存在，不覆盖已交付结果')
    cfg = yaml.safe_load((FEATURE_PARENT / 'config.yaml').read_text())
    frozen_cfg = yaml.safe_load((FROZEN / 'config.yaml').read_text())
    end = pd.Timestamp(registration['end'])
    start = pd.Timestamp(registration['continuation_start'])
    metadata = pq.ParquetFile(panel).metadata
    date_column = pq.ParquetFile(panel).schema.names.index('date')
    latest = max(pd.Timestamp(metadata.row_group(i).column(date_column).statistics.max) for i in range(metadata.num_row_groups))
    if latest != end:
        raise ValueError(f'本地最新日期{latest.date()}与已登记终点{end.date()}不同，不能无声扩展窗口')
    columns = ['date', 'ts_code', 'hfq_close', 'total_mv_yuan', 'turnover_rate']
    raw = pd.read_parquet(panel, columns=columns,
        filters=[('date', '>=', pd.Timestamp(cfg['history'][0])), ('date', '<=', end)])
    raw_hash = hashlib.sha256(pd.util.hash_pandas_object(raw, index=False).values.tobytes()).hexdigest()
    basic_path, names_path = ROOT / 'data/raw/stock_basic_all.parquet', ROOT / 'data/raw/namechange_all.parquet'
    basics = pd.read_parquet(basic_path, columns=['ts_code', 'list_date', 'delist_date'])
    names = pd.read_parquet(names_path, columns=['ts_code', 'name', 'start_date', 'end_date'])
    print('Local market and frozen metadata loaded', len(raw), 'rows', flush=True)
    table, hold = build_inputs(raw, basics, names, cfg, start, end)
    print('All-A monthly features and continuous hold states built', len(table), 'rows', flush=True)
    models_path = FROZEN / 'mainboard_execution_models.joblib'
    model = joblib.load(models_path)['low_turnover']
    predictions = make_predictions(table, model, frozen_cfg)
    signals = predictions.pivot(index='date', columns='ts_code', values='low_turnover__score').reindex(index=hold.index, columns=hold.columns)
    forecasts = predictions.pivot(index='date', columns='ts_code', values='low_turnover__expected_return').reindex(index=hold.index, columns=hold.columns)
    checks = check_prefix(table, hold, predictions, signals, forecasts)
    passed = all(check['passed'] for check in checks)
    (run / 'prefix_exact_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    source_paths = [Path(__file__), ROOT / 'src/trend_entry.py', ROOT / 'src/patient_trend.py',
        ROOT / 'src/divide_selection.py', ROOT / 'src/value_forecast.py',
        ROOT / 'scripts/run_patient_value_execution.py', FEATURE_PARENT / 'config.yaml',
        FROZEN / 'config.yaml', models_path, basic_path, names_path,
        FEATURE_PARENT / 'execution/pit_review_features.parquet', FROZEN / 'execution_predictions.parquet']
    source_paths += [FROZEN / 'execution' / (name + '.parquet') for name in
                    ['monthly_hold_allowed', 'low_turnover_signals', 'low_turnover_point_forecasts']]
    report = dict(status='ready' if passed else 'blocked_prefix_mismatch', no_download=True, no_training=True,
        no_account_execution=True, user_authorized_2024_plus_evaluation=True,
        continuation_start=str(start.date()), evaluation_start=registration['start'], market_end=str(end.date()),
        history_start=cfg['history'][0], all_A_monthly_feature_rows=len(table), mainboard_prediction_rows=len(predictions),
        monthly_reviews=len(hold), hold_codes=len(hold.columns), old_prefix_exact=passed,
        source_hashes={str(path): sha(path) for path in source_paths}, market_projection_hash=raw_hash,
        market_columns=columns, frozen_model_metadata=model.metadata,
        constant_expected_return=float(model.coefficients.loc['intercept', 'expected_return']),
        constant_expected_adverse=float(model.coefficients.loc['intercept', 'expected_adverse']),
        semantics='same_all_A_PIT_rank_monthly_unknown_breaks_confirmation_frozen_2021_constant60',
        checks=checks)
    if passed:
        table.to_parquet(run / 'pit_review_features.parquet', index=False)
        predictions.to_parquet(run / 'execution_predictions.parquet', index=False)
        hold.to_parquet(output / 'monthly_hold_allowed.parquet')
        signals.to_parquet(output / 'low_turnover_signals.parquet')
        forecasts.to_parquet(output / 'low_turnover_point_forecasts.parquet')
        joblib.dump(model, run / 'frozen_low_turnover_model.joblib')
        for path in source_paths:
            if path.is_relative_to(ROOT) and (path.suffix == '.py' or path.suffix == '.yaml'):
                dest = run / 'input_sources' / path.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(path.read_bytes())
        report['output_hashes'] = {str(path.relative_to(run)): sha(path) for path in [
            run / 'pit_review_features.parquet', run / 'execution_predictions.parquet', run / 'frozen_low_turnover_model.joblib',
            output / 'monthly_hold_allowed.parquet', output / 'low_turnover_signals.parquet', output / 'low_turnover_point_forecasts.parquet']}
    else:
        # 失败产物隔离，不伪装成可供账户读取的正式预测。
        failed = run / 'input_prefix_failure'
        failed.mkdir(exist_ok=True)
        table[table.date.lt('2024-01-01')].to_parquet(failed / 'recomputed_old_monthly_features.parquet', index=False)
        predictions[predictions.date.lt('2024-01-01')].to_parquet(failed / 'recomputed_old_predictions.parquet', index=False)
    (run / 'inputs_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise ValueError('旧窗口精确比较失败；已隔离差异，禁止下游运行或自行放宽误差')
    print('Inputs ready; all frozen 2022-2023 prefix checks exactly passed', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-panel', type=Path, default=ROOT / 'data/panel/daily_panel.parquet')
    args = parser.parse_args()
    run_preparation(args.run, args.source_panel)


if __name__ == '__main__':
    main()
