"""独立核验四日两列修复、月度覆盖准入，再调用冻结的账户核验器。"""
import argparse
from itertools import zip_longest
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATES = ['20240819', '20250120', '20250421', '20260129']
COLUMNS = ['date', 'ts_code', 'hfq_close', 'total_mv_yuan', 'turnover_rate']


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_repair(run):
    """按原始行序分块逐格比较，不调用修复生成代码。"""
    repair = run / 'data_repair'
    registration = json.loads((run / 'manifest.json').read_text())
    parent = Path(registration['parent_invalid_run'])
    prior = json.loads((parent / 'inputs_manifest.json').read_text())
    source = ROOT / 'data/panel/daily_panel.parquet'
    source_stat = source.stat()
    current_file_sha = sha(source)
    repair_manifest = json.loads((repair / 'repair_manifest.json').read_text())
    require(repair_manifest['dates'] == DATES, '修复日期偏离限定四日')
    for name, expected in repair_manifest['artifacts'].items():
        require(sha(repair / name) == expected, '修复产物哈希不符: ' + name)
    raw_api = {}
    for date in DATES:
        table = pd.read_parquet(repair / f'daily_basic_{date}.parquet')
        require(not table.ts_code.duplicated().any(), 'API重复股票')
        require(table.trade_date.astype(str).eq(date).all(), 'API含其他日期')
        require(0 < len(table) < 5800, 'API空响应或可能截断')
        raw_api[pd.Timestamp(date)] = table.set_index('ts_code')
    original = pq.ParquetFile(source)
    fixed = pq.ParquetFile(repair / 'repaired_feature_projection.parquet')
    require(original.metadata.num_rows == fixed.metadata.num_rows, '投影行数变化')
    require(fixed.schema_arrow.names == COLUMNS, '投影列范围变化')
    original_hash = hashlib.sha256()
    differences, counts = [], {day: 0 for day in DATES}
    total_rows, preserved_rows = 0, 0
    batches_old = original.iter_batches(batch_size=250000, columns=COLUMNS)
    batches_new = fixed.iter_batches(batch_size=250000, columns=COLUMNS)
    for old_batch, new_batch in zip_longest(batches_old, batches_new):
        require(old_batch is not None and new_batch is not None, '分块数量变化')
        old, new = old_batch.to_pandas(), new_batch.to_pandas()
        require(len(old) == len(new), '原投影行序或分块长度不同')
        original_hash.update(pd.util.hash_pandas_object(old, index=False).values.tobytes())
        pd.testing.assert_frame_equal(old[['date', 'ts_code', 'hfq_close']],
                                      new[['date', 'ts_code', 'hfq_close']], check_exact=True)
        affected = old.date.isin(raw_api)
        pd.testing.assert_frame_equal(old.loc[~affected], new.loc[~affected], check_exact=True)
        preserved_rows += int((~affected).sum())
        total_rows += len(old)
        for day, api in raw_api.items():
            selected = old.date.eq(day)
            if not selected.any():
                continue
            counts[day.strftime('%Y%m%d')] += int(selected.sum())
            codes = old.loc[selected, 'ts_code']
            require(codes.isin(api.index).all(), 'API漏掉原面板股票')
            for column, source_column, scale in [('total_mv_yuan', 'total_mv', 10000.),
                                                   ('turnover_rate', 'turnover_rate', 1.)]:
                require(old.loc[selected, column].isna().all(), '修复覆盖了非缺失旧值')
                expected = codes.map(api[source_column]).to_numpy(float) * scale
                actual = new.loc[selected, column].to_numpy(float)
                require(np.isfinite(expected).all() and (expected >= 0).all(), 'API字段无效')
                require(np.array_equal(actual, expected), '修复值不等于API原值及规定单位转换')
                require(np.isfinite(actual).all(), '修复仍有缺失')
                for code, value in zip(codes, expected):
                    differences.append({'date': day.strftime('%Y%m%d'), 'ts_code': code,
                                        'field': column, 'old_value': np.nan, 'new_value': value})
    require(original_hash.hexdigest() == prior['market_projection_hash'],
            '当前原面板五列内容已不同于修复前冻结输入')
    recorded = pd.read_csv(repair / 'changed_cells.csv', dtype={'date': str}, float_precision='round_trip')
    rebuilt = pd.DataFrame(differences)
    keys = ['date', 'ts_code', 'field']
    require(not recorded.duplicated(keys).any(), '变更账本重复格')
    pd.testing.assert_frame_equal(recorded.sort_values(keys).reset_index(drop=True),
                                  rebuilt.sort_values(keys).reset_index(drop=True), check_exact=True)
    require(len(rebuilt) == repair_manifest['changed_cells'], '变更格数量不符')
    for date, count in counts.items():
        require(count == len(raw_api[pd.Timestamp(date)]), '四日API股票集合不等于原面板股票集合')
    after = source.stat()
    require((after.st_size, after.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns),
            '独立检查期间原面板发生写入')
    report = dict(status='passed', original_rows=total_rows, unchanged_full_projection_rows=preserved_rows,
                  changed_cells=len(rebuilt), per_date_rows=counts,
                  raw_panel_five_column_hash_matches_before_repair=True,
                  raw_panel_five_column_hash=original_hash.hexdigest(),
                  all_projected_hfq_prices_unchanged=True,
                  all_other_projected_cells_unchanged=True, all_replacements_from_original_nan=True,
                  exact_api_values_and_units=True, changed_cells_ledger_exact=True,
                  current_raw_panel_file_sha256=current_file_sha,
                  raw_panel_file_bytes=source_stat.st_size, raw_panel_mtime_ns=source_stat.st_mtime_ns,
                  raw_panel_stat_unchanged_during_verification=True,
                  full_73_column_historical_byte_hash_available=False,
                  evidence_boundary='修复前保存五列内容哈希而非整份原面板字节哈希；已证五列仍与旧冻结一致，不能凭空认证其余68列历史字节。修复仅写独立五列投影。',
                  repair_artifact_hashes={p.name: sha(p) for p in repair.iterdir() if p.is_file()},
                  verifier_sha256=sha(Path(__file__)))
    (run / 'independent_repair_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def verify_coverage(run):
    """在调用账户核验前检查每个既定review日及冻结矩阵覆盖。"""
    out = run / 'execution'
    predictions = pd.read_parquet(run / 'execution_predictions.parquet')
    predictions['date'] = pd.to_datetime(predictions.date)
    hold = pd.read_parquet(out / 'monthly_hold_allowed.parquet')
    signals = pd.read_parquet(out / 'low_turnover_signals.parquet')
    forecasts = pd.read_parquet(out / 'low_turnover_point_forecasts.parquet')
    require(hold.index.equals(signals.index) and hold.columns.equals(signals.columns), '持有与信号矩阵未对齐')
    require(forecasts.index.equals(signals.index) and forecasts.columns.equals(signals.columns), '预测收益与信号未对齐')
    require(not predictions.duplicated(['date', 'ts_code']).any(), '预测键重复')
    rows = []
    for date in hold.index:
        known = predictions[predictions.date.eq(date)]
        rows.append({'review_date': str(pd.Timestamp(date).date()), 'prediction_rows': len(known),
                     'signal_nonmissing': int(signals.loc[date].notna().sum()),
                     'forecast_nonmissing': int(forecasts.loc[date].notna().sum()),
                     'hold_nonmissing': int(hold.loc[date].notna().sum())})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'independent_prediction_review_coverage.csv', index=False)
    missing = frame[frame.prediction_rows.eq(0)]
    require(missing.empty, '预测仍完全缺少既定复核日: ' + ','.join(missing.review_date))
    require(not (signals.notna() & forecasts.isna()).any().any(), '有效候选缺少点预测')
    source_gate = run / 'input_coverage_audit/coverage_gate.json'
    require(source_gate.exists(), '缺少独立上游原始字段/rolling覆盖准入报告')
    gate = json.loads(source_gate.read_text())
    require(gate.get('status') in ['passed', 'pass'], '上游覆盖门未通过')
    for name, expected in gate['input_hashes'].items():
        require(sha(run / name) == expected, '上游覆盖门引用输入已漂移: ' + name)
    gate_script = source_gate.parent / 'validate_recent_input_coverage.py'
    require(sha(gate_script) == gate['script_sha256'], '上游覆盖门脚本快照哈希不同')
    report = dict(status='passed', review_count=len(frame), zero_prediction_review_count=0,
                  minimum_predictions_per_review=int(frame.prediction_rows.min()),
                  matrix_alignment_passed=True, source_coverage_gate=str(source_gate),
                  source_coverage_gate_sha256=sha(source_gate), verifier_sha256=sha(Path(__file__)))
    (run / 'independent_coverage_admission.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repair-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    repair = verify_repair(run)
    if args.repair_only:
        print(json.dumps({'status': 'repair_passed', 'changed_cells': repair['changed_cells']}))
        return
    coverage = verify_coverage(run)
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_recent_low_turnover.py'), '--run', str(run)], check=True)
    ledger = json.loads((run / 'execution/independent_recent_verification.json').read_text())
    require(ledger['status'] == 'complete', '独立账户核验未完成')
    report = dict(status='complete', repair='passed', coverage='passed', accounts='passed',
                  cases=len(ledger['cases']), reviews=coverage['review_count'],
                  component_hashes={name: sha(run / name) for name in ['independent_repair_verification.json',
                    'independent_coverage_admission.json', 'execution/independent_recent_verification.json']},
                  wrapper_sha256=sha(Path(__file__)))
    (run / 'independent_combined_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
