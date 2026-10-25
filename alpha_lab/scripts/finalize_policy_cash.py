#!/usr/bin/env python3
"""核对P2/P3完成证据并追加全部实验事件；不晋升策略或改变已冻结输入。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from ledger import ExperimentLedger
from run_policy_cash_execution import frozen_check


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    read = lambda name: json.loads((run/name).read_text())
    root_manifest = read('manifest.json')
    cash, confirm = read('cash_execution/manifest.json'), read('cash_confirm/manifest.json')
    assert cash['status'] == confirm['status'] == 'complete'
    assert cash['completed_cases'] == 16 and confirm['completed_cases'] == 5
    assert cash['independent_verification_pass'] and cash['order_invariance_pass']
    assert confirm['independent_verification_pass'] and confirm['trigger_verification_pass']
    frozen_check(run, cash)
    frozen_check(run/'cash_confirm/workspace', confirm)
    assert read('limits_manifest.json')['status'] == 'complete'
    assert read('policy_labels_manifest.json')['status'] == 'complete'
    assert read('verification/policy_label_verification.json')['status'] == 'complete_pass'
    review = read('verification/policy_model_review.json')
    assert review['status'] == 'passed' and review['models_verified'] == 8
    decision = read('policy_target_decision.json')
    assert decision['status'] == 'evaluated' and not decision['passed']
    assert decision['action'] == 'stop_no_incremental_support'
    assert read('fit_policy_manifest.json')['model_count'] == 8
    tests = (run/'full_test_suite.log').read_text()
    assert 'Ran 239 tests' in tests and tests.rstrip().endswith('OK')
    for manifest_name in ['policy_labels_manifest.json','verification/policy_model_review.json']:
        manifest = read(manifest_name)
        for name, expected in manifest['source_hashes'].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT/path) == expected, name
    report = ROOT/'docs/POLICY_CASH_VALIDATION_20260913.md'
    assert report.exists()
    ledger = ExperimentLedger()
    logged = set(ledger.df.get('experiment_id', pd.Series(dtype=str)).dropna())
    identifiers = []
    def event(name, label, period, params, sources, inputs, verdict='NOT_ADMITTED'):
        identifier = f'{run.name}:{name}'
        identifiers.append(identifier)
        if identifier in logged:
            return
        ledger.log('policy_cash_'+name.split(':')[0], label, period,
            float('nan'), float('nan'), float('nan'), float('nan'), params=params,
            verdict=verdict, experiment_id=identifier,
            parent_experiment='symmetric_phase_20260913_P2_P3',
            source_hashes=sources, input_hashes=inputs,
            note='模型/资金/倒序事件相互依赖；追加P3b在P3之后登记；全部结果保留，未准入。',
            selection_note='重复开发窗口；无TEST；事件计数不作独立试验数；政策目标停止门未过。')
    for folder, manifest in [('cash_execution',cash),('cash_confirm',confirm)]:
        summary = pd.read_csv(run/folder/'execution_summary.csv')
        assert len(summary) == manifest['completed_cases']
        for row in json.loads(summary.to_json(orient='records')):
            event(folder+':'+row['name'], 'patient_hold_actual_cash_share_ledger',
                '2022-2023_reused_development', dict(output=str(run/folder), result=row,
                    adaptive_followup_after_P3=folder=='cash_confirm'),
                manifest['source_hashes'], manifest['input_hashes'])
    metadata = read('policy_model_metadata.json')
    fit_manifest = read('fit_policy_manifest.json')
    for stage, info in metadata.items():
        for policy, model in info['models'].items():
            event(f'target_model:{stage}:{policy}', 'gross_adjusted_unit_policy_120_proxy',
                stage+'_reused_development', dict(output=str(run), model=model,
                    common_training_rows=info['common_training_rows'],
                    common_training_dates=info['common_training_dates'],
                    common_cohort_hash=info['common_cohort_hash'], decision='no_additional_replay'),
                fit_manifest['source_hashes'], {'policy_labels.parquet':sha(run/'policy_labels.parquet')})
    for repair, note in [
        ('limits_scope','未使用5800行截断响应；忽略offset经探针确认，改固定代码范围查询。'),
        ('hold_dtype','准备阶段混合bool/float序列化修复，未改变经济输入。'),
        ('index_frequency','忽略Parquet不保存的索引频率元数据；中止已完成一组与最终18CSV一致。')]:
        event('repair:'+repair, 'pipeline_repair_not_strategy_trial', '20260913',
            dict(output=str(run), note=note), {}, {}, verdict='IMPLEMENTATION_VERIFIED')
    files = [run/name for name in ['config.yaml','p2_decision_protocol.json','limits_manifest.json',
        'limits_download.csv','label_preparation.json','policy_labels_manifest.json',
        'fit_policy_manifest.json','policy_model_metadata.json','policy_target_gate.csv',
        'policy_target_decision.json','cash_execution/manifest.json','cash_confirm/manifest.json',
        'cash_held_baskets/manifest.json','cash_held_baskets/alarm_held_states.csv',
        'cash_held_baskets/alarm_held_states_summary.csv','full_test_suite.log',
        'cash_action_paths.png','cash_capital_comparison.csv','cash_process_comparison.csv',
        'policy_label_process_summary.csv']]
    files.extend(path for path in (run/'verification').iterdir() if path.is_file())
    supplemental = [report,ROOT/'README.md',ROOT/'docs/POLICY_CASH_PROTOCOL_20260913.md',
        Path(__file__).resolve(),ROOT/'scripts/summarize_policy_cash.py',
        ROOT/'scripts/summarize_policy_labels.py',ROOT/'scripts/evaluate_policy_target_gate.py',
        ROOT/'scripts/diagnose_alarm_held_states.py',ROOT/'scripts/fetch_policy_limits.py']
    root_manifest.update(status='complete', admission='NOT_ADMITTED', model_count=8,
        cash_cases=16, adaptive_confirm_cases=5, tests_passed=239,
        policy_target_gate='failed_2_of_4; conditional_8_replays_not_triggered',
        independent_label_and_model_checks=True, independent_all_21_execution_checks=True,
        test_market_data_used=False, fresh_holdout=False,
        ledger_experiment_ids=identifiers, ledger_semantics='32 dependent research/verification/repair events',
        report=str(report.relative_to(ROOT)),
        output_hashes={str(path.relative_to(run)):sha(path) for path in files},
        supplemental_hashes={str(path.relative_to(ROOT)):sha(path) for path in supplemental})
    (run/'manifest.json').write_text(json.dumps(root_manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status='complete',model_count=8,cash_cases=21,tests=239,
        ledger_events=len(identifiers),admission='NOT_ADMITTED')))


if __name__=='__main__':
    main()
