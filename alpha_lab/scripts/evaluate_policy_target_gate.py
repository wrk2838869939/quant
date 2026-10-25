#!/usr/bin/env python3
"""执行模型拟合前登记的政策目标停止条件，全部窗口/对照共同决定。"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    protocol = run / 'p2_decision_protocol.json'
    assert json.loads(protocol.read_text())['before_policy_labels_and_models']
    records = []
    for stage in ['development', 'execution']:
        table = pd.read_csv(run / f'{stage}_diagnostics/top_k_dates.csv', parse_dates=['date'])
        flags = pd.read_parquet(run / f'{stage}_label_coverage.parquet')
        matured = flags.groupby('date').mature_120.any()
        complete_dates = matured.index[matured]
        part = table[table.k.eq(5) & table.target_kind.eq('common_patient120') &
            table.policy.isin(['fixed60','fixed120','patient120'])]
        coverage = part.pivot(index='date', columns='policy', values='complete_cash_slots')
        common_dates = coverage.index[coverage.fillna(False).all(axis=1) & coverage.index.isin(complete_dates)]
        observed = part[part.date.isin(common_dates)].groupby('policy')[['cash_slot_return','cash_slot_optimism']].mean()
        ratio = len(common_dates) / len(complete_dates) if len(complete_dates) else 0.
        for baseline in ['fixed60', 'fixed120']:
            if len(observed) == 3:
                candidate, control = observed.loc['patient120'], observed.loc[baseline]
                delta = candidate.cash_slot_return-control.cash_slot_return
                candidate_error, control_error = abs(candidate.cash_slot_optimism), abs(control.cash_slot_optimism)
            else:
                delta = candidate_error = control_error = float('nan')
            records.append(dict(stage=stage, baseline=baseline, paired_dates=len(common_dates),
                mature_dates=len(complete_dates), coverage=ratio,
                patient_minus_baseline_cash_slot_return=delta,
                patient_absolute_mean_optimism=candidate_error, baseline_absolute_mean_optimism=control_error,
                passed=bool(ratio >= .8 and delta > 0 and candidate_error < control_error)))
    result = pd.DataFrame(records)
    result.to_csv(run / 'policy_target_gate.csv', index=False)
    (run / 'policy_target_decision.json').write_text(json.dumps(dict(
        status='evaluated', protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        checks=len(result), passed=bool(result.passed.all()), strategy_admission='NOT_ADMITTED',
        action='run_registered_8_research_replays' if result.passed.all() else 'stop_no_incremental_support',
        extra_replay_not_run_reason=None if result.passed.all() else 'registered_target_improvement_gate_failed',
        rows=result.to_dict(orient='records')), indent=2))
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
