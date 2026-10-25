"""仅替换预登记持有矩阵，复用原账户引擎和月度选股。"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import portfolio
import selection_first
import run_recent_low_turnover as driver
from run_patient_value_execution import digest


def effective_hold(original, subset):
    """保持原全A接口，仅替换原Top20并集可能使用的列。"""
    if not original.index.equals(subset.index) or not subset.columns.isin(original.columns).all():
        raise ValueError('持有变体不属于原信号矩阵')
    result = original.copy()
    result.loc[:, subset.columns] = subset
    return result


@contextmanager
def actual_reverse(enabled, records):
    """在真实接收端保存反序证据；原driver的反序包装在调用引擎前停用。"""
    original = selection_first.plan_selection_first
    if enabled:
        def wrapped(candidates, **kwargs):
            actual = candidates.iloc[::-1].copy()
            records.append(dict(before=candidates.ts_code.tolist(), actual=actual.ts_code.tolist(),
                                function_module='selection_first'))
            return original(actual, **kwargs)
        selection_first.plan_selection_first = wrapped
    try:
        yield
    finally:
        selection_first.plan_selection_first = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--case-name', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, expected in manifest['source_hashes'].items():
        assert digest(ROOT / name) == expected, '源码漂移: ' + name
    for name, expected in manifest['input_hashes'].items():
        assert digest(run / name) == expected, '输入漂移: ' + name
    case = next(c for c in manifest['cases'] if c['name'] == args.case_name)
    hold = pd.read_parquet(run / 'hold_variants' / (case['variant'] + '.parquet'))
    def execute(*args, **kwargs):
        kwargs['hold_allowed'] = effective_hold(kwargs['hold_allowed'], hold)
        return portfolio.run_backtest(*args, **kwargs)
    driver.run_backtest = execute
    records = []
    # 防止父driver再次反转；反序只在实际selector调用一次。
    from contextlib import nullcontext
    driver.reversed_candidate_input = lambda enabled: nullcontext()
    with actual_reverse(case['reverse_input'], records):
        driver.run_case(run, manifest, case)
    if case['reverse_input']:
        (run / 'execution' / case['name'] / 'reverse_input_evidence.json').write_text(
            json.dumps(dict(records=records, input_reversal_performed=True), indent=2))


if __name__ == '__main__':
    main()
