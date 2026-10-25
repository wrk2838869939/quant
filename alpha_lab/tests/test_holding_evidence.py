"""C1证据记忆的独立状态机和原执行引擎边界验证。"""
import ast
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from holding_evidence import evidence_hold_states
import test_patient_execution as execution_fixture


def original_states():
    """仅编译原函数，不导入原驱动的行情接口依赖。"""
    tree = ast.parse((ROOT/'scripts/run_patient_value_execution.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'confirmed_hold_states')
    scope = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<original>', 'exec'), scope)
    return scope['confirmed_hold_states']


class HoldingEvidenceTest(unittest.TestCase):
    def inputs(self, sequence):
        dates = pd.date_range('2022-01-01', periods=len(sequence), freq='MS')
        return (pd.DataFrame({'a': [s == 1 for s in sequence]}, index=dates),
                pd.DataFrame({'a': [s is not None for s in sequence]}, index=dates))

    def test_failure_unknown_failure_and_recovery(self):
        trend, known = self.inputs([0, None, 0, 1, 0, None, None])
        state, before, after = evidence_hold_states(trend, known, 2, True)
        np.testing.assert_allclose(state.a, [1, np.nan, 0, 1, 1, np.nan, np.nan], equal_nan=True)
        np.testing.assert_array_equal(after.a, [1, 1, 2, 0, 1, 1, 1])
        np.testing.assert_array_equal(before.a, [0, 1, 1, 2, 0, 1, 1])
        np.testing.assert_allclose(original_states()(trend, known, 2).a,
                                   [1, np.nan, 1, 1, 1, np.nan, np.nan], equal_nan=True)

    def test_all_short_sequences_against_independent_scalar_and_prefixes(self):
        original = original_states()
        for sequence in itertools.product([0, 1, None], repeat=5):
            trend, known = self.inputs(sequence)
            result = evidence_hold_states(trend, known, 2)
            count, expected = 0, []
            for value in sequence:
                if value is None:
                    expected.append(np.nan)
                    continue
                count = 0 if value else count + 1
                expected.append(float(count < 2))
            np.testing.assert_allclose(result.a, expected, equal_nan=True)
            for stop in range(1, 6):
                pd.testing.assert_frame_equal(result.iloc[:stop],
                    evidence_hold_states(trend.iloc[:stop], known.iloc[:stop], 2))
            if None not in sequence:
                pd.testing.assert_frame_equal(result, original(trend, known, 2))

    def test_good_unknown_then_bad_and_terminal_unknown(self):
        trend, known = self.inputs([1, None, 0, None, None])
        state, _, counts = evidence_hold_states(trend, known, 2, True)
        np.testing.assert_allclose(state.a, [1, np.nan, 1, np.nan, np.nan], equal_nan=True)
        np.testing.assert_array_equal(counts.a, [0, 0, 1, 1, 1])

    def test_unknown_cannot_cancel_submitted_exit_in_original_engine(self):
        fixture = execution_fixture.PatientExecutionTest()
        fixture.setUp()
        fixture.signals = fixture.signals.reindex(fixture.dates[[0, 1, 2, 3]])
        fixture.signals.loc[:] = [2., 1.]
        fixture.forecasts = fixture.forecasts.reindex(fixture.signals.index)
        fixture.forecasts.loc[:] = [.1, -.1]
        trend = pd.DataFrame([[True, True], [False, True], [False, True], [False, True]],
                             index=fixture.signals.index, columns=fixture.codes)
        known = pd.DataFrame(True, index=trend.index, columns=trend.columns)
        known.loc[fixture.dates[3], fixture.codes[0]] = False
        fixture.hold = evidence_hold_states(trend, known, 2)
        mask = fixture.limits.date.eq(fixture.dates[3]) & fixture.limits.ts_code.eq(fixture.codes[0])
        fixture.limits.loc[mask, 'down_limit'] = 10.
        result = fixture.run_case()
        sold = result.trades[result.trades.side.eq('sell')]
        self.assertEqual(list(sold.date), [fixture.dates[4]])
        requests = result.events[result.events.kind.eq('hold_exit_requested')]
        self.assertEqual(requests.iloc[-1].reason, 'confirmed_pending')


if __name__ == '__main__':
    unittest.main()
