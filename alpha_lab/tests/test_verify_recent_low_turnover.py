"""扩窗独立核验的边界与执行证据拒绝测试。"""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import unittest
import tempfile

PATH = Path(__file__).resolve().parents[1] / 'scripts/verify_recent_low_turnover.py'
SPEC = importlib.util.spec_from_file_location('verify_recent_low_turnover', PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def empty_account(folder):
    """冻结两日空仓账户，便于单独验证日期和现金守恒。"""
    (folder / 'summary.json').write_text(json.dumps({'policy': 'low_turnover', 'capital': 50000}))
    pd.DataFrame({'date': ['2024-01-02', '2024-01-03'], 'cash': [50000, 50000],
                  'equity': [50000, 50000], 'stock_value': [0, 0], 'receivables': [0, 0],
                  'n_hold': [0, 0]}).to_csv(folder / 'daily.csv', index=False)
    for name, columns in {'trades': ['date', 'ts_code', 'side'],
                          'events': ['date', 'ts_code', 'kind'],
                          'holdings': ['date', 'ts_code', 'shares', 'value'],
                          'entry_phase_episodes': ['ts_code']}.items():
        pd.DataFrame(columns=columns).to_csv(folder / f'{name}.csv', index=False)
    return pd.DataFrame(columns=['date', 'ts_code'])


class RecentVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_prefix_all_null_event_columns_allowed_but_values_rejected(self):
        old = pd.DataFrame({'date': ['2023-12-29'], 'amount': [1]})
        new = old.assign(future_action=float('nan'))
        result = VERIFY.compare_prefix_frame(old, new, 'events.csv')
        self.assertEqual(result['added_all_null_columns'], ['future_action'])
        new.loc[0, 'future_action'] = 0
        with self.assertRaisesRegex(AssertionError, '新增列在旧前缀含有效值'):
            VERIFY.compare_prefix_frame(old, new, 'events.csv')

    def test_prefix_rejects_economic_drift(self):
        old = pd.DataFrame({'equity': [50000.]})
        new = pd.DataFrame({'equity': [50000.00001]})
        with self.assertRaises(AssertionError):
            VERIFY.compare_prefix_frame(old, new, 'daily.csv')

    def test_copied_state_machine_only_changes_explicit_window(self):
        evidence = VERIFY.verify_copy_provenance()
        assert evidence['sole_change']['new'].count('verification_window') == 1


    def test_registered_recent_window_passes_and_other_window_fails(self):
        tmp_path = self.folder
        predictions = empty_account(tmp_path)
        cfg = {'verification_window': (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-31'))}
        assert VERIFY.verify_case(tmp_path, predictions, cfg, {})['verified']
        cfg['verification_window'] = (pd.Timestamp('2022-01-01'), pd.Timestamp('2023-12-31'))
        with self.assertRaisesRegex(AssertionError, '账户日期超出'):
            VERIFY.verify_case(tmp_path, predictions, cfg, {})


    def test_recent_window_does_not_relax_cash_reconciliation(self):
        tmp_path = self.folder
        predictions = empty_account(tmp_path)
        daily = pd.read_csv(tmp_path / 'daily.csv')
        daily.loc[1, 'cash'] = 50001
        daily.to_csv(tmp_path / 'daily.csv', index=False)
        cfg = {'verification_window': (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-31'))}
        with self.assertRaisesRegex(AssertionError, '每日现金'):
            VERIFY.verify_case(tmp_path, predictions, cfg, {})


    def test_official_limit_blocks_real_fill_at_up_limit(self):
        tmp_path = self.folder
        empty_account(tmp_path)
        pd.DataFrame([{'date': '2024-01-03', 'signal_date': '2024-01-02', 'ts_code': '600828.SH',
                       'side': 'buy', 'shares': 100, 'price': 10}]).to_csv(tmp_path / 'trades.csv', index=False)
        raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-03']),
                            'ts_code': ['600828.SH'] * 2, 'open': [10., 10.],
                            'close': [10., 10.], 'vol_share': [100000., 100000.]})
        limits = raw[['date', 'ts_code']].assign(up_limit=11., down_limit=9.)
        spec = {'require_official_limits': True, 'strict_actions': True, 'price_tick': .01,
                'max_volume_participation': .01, 'boards': {'main': {'min_buy': 100, 'buy_step': 100}}}
        market, bound = VERIFY.prepare_execution_evidence(raw, limits)
        window = (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-31'))
        assert VERIFY.verify_market_fills(tmp_path, spec, market, bound, window)['official_limit_and_raw_open_checks'] == 1
        limits.loc[1, 'up_limit'] = 10.
        market, bound = VERIFY.prepare_execution_evidence(raw, limits)
        with self.assertRaisesRegex(AssertionError, '涨停开盘买入'):
            VERIFY.verify_market_fills(tmp_path, spec, market, bound, window)

if __name__ == "__main__":
    unittest.main()
