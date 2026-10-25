"""独立产物审计器须捕获现金、逐代码股份、预测混用及事后阶段篡改。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.append(str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_divide_selection_execution import verify_case


class ExecutionVerifierTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.pred = pd.DataFrame([dict(date=pd.Timestamp('2022-01-03'), ts_code='A',
            phase='recovery', toy__score=.9, toy__expected_return=.1)])
        self.ecfg = dict(portfolio=dict(buy_cost=.001, sell_cost=.002, min_fee=5.))
        (self.path/'summary.json').write_text(json.dumps(dict(policy='toy', capital=5000.)))
        self.frames = {
            'daily': pd.DataFrame([
                dict(date='2022-01-04', cash=3995., receivables=0., stock_value=1000., equity=4995., n_hold=1),
                dict(date='2022-01-05', cash=5090., receivables=0., stock_value=0., equity=5090., n_hold=0)]),
            'trades': pd.DataFrame([
                dict(date='2022-01-04', signal_date='2022-01-03', ts_code='A', side='buy', shares=100,
                     gross=1000., price=10., fee=5., cash_after=3995.),
                dict(date='2022-01-05', signal_date='2022-01-04', ts_code='A', side='sell', shares=100,
                     gross=1100., price=11., fee=5., cash_after=5090.)]),
            'events': pd.DataFrame([
                dict(date='2022-01-04', signal_date='2022-01-03', ts_code='A', kind='selection_decision',
                     reason='selected', priority=.9, expected_return=.1, reserved_shares=100, reserved_net_profit=90.),
                dict(date='2022-01-05', signal_date='2022-01-04', ts_code='A', kind='hold_exit_requested')]),
            'holdings': pd.DataFrame([dict(date='2022-01-04', ts_code='A', shares=100, value=1000., price=10.)]),
            'entry_phase_episodes': pd.DataFrame([dict(ts_code='A', entry_date='2022-01-04',
                signal_date='2022-01-03', close_date='2022-01-05', entry_phase='recovery', pnl_yuan=90.)])}

    def verify(self):
        for name, frame in self.frames.items():
            frame.to_csv(self.path/f'{name}.csv', index=False)
        return verify_case(self.path, self.pred, {}, self.ecfg)

    def test_clean_path_reconciles(self):
        self.assertTrue(self.verify()['verified'])

    def test_equal_total_shares_but_wrong_stock_is_rejected(self):
        self.frames['holdings'].loc[0, 'ts_code'] = 'B'
        with self.assertRaisesRegex(AssertionError, '逐代码'):
            self.verify()

    def test_trade_cash_mismatch_is_rejected(self):
        self.frames['trades'].loc[0, 'cash_after'] += 1
        with self.assertRaisesRegex(AssertionError, '逐笔现金'):
            self.verify()

    def test_rank_cannot_replace_mu(self):
        self.frames['events'].loc[0, 'expected_return'] = .9
        with self.assertRaisesRegex(AssertionError, '真实μ'):
            self.verify()

    def test_episode_phase_cannot_be_relabelled_after_entry(self):
        self.frames['entry_phase_episodes'].loc[0, 'entry_phase'] = 'accelerating'
        with self.assertRaisesRegex(AssertionError, '入场时已知阶段'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
