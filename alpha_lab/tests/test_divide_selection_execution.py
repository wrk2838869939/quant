"""冻结分数/收益分离、真实阶段持仓段和倒序wrapper恢复的独立测试。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from portfolio import BacktestResult
from run_divide_selection_execution import execution_cases,make_matrices,reversed_candidate_input,entry_phase_attribution


class DivideSelectionExecutionTest(unittest.TestCase):
    def test_twenty_cases_and_five_real_reverse_reruns(self):
        cfg=dict(capitals=[20000.,50000.,100000.],reference_capital=50000.,policies=list('abcde'))
        cases=execution_cases(cfg)
        self.assertEqual(len(cases),20)
        self.assertEqual(sum(c['reverse_input'] for c in cases),5)
        self.assertTrue(all(c['capital']==50000 for c in cases if c['reverse_input']))

    def test_score_and_true_return_are_preserved_separately(self):
        date=pd.Timestamp('2022-01-04')
        pred=pd.DataFrame(dict(date=[date,date],ts_code=['A','B'],policy__score=[10.,1.],policy__expected_return=[.03,.8]))
        hold=pd.DataFrame(True,index=[date],columns=['A','B','old'])
        signals,mu=make_matrices(pred,hold,dict(execution=['2022-01-01','2023-12-31'],policies=['policy']))['policy']
        self.assertEqual(signals.loc[date].idxmax(),'A')
        self.assertEqual(mu.loc[date].idxmax(),'B')
        self.assertTrue(pd.isna(mu.loc[date,'old']))
        pd.testing.assert_index_equal(signals.columns,hold.columns)

    def test_reversed_wrapper_preserves_priorities_and_restores_after_error(self):
        import selection_first
        calls=[];priorities={'A':2.,'B':1.}
        def fake(frame,**kwargs):
            calls.append((frame.ts_code.tolist(),kwargs['priorities']))
            raise RuntimeError('synthetic')
        with patch.object(selection_first,'plan_selection_first',fake):
            with self.assertRaises(RuntimeError):
                with reversed_candidate_input(True):
                    selection_first.plan_selection_first(pd.DataFrame({'ts_code':['A','B']}),priorities=priorities)
            self.assertIs(selection_first.plan_selection_first,fake)
        self.assertEqual(calls[0][0],['B','A'])
        self.assertIs(calls[0][1],priorities)

    def test_phase_is_fixed_per_real_entry_not_later_market_stage(self):
        dates=pd.bdate_range('2022-01-04',periods=4)
        daily=pd.DataFrame(dict(cash=[1000.,2100.,1000.,1000.],stock_value=[1000.,0.,1100.,1200.],
            receivables=0.,equity=[2000.,2100.,2100.,2200.],n_hold=[1,0,1,1]),index=dates)
        trades=pd.DataFrame([dict(date=dates[0],signal_date=dates[0],ts_code='A',side='buy',shares=100,gross=1000.,price=10.,fee=0.),
            dict(date=dates[1],signal_date=dates[1],ts_code='A',side='sell',shares=100,gross=1100.,price=11.,fee=0.),
            dict(date=dates[2],signal_date=dates[2],ts_code='A',side='buy',shares=100,gross=1100.,price=11.,fee=0.)])
        holdings=pd.DataFrame([dict(date=dates[i],ts_code='A',shares=100,value=value) for i,value in [(0,1000.),(2,1100.),(3,1200.)]])
        events=pd.DataFrame([dict(date=dates[i],signal_date=dates[i],ts_code='A',kind='selection_decision',reason='selected',
            priority=1.,expected_return=.1,reserved_shares=100,reserved_net_profit=100.) for i in [0,2]])
        predictions=pd.DataFrame(dict(date=dates,ts_code='A',phase=['recovery','decelerating','accelerating','decelerating']))
        signals=pd.DataFrame({'A':[1.,1.,1.,1.]},index=dates)
        result=BacktestResult(daily,trades,events,holdings,{'capital':2000.})
        episodes,phases,buys,error=entry_phase_attribution(result,predictions,signals)
        self.assertEqual(error,0.)
        self.assertEqual(list(episodes.entry_phase),['recovery','accelerating'])
        self.assertEqual(list(episodes.pnl_yuan),[100.,100.])
        self.assertEqual(list(buys.priority_rank),[1,1])
        self.assertEqual(phases.pnl_yuan.sum(),200.)


if __name__=='__main__':unittest.main()
