"""首次复核插入不能改写未来月度规则、错用未来信号或丢失现金等待。"""
import unittest
import pandas as pd

from startup_timing import startup_matrices


def fixture():
    columns=['600000.SH','601988.SH']
    def frames(dates,values):
        x=pd.DataFrame(values,index=pd.DatetimeIndex(dates,name='date'),columns=columns)
        return dict(signals=x,forecasts=x*0+.03,hold=x*0+1)
    base=frames(['2024-01-02','2024-02-01','2024-03-01'],[[.2,.3],[.4,.2],[.3,.1]])
    shift=frames(['2024-01-15','2024-02-15','2024-03-15'],[[.5,.1],[.1,.5],[.2,.4]])
    risk=pd.DataFrame(.8,index=base['signals'].index.union(shift['signals'].index),columns=columns)
    base['hold'].loc['2024-02-01','600000.SH']=0.
    return base,shift,risk


class StartupTests(unittest.TestCase):
    def test_wait_excludes_pre_start_month_signal(self):
        b,s,r=fixture();m,_,line=startup_matrices(b,s,r,'2024-01-15','2024-01-08','2024-03-29','wait')
        self.assertEqual(m['signals'].index[0],pd.Timestamp('2024-02-01'))

    def test_fresh_insertion_preserves_all_later_hold_values(self):
        b,s,r=fixture();m,risk,line=startup_matrices(b,s,r,'2024-01-15','2024-01-08','2024-03-29','fresh_once')
        pd.testing.assert_series_equal(m['signals'].loc['2024-01-15'],s['signals'].loc['2024-01-15'])
        self.assertEqual(m['hold'].loc['2024-02-01','600000.SH'],0.)
        self.assertFalse(line['future_source_used'])

    def test_stale_retains_old_values_with_explicit_lineage(self):
        b,s,r=fixture();m,_,line=startup_matrices(b,s,r,'2024-01-15','2024-01-08','2024-03-29','stale_once')
        self.assertEqual(line['source_date'],'2024-01-02')
        self.assertEqual(m['signals'].loc['2024-01-15','600000.SH'],.2)

    def test_future_rows_cannot_change_first_order_inputs(self):
        b,s,r=fixture();first,_,_=startup_matrices(b,s,r,'2024-01-15','2024-01-08','2024-03-29','fresh_once')
        s['signals'].loc['2024-02-15':]=999
        second,_,_=startup_matrices(b,s,r,'2024-01-15','2024-01-08','2024-03-29','fresh_once')
        pd.testing.assert_frame_equal(first['signals'],second['signals'])

    def test_shift_moves_future_calendar_only_after_same_initial_signal(self):
        b,s,r=fixture()
        one,_,_=startup_matrices(b,s,r,'2024-01-15','2024-01-08','2024-03-29','fresh_once')
        all_,_,_=startup_matrices(b,s,r,'2024-01-15','2024-01-08','2024-03-29','shifted_calendar')
        pd.testing.assert_series_equal(one['signals'].iloc[0],all_['signals'].iloc[0])
        self.assertNotEqual(one['signals'].index[1],all_['signals'].index[1])


if __name__=='__main__':unittest.main()
