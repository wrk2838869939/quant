"""历史别名限定边界、冲突拒绝与read adapter严格作用域测试。"""
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.append(str(ROOT/'scripts'))
from run_policy_alias_refit import rename_historical_limit,scoped_limit_alias_reader


class PolicyAliasTest(unittest.TestCase):
    def frame(self,date='20191213'):
        return pd.DataFrame(dict(trade_date=[date,date],ts_code=['000043.SZ','600000.SH'],
            up_limit=[21.85,12.],down_limit=[17.87,10.]),index=[8,3])

    def test_before_boundary_only_alias_changes_all_values_preserved(self):
        original=self.frame();snapshot=original.copy(deep=True)
        fixed,changed=rename_historical_limit(original,'2019-12-13')
        self.assertTrue(changed)
        self.assertEqual(fixed.loc[8,'ts_code'],'001914.SZ')
        pd.testing.assert_frame_equal(original,snapshot)
        pd.testing.assert_frame_equal(fixed.drop(columns='ts_code'),snapshot.drop(columns='ts_code'))

    def test_effective_day_and_later_are_untouched(self):
        for date in ['20191216','20191217']:
            frame=self.frame(date);fixed,changed=rename_historical_limit(frame,pd.to_datetime(date))
            self.assertFalse(changed);self.assertIs(fixed,frame)

    def test_conflicts_duplicates_and_wrong_date_fail_closed(self):
        for frame in [pd.concat([self.frame(),self.frame().iloc[:1]]),
                      self.frame().assign(ts_code=['000043.SZ','001914.SZ'])]:
            with self.assertRaises(ValueError):rename_historical_limit(frame,'2019-12-13')
        with self.assertRaises(ValueError):rename_historical_limit(self.frame(),'2019-12-12')

    def test_no_old_code_does_not_invent_a_record(self):
        frame=self.frame().iloc[1:];fixed,changed=rename_historical_limit(frame,'2019-12-13')
        self.assertFalse(changed);pd.testing.assert_frame_equal(fixed,frame)

    def test_only_registered_caller_and_directory_are_adapted_and_exception_restores(self):
        def build_reader(path):return pd.read_parquet(path)
        def other_reader(path):return pd.read_parquet(path)
        original=pd.read_parquet
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'limits';folder.mkdir();path=folder/'20191213.parquet'
            elsewhere=Path(temp)/'20191213.parquet';frame=self.frame()
            frame.to_parquet(path);frame.to_parquet(elsewhere)
            with self.assertRaisesRegex(RuntimeError,'forced'):
                with scoped_limit_alias_reader(build_reader,folder) as audit:
                    self.assertEqual(build_reader(path).iloc[0].ts_code,'001914.SZ')
                    self.assertEqual(other_reader(path).iloc[0].ts_code,'000043.SZ')
                    self.assertEqual(build_reader(elsewhere).iloc[0].ts_code,'000043.SZ')
                    self.assertEqual(len(audit),1)
                    raise RuntimeError('forced')
            self.assertIs(pd.read_parquet,original)
            pd.testing.assert_frame_equal(pd.read_parquet(path),frame)


if __name__=='__main__':unittest.main()
