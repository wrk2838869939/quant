"""新候选从全截面重建，验证退出屏蔽、缺数与未来数据不变性。"""
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import yaml
from shadow_inputs import build_bundle
from shadow_planner import ShadowError
from test_shadow_inputs import ShadowInputsTests
ROOT=Path(__file__).resolve().parents[1]


class MonthlyRiskInputsTests(unittest.TestCase):
    def setUp(self):
        base=ShadowInputsTests();base.setUp();self.base=base
        self.cfg=yaml.safe_load((ROOT/'configs/monthly_risk_manual.yaml').read_text());self.cfg['schedule']['anchor']='2023-01-01'
        dates=pd.DatetimeIndex(base.cal['sessions']);self.index=pd.Series(3000*np.exp(np.arange(len(dates))*.001+np.sin(np.arange(len(dates)))*.002),index=dates)
        self.receipt=deepcopy(base.receipt);self.receipt['market_index']=dict(code='000300.SH',asof=base.date,complete=True,source='synthetic',sha256='synthetic',downloaded_at=base.date+'T18:00:00+08:00')
        template=base.raw[base.raw.ts_code.eq('600000.SH')];rows=[]
        self.codes=[f'{600000+j}.SH' for j in range(25)]
        for j,c in enumerate(self.codes):
            x=template.copy();x['ts_code']=c;x['turnover_rate']=1.+j;rows.append(x)
        self.raw=pd.concat(rows,ignore_index=True)
        self.basics=pd.DataFrame([dict(ts_code=c,list_date=pd.Timestamp('2000-01-01'),delist_date=pd.NaT) for c in self.codes]);self.receipt['expected_rows']=25

    def build(self,raw=None,index=None):
        b=self.base;x=self.raw if raw is None else raw
        return build_bundle(x,x,self.basics,b.st,b.actions,b.date,b.cal,self.cfg,b.model,self.codes,self.receipt,self.index if index is None else index)

    def test_rank_exit_is_also_entry_mask_without_pool_expansion(self):
        result=self.build()
        self.assertEqual(set(result['signals']),set(self.codes[:10]))
        self.assertFalse(result['holds'][self.codes[10]])
        self.assertTrue(result['holds'][self.codes[0]])
        self.assertTrue(result['stock_recovery'][self.codes[0]]['known'])

    def test_future_market_and_index_do_not_change_bundle(self):
        old=self.build();raw=self.raw.copy();idx=self.index.copy()
        raw.loc[raw.date.gt(self.base.date),['hfq_close','close','turnover_rate']]*=100
        idx.loc[idx.index>self.base.date]*=100
        self.assertEqual(old,self.build(raw,idx))

    def test_missing_day_or_index_source_blocks(self):
        with self.assertRaises(ShadowError):self.build(self.raw[~self.raw.date.eq('2024-02-06')])
        with self.assertRaises(ShadowError):self.build(index=self.index.drop(pd.Timestamp('2024-02-06')))
        self.receipt.pop('market_index')
        with self.assertRaises(ShadowError):self.build()

    def test_missing_recent_security_cannot_silently_shrink_universe(self):
        self.basics=self.basics.iloc[1:]
        with self.assertRaises(ShadowError):self.build()


if __name__=='__main__':unittest.main()
