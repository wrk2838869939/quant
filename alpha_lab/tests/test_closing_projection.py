"""真实时点契约与未来数据隔离；全部为明确标记的合成报价。"""
from copy import deepcopy
import unittest

import pandas as pd

from closing_projection import project, proxy_receipt, parse_sina
from shadow_inputs import build_bundle
from shadow_planner import ShadowError
from test_monthly_risk_inputs import MonthlyRiskInputsTests


class ClosingProjectionTests(unittest.TestCase):
    def setUp(self):
        fixture = MonthlyRiskInputsTests(); fixture.setUp(); self.f = fixture
        self.date = fixture.base.date
        day = pd.Timestamp(self.date)
        self.history = fixture.raw.loc[fixture.raw.date.lt(day), ['date','ts_code','hfq_close','total_mv_yuan','turnover_rate']].copy()
        prior = fixture.raw[fixture.raw.date.eq('2024-03-29')]
        self.basis = prior[['date','ts_code','close','hfq_close']].copy()
        self.basis['total_share']=1e8;self.basis['float_share']=5e7
        self.basis['total_mv_yuan']=1e9
        self.index = fixture.index[fixture.index.index<day]
        self.quotes = pd.DataFrame([dict(ts_code=r.ts_code,name='synthetic',pre_close=r.close,
            close=r.close*1.001,vol_share=1e6,timestamp=self.date+'T14:30:30+08:00') for r in prior.itertuples()]+[
            dict(ts_code='000300.SH',name='synthetic index',pre_close=float(self.index.iloc[-1]),
                close=float(self.index.iloc[-1])*1.001,vol_share=1e8,timestamp=self.date+'T14:30:30+08:00')])
        self.receipt = proxy_receipt(self.quotes,self.date,self.date+'T14:31:00+08:00','synthetic only')
        self.cfg=deepcopy(fixture.cfg);self.cfg['execution_session']='closing_auction'

    def project(self):
        return project(self.history,self.index,self.basis,self.quotes,self.receipt,self.f.base.cal)

    def test_build_uses_real_time_price_and_known_share_counts(self):
        raw,index,market = self.project()
        today=raw[raw.date.eq(self.date)]
        self.assertTrue(today.turnover_rate.eq(2.).all())
        self.assertTrue((today.total_mv_yuan-1.001e9).abs().lt(1e-5).all())
        self.assertEqual(len(today),25)
        b=self.f.base
        bundle=build_bundle(raw,market,self.f.basics,b.st,b.actions,self.date,b.cal,self.cfg,b.model,[],self.receipt,index)
        self.assertFalse(bundle['eod_complete'])
        self.assertEqual(bundle['input_kind'],'intraday_proxy')
        self.assertEqual(bundle['market_risk_history'][-1]['date'],self.date)

    def test_final_eod_and_future_rows_are_rejected_before_projection(self):
        for date in [self.date,'2024-04-02']:
            old=self.history
            extra=old.tail(1).copy();extra['date']=pd.Timestamp(date)
            self.history=pd.concat([old,extra])
            with self.assertRaises(ShadowError):self.project()
            self.history=old
        self.basis['date']=pd.Timestamp(self.date)
        with self.assertRaises(ShadowError):self.project()

    def test_missing_stale_mutated_and_adjustment_inputs_block(self):
        old=self.quotes.copy()
        self.quotes=self.quotes.iloc[1:]
        with self.assertRaises(ShadowError):self.project()
        self.quotes=old.copy();self.quotes.loc[0,'close']*=2
        with self.assertRaises(ShadowError):self.project()
        self.quotes=old.copy();self.quotes.loc[0,'timestamp']=self.date+'T14:28:00+08:00'
        with self.assertRaises(ShadowError):proxy_receipt(self.quotes,self.date,self.date+'T14:31:00+08:00','fixture')
        self.quotes=old;self.basis.iloc[0,self.basis.columns.get_loc('close')]*=2
        with self.assertRaises(ShadowError):self.project()

    def test_freeze_proxy_history_not_replaced_by_later_eod(self):
        raw,index,market=self.project()
        # 次日追加仍延续前一日盘中代理，既有月度概率不能改写。
        from monthly_risk_live import risk_history
        before=risk_history(index,self.date,self.f.base.cal,self.cfg)
        tomorrow=pd.Timestamp('2024-04-02')
        idx=pd.concat([index,pd.Series([float(index.iloc[-1])*1.1],index=[tomorrow])])
        self.assertEqual(before,risk_history(idx,str(tomorrow.date()),self.f.base.cal,self.cfg))

    def test_parser_retains_time_and_share_unit(self):
        fields=['fixture','10','10','10.1','11','9','0','0','123456','999999']+['0']*20+['2024-04-01','14:30:01','00']
        data=parse_sina('var hq_str_sh600000="'+','.join(fields)+'";')
        self.assertEqual(data.iloc[0].vol_share,123456)
        self.assertEqual(data.iloc[0].timestamp,'2024-04-01T14:30:01+08:00')

    def test_broker_quantities_preserved_and_valuation_uses_frozen_quote(self):
        from closing_projection import mark_account
        broker=dict(account_id='fixture',asof=self.date,timestamp=self.date+'T14:32:00+08:00',
            source='synthetic',evidence='fixture',cash=1000.,receivables=0.,equity=9999.,
            frozen_cash=0.,open_orders=[],positions={'600000.SH':dict(shares=100,sellable_today=0)})
        marked=mark_account(broker,self.quotes,self.receipt,self.date+'T14:33:00+08:00')
        self.assertEqual(marked['positions']['600000.SH']['sellable_today'],0)
        self.assertEqual(marked['broker_reported_equity'],9999.)
        self.assertEqual(marked['equity'],1000.+100*self.quotes.set_index('ts_code').at['600000.SH','close'])
        broker['timestamp']=self.date+'T14:29:00+08:00'
        with self.assertRaises(ShadowError):mark_account(broker,self.quotes,self.receipt,self.date+'T14:33:00+08:00')


if __name__ == '__main__': unittest.main()
