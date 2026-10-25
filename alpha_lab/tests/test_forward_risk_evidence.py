"""未来目标的时点、样本清洗及特征截断验证。"""
import unittest
import numpy as np
import pandas as pd
from forward_risk_evidence import market_features, forward_outcomes, matured_weekly_rows, FAMILIES


class ForwardRiskEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.bdate_range('2017-01-02',periods=200)
        x=np.arange(200)
        self.close=pd.DataFrame({str(k):100+np.sin(x/(k+2))+x*.02 for k in range(4)},index=self.dates)
        self.amount=pd.DataFrame(1e8,index=self.dates,columns=self.close.columns)
        self.index=pd.Series(100+np.sin(x/5)+x*.05,index=self.dates)

    def test_features_future_mutation_and_prefix_are_exact(self):
        cutoff=self.dates[169]
        base=market_features(self.close,self.amount,self.index)
        close=self.close.copy();amount=self.amount.copy();index=self.index.copy()
        close.loc[close.index>cutoff]*=3;amount.loc[amount.index>cutoff]*=20;index.loc[index.index>cutoff]*=.6
        changed=market_features(close,amount,index)
        truncated=market_features(self.close.loc[:cutoff],self.amount.loc[:cutoff],self.index.loc[:cutoff])
        pd.testing.assert_frame_equal(base.loc[:cutoff],changed.loc[:cutoff],check_exact=True)
        pd.testing.assert_frame_equal(base.loc[:cutoff],truncated,check_exact=True)

    def test_future_only_stock_cannot_change_prior_aggregates(self):
        cutoff=self.dates[169]
        base=market_features(self.close,self.amount,self.index)
        close=self.close.copy();amount=self.amount.copy()
        close['future']=np.nan;amount['future']=np.nan
        close.loc[close.index>cutoff,'future']=100
        amount.loc[amount.index>cutoff,'future']=1e12
        changed=market_features(close,amount,self.index)
        pd.testing.assert_frame_equal(base.loc[:cutoff],changed.loc[:cutoff],check_exact=True)

    def test_label_starts_next_open_and_records_maturity(self):
        dates=self.dates[:8]
        op=pd.Series([100,10,9,9,11,12,11,10],index=dates,dtype=float)
        cl=pd.Series([1,9,8,10,11,12,11,10],index=dates,dtype=float)
        labels=forward_outcomes(op,cl,3)
        self.assertAlmostEqual(labels.at[dates[0],'worst'],-.2)
        self.assertAlmostEqual(labels.at[dates[0],'terminal'],.1)
        self.assertEqual(labels.at[dates[0],'label_end'],dates[4])
        self.assertTrue(pd.isna(labels.iloc[-1].loss_event))

    def test_missing_forward_price_is_unknown_not_safe(self):
        op=self.index.copy();cl=self.index.copy();cl.iloc[3]=np.nan
        labels=forward_outcomes(op,cl,5)
        self.assertTrue(pd.isna(labels.iloc[0].loss_event))

    def test_training_rows_never_cross_declared_end(self):
        features=pd.DataFrame(1.,index=self.dates,columns=FAMILIES['joint'])
        features['price_coverage']=2000
        targets=forward_outcomes(self.index,self.index)
        cutoff=self.dates[100]
        rows=matured_weekly_rows(features,targets,self.dates[0],cutoff)
        self.assertTrue(targets.loc[rows,'label_end'].le(cutoff).all())
        self.assertTrue((rows.dayofweek==0).all())
        self.assertLess(rows.max(),cutoff)


if __name__=='__main__':unittest.main()
