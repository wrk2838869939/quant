"""日期权重、交易日聚簇及特征时间契约的反例测试。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sector_candidate_quality import assign_clusters,date_equal,build_features


class QualityTests(unittest.TestCase):
    def test_same_day_duplication_has_no_date_weight_effect(self):
        x=pd.DataFrame({'date':pd.to_datetime(['2020-01-01','2020-01-01','2020-01-02']),'y':[.1,.3,-.1]})
        self.assertAlmostEqual(date_equal(x,'y').mean(),.05)
        repeated=pd.concat([x,x.iloc[:2],x.iloc[:2]],ignore_index=True)
        self.assertAlmostEqual(date_equal(repeated,'y').mean(),.05)

    def test_clusters_use_trading_sessions_and_transitive_links(self):
        cal=pd.bdate_range('2020-01-01',periods=100)
        x=assign_clusters(cal[[0,15,30,60]],cal,20)
        self.assertEqual(x.cluster_id.tolist(),[0,0,0,1])

    def test_prefix_and_unknown_breadth(self):
        cfg=yaml.safe_load((ROOT/'configs/sector_candidate_quality_v1.yaml').read_text())
        dates=pd.bdate_range('2020-01-01',periods=10)
        panel=pd.DataFrame(dict(date=dates,sector_code='S',segment_id=1,index_valid=True,
            up_share=np.arange(10)/10,breadth_members=10,known_members=10,turnover_members=10,turnover_pct=1.))
        signals=pd.DataFrame(dict(date=[dates[5]],sector_code='S',segment_id=1,drawdown=-.2,index_close=100.,
            peak_price=125.,pullback_days=10,recent_loss_mean=.01,previous_loss_mean=.03,
            platform_low=95.,platform_high=105.,reference_high=110.))
        full=build_features(panel,signals,cfg)
        short=build_features(panel.iloc[:6],signals,cfg)
        pd.testing.assert_frame_equal(full,short)
        panel.loc[5,'breadth_members']=0
        missing=build_features(panel,signals,cfg)
        self.assertTrue(np.isnan(missing.breadth_change.iloc[0]))
        self.assertEqual(full.price_zone.iloc[0],'平台区间内')


if __name__=='__main__':unittest.main()
