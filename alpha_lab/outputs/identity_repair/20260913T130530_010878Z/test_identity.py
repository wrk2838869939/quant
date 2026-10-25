import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,'/root/autodl-tmp/alpha_lab/scripts')
from repair_historical_identity import repair,EVENTS
R=Path(__file__).resolve().parent
class IdentityChecks(unittest.TestCase):
 def setUp(self):
  self.b=pd.read_parquet(R/'inputs/stock_basic_all.parquet');self.n=pd.read_parquet(R/'inputs/namechange_all.parquet')
 def test_retired_alias_is_not_delisting_or_ipo(self):
  b,n,i,a=repair(self.b,self.n)
  self.assertTrue(a.delist_date.isna().all());self.assertTrue(a.list_status.eq('R').all())
  for e in EVENTS:
   self.assertEqual(a.set_index('ts_code').loc[e['old_code'],'list_date'],pd.Timestamp(e['original_list_date']).strftime('%Y%m%d'))
   self.assertTrue(pd.to_datetime(n[n.ts_code.eq(e['old_code'])].end_date).lt(pd.Timestamp(e['effective_date'])).all())
 def test_rejects_incorrect_original_listing_age(self):
  self.b.loc[self.b.ts_code.eq('001872.SZ'),'list_date']='20181226'
  with self.assertRaises(AssertionError):repair(self.b,self.n)
 def test_rejects_duplicate_alias_insertion(self):
  b,n,_,_=repair(self.b,self.n)
  with self.assertRaises(AssertionError):repair(b,n)
if __name__=='__main__':unittest.main()
