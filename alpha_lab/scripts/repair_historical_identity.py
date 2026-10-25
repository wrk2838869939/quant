"""生成独立历史代码身份覆盖；换码不是退市/IPO，不改价格或原始元数据。"""
import argparse,hashlib,json,urllib.request
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'port_2018_listing.pdf':'https://static.cninfo.com.cn/finalpage/2018-12-24/1205685040.PDF',
 'property_2019_change.pdf':'https://static.cninfo.com.cn/finalpage/2019-12-06/1207141416.PDF',
 'property_2023_report.pdf':'https://static.cninfo.com.cn/finalpage/2023-08-25/1217629427.PDF',
 'avic_2025_change.pdf':'https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-02-07/e2d8b9d3-879b-4587-9e1e-235ebf4c8a44.PDF',
 'avic_2010_listing.html':'https://www.szse.cn/aboutus/trends/news/t20100827_518017.html'}
EVENTS=[dict(old_code='000022.SZ',new_code='001872.SZ',effective_date='2018-12-26',announcement_date='2018-12-24',original_list_date='1993-05-05',market='主板',last_old_name='深赤湾A',listing_source='port_2018_listing.pdf',change_source='port_2018_listing.pdf'),dict(old_code='000043.SZ',new_code='001914.SZ',effective_date='2019-12-16',announcement_date='2019-12-06',original_list_date='1994-09-28',market='主板',last_old_name='中航善达',listing_source='property_2023_report.pdf',change_source='property_2019_change.pdf'),dict(old_code='300114.SZ',new_code='302132.SZ',effective_date='2025-02-17',announcement_date='2025-02-07',original_list_date='2010-08-27',market='创业板',last_old_name='中航电测',listing_source='avic_2010_listing.html',change_source='avic_2025_change.pdf')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def repair(basics,names):
 """原行不变；只新增已证明的历史代码。代码停用单列，绝不写成delist_date。"""
 additions=[];name_add=[];intervals=[]
 for e in EVENTS:
  assert not basics.ts_code.eq(e['old_code']).any(),'拒绝重复覆盖已有身份'
  current=basics[basics.ts_code.eq(e['new_code'])]
  assert len(current)==1
  assert pd.Timestamp(current.list_date.iloc[0])==pd.Timestamp(e['original_list_date'])
  assert current.market.iloc[0]==e['market']
  assert pd.Timestamp(e['announcement_date'])<=pd.Timestamp(e['effective_date'])
  row={c:None for c in basics};row.update(ts_code=e['old_code'],symbol=e['old_code'][:6],name=e['last_old_name'],market=e['market'],exchange='SZSE',list_status='R',list_date=pd.Timestamp(e['original_list_date']).strftime('%Y%m%d'),delist_date=None)
  additions.append(row)
  intervals.append(dict(security_id=e['new_code'],ts_code=e['old_code'],code_valid_from=e['original_list_date'],code_valid_to=(pd.Timestamp(e['effective_date'])-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),original_list_date=e['original_list_date'],change_is_delisting=False))
  intervals.append(dict(security_id=e['new_code'],ts_code=e['new_code'],code_valid_from=e['effective_date'],code_valid_to=None,original_list_date=e['original_list_date'],change_is_delisting=False))
  hist=names[names.ts_code.eq(e['new_code'])].copy()
  starts=pd.to_datetime(hist.start_date);hist=hist[starts.lt(pd.Timestamp(e['effective_date']))].copy()
  assert len(hist)>0 and pd.to_datetime(hist.start_date).min()==pd.Timestamp(e['original_list_date'])
  hist['ts_code']=e['old_code'];last=pd.Timestamp(e['effective_date'])-pd.Timedelta(days=1)
  hist['end_date']=pd.to_datetime(hist.end_date).fillna(last).clip(upper=last).dt.strftime('%Y%m%d')
  assert (pd.to_datetime(hist.ann_date)<=pd.to_datetime(hist.start_date)).all()
  name_add.append(hist)
 result=pd.concat([basics,pd.DataFrame(additions)],ignore_index=True)
 resultnames=pd.concat([names,*name_add],ignore_index=True)
 pd.testing.assert_frame_equal(result.iloc[:len(basics)].reset_index(drop=True),basics.reset_index(drop=True))
 pd.testing.assert_frame_equal(resultnames.iloc[:len(names)].reset_index(drop=True),names.reset_index(drop=True))
 assert not result.ts_code.duplicated().any()
 return result,resultnames,pd.DataFrame(intervals),pd.DataFrame(additions)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--fetch-public-sources',action='store_true');a=ap.parse_args();out=a.run;out.mkdir(parents=True,exist_ok=True)
 if (out/'repair_manifest.json').exists():raise ValueError('已封存运行不覆盖')
 (out/'inputs').mkdir(exist_ok=True);(out/'sources').mkdir(exist_ok=True)
 evidence=[]
 for name,url in SOURCES.items():
  p=out/'sources'/name
  if not p.exists():
   if not a.fetch_public_sources:raise ValueError('缺公开原件：'+name)
   req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
   with urllib.request.urlopen(req,timeout=30) as r:p.write_bytes(r.read())
  assert p.stat().st_size>1000
  evidence.append(dict(file=name,url=url,sha256=sha(p),bytes=p.stat().st_size))
 inputs={}
 for name in ['stock_basic_all.parquet','namechange_all.parquet']:
  source=ROOT/'data/raw'/name;target=out/'inputs'/name
  if not target.exists():target.write_bytes(source.read_bytes())
  assert sha(source)==sha(target);inputs[str(source)]=sha(source)
 b=pd.read_parquet(out/'inputs/stock_basic_all.parquet');n=pd.read_parquet(out/'inputs/namechange_all.parquet')
 rb,rn,intervals,added=repair(b,n)
 rb.to_parquet(out/'stock_basic_identity_overlay.parquet',index=False);rn.to_parquet(out/'namechange_identity_overlay.parquet',index=False);intervals.to_csv(out/'security_code_intervals.csv',index=False);added.to_csv(out/'added_basics.csv',index=False)
 # 供应商已用现代码回溯旧历史：必须去掉重复旧代码，而非新增三家证券。
 price_cols=['date','ts_code','open','close','pre_close','vol_share','adj_factor','hfq_close','total_mv_yuan','turnover_rate']
 prices=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',columns=price_cols,filters=[('ts_code','in',intervals.ts_code.tolist())]);checks=[];differences=[]
 for e in EVENTS:
  old=prices[prices.ts_code.eq(e['old_code'])].set_index('date');new=prices[prices.ts_code.eq(e['new_code'])].set_index('date')
  assert old.index.isin(new.index).all()
  assert (old.index < pd.Timestamp(e['effective_date'])).all()
  rhs=new.reindex(old.index)
  for col in ['open','close','pre_close','vol_share','total_mv_yuan','turnover_rate']:
   assert ((old[col]==rhs[col])|(old[col].isna()&rhs[col].isna())).all(),(e['old_code'],col)
  delta=old.adj_factor.ne(rhs.adj_factor)
  for date in old.index[delta]:differences.append(dict(old_code=e['old_code'],new_code=e['new_code'],date=str(date.date()),old_adj=float(old.loc[date,'adj_factor']),canonical_adj=float(rhs.loc[date,'adj_factor']),relative_difference=float(rhs.loc[date,'adj_factor']/old.loc[date,'adj_factor']-1)))
  checks.append(dict(old_code=e['old_code'],canonical_code=e['new_code'],duplicate_rows=len(old),raw_prices_and_basic_fields_exact=True,adj_differing_rows=int(delta.sum()),canonical_first=str(new.index.min().date()),old_last=str(old.index.max().date())))
 pd.DataFrame(differences).to_csv(out/'alias_adjustment_differences.csv',index=False)
 projection=ROOT/'outputs/recent_low_turnover/20260913T090336_282870Z/data_repair/repaired_feature_projection.parquet'
 raw=pd.read_parquet(projection);drop=raw.ts_code.isin([e['old_code'] for e in EVENTS]);canonical=raw.loc[~drop].copy()
 assert not canonical.duplicated(['date','ts_code']).any()
 canonical.to_parquet(out/'canonical_feature_projection.parquet',index=False)
 assert canonical.ts_code.isin(rb.ts_code).all()
 inputs[str(projection)]=sha(projection)
 # 原baseline已排除旧代码，因此canonical历史保持原值；不合并复权差异、更不按收益选版本。
 # 新增别名basics只允许与此去重投影配套；实际交易代码通过区间单独映射。
 (out/'repair_manifest.json').write_text(json.dumps(dict(status='complete',inputs=inputs,sources=evidence,events=EVENTS,price_identity_checks=checks,added_basics=len(rb)-len(b),added_name_intervals=len(rn)-len(n),original_rows_unchanged=True,price_data_modified=False,code_retires_without_delisting=True,price_stitching=False,execution_code_rollover_implemented=False,removed_duplicate_rows=int(drop.sum()),canonical_projection_rows=len(canonical),canonical_retained_values_unchanged=True,require_canonical_projection_with_overlay=True,classification='isolated historical metadata overlay; R is retired alias, not exchange delisting',outputs={p.name:sha(p) for p in [out/'stock_basic_identity_overlay.parquet',out/'namechange_identity_overlay.parquet',out/'security_code_intervals.csv',out/'added_basics.csv',out/'canonical_feature_projection.parquet',out/'alias_adjustment_differences.csv']},script_sha256=sha(__file__)),ensure_ascii=False,indent=2))
 (out/'sources/repair_historical_identity.py').write_bytes(Path(__file__).read_bytes());print(out)
if __name__=='__main__':main()
