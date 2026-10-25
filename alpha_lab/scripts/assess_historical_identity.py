"""隔离身份去重层影响；不拟合、不读取收益结果。"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from prepare_recent_low_turnover_inputs import build_inputs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();out=a.run
 cfg=yaml.safe_load((ROOT/'outputs/autonomous_value_research/20260912T160734_658714Z/config.yaml').read_text())
 def build(projection,basic,names):
  raw=pd.read_parquet(projection);b=pd.read_parquet(basic);n=pd.read_parquet(names)
  result,hold=build_inputs(raw,b,n,cfg,pd.Timestamp('2014-01-01'),pd.Timestamp('2026-09-08'));return result,hold
 original=ROOT/'outputs/recent_low_turnover/20260913T090336_282870Z/data_repair/repaired_feature_projection.parquet'
 print('build original full history',flush=True)
 before,h1=build(original,out/'inputs/stock_basic_all.parquet',out/'inputs/namechange_all.parquet')
 print('build identity-deduplicated full history',flush=True)
 after,h2=build(out/'canonical_feature_projection.parquet',out/'stock_basic_identity_overlay.parquet',out/'namechange_identity_overlay.parquet')
 key=['date','ts_code'];b=before.set_index(key);c=after.set_index(key);keysame=b.index.equals(c.index)
 assert keysame,'资格行改变，需要单独审查'
 delta=~((b==c)|(b.isna()&c.isna()));summary={col:int(delta[col].sum()) for col in delta};changed=delta.any(axis=1)
 before[changed.to_numpy()].to_parquet(out/'before_changed_review_features.parquet',index=False);after[changed.to_numpy()].to_parquet(out/'after_changed_review_features.parquet',index=False)
 # raw momentum rank uses all quoted aliases; downstream low_turnover score only uses eligible x_turnover.
 expected=[col for col,n in summary.items() if n and col!='momentum'];assert not expected,expected
 allfeatures=[x for x in b if x.startswith('x_')]
 comparisons=[]
 for date in sorted(before.date.unique()):
  x=before[before.date.eq(date)];y=after[after.date.eq(date)]
  # 与实际执行源码相同的主板代码范围；先比较实际字符串，再对照日期映射。
  pattern=r'^(000|001|002|003|600|601|603|605)'
  def selected(z):return z[z.ts_code.str.match(pattern)&z.trend].sort_values(['x_turnover','ts_code'],kind='stable').head(20).ts_code.tolist()
  bx,cy=selected(x),selected(y);assert bx==cy
  comparisons.append(dict(date=str(pd.Timestamp(date).date()),all_A_rows=len(x),mainboard_top20_count=len(bx),top20_exact=True,eligible_exact=True))
 pd.DataFrame(comparisons).to_csv(out/'monthly_identity_comparison.csv',index=False)
 # 旧代码剔除后独立持有状态的canonical列保持逐值相同。
 common=h2.columns;hold_equal=((h1[common]==h2)|(h1[common].isna()&h2.isna())).to_numpy().all();assert hold_equal
 # 当时可交易代码映射以公告生效日为界；canonical只是内部ID，不提前使用新证券代码下单。
 events=json.loads((out/'repair_manifest.json').read_text())['events'];px=pd.read_parquet(out/'canonical_feature_projection.parquet',columns=['date','ts_code'],filters=[('ts_code','in',[e['new_code'] for e in events])]);px['security_id']=px.ts_code;px['tradable_code']=px.ts_code
 for e in events:px.loc[px.ts_code.eq(e['new_code'])&px.date.lt(pd.Timestamp(e['effective_date'])),'tradable_code']=e['old_code']
 px.to_parquet(out/'canonical_quote_identity.parquet',index=False)
 # 追溯全部复权差异回原按日cache，不自行挑选更有利的一个版本。
 differences=pd.read_csv(out/'alias_adjustment_differences.csv',parse_dates=['date']);provenance=[]
 for date,g in differences.groupby('date'):
  file=ROOT/'data/raw/by_date/adj_factor'/f'{date:%Y%m%d}.parquet';raw=pd.read_parquet(file).set_index('ts_code')
  for r in g.itertuples():
   ok=np.isclose(raw.loc[r.old_code,'adj_factor'],r.old_adj,atol=1e-12,rtol=0) and np.isclose(raw.loc[r.new_code,'adj_factor'],r.canonical_adj,atol=1e-12,rtol=0);assert ok
   provenance.append(dict(date=str(date.date()),old_code=r.old_code,new_code=r.new_code,source_file=str(file),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),source_values_match=True))
 pd.DataFrame(provenance).to_csv(out/'adjustment_difference_provenance.csv',index=False)
 report=dict(status='complete',review_rows=len(before),review_dates=int(before.date.nunique()),eligible_identity_exact=keysame,changed_columns=summary,rank_feature_columns=allfeatures,rank_features_exact=all(summary[k]==0 for k in allfeatures),all_mainboard_top20_exact=True,canonical_hold_exact=bool(hold_equal),tradable_identity_rows=len(px),adjustment_disagreements=len(differences),adjustment_source_reproduced=True,adjustment_authoritative_version='unresolved; canonical values retained to isolate identity repair, not selected by returns',new_training_entities=0,no_account_execution=True,no_model_fit=True)
 (out/'impact.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(out/'sources/assess_historical_identity.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
