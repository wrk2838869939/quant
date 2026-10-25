"""使用本次拉取的名称历史复核资格，再调用既有截面排序与选取函数。"""
from pathlib import Path
import sys,importlib.util,json,hashlib
import pandas as pd,numpy as np
ROOT=Path('/root/autodl-tmp');OUT=Path(__file__).resolve().parent
sys.argv=[sys.argv[0],'--scan',str(ROOT/'alpha_lab/outputs/sector_live_scan/20261019T144154_853953Z')]
spec=importlib.util.spec_from_file_location('existing_eval',ROOT/'alpha_lab/outputs/strategy_2026_opportunity_scan/20261019_eod/evaluate_existing_models.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
asof=pd.Timestamp('2026-10-19');allrows=pd.read_csv(OUT/'all_members_detailed.csv');codes=allrows.ts_code.tolist()
nh=pd.read_parquet(OUT/'fresh_namechange.parquet')
for key in ['start_date','end_date','ann_date']:nh[key]=pd.to_datetime(nh[key],format='%Y%m%d',errors='coerce')
normal=m.announced_normal_names(nh,asof,codes)
meta=pd.read_parquet(ROOT/'datahub/releases/selection_live_20261019_eod/datasets/meta/stock_basic.parquet').set_index('ts_code').reindex(codes)
listed=pd.to_datetime(meta.list_date,format='%Y%m%d',errors='coerce');delisted=pd.to_datetime(meta.delist_date,format='%Y%m%d',errors='coerce')
raw=pd.read_parquet(OUT/'stock_history.parquet');latest=raw[raw.date.eq(asof)].set_index('ts_code').reindex(codes)
qualified=normal & listed.le(asof)&(delisted.isna()|delisted.gt(asof))&latest.close.gt(0)&latest.vol_share.gt(0)
allrows['qualified_fresh']=allrows.ts_code.map(qualified)
assert allrows.qualified.eq(allrows.qualified_fresh).all(),'新名称历史导致资格变化，需重新审视分数差异'
d=allrows[allrows.qualified_fresh].copy().reset_index(drop=True)
cfg=m.yaml.safe_load((m.STOCK/'config.yaml').read_text())
feats,names=m.cross_section_features(d,cfg['model']);weights=np.load(m.STOCK/'models/2026.npz');assert names==weights['features'].tolist()
d['model_score']=feats[names].to_numpy()@weights['coef'];d['observation_candidate']=False
d.loc[m.select_positive_top(d,'model_score',cfg),'observation_candidate']=True
old=pd.read_csv(OUT/'stock_scores_old_name_history.csv').set_index('ts_code')
np.testing.assert_allclose(d.model_score.to_numpy(),old.reindex(d.ts_code).model_score.to_numpy(),atol=1e-12,rtol=1e-10)
d['official_strategy_entry']=False
p=m.yaml.safe_load((m.SECTOR/'sources/left_side_trend_v2.yaml').read_text())['stock']
def reason(r):
 reasons=[]
 if not r.qualified_fresh:return 'ST资格排除'
 if r.early_return>-p['minimum_early_fall']:reasons.append('前3日跌幅不足1%')
 if not np.isfinite(r.slowdown) or r.slowdown>p['slowdown_ratio']:reasons.append('下跌减速未过')
 if r.late_return>p['maximum_recent_gain']:reasons.append('近2日涨幅超过2%')
 if r.day_return<-p['maximum_current_daily_fall']:reasons.append('当日跌幅超过2%')
 if r.model_score<=0:reasons.append('个股排序分数非正')
 return '；'.join(reasons) if reasons else '个股条件通过；板块/首次事件入口不通过'
allrows=allrows.merge(d[['ts_code','model_score','observation_candidate','official_strategy_entry']],on='ts_code',how='left')
allrows['reason']=[reason(r) for r in allrows.itertuples()]
allrows.to_csv(OUT/'all_members_final.csv',index=False)
candidates=d[d.observation_candidate].sort_values('model_score',ascending=False)
candidates.to_csv(OUT/'observation_candidates.csv',index=False)
feats.to_parquet(OUT/'cross_section_features.parquet',index=False)
checks={}
for filename in ['pipeline_hashes_before.json','model_hashes_before.json']:
 for name,h in json.loads((ROOT/'alpha_lab/outputs/strategy_2026_opportunity_scan/20261019_eod'/filename).read_text()).items():
  assert m.digest(ROOT/name)==h,name
  checks[name]=h
summary=dict(asof='2026-10-19',sector='酒店餐饮',members=len(allrows),qualified=len(d),price_pass=int(d.price.sum()),
 observation_candidates=candidates.ts_code.tolist(),official_strategy_candidates=0,model_year=2026,model_last_label='2025-11-19',
 scenario='19日横截面独立观察；不是原策略新入场',name_history_refetched_for_all_members=True,qualification_unchanged=True,
 no_retraining=True,no_engineering_changes=True,model_condition_applicability='未验证板块模型拒绝后、重复确认日的排序有效性')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',hashes_unchanged=checks,qualification_unchanged=True),ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False));print(candidates[['name','ts_code','close','day_return','ret20','volatility','early_return','late_return','slowdown','model_score']].to_string(index=False))
