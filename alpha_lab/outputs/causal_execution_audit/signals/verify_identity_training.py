from pathlib import Path
import sys,json,hashlib
import pandas as pd,numpy as np,joblib,yaml
R=Path('/root/autodl-tmp/alpha_lab');O=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'src'))
from value_forecast import fit_value_model
repair=R/'outputs/identity_repair/20260913T130530_010878Z';cfg=yaml.safe_load((R/'configs/symmetric_phase_research.yaml').read_text())
old=pd.read_parquet(R/'data/raw/stock_basic_all.parquet');new=pd.read_parquet(repair/'stock_basic_identity_overlay.parquet');print('new fields',new.columns.tolist())
for f in ['list_date','delist_date']:
 old[f]=pd.to_datetime(old[f],errors='coerce');new[f]=pd.to_datetime(new[f],errors='coerce')
left=old.set_index('ts_code')[['list_date','delist_date']].sort_index();right=new.set_index('ts_code')[['list_date','delist_date']].reindex(left.index)
pd.testing.assert_frame_equal(left,right,check_exact=True)
pairs={'000022.SZ':'001872.SZ','000043.SZ':'001914.SZ','300114.SZ':'302132.SZ'}
expected={'000022.SZ':'1993-05-05','000043.SZ':'1994-09-28','300114.SZ':'2010-08-27'}
for code,day in expected.items():
 assert new.set_index('ts_code').loc[code,'list_date']==pd.Timestamp(day)
 assert new.set_index('ts_code').loc[pairs[code],'list_date']==pd.Timestamp(day)
projection=pd.read_parquet(repair/'canonical_feature_projection.parquet',columns=['date','ts_code'])
assert not projection.ts_code.isin(pairs).any();assert not projection.duplicated(['date','ts_code']).any()
# Check original/canonical ST intervals for all original identity codes exactly unchanged.
oldnames=pd.read_parquet(R/'data/raw/namechange_all.parquet');newnames=pd.read_parquet(repair/'namechange_identity_overlay.parquet')
cols=list(oldnames.columns)
a=oldnames.copy().sort_values(cols,na_position='first').reset_index(drop=True)
b=newnames[~newnames.ts_code.isin(pairs)][cols].copy().sort_values(cols,na_position='first').reset_index(drop=True)
# New overlay may normalize dates; make an explicit semantic date comparison.
for f in ['start_date','end_date','ann_date']:
 a[f]=pd.to_datetime(a[f],errors='coerce');b[f]=pd.to_datetime(b[f],errors='coerce')
pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
cols=['date','ts_code','monthly','trend','label_end_60','ret_60','mae_60','entry_missing_60']
panel=pd.read_parquet(R/cfg['source_panel'],columns=cols)
train=panel[panel.monthly&panel.trend&panel.ts_code.str.match(cfg['mainboard_pattern'])].copy()
mature=train[train.label_end_60.notna()&train.label_end_60.le('2021-12-31')&train.entry_missing_60.eq(False)]
oldopp=pd.read_csv(R/'outputs/independent_leak_audit/20260913/signals/missing_identity_observable_months.csv',parse_dates=['date'])
oldopp=oldopp[oldopp.trend&oldopp.would_have_mature_label_by_2021].copy();oldopp['canonical_code']=oldopp.ts_code.map(pairs)
oldopp=oldopp.merge(mature[['date','ts_code']].rename(columns={'ts_code':'canonical_code'}).assign(already_in_actual_training=True),on=['date','canonical_code'],how='left')
oldopp.to_csv(O/'previous_potential_missing_rows_resolved.csv',index=False)
main=oldopp[oldopp.ts_code.isin(['000022.SZ','000043.SZ'])]
model=joblib.load(R/'outputs/recent_low_turnover/20260913T090336_282870Z/frozen_low_turnover_model.joblib')
refit=fit_value_model(train,[],asof='2021-12-31',**cfg['forecast'])
pd.testing.assert_frame_equal(refit.coefficients,model.coefficients,check_exact=True)
report={'status':'complete','identity_repair':str(repair),'canonical_codes_existing_listing_dates_exact':True,
'old_alias_listing_dates_match_official_evidence':expected,'canonical_namechange_history_unchanged':True,
'canonical_projection_no_duplicate_date_security_rows':True,'projection_rows':len(projection),
'previous_potential_mainboard_missing_rows':len(main),'already_present_under_canonical_id':int(main.already_in_actual_training.eq(True).sum()),
'canonical_training_counts':mature[mature.ts_code.isin(pairs.values())].groupby('ts_code').size().to_dict(),
'original_training_rows':len(mature),'frozen_constant_refit_exact':True,'constant_return':float(refit.coefficients.iloc[0,0]),
'constant_adverse':float(refit.coefficients.iloc[0,1]),'interpretation':'No new securities or training samples should be added; remove old duplicate aliases with companion identity overlay. Original training constant unchanged. Full feature/eligibility re-run independently owned by data reviewer.'}
(O/'identity_training_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
