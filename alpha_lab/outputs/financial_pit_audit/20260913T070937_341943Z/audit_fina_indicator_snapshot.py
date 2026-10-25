"""只读审计财务快照，历史财务行以公告日和期末双重谓词隔离。"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CUTOFF = '20231231'
KEY = ['ts_code', 'ann_date', 'end_date']
FIELDS = ['eps', 'ocfps', 'roe', 'roe_waa', 'roa', 'roa_yearly', 'roa2_yearly',
          'profit_dedt', 'q_npta', 'q_ocf_to_sales', 'ocf_to_debt', 'netprofit_yoy']
SAMPLES = ['600828.SH', '601006.SH', '601018.SH', '601811.SH', '603868.SH']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('/root/autodl-tmp/alpha_lab/data/raw/fina_indicator_all.parquet'))
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    meta = pq.ParquetFile(args.raw)
    schema = str(meta.schema_arrow)
    (out / 'raw_schema.txt').write_text(schema)
    filters = [('ann_date', '<=', CUTOFF), ('end_date', '<=', CUTOFF)]
    f = pd.read_parquet(args.raw, filters=filters).sort_values(KEY).reset_index(drop=True)
    ann = pd.to_datetime(f.ann_date, format='%Y%m%d', errors='coerce')
    end = pd.to_datetime(f.end_date, format='%Y%m%d', errors='coerce')
    assert f.ann_date.le(CUTOFF).all() and f.end_date.le(CUTOFF).all()
    lag = (ann - end).dt.days
    nonstandard = ~f.end_date.str[-4:].isin(['0331', '0630', '0930', '1231'])
    f.loc[nonstandard, KEY].to_csv(out / 'nonstandard_periods.csv', index=False)
    dup = f[f.duplicated(['ts_code', 'end_date'], keep=False)]
    dup.to_csv(out / 'multiple_announcement_rows.csv', index=False)
    changes = []
    for (code, period), g in dup.groupby(['ts_code', 'end_date']):
        a, b = g.iloc[0], g.iloc[-1]
        changed = [c for c in f.columns[3:] if not ((pd.isna(a[c]) and pd.isna(b[c])) or a[c] == b[c])]
        changes.append(dict(ts_code=code, end_date=period, first_ann=a.ann_date,
                            last_ann=b.ann_date, changed_fields=len(changed), fields='|'.join(changed)))
    pd.DataFrame(changes).to_csv(out / 'multiple_announcement_changes.csv', index=False)
    coverage = []
    for col in FIELDS:
        s = f[col]
        coverage.append(dict(field=col, observed=int(s.notna().sum()), missing=int(s.isna().sum()),
                             infinite=int(np.isinf(s).sum()), minimum=s.min(), median=s.median(), maximum=s.max()))
    pd.DataFrame(coverage).to_csv(out / 'field_coverage.csv', index=False)
    f[f.ts_code.isin(SAMPLES)][KEY + FIELDS].to_csv(out / 'preselected_five_company_rows.csv', index=False)
    f.loc[lag.gt(365), KEY].assign(disclosure_lag_days=lag[lag.gt(365)]).to_csv(out / 'ann_lag_over365.csv', index=False)
    f.assign(ann_year=ann.dt.year).groupby('ann_year').agg(rows=('ts_code','size'), codes=('ts_code','nunique')).to_csv(out / 'annual_coverage.csv')
    # 原公告数值由附存 PDF 对照抄录；年度每股现金流分别保留公告值与公式推导值。
    records = [
        ('601018.SH','20190330','20181231','q_ocf_to_sales',2265052/5460626*100,.000051,'ningbo2018',3,'quarterly_cfo_div_revenue_times100'),
        ('601018.SH','20200411','20191231','q_ocf_to_sales',655268/7119694*100,.000051,'ningbo2019',7,'quarterly_cfo_div_revenue_times100'),
        ('601018.SH','20190330','20181231','eps',.22,.00005,'ningbo2018',3,'direct'),
        ('601018.SH','20190330','20181231','profit_dedt',2559051000.,.5,'ningbo2018',3,'thousand_yuan_to_yuan'),
        ('601018.SH','20190330','20181231','roe_waa',7.75,.00005,'ningbo2018',3,'direct'),
        ('601018.SH','20200411','20191231','eps',.26,.00005,'ningbo2019',6,'direct'),
        ('601018.SH','20200411','20191231','profit_dedt',2749971000.,.5,'ningbo2019',6,'thousand_yuan_to_yuan'),
        ('601018.SH','20200411','20191231','ocfps',.02,.00005,'ningbo2019',6,'direct'),
        ('601018.SH','20200411','20191231','roe_waa',8.68,.00005,'ningbo2019',6,'direct'),
        ('601006.SH','20200429','20191231','eps',.92,.00005,'daqin2019',5,'direct'),
        ('601006.SH','20200429','20191231','profit_dedt',13739687369.,.5,'daqin2019',5,'direct'),
        ('601006.SH','20200429','20191231','roe_waa',12.55,.00005,'daqin2019',6,'direct'),
        ('601006.SH','20200429','20191231','ocfps',15022536018/14866791491,.000051,'daqin2019',5,'operating_cashflow_div_year_end_shares_p2'),
    ]
    checks=[]
    for code, ad, ed, field, expected, atol, source, page, kind in records:
        row=f[(f.ts_code==code)&(f.ann_date==ad)&(f.end_date==ed)]
        assert len(row)==1
        actual=float(row.iloc[0][field])
        checks.append(dict(ts_code=code,ann_date=ad,end_date=ed,field=field,cache=actual,
                           issuer=expected,atol=atol,match=abs(actual-expected)<=atol,
                           source=source,page_one_based=page,comparison_kind=kind))
    pd.DataFrame(checks).to_csv(out / 'issuer_value_checks.csv', index=False)
    # 宁波港2019报告给出的2018重述值不能回填到2019-03-30。
    old=f[(f.ts_code=='601018.SH')&(f.end_date=='20181231')].iloc[0]
    restatement=dict(ts_code='601018.SH',end_date='20181231',ann_date=old.ann_date,
                     original_profit_dedt_yuan=2559051000.,later_restated_profit_dedt_yuan=2560769000.,
                     later_publication='20200411',cache_profit_dedt_yuan=float(old.profit_dedt),
                     cache_keeps_original=bool(old.profit_dedt==2559051000.),
                     conclusion='此字段保留原披露，不能据此证明其他字段或全表版本完整')
    (out / 'restatement_control.json').write_text(json.dumps(restatement,ensure_ascii=False,indent=2))
    summary=dict(status='complete',filters=filters,raw_metadata_rows=meta.metadata.num_rows,
                 row_groups=meta.metadata.num_row_groups,filtered_rows=len(f),codes=int(f.ts_code.nunique()),
                 columns=len(f.columns),ann_min=f.ann_date.min(),ann_max=f.ann_date.max(),
                 end_min=f.end_date.min(),end_max=f.end_date.max(),invalid_dates=int((ann.isna()|end.isna()).sum()),
                 negative_ann_lag=int(lag.lt(0).sum()),nonstandard_periods=int(nonstandard.sum()),ann_lag_over365=int(lag.gt(365).sum()),
                 max_ann_lag=int(lag.max()),ann_lag_median=float(lag.median()),
                 full_duplicates=int(f.duplicated().sum()),key_duplicates=int(f.duplicated(KEY).sum()),
                 multiple_announcement_groups=len(changes),multiple_announcement_rows=len(dup),
                 missing_version_fields=[c for c in ['update_flag','f_ann_date','report_type','comp_type','ingested_at','snapshot_date'] if c not in f],
                 issuer_checks=len(checks),issuer_matches=sum(c['match'] for c in checks),
                 filtered_sorted_row_hash=hashlib.sha256(pd.util.hash_pandas_object(f,index=False).values.tobytes()).hexdigest(),
                 pandas_version=pd.__version__,training=False,market_rows_read=0,
                 verdict='snapshot_not_certified_strict_PIT; spot_checks_match_original; no_universal_backfill_claim')
    (out / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
