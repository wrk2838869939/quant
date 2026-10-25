#!/usr/bin/env python3
"""合并有出处的披露事实，保留原vintage描述差分，估算扩展原机会池的文档规模。"""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from financial_events import normalize_events, asof_events, verify_event_sources, append_events


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve();out=run/'assembled';out.mkdir(exist_ok=True)
    if (out/'events.parquet').exists():raise ValueError('禁止覆盖已组装事件')
    folders=[('pilot_601006_601018','announcement_facts.csv'),('pilot_600828_601811_603868','facts_long.csv')]
    facts=[];bridges=[]
    for folder,name in folders:
        frame=pd.read_csv(run/folder/name)
        for row in frame.to_dict('records'):
            token=str(row['original_number'])
            canonical=token.replace(',','').replace('−','-').strip()
            if canonical.startswith('(') and canonical.endswith(')'):canonical='-'+canonical[1:-1]
            actual=Decimal(canonical)*Decimal(str(row['unit_multiplier']))
            assert abs(actual-Decimal(str(row['value_yuan'])))<=Decimal('.01')
            role='disclosed_value' if row['role']=='current' or row.get('disclosure_status')=='newly_disclosed_restatement' else 'historical_comparator'
            record=dict(ts_code=row['ts_code'],period_end=row['period_end'],field=row['field'],scope=row['scope'],
                published_date=row['published_date'],value=row['value_yuan'],unit='CNY',
                basis_id='document:'+row['source_sha256'],source_path=str((run/folder/row['source_file']).resolve()),
                source_sha256=row['source_sha256'],pdf_page=int(row['pdf_page']),source_token=token,
                source_url=row['source_url'],event_role=role)
            facts.append(record);bridges.append(dict(original_fact_id=row['fact_id'],folder=folder,
                published_date=row['published_date'],version=row.get('version',row.get('version_label')),
                event_role=role,accounting_notes=row['accounting_notes']))
    pilot=normalize_events(pd.DataFrame(facts));assert len(pilot)==98
    checked=verify_event_sources(pilot)
    normalized_one=[normalize_events(pd.DataFrame([row])).event_id.iloc[0] for row in facts]
    pd.DataFrame(bridges).assign(event_id=normalized_one).to_csv(out/'fact_to_event.csv',index=False)
    controls=pd.read_parquet(run/'event_recovery/events.parquet')
    events=append_events(controls,pilot);assert len(events)==107
    pd.testing.assert_frame_equal(events,append_events(controls,pilot.iloc[::-1]))
    req=pilot[['ts_code','period_end','field','scope']].drop_duplicates().assign(date=pd.Timestamp('2022-01-25'))
    latest=asof_events(pilot,req)
    disclosed=pilot[pilot.event_role.eq('disclosed_value')]
    for row in latest[latest.status.eq('available')].itertuples():
        subset=disclosed[(disclosed.ts_code==row.ts_code)&(disclosed.period_end==row.period_end)&(disclosed.field==row.field)&(disclosed.scope==row.scope)&(disclosed.published_date<row.date)]
        last=subset[subset.published_date==subset.published_date.max()]
        assert last.value.eq(row.value).all()
    amended=pd.DataFrame(bridges).assign(event_id=normalized_one).query("version == 'restated_in_2021'")
    assert len(amended)==10 and amended.event_role.eq('disclosed_value').all()
    selected=set('|'.join(latest.lineage).split('|'))
    assert set(amended.event_id).issubset(selected)
    # 改码/时间线试点和五股试点合并；原事实与原比较列均保留。
    events.to_parquet(out/'events.parquet',index=False);latest.to_csv(out/'asof_20220125.csv',index=False)
    a=pd.read_csv(run/folders[0][0]/'metric_differences.csv')
    a=a.rename(columns={'descriptive_delta_pp':'original_vintage_delta_pp','strict_delta_pp':'comparable_delta_pp','strict_comparable_ready':'document_comparability_supported','reason':'accounting_notes'})
    b=pd.read_csv(run/folders[1][0]/'decision_features.csv').rename(columns={'observed_initial_report_delta_pp':'original_vintage_delta_pp','comparable_known':'document_comparability_supported'})
    summary=pd.concat([a,b],ignore_index=True)[['ts_code','field','original_vintage_delta_pp','comparable_delta_pp','document_comparability_supported','accounting_notes']]
    summary['history_complete']=False;summary['model_ready']=False
    assert len(summary)==10 and summary.document_comparability_supported.sum()==3
    summary.to_csv(out/'five_stock_feature_evidence.csv',index=False)
    # 只读上一轮来源日期。股票—报告期是取证组合，一份公告可覆盖多个期间。
    coverage=ROOT/'outputs/financial_pit_audit/20260913T070937_341943Z/opportunity_coverage/financial_candidates.parquet'
    opportunities=pd.read_parquet(coverage,columns=['date','ts_code','report_end','prior_end','trend'])
    required=set()
    for row in opportunities.itertuples():
        for end in [row.report_end,row.prior_end]:
            if pd.isna(end) or not end.is_quarter_end:continue
            required.add((row.ts_code,end))
            if end.quarter>1:required.add((row.ts_code,(end.to_period('Q')-1).end_time.normalize()))
    requirements=pd.DataFrame(sorted(required),columns=['ts_code','period_end'])
    requirements.to_csv(out/'stock_period_evidence_requirements.csv',index=False)
    manifest=dict(status='complete_recovery_and_scale_pilot',pilot_document_count=20,pilot_facts=checked,
        total_event_count=len(events),restated_values_activated_at_2021_dates=10,
        original_vintage_differences=10,document_comparability_supported=3,model_ready_features=0,
        source_history_complete=False,models_run=0,market_or_target_labels_read=False,
        original_opportunities=len(opportunities),distinct_stock_period_evidence_requirements=len(requirements),
        document_requirement_semantics='股票—报告期取证组合；不是PDF数量或其下限。一份公告可覆盖多个期末，修订/更正搜索还会增加取证工作。',
        source_hashes={str(p):sha(p) for p in [Path(__file__).resolve(),ROOT/'src/financial_events.py',coverage]+[run/f/n for f,n in folders]},
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({key:manifest[key] for key in ['status','pilot_facts','total_event_count','distinct_stock_period_evidence_requirements','models_run']}))


if __name__=='__main__':main()
