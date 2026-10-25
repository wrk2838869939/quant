#!/usr/bin/env python3
"""从已核验康美原PDF构建字段披露事件并回放真实修订边界，不调用行情。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from financial_events import normalize_events, append_events, asof_events, verify_event_sources


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve();out=run/'event_recovery';out.mkdir(exist_ok=True)
    if (out/'events.parquet').exists():raise ValueError('禁止覆盖已恢复事件')
    old=ROOT/'outputs/financial_pit_audit/20260913T070937_341943Z/restatement_evidence'
    registry={row['name']:row for row in json.loads((old/'sources.json').read_text())}
    registration=dict(purpose='已核实披露事件恢复，不假定公告集合完整',issuer='600518.SH',
        event_fields=['cfo_ytd','eps'],test_dates=['2018-04-26','2018-04-27','2019-04-29','2019-04-30',
        '2019-05-01','2020-06-17','2020-06-18','2020-06-19'],
        same_day_rule='published_date严格早于查询日',model_fits=0,market_data_read=False)
    (out/'registration.json').write_text(json.dumps(registration,ensure_ascii=False,indent=2))
    sources=out/'sources';sources.mkdir(exist_ok=True)
    for name in ['annual_2017','annual_2018','annual_2019']:
        src=old/f'{name}.pdf';assert sha(src)==registry[name]['sha256']
        shutil.copy2(src,sources/src.name)
    rows=[]
    def add(name,page,period,field,value,token,role='disclosed_value',basis=None):
        date={'annual_2017':'2018-04-26','annual_2018':'2019-04-30','annual_2019':'2020-06-18'}[name]
        rows.append(dict(ts_code='600518.SH',period_end=period,field=field,scope='consolidated',
            published_date=date,value=value,unit='CNY' if field=='cfo_ytd' else 'CNY/share',
            basis_id=basis or name,source_path=str((sources/f'{name}.pdf').resolve()),
            source_sha256=registry[name]['sha256'],pdf_page=page,source_token=token,
            source_url=registry[name]['url'],event_role=role))
    add('annual_2017',7,'2017-12-31','eps',.784,'0.784')
    add('annual_2018',7,'2017-12-31','eps',.388,'0.388')
    add('annual_2019',7,'2017-12-31','eps',.2397,'0.2397')
    add('annual_2018',6,'2018-12-31','cfo_ytd',-3191529576.31,'-3,191,529,576.31')
    add('annual_2018',7,'2018-12-31','eps',.184,'0.184')
    add('annual_2019',7,'2018-12-31','cfo_ytd',305616727.44,'305,616,727.44')
    add('annual_2019',7,'2018-12-31','eps',.0302,'0.0302')
    add('annual_2019',7,'2018-12-31','cfo_ytd',-3191529576.31,'-3,191,529,576.31','historical_comparator','annual_2018')
    add('annual_2019',7,'2018-12-31','eps',.184,'0.1840','historical_comparator','annual_2018')
    events=normalize_events(pd.DataFrame(rows))
    checked=verify_event_sources(events)
    split=events.published_date.lt('2020-01-01')
    appended=append_events(events[split],events[~split])
    pd.testing.assert_frame_equal(events,appended)
    requests=pd.DataFrame([dict(date=date,ts_code='600518.SH',period_end=period,field=field,scope='consolidated')
        for date in registration['test_dates'] for period,field in
        [('2017-12-31','eps'),('2018-12-31','eps'),('2018-12-31','cfo_ytd')]])
    replay=asof_events(events,requests)
    pd.testing.assert_frame_equal(replay,asof_events(events.iloc[::-1],requests))
    earlier=requests[pd.to_datetime(requests.date).lt('2020-06-18')]
    pd.testing.assert_frame_equal(asof_events(events,earlier),asof_events(events[split],earlier))
    cfo=replay[replay.field.eq('cfo_ytd')].set_index('date')
    assert cfo.loc['2019-04-30','status']=='unavailable'
    assert cfo.loc['2019-05-01','value']==cfo.loc['2020-06-18','value']==-3191529576.31
    assert cfo.loc['2020-06-19','value']==305616727.44
    eps=replay[replay.field.eq('eps')&replay.period_end.eq('2018-12-31')].set_index('date')
    assert eps.loc['2020-06-18','value']==.184 and eps.loc['2020-06-19','value']==.0302
    events.to_parquet(out/'events.parquet',index=False);events.to_csv(out/'events.csv',index=False)
    replay.to_csv(out/'asof_replay.csv',index=False)
    (out/'source_checks.json').write_text(json.dumps(dict(status='passed',events=checked,asof_queries=len(replay),
        source_hash_and_page_token_verified=True,original_cells_visually_reviewed_in_prior_audit=True,
        history_complete=False,training_admission=False,append_and_reverse_and_future_checks=True),indent=2))
    files=[p for p in out.rglob('*') if p.is_file()]
    manifest=dict(status='complete_event_recovery_pilot',event_count=len(events),asof_queries=len(replay),
        source_coverage='known_verified_filings_only_not_exhaustive_correction_history',model_fits=0,
        source_hashes={str(p):sha(p) for p in [Path(__file__).resolve(),ROOT/'src/financial_events.py',ROOT/'tests/test_financial_events.py']},
        artifacts={str(p.relative_to(out)):sha(p) for p in files})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status=manifest['status'],events=len(events),queries=len(replay))))


if __name__=='__main__':main()
