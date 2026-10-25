"""复验旧封存、真实行情增量与合成CSV跨进程闭环，独立保存本轮证据。"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from broker_csv import BrokerAccount, KINDS
from execution_market_feed import file_hash, require_current_metadata
from shadow_planner import ShadowError, canonical
from test_broker_csv import exports, encode
from test_shadow_execution import fixture, preflight


def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,default=ROOT/'outputs/live_ingestion/20260914_market_stage')
    args=parser.parse_args()
    out=ROOT/'outputs/live_ingestion'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    print(str(out),flush=True)
    prior=ROOT/'outputs/shadow_execution/20260914T054517_770961Z'
    completion=json.loads((prior/'completion.json').read_text())
    for name,digest in completion['artifact_sha256'].items():
        assert file_hash(prior/name)==digest,name
    verification=json.loads((prior/'verification.json').read_text())
    for name,digest in verification['source_hashes'].items():assert file_hash(ROOT/name)==digest,name
    assert file_hash(ROOT.parent/'AGENTS.md')==verification['root_agents_sha256']
    package=json.loads((args.stage/'validated/package.json').read_text())
    for name,digest in package['input_sha256'].items():assert file_hash(name)==digest,name
    for name,digest in package['artifacts'].items():assert file_hash(args.stage/'validated'/name)==digest,name
    assert file_hash(ROOT/'scripts/validate_execution_market.py')==package['source_sha256']
    assert file_hash(ROOT/'src/execution_market_feed.py')==package['module_sha256']
    f=pd.read_parquet(args.stage/'validated/feature_increment.parquet').set_index(['date','ts_code'])
    m=pd.read_parquet(args.stage/'validated/market_increment.parquet').set_index(['date','ts_code'])
    # 从原始列另行核算单位转换及连接，不调用normalize_day。
    for check in package['checks']:
        date=check['date'];key=date.replace('-','')
        d=pd.read_parquet(args.stage/f'daily_{key}.parquet').set_index('ts_code').sort_index()
        b=pd.read_parquet(args.stage/f'daily_basic_{key}.parquet').set_index('ts_code').loc[d.index]
        a=pd.read_parquet(args.stage/f'adj_factor_{key}.parquet').set_index('ts_code').loc[d.index]
        actual=f.loc[pd.Timestamp(date)].loc[d.index];market=m.loc[pd.Timestamp(date)].loc[d.index]
        for left,right in [(actual.hfq_close,d.close*a.adj_factor),(actual.total_mv_yuan,b.total_mv*10000),
                           (market.vol_share,d.vol*100),(market.amount_yuan,d.amount*1000)]:
            assert ((left-right).abs()<1e-6).all()
    try:require_current_metadata(package,package['asof'])
    except ShadowError as exc:metadata_block=str(exc)
    else:raise AssertionError('元数据缺失未停止')
    followup=args.stage/'absence_followup'
    follow=json.loads((followup/'receipt.json').read_text())
    for item in follow['queries']:
        assert item['status']=='downloaded'
        assert file_hash(followup/item['file'])==item['sha256']
    assessments=pd.read_csv(followup/'assessments.csv')
    for item in assessments.to_dict('records'):
        source=pd.read_parquet(followup/('suspend_d_'+item['date']+'.parquet'))
        matched=source.loc[source.ts_code.eq(item['ts_code']) & source.trade_date.astype(str).eq(item['date'].replace('-','')) & source.suspend_type.eq('S')]
        assert not matched.empty and (matched.suspend_timing.isna()|matched.suspend_timing.eq('')).any()
        assert item['assessment']=='provider_reported_full_day_suspension'
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','test_broker_csv','test_execution_market_feed',
        'test_shadow_execution','test_shadow_inputs','-v'],env=env,capture_output=True,text=True)
    (out/'tests.log').write_text(tests.stdout+tests.stderr)
    assert tests.returncode==0,'tests failed'
    demo=out/'synthetic_csv_demo';demo.mkdir()
    db=demo/'account.sqlite';account=BrokerAccount(db)
    s,b,cal,cfg=fixture();account.import_snapshot(s)
    plan=account.plan(b,cal,cfg);account.preflight(plan['plan_id'],preflight(plan))
    before=account.status();account.close()
    save(demo/'initial_snapshot.json',s);save(demo/'plan.json',plan)
    profile,records=exports(plan)
    def write_exports(folder,rows):
        folder.mkdir();p,files=encode(profile,rows)
        (folder/'profile.yaml').write_bytes(p)
        for name,data in files.items():(folder/(name+'.csv')).write_bytes(data)
    bad=deepcopy(records);bad['account'][0]['cash']+=1;bad['account'][0]['equity']+=1
    write_exports(demo/'wrong_snapshot',bad);write_exports(demo/'correct',records)
    calls=[]
    for label,folder,expected in [('wrong_snapshot',demo/'wrong_snapshot',1),('correct',demo/'correct',0),('repeat',demo/'correct',0)]:
        command=[sys.executable,str(ROOT/'scripts/import_broker_csv.py'),'--db',str(db),'--profile',str(folder/'profile.yaml'),
            '--archive',str(demo/'archive')]
        for name in KINDS:command.extend(['--'+name,str(folder/(name+'.csv'))])
        run=subprocess.run(command,capture_output=True,text=True)
        (demo/(label+'.log')).write_text(run.stdout+run.stderr)
        assert run.returncode==expected,label
        calls.append(dict(case=label,returncode=run.returncode,result=json.loads(run.stdout)))
        account=BrokerAccount(db)
        if label=='wrong_snapshot':assert account.status()==before
        if label=='correct':after=account.status()
        if label=='repeat':assert account.status()==after
        account.close()
    for archive in (demo/'archive').iterdir():
        manifest=json.loads((archive/'manifest.json').read_text())
        for name,digest in manifest['sha256'].items():assert file_hash(archive/name)==digest
    save(demo/'final_state.json',after)
    save(demo/'verification.json',dict(synthetic=True,real_broker_exports=False,calls=calls,
        wrong_snapshot_rollback=True,repeated_import_no_change=True))
    sources=['src/broker_csv.py','src/execution_market_feed.py','configs/broker_csv_template.yaml',
        'scripts/import_broker_csv.py','scripts/stage_execution_market.py','scripts/validate_execution_market.py',
        'scripts/check_execution_absences.py','scripts/validate_live_ingestion.py',
        'tests/test_broker_csv.py','tests/test_execution_market_feed.py']
    for name in sources:
        target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copytree(args.stage,out/'market_stage')
    result=dict(status='passed',prior_artifacts_verified=len(completion['artifact_sha256']),
        frozen_sources_verified=len(verification['source_hashes']),root_agents_unchanged=True,
        price_asof=package['asof'],price_rows=len(f),missing_previous_quote_rows=len(assessments),
        absence_explanation='provider_reported_full_day_suspension',metadata_block=metadata_block,
        tests_passed=45,cli_cases=3,real_broker_exports=False,live_orders=False,strategy_admitted=False,
        sources_sha256={name:file_hash(ROOT/name) for name in sources})
    save(out/'verification.json',result)
    print(canonical(result))


if __name__=='__main__':main()
