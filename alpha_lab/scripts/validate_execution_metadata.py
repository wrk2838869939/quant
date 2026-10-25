"""核验元数据原件、最新观察包、拒绝过窗的真实CLI和已有封存。"""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from execution_metadata import inspect_metadata
from execution_market_feed import file_hash
from shadow_planner import ShadowError,fingerprint


def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--observation',type=Path,required=True)
    args=p.parse_args()
    out=ROOT/'outputs/execution_metadata'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False);print(str(out),flush=True)
    counts={}
    for name in ['shadow_execution/20260914T054517_770961Z','live_ingestion/20260914T060330_024634Z']:
        prior=ROOT/'outputs'/name
        record=json.loads((prior/'completion.json').read_text())
        for file,digest in record['artifact_sha256'].items():assert file_hash(prior/file)==digest,file
        counts[name]=len(record['artifact_sha256'])
    old=json.loads((ROOT/'outputs/shadow_execution/20260914T054517_770961Z/verification.json').read_text())
    assert file_hash(ROOT.parent/'AGENTS.md')==old['root_agents_sha256']
    for name,digest in old['source_hashes'].items():assert file_hash(ROOT/name)==digest
    report,data=inspect_metadata(args.stage)
    stored=json.loads((args.observation/'metadata_report.json').read_text())
    assert report==stored
    bundle=json.loads((args.observation/'bundle.json').read_text())
    manifest=json.loads((args.observation/'manifest.json').read_text())
    assert manifest['historical_backfill'] and fingerprint(bundle)==manifest['bundle_sha256']
    for name,digest in manifest['files_sha256'].items():assert file_hash(name)==digest
    assert report['first_available_at']>bundle['asof']
    assert report['current_market_status_complete'] is False
    assert report['action_scope']==sorted(bundle['signals'])
    # 持续空接口和损坏原件必须保留不可用，不能用布尔完成标记覆盖。
    with tempfile.TemporaryDirectory() as temp:
        copy=Path(temp)/'stage';shutil.copytree(args.stage,copy)
        first=json.loads((copy/'receipt.json').read_text())['requests'][0]['file']
        with (copy/first).open('ab') as f:f.write(b'tampered')
        try:inspect_metadata(copy)
        except ShadowError:pass
        else:raise AssertionError('tampered file accepted')
    # 以原始行情独立核对本窗口候选唯一除息事件，只有同一数据商内部交叉核对。
    event=data['actions'].loc[data['actions'].ex_date.between('2026-09-09','2026-09-11')]
    market=ROOT/'outputs/live_ingestion/20260914_market_stage'
    checks=[]
    for row in event.itertuples():
        day=row.ex_date.strftime('%Y%m%d')
        prior_day=pd.Timestamp(row.ex_date)-pd.offsets.BDay(1)
        today=pd.read_parquet(market/('daily_'+day+'.parquet')).set_index('ts_code').loc[row.ts_code]
        previous=pd.read_parquet(market/('daily_'+prior_day.strftime('%Y%m%d')+'.parquet')).set_index('ts_code').loc[row.ts_code]
        theoretical=(previous.close-row.cash_div_tax)/(1+row.stk_div)
        error=float(today.pre_close-theoretical)
        assert abs(error)<=.011
        checks.append(dict(ts_code=row.ts_code,ex_date=str(row.ex_date.date()),cash_div_tax=row.cash_div_tax,
            previous_close=float(previous.close),reported_pre_close=float(today.pre_close),
            theoretical_ex_price=float(theoretical),rounding_error=error,independent_provider=False))
    save(out/'corporate_action_price_checks.json',checks)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','test_execution_metadata','test_broker_csv',
        'test_execution_market_feed','test_shadow_execution','test_shadow_inputs','-v'],
        capture_output=True,text=True,env=env)
    (out/'tests.log').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
    n=int(re.search(r'Ran (\d+) tests',tests.stderr).group(1))
    snapshot=dict(account_id='synthetic_guard_test',asof=bundle['asof'],positions={},cash=50000,
        equity=50000,receivables=0,frozen_cash=0,open_orders=[],source='synthetic verification')
    save(out/'synthetic_snapshot.json',snapshot)
    run=subprocess.run([sys.executable,str(ROOT/'scripts/guarded_shadow_plan.py'),'--observation',str(args.observation),
        '--snapshot',str(out/'synthetic_snapshot.json'),'--db',str(out/'must_not_be_created.sqlite')],
        capture_output=True,text=True)
    (out/'guard_cli.log').write_text(run.stdout+run.stderr)
    assert run.returncode==2 and not (out/'must_not_be_created.sqlite').exists()
    assert '错过' in json.loads(run.stdout)['reason']
    sources=['src/execution_metadata.py','scripts/stage_execution_metadata.py','scripts/build_execution_observation.py',
        'scripts/guarded_shadow_plan.py','scripts/validate_execution_metadata.py','tests/test_execution_metadata.py']
    for name in sources:
        target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copytree(args.stage,out/'raw_metadata');shutil.copytree(args.observation,out/'observation')
    result=dict(status='passed',tests_passed=n,prior_artifacts_verified=counts,root_agents_unchanged=True,
        real_account_supplied=False,live_orders=False,strategy_admitted=False,
        raw_hash_tampering_rejected=True,expired_cli_blocked_without_database=True,
        metadata_report=report,comparison=json.loads((args.observation/'comparison.json').read_text()),
        source_sha256={name:file_hash(ROOT/name) for name in sources})
    save(out/'verification.json',result);print(json.dumps(dict(tests=n,output=str(out)),ensure_ascii=False))


if __name__=='__main__':main()
