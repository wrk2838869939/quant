"""在检查当前元数据与实际时刻后调用冻结计划器，检查失败不打开账户数据库。"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from execution_metadata import inspect_metadata,guard_plan
from execution_market_feed import file_hash
from shadow_account import ShadowAccount
from shadow_planner import ShadowError,canonical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['observation','snapshot','db']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    try:
        manifest=json.loads((args.observation/'manifest.json').read_text())
        for name,digest in manifest['files_sha256'].items():
            if file_hash(name)!=digest:raise ShadowError('观察包依赖文件改变')
        metadata,_=inspect_metadata(manifest['metadata_stage'])
        if metadata['receipt_sha256']!=manifest['metadata_receipt_sha256']:
            raise ShadowError('元数据回执改变')
        bundle=json.loads((args.observation/'bundle.json').read_text())
        calendar=json.loads((args.observation/'calendar.json').read_text())
        snapshot=json.loads(args.snapshot.read_text())
        guard_plan(bundle,snapshot,calendar,metadata,pd.Timestamp.now(tz='UTC'),manifest)
        account=ShadowAccount(args.db)
        try:
            if account.state()['snapshot']!=snapshot:raise ShadowError('快照不是已核对账户状态')
            plan=account.plan(bundle,calendar,yaml.safe_load((ROOT/'configs/shadow_execution.yaml').read_text()))
        finally:account.close()
        print(canonical(dict(status='planned',plan_id=plan['plan_id'],live_submission=False)))
        return 0
    except (ShadowError,ValueError,KeyError) as exc:
        print(canonical(dict(status='blocked',reason=str(exc),live_submission=False)))
        return 2


if __name__=='__main__':sys.exit(main())
