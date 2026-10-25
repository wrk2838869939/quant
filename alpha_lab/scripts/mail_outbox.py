"""邮件入队与受控发送；默认不发送，凭据仅在运行时读取。"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mail_outbox import MailOutbox,settings_from_env,settings_from_legacy
from shadow_planner import canonical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,required=True)
    p.add_argument('--legacy-settings',type=Path)
    subs=p.add_subparsers(dest='command',required=True)
    q=subs.add_parser('enqueue')
    q.add_argument('--plan',type=Path,required=True);q.add_argument('--metadata',type=Path,required=True)
    q=subs.add_parser('dispatch');q.add_argument('--id',required=True);q.add_argument('--send',action='store_true')
    subs.add_parser('status')
    q=subs.add_parser('resolve')
    q.add_argument('--id',required=True);q.add_argument('--outcome',choices=['accepted','retry'],required=True);q.add_argument('--evidence',required=True)
    args=p.parse_args();box=MailOutbox(args.db)
    try:
        if args.command=='status':result=box.status()
        elif args.command=='resolve':
            box.resolve(args.id,args.outcome,args.evidence);result=dict(status='resolved',mail_sent=False)
        elif args.command=='dispatch' and not args.send:
            result=dict(status='dry_run',mail_sent=False,queue=box.status())
        else:
            settings=settings_from_legacy(args.legacy_settings) if args.legacy_settings else settings_from_env()
            if args.command=='enqueue':
                result=dict(ids=box.enqueue(json.loads(args.plan.read_text()),json.loads(args.metadata.read_text()),
                    settings,pd.Timestamp.now(tz='UTC'),yaml.safe_load((ROOT/'configs/manual_execution.yaml').read_text())),mail_sent=False)
            else:result=box.dispatch(args.id,settings,pd.Timestamp.now(tz='UTC'))
        print(canonical(result))
        return 0
    except Exception as exc:
        print(canonical(dict(status='failed',error_type=type(exc).__name__,detail='请检查本地配置或发件箱状态；不输出认证信息。')))
        return 1
    finally:box.close()


if __name__=='__main__':sys.exit(main())
