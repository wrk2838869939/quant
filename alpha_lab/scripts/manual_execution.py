"""生成邮件草稿，记录人工核对/未执行/实际成交；不发送邮件。"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from manual_execution import ManualAccount,bulletin
from shadow_planner import canonical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    m=subs.add_parser('draft')
    for key in ['plan','metadata','output']:m.add_argument('--'+key,type=Path,required=True)
    m.add_argument('--historical',action='store_true')
    m=subs.add_parser('prepare')
    for key in ['db','bundle','calendar','output']:m.add_argument('--'+key,type=Path,required=True)
    m.add_argument('--config',type=Path,default=ROOT/'configs/shadow_execution.yaml')
    for cmd in ['review','skip','fill','snapshot','terminal','adjustment']:
        m=subs.add_parser(cmd);m.add_argument('--db',type=Path,required=True);m.add_argument('--input',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':
        account=ManualAccount(args.db)
        try:
            result=account.prepare(json.loads(args.bundle.read_text()),json.loads(args.calendar.read_text()),
                yaml.safe_load(args.config.read_text()),pd.Timestamp.now(tz='UTC'))
            payload=canonical(result)
            if args.output.exists():
                if args.output.read_text()!=payload:raise ValueError('计划输出已存在不同内容')
            else:
                with args.output.open('x') as target:target.write(payload)
            print(canonical(dict(status='planned_for_manual_review',plan_id=result['plan_id'],submitted=False)))
        finally:account.close()
        return
    if args.command=='draft':
        plan=json.loads(args.plan.read_text());metadata=json.loads(args.metadata.read_text())
        cfg=yaml.safe_load((ROOT/'configs/manual_execution.yaml').read_text())
        result=bulletin(plan,metadata,pd.Timestamp.now(tz='UTC'),cfg,backfill=args.historical)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'briefing.txt').write_text(result['body'])
        (args.output/'briefing.eml').write_bytes(result.pop('eml'))
        (args.output/'receipt.json').write_text(canonical(result))
        print(canonical(dict(status='draft_written',historical=result['historical'],mail_sent=False)))
        return
    item=json.loads(args.input.read_text());a=ManualAccount(args.db)
    try:
        if args.command=='review':result=a.review_order(item['order_id'],item['review'])
        elif args.command=='skip':result=a.skip_order(item['order_id'],item['evidence'])
        elif args.command=='fill':result=a.record_fill(item)
        elif args.command=='snapshot':result=a.import_snapshot(item)
        elif args.command=='adjustment':result=a.adjustment(item)
        else:result=a.terminal(item['order_id'],item['status'],item['evidence'])
        print(canonical(result))
    finally:a.close()


if __name__=='__main__':main()
