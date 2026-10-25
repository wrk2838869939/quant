"""影子账户CLI：生成计划、盘前检查、导入成交和收盘核对，不提交任何委托。"""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shadow_account import ShadowAccount
from shadow_inputs import build_bundle
from shadow_planner import canonical, ShadowError


def load(path):
    return json.loads(Path(path).read_text())


def output(path, data):
    """同内容幂等导出，已有不同内容不覆盖；同目录原子发布。"""
    path = Path(path)
    payload = canonical(data) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        temporary = handle.name
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != payload:
                raise ShadowError('导出目标存在不同内容，拒绝覆盖')
    finally:
        os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path)
    subs = parser.add_subparsers(dest='command', required=True)
    for command in ['snapshot','fill','preflight','adjustment']:
        part = subs.add_parser(command); part.add_argument('--input', required=True)
    part = subs.add_parser('terminal')
    part.add_argument('--order-id', required=True)
    part.add_argument('--status', required=True, choices=['cancelled','rejected','expired'])
    part.add_argument('--evidence', required=True)
    part = subs.add_parser('plan')
    part.add_argument('--bundle', required=True); part.add_argument('--calendar', required=True)
    part.add_argument('--config', default=str(ROOT/'configs/shadow_execution.yaml'))
    part.add_argument('--output', required=True)
    subs.add_parser('status')
    part = subs.add_parser('build-bundle')
    for name in ['feature-panel','market','basics','st','actions','calendar','model','snapshot','receipt','output']:
        part.add_argument('--'+name, required=True)
    part.add_argument('--config', default=str(ROOT/'configs/shadow_execution.yaml'))
    part.add_argument('--index-close', help='新候选必需：parquet，date与close列，沪深300单指数')
    args = parser.parse_args()
    if args.command == 'build-bundle':
        snapshot = load(args.snapshot); date = snapshot['asof']
        filtered = [('date','<=',pd.Timestamp(date))]
        raw = pd.read_parquet(args.feature_panel, columns=['date','ts_code','hfq_close','total_mv_yuan','turnover_rate'], filters=filtered)
        market = pd.read_parquet(args.market, columns=['date','ts_code','close','vol_share'], filters=filtered)
        index = None
        if args.index_close:
            from monthly_risk_live import read_index_close
            from execution_market_feed import file_hash
            if file_hash(args.index_close) != load(args.receipt).get('market_index', {}).get('sha256'):
                raise ShadowError('指数文件与获取回执散列不一致')
            index = read_index_close(args.index_close)
        result = build_bundle(raw, market, pd.read_parquet(args.basics), pd.read_parquet(args.st),
            pd.read_parquet(args.actions), date, load(args.calendar), yaml.safe_load(Path(args.config).read_text()),
            load(args.model), snapshot['positions'], load(args.receipt),
            index)
        output(args.output, result)
        print(canonical(dict(status='bundle_written', asof=date, signal_date=result['signal_date'], live_submission=False)))
        return
    if not args.db:
        parser.error('账户命令需要--db')
    account = ShadowAccount(args.db)
    try:
        if args.command == 'snapshot': result = account.import_snapshot(load(args.input))
        elif args.command == 'fill': result = account.record_fill(load(args.input))
        elif args.command == 'adjustment': result = account.adjustment(load(args.input))
        elif args.command == 'terminal': result = account.terminal(args.order_id, args.status, args.evidence)
        elif args.command == 'preflight':
            item = load(args.input); result = account.preflight(item['plan_id'], item['market'])
        elif args.command == 'status': result = account.status()
        else:
            result = account.plan(load(args.bundle), load(args.calendar), yaml.safe_load(Path(args.config).read_text()))
            output(args.output, result)
        print(canonical(result))
    finally:
        account.close()


if __name__ == '__main__':
    main()
