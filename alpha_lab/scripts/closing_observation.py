"""14:30后采集全A快照并构建收盘竞价观察包；不下单、不发送邮件。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import json
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from closing_projection import parse_sina, project, proxy_receipt, mark_account
from execution_clock import check_proxy, local_stamp
from execution_market_feed import file_hash
from monthly_risk_live import read_index_close
from shadow_inputs import build_bundle
from shadow_planner import canonical, fingerprint, ShadowError


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(canonical(data))


def verify(directory):
    manifest = read(directory/'manifest.json')
    for name, digest in manifest['output_sha256'].items():
        if file_hash(directory/name) != digest:
            raise ShadowError('已封存输入发生变化：'+name)
    return manifest


def status_input(path, date, now):
    """全市场停牌/证券清单由有时点的外部状态回执提供；不能猜测空返回。"""
    status = read(path)
    at = local_stamp(status.get('timestamp'))
    if (status.get('asof') != date or status.get('complete') is not True
            or not status.get('source') or not status.get('evidence')
            or str(at.date()) != date or at > local_stamp(now)):
        raise ShadowError('缺当天已获取的证券状态清单')
    for key in ['universe', 'suspended']:
        if len(status[key]) != len(set(status[key])):
            raise ShadowError('状态清单代码重复')
    if not set(status['suspended']) <= set(status['universe']):
        raise ShadowError('停牌清单不在证券全集中')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('status')
    p.add_argument('--metadata-stage', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('account-snapshot')
    for name in ['broker','quotes','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p = sub.add_parser('collect')
    p.add_argument('--status', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('build')
    for name in ['previous','quotes','basis','index-history','model','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--snapshot', type=Path)
    p.add_argument('--config', type=Path, default=ROOT/'configs/monthly_risk_close_day15.yaml')
    args = parser.parse_args()
    now = pd.Timestamp.now(tz='Asia/Shanghai')
    if args.command == 'account-snapshot':
        verify(args.quotes)
        snapshot=mark_account(read(args.broker),pd.read_parquet(args.quotes/'quotes.parquet'),
            read(args.quotes/'receipt.json'),now)
        with args.output.open('x') as f:f.write(canonical(snapshot))
        print(canonical(dict(status='marked_account_snapshot',live_orders=False)))
        return
    if args.command == 'status':
        from execution_metadata import inspect_metadata
        report, data = inspect_metadata(args.metadata_stage)
        date = str(now.date())
        if (report['observation_date'] != date or report['price_date'] != date
                or not report['current_market_status_complete']
                or local_stamp(report['first_available_at']) > now):
            raise ShadowError('必须使用当天已完整取得的证券元数据')
        source = read(args.metadata_stage/'receipt.json')
        def raw(api):
            matches=[r for r in source['requests'] if r['api']==api]
            if api == 'dividend':
                matches=[r for r in matches if r['params'].get('ex_date')==date.replace('-','')]
            if len(matches)!=1:raise ShadowError('状态原件不唯一')
            return pd.read_parquet(args.metadata_stage/matches[0]['file'])
        suspended=raw('suspend_d')
        # 分时临停不是全天停牌，未知时间描述不能简化成整日删除。
        if (not suspended.suspend_type.isin(['S','R']).all()
                or suspended.suspend_timing.fillna('').ne('').any()):
            raise ShadowError('存在分时/未知停牌安排，需要核对当前可交易范围')
        master=data['basics']
        universe=sorted(master.loc[master.list_date.le(date)&(master.delist_date.isna()|master.delist_date.gt(date)),'ts_code'])
        status=dict(asof=date,timestamp=report['first_available_at'],complete=True,
            source='configured Tushare timestamped security metadata',
            evidence=str((args.metadata_stage/'receipt.json').resolve()),
            metadata_sha256=report['receipt_sha256'],universe=universe,
            suspended=sorted(suspended.loc[suspended.suspend_type.eq('S'),'ts_code']),
            st_codes=sorted(data['st'][date].ts_code.tolist()),
            corporate_action_codes=sorted(raw('dividend').ts_code.unique()),
            basics=json.loads(master.to_json(orient='records',date_format='iso')))
        args.output.mkdir(parents=True,exist_ok=False)
        write(args.output/'status.json',status)
        write(args.output/'manifest.json',dict(output_sha256={'status.json':file_hash(args.output/'status.json')},live_orders=False))
        print(canonical(dict(status='security_status_prepared',asof=date,universe=len(universe),live_orders=False)))
        return
    if args.command == 'collect':
        if not '14:30:00' <= now.strftime('%H:%M:%S') < '14:53:00':
            raise ShadowError('实际采集只在14:30至14:53启动，预留计算与人工操作时间')
        date = str(now.date()); status = status_input(args.status, date, now)
        codes = sorted(set(status['universe']) - set(status['suspended'])) + ['000300.SH']
        args.output.mkdir(parents=True, exist_ok=False)
        symbols = [c.split('.')[1].lower()+c.split('.')[0] for c in codes]
        batches = [symbols[i:i+80] for i in range(0, len(symbols), 80)]
        def fetch(item):
            i, batch = item
            request = Request('https://hq.sinajs.cn/list='+','.join(batch),
                headers={'Referer':'https://finance.sina.com.cn', 'User-Agent':'Mozilla/5.0'})
            with urlopen(request, timeout=8) as response:
                raw = response.read()
            (args.output/f'raw_{i:03d}.txt').write_bytes(raw)
            return parse_sina(raw.decode('gbk'))
        with ThreadPoolExecutor(max_workers=3) as pool:
            quotes = pd.concat(list(pool.map(fetch, enumerate(batches))), ignore_index=True)
        received = pd.Timestamp.now(tz='Asia/Shanghai')
        if quotes.ts_code.duplicated().any() or set(quotes.ts_code) != set(codes):
            raise ShadowError('接口未完整返回全A证券与指数，不生成有效回执')
        receipt = proxy_receipt(quotes, date, received.isoformat(), 'Sina timestamped raw snapshots')
        quotes.to_parquet(args.output/'quotes.parquet', index=False)
        write(args.output/'receipt.json', receipt)
        write(args.output/'status.json', status)
        write(args.output/'manifest.json', dict(output_sha256={f.name:file_hash(f) for f in args.output.iterdir()},
            input_sha256={str(args.status):file_hash(args.status)}, live_orders=False))
        print(canonical(dict(status='captured', asof=date, rows=len(quotes), live_orders=False)))
        return
    previous = verify(args.previous); verify(args.quotes)
    receipt = read(args.quotes/'receipt.json'); date = receipt['asof']
    check_proxy(receipt, date, now)
    status = status_input(args.quotes/'status.json', date, now)
    cfg = yaml.safe_load(args.config.read_text())
    if cfg['execution_session'] != 'closing_auction':
        raise ShadowError('只能使用收盘竞价配置')
    old_bundle = read(args.previous/'bundle.json')
    if old_bundle.get('input_kind') == 'intraday_proxy':
        if old_bundle['strategy_config'] != fingerprint(cfg):
            raise ShadowError('盘中观察历史配置变化，不静默迁移')
        if file_hash(args.index_history) != previous['output_sha256']['index_close.parquet']:
            raise ShadowError('必须延续已封存的盘中指数历史')
        receipt['prior_index_is_proxy'] = True
    basis = pd.read_parquet(args.basis)
    basics = pd.DataFrame(status['basics']) if status.get('basics') else pd.read_parquet(args.previous/'basics.parquet')
    for column in ['list_date','delist_date']:
        basics[column]=pd.to_datetime(basics[column])
    # 开始前的主表不能证明之后IPO/退市状态；当天证券全集是额外必需输入。
    if set(basis.ts_code) != set(status['universe']):
        raise ShadowError('盘前股本依据与当天证券全集不一致，先补齐/核对变更')
    if not set(status['universe']) <= set(basics.ts_code):
        raise ShadowError('证券主表缺当日股票，先更新已知主表')
    quotes = pd.read_parquet(args.quotes/'quotes.parquet')
    calendar = read(args.previous/'calendar.json')
    history, index, market = project(pd.read_parquet(args.previous/'feature_panel.parquet'),
        read_index_close(args.index_history), basis, quotes, receipt, calendar, status['suspended'])
    # 历史ST观察只追加今天，旧复核日状态不会被今日名称改写。
    st = pd.read_parquet(args.previous/'st.parquet')
    names = set(status.get('st_codes',[])) | set(quotes.loc[quotes.name.str.contains('ST', case=False, regex=False), 'ts_code'])
    extra = pd.DataFrame([dict(ts_code=c, start=pd.Timestamp(date), end=pd.Timestamp(date),
        known_date=pd.Timestamp(date)) for c in names])
    if len(extra):
        st = pd.concat([st, extra], ignore_index=True)
    actions = pd.read_parquet(args.previous/'actions.parquet')
    if status.get('corporate_action_codes'):
        actions=pd.concat([actions,pd.DataFrame([dict(ts_code=c,known_date=pd.Timestamp(date),ex_date=pd.Timestamp(date))
            for c in status['corporate_action_codes']])],ignore_index=True)
    snapshot = read(args.snapshot) if args.snapshot else None
    bundle = build_bundle(history, market, basics, st, actions, date, calendar, cfg,
        read(args.model), snapshot['positions'] if snapshot else [], receipt, index)
    check_proxy(receipt, date, pd.Timestamp.now(tz='Asia/Shanghai'))
    args.output.mkdir(parents=True, exist_ok=False)
    history.to_parquet(args.output/'feature_panel.parquet', index=False)
    market.to_parquet(args.output/'market.parquet', index=False)
    pd.DataFrame(dict(date=index.index, ts_code='000300.SH', close=index.values)).to_parquet(args.output/'index_close.parquet', index=False)
    for name, data in [('basics',basics), ('st',st), ('actions',actions)]:
        data.to_parquet(args.output/(name+'.parquet'), index=False)
    for name, data in [('bundle',bundle), ('calendar',calendar), ('receipt',receipt),
                       ('metadata_report',dict(all_corporate_actions_covered=False,
                         blockers=['逐股核对当日ST/停牌/公司行动/限价；自动覆盖不完整']))]:
        write(args.output/(name+'.json'), data)
    sources = [args.basis, args.index_history, args.model, args.config,
        args.previous/'manifest.json', args.quotes/'manifest.json']
    if args.snapshot: sources.append(args.snapshot)
    write(args.output/'manifest.json', dict(asof=date, input_kind='intraday_proxy',
        generated_at=pd.Timestamp.now(tz='UTC').isoformat(),
        input_sha256={str(f):file_hash(f) for f in sources},
        output_sha256={f.name:file_hash(f) for f in args.output.iterdir()},
        live_orders=False, source_vintage_externally_certified=False))
    print(canonical(dict(status='proxy_observation_built', asof=date,
        candidates=len(bundle['signals']), is_review=bundle['is_review'], live_orders=False)))


if __name__ == '__main__':
    main()
