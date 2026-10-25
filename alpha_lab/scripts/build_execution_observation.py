"""重建最新收盘观察包；回补数据仅供核对，不生成账户订单。"""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import sys

import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from execution_metadata import inspect_metadata
from execution_market_feed import file_hash
from shadow_inputs import build_bundle
from shadow_planner import fingerprint, canonical


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--market-stage',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report,data=inspect_metadata(args.metadata)
    (args.output/'metadata_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    for name in ['basics','actions','quarantine']:data[name].to_parquet(args.output/(name+'.parquet'),index=False)
    print(canonical(report),flush=True)
    package=json.loads((args.market_stage/'validated/package.json').read_text())
    for name,digest in package['artifacts'].items():
        if file_hash(args.market_stage/'validated'/name)!=digest:raise ValueError('price artifact hash')
    old=ROOT/'outputs/recent_low_turnover/20260913T090336_282870Z/data_repair/repaired_feature_projection.parquet'
    feature=pd.read_parquet(old)
    increment=pd.read_parquet(args.market_stage/'validated/feature_increment.parquet')
    if feature.date.max()>=increment.date.min():raise ValueError('历史与增量重叠')
    raw=pd.concat([feature,increment],ignore_index=True)
    market=pd.read_parquet(args.market_stage/'validated/market_increment.parquet')
    date=package['asof']
    unknown=set(raw.ts_code)-set(data['basics'].ts_code)
    # 旧代码历史仍保留；若信号日至今仍有报价，则缺主表必须阻断。
    recent=set(raw.loc[raw.date.ge(pd.Timestamp(report['price_date'])-pd.Timedelta(days=200)),'ts_code'])
    if unknown & recent:raise ValueError('主表未覆盖近期有报价代码')
    missing_history=raw.loc[raw.ts_code.isin(unknown)].groupby('ts_code').date.max()
    (args.output/'inactive_missing_basics.json').write_text(json.dumps(
        {c:str(d.date()) for c,d in missing_history.items()},ensure_ascii=False,indent=2))
    oldcalendar=json.loads((ROOT/'outputs/shadow_execution/20260914T054517_770961Z/calendar.json').read_text())
    sessions=sorted(set([d for d in oldcalendar['sessions'] if d<'2026-01-01']+data['sessions']))
    calendar=dict(source=report['calendar_basis'],sessions=sessions)
    st=pd.concat([frame.assign(start=pd.Timestamp(day),end=pd.Timestamp(day),known_date=pd.Timestamp(day))[
        ['ts_code','start','end','known_date']] for day,frame in data['st'].items() if day<=date],ignore_index=True)
    # 此处有效日为历史快照查询日，实际首次获取时刻在manifest中单独记录。
    # 不能将此观察包用于宣称历史可见或真实前向。
    modelpath=ROOT/'outputs/shadow_execution/20260914T054517_770961Z/model_2026.json'
    model=json.loads(modelpath.read_text())
    receipt=dict(asof=date,complete=True,source='verified price increment; retrospective metadata snapshots',
        downloaded_at=report['first_available_at'],expected_rows=int(raw.date.eq(pd.Timestamp(date)).sum()))
    config=yaml.safe_load((ROOT/'configs/shadow_execution.yaml').read_text())
    bundle=build_bundle(raw,market,data['basics'],st,data['actions'],date,calendar,config,model,[],receipt)
    previous=json.loads((ROOT/'outputs/shadow_execution/20260914T054517_770961Z/example_rebuilt_bundle.json').read_text())
    comparison=dict(previous_signal_date=previous['signal_date'],signal_date=bundle['signal_date'],
        rank_order_same=list(previous['signals'])==list(bundle['signals']),
        new_candidates=sorted(set(bundle['signals'])-set(previous['signals'])),
        removed_candidates=sorted(set(previous['signals'])-set(bundle['signals'])),
        max_shared_score_change=max(abs(bundle['signals'][c]-previous['signals'][c]) for c in set(bundle['signals'])&set(previous['signals'])),
        historical_backfill=True,real_account_supplied=False)
    for name,value in [('bundle.json',bundle),('calendar.json',calendar),('comparison.json',comparison)]:
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    manifest=dict(historical_backfill=True,generated_at=datetime.now(timezone.utc).isoformat(),
        bundle_sha256=fingerprint(bundle),snapshot_sha256=None,metadata_receipt_sha256=report['receipt_sha256'],
        metadata_stage=str(args.metadata.resolve()),price_asof=date,real_account_supplied=False,
        files_sha256={str(p.resolve()):file_hash(p) for p in [old,modelpath,args.market_stage/'validated/package.json',
            args.market_stage/'validated/feature_increment.parquet',args.market_stage/'validated/market_increment.parquet',
            ROOT/'configs/shadow_execution.yaml']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(canonical(comparison),flush=True)


if __name__=='__main__':main()
