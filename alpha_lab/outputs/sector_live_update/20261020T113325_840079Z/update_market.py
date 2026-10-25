#!/usr/bin/env python3
"""按交易日补齐实时研究行情，保留冻结版本，拒绝将 HTTP 错误缓存成空日。"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import json
import time
import shutil
import os
import argparse
import requests
import numpy as np
import pandas as pd
import yaml
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / 'alpha_lab/src'))
sys.path.insert(0, str(PROJECT / 'datahub/scripts'))
sys.path.insert(0, str(PROJECT / 'datahub/src'))
from ts_client import get_pro
from update_data import transform_raw
from build_dataset import CN_COLUMNS, add_board
from sector_dataset import digest
from datahub.frozen import create_snapshot


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def query(api, cfg, **params):
    """直接访问标准 POST 入口，避免本机改版 SDK 自动追加接口名。"""
    credential = getattr(get_pro(), '_DataApi__token')
    for attempt in range(cfg['retries']):
        try:
            response = requests.post(cfg['endpoint'], json=dict(api_name=api, token=credential, params=params, fields=''), timeout=cfg['timeout_seconds'])
            response.raise_for_status()
            payload = response.json()
            if payload.get('code') != 0:
                message = str(payload.get('msg', '')).replace(credential, '[redacted]')
                raise RuntimeError(f'{api}: {message[:150]}')
            value = payload['data']
            return pd.DataFrame(value['items'], columns=value['fields'])
        except Exception:
            if attempt + 1 == cfg['retries']:
                # 部分接口的旧分路仍可用；新标准入口失败时明确尝试兼容分路。
                pro = get_pro()
                pro._DataApi__timeout = cfg['timeout_seconds']
                result = getattr(pro, api)(**params)
                if result.empty:
                    raise RuntimeError(f'{api} 标准入口和兼容分路均无有效数据')
                return result
            time.sleep(1 + attempt)


def atomic_parquet(frame, path):
    tmp = path.with_suffix('.updating.parquet')
    frame.to_parquet(tmp, index=False, compression='zstd')
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--config', type=Path, default=HERE / 'config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    # 盘中不发布当日未完成的日线。
    now = pd.Timestamp.now(tz='Asia/Shanghai')
    assert pd.Timestamp(cfg['asof']).date() < now.date() or (pd.Timestamp(cfg['asof']).date() == now.date() and now.hour >= 17), '当日日线尚未到完整发布检查时点'
    out = args.resume.resolve() if args.resume else PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if args.resume:
        assert json.loads((out / 'manifest.json').read_text())['status'] == 'building'
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    print('OUTPUT', out, flush=True)
    shutil.copy2(args.config, out / 'config.yaml')
    shutil.copy2(Path(__file__), out / 'update_market.py')
    write_json(out / 'manifest.json', dict(status='building', config=cfg))
    calendar = pd.read_parquet(PROJECT / 'datahub/datasets/calendar/cn_calendar.parquet')
    dates = pd.to_datetime(calendar.loc[pd.to_numeric(calendar.is_open).eq(1), 'cal_date'])
    days = dates[(dates > cfg['previous_end']) & (dates <= cfg['asof'])].dt.strftime('%Y%m%d').tolist()
    assert days and days[-1] == cfg['asof'].replace('-', '')
    raw = out / 'raw'
    raw.mkdir(exist_ok=True)
    frames, rows = {}, []
    def cached_query(api, day):
        path = raw / f'{api}_{day}.parquet'
        if path.exists():
            value = pd.read_parquet(path)
            if not value.empty:
                return value
        return query(api, cfg, trade_date=day)
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs = {pool.submit(cached_query, api, day): (api, day) for day in days for api in cfg['apis']}
        for future in as_completed(jobs):
            api, day = jobs[future]
            frame = future.result()
            if frame.empty:
                raise RuntimeError(f'{api} {day} 返回空表，未发布更新')
            assert frame.trade_date.astype(str).eq(day).all()
            assert not frame.duplicated(['ts_code', 'trade_date']).any()
            frame.to_parquet(raw / f'{api}_{day}.parquet', index=False)
            frames[api, day] = frame
            rows.append(dict(api=api, date=day, rows=len(frame)))
            print('FETCH', api, day, len(frame), flush=True)
    assembled, quality = [], []
    for day in days:
        daily = frames['daily', day].copy()
        invalid = daily[['open', 'high', 'low', 'close']].le(0).any(axis=1) | daily[['open', 'high', 'low', 'close']].isna().any(axis=1)
        # 代理有时返回停牌占位：开高低全零、无成交，不能把沿用收盘当真实报价。
        placeholders = daily[['open', 'high', 'low']].eq(0).all(axis=1) & daily.vol.eq(0) & daily.amount.eq(0)
        assert not (invalid & ~placeholders).any(), f'{day} 存在非停牌占位的异常价格'
        if placeholders.any():
            daily[placeholders].to_parquet(out / f'quarantined_nontrading_{day}.parquet', index=False)
        daily = daily[~placeholders].copy()
        adj = frames['adj_factor', day][['ts_code', 'trade_date', 'adj_factor']]
        basic = frames['daily_basic', day].drop(columns='close', errors='ignore')
        money = frames['moneyflow', day]
        assert len(daily) >= cfg['minimum_daily_rows']
        frame = daily.merge(adj, on=['ts_code', 'trade_date'], how='left', validate='one_to_one').merge(
            basic, on=['ts_code', 'trade_date'], how='left', validate='one_to_one').merge(
            money, on=['ts_code', 'trade_date'], how='left', validate='one_to_one')
        assert frame.adj_factor.gt(0).all()
        for col in ['open', 'high', 'low', 'close']:
            assert frame[col].gt(0).all()
        assert (frame.high + 1e-8 >= frame[['open', 'close', 'low']].max(axis=1)).all()
        assert (frame.low - 1e-8 <= frame[['open', 'close', 'high']].min(axis=1)).all()
        basic_coverage = (frame.total_mv.gt(0) & frame.float_share.gt(0)).mean()
        assert basic_coverage >= cfg['minimum_basic_coverage']
        assert frame.vol.ge(0).all() and frame.amount.ge(0).all()
        quality.append(dict(date=day, rows=len(frame), nontrading_placeholders_removed=int(placeholders.sum()), basic_coverage=basic_coverage,
                            turnover_coverage=frame.turnover_rate.notna().mean(), adjustment_coverage=frame.adj_factor.notna().mean()))
        assembled.append(add_board(transform_raw(frame))[CN_COLUMNS])
    delta = pd.concat(assembled, ignore_index=True)
    delta.to_parquet(out / 'cn_delta.parquet', index=False)
    pd.DataFrame(rows).to_csv(out / 'fetch_inventory.csv', index=False)
    pd.DataFrame(quality).to_csv(out / 'daily_quality.csv', index=False)
    # 当前主档只增加/刷新元数据；旧冻结主档保持不动。
    latest_basic = query('stock_basic', cfg, exchange='', list_status='L')
    assert len(latest_basic) >= cfg['minimum_daily_rows'] and latest_basic.ts_code.is_unique
    latest_basic.to_parquet(raw / 'stock_basic_L.parquet', index=False)
    hub = PROJECT / 'datahub'
    path = hub / 'datasets/cn_daily/year=2026/data.parquet'
    prior = pd.read_parquet(path)
    assert prior.date.max() <= pd.Timestamp(cfg['previous_end']), '活动行情已经更新，请核验而非覆盖'
    combined = pd.concat([prior, delta], ignore_index=True).sort_values(['ts_code', 'date'], kind='stable').reset_index(drop=True)
    assert not combined.duplicated(['ts_code', 'date']).any()
    old = combined[combined.date <= cfg['previous_end']].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, prior.sort_values(['ts_code', 'date'], kind='stable').reset_index(drop=True), check_dtype=False)
    backup = out / 'before_update'
    backup.mkdir()
    for target in [path, hub / 'datasets/meta/stock_basic.parquet', hub / 'build_manifest.json']:
        shutil.copy2(target, backup / target.name)
    master = pd.read_parquet(hub / 'datasets/meta/stock_basic.parquet')
    latest_basic = add_board(latest_basic).assign(list_status='L')
    # 当前接口未返回的主档列保留旧值，不用 NaN 覆盖既有资料。
    master = latest_basic.set_index('ts_code').combine_first(master.set_index('ts_code')).reset_index()
    atomic_parquet(combined, path)
    atomic_parquet(master, hub / 'datasets/meta/stock_basic.parquet')
    bm = json.loads((hub / 'build_manifest.json').read_text())
    bm.update(built_at=datetime.now(timezone.utc).isoformat(), updated_by='sector_live_v1/update_market.py',
              cn_daily_end=cfg['asof'], appended_sessions=len(days), update_log=str(out),
              other_datasets_note='本次更新A股日线四接口与当前在市主档，指数、美股和历史行业不变')
    write_json(hub / 'build_manifest.json', bm)
    print('FREEZE', cfg['release'], flush=True)
    frozen = create_snapshot(hub, PROJECT / cfg['release'])
    write_json(out / 'manifest.json', dict(status='complete', config=cfg, release=cfg['release'],
        content_id=frozen['content_id'], sessions=len(days), delta_rows=len(delta), asof=cfg['asof'], old_prefix_unchanged=True,
        artifacts={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    (HERE / 'LATEST_UPDATE').write_text(str(out.relative_to(PROJECT)) + '\n')
    print('COMPLETE', out, frozen['content_id'], flush=True)


if __name__ == '__main__':
    main()
