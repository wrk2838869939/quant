#!/usr/bin/env python3
"""按冻结开发日历补官方限价；两并发、断点缓存、失败不泄露远端异常。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    lo, hi = map(pd.Timestamp, cfg['history'])
    if hi >= pd.Timestamp('2024-01-01'):
        raise ValueError('禁止TEST日历')
    raw = pd.read_parquet(ROOT / 'data/panel/daily_panel.parquet', columns=['date'],
        filters=[('date', '>=', lo), ('date', '<=', hi)])
    dates = pd.DatetimeIndex(sorted(raw.date.unique()))
    assert dates.max() <= hi and dates.min() >= lo
    del raw
    cache = ROOT / 'data/raw/policy_limits_by_date'
    cache.mkdir(exist_ok=True)
    from ts_client import get_pro
    spec = cfg['limits_api']
    # 代理忽略offset，禁止把5800条截断响应当全量；按事前冻结研究代码查询。
    features = pd.read_parquet(run / 'policy_features.parquet', columns=['ts_code'])
    requested_codes = sorted(features.ts_code.unique())
    if len(requested_codes) >= spec['row_limit']:
        raise ValueError('研究代码数达到上限，须显式分块')
    requested_set = set(requested_codes)
    def worker(date):
        path = cache / f'{date:%Y%m%d}.parquet'
        try:
            if path.exists():
                frame = pd.read_parquet(path)
            else:
                frame = get_pro().stk_limit(trade_date=date.strftime('%Y%m%d'), ts_code=','.join(requested_codes))
                if frame is None or not len(frame):
                    raise ValueError('空限价响应')
                if len(frame) >= spec['row_limit']:
                    raise ValueError('响应可能截断')
                if not pd.to_datetime(frame.trade_date).eq(date).all():
                    raise ValueError('限价日期越界')
                if frame.ts_code.duplicated().any():
                    raise ValueError('限价代码重复')
                if not set(frame.ts_code).issubset(requested_set):
                    raise ValueError('代码范围参数未生效')
                frame.to_parquet(path.with_suffix('.tmp.parquet'), index=False)
                path.with_suffix('.tmp.parquet').replace(path)
                time.sleep(spec['interval_seconds'])
            return dict(date=str(date.date()), rows=len(frame), status='ok',
                path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception as error:
            return dict(date=str(date.date()), rows=0, status='failed', error=type(error).__name__)
    results = []
    with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
        pending = [pool.submit(worker, date) for date in dates]
        for index, future in enumerate(as_completed(pending), 1):
            results.append(future.result())
            if index % 100 == 0 or index == len(dates):
                pd.DataFrame(results).sort_values('date').to_csv(run / 'limits_download.csv', index=False)
                print(f'limits {index}/{len(dates)} failures={sum(r["status"] != "ok" for r in results)}', flush=True)
    failures = sum(r['status'] != 'ok' for r in results)
    (run / 'limits_manifest.json').write_text(json.dumps(dict(status='complete' if not failures else 'incomplete',
        source='Tushare stk_limit by trade_date; uncached requests scoped to frozen opportunity codes',
        requested_codes=len(requested_codes), retained_earlier_full_market_nontruncated_cache=True,
        ignored_offset_probe='limit_pagination_probe', dates=len(dates), failures=failures,
        data_bounds=[str(lo.date()), str(hi.date())], test_market_data_used=False,
        downloader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    if failures:
        raise RuntimeError(f'{failures} dates incomplete; keep unavailable explicit')


if __name__ == '__main__':
    main()
