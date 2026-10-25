#!/usr/bin/env python3
"""实测细分板块接口，保存原始响应和摘要；凭证不进入输出。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import time
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import ts_client


def main():
    config = ROOT / 'configs/fine_sector_probe_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    run = ROOT / 'outputs/fine_sector_probe' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    (run / config.name).write_text(config.read_text())
    client = ts_client.get_pro()
    client._DataApi__timeout = cfg['timeout_seconds']
    print('RUN=' + str(run), flush=True)

    def query(q):
        started = time.monotonic()
        row = dict(q, retrieved_at=datetime.now(timezone.utc).isoformat())
        try:
            df = client.query(q['api'], **q['params'])
            if df is None:
                raise ValueError('empty_response_object')
            path = run / (q['name'] + '.parquet')
            df.to_parquet(path, index=False)
            row.update(status='ok', rows=len(df), columns=list(df),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for field in ['trade_date', 'in_date', 'out_date']:
                if field in df and df[field].notna().any():
                    s = df[field].dropna().astype(str)
                    row[field + '_range'] = [s.min(), s.max()]
        except Exception as error:
            # 不输出可能包含服务地址、凭证或完整请求的异常文本。
            message = str(error)
            category = ('permission' if any(x in message for x in ['权限', '积分', 'permission']) else
                        'timeout' if 'timed out' in message.lower() else 'request_error')
            row.update(status='error', error_type=type(error).__name__, error_category=category)
        row['seconds'] = round(time.monotonic() - started, 3)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        rows = list(pool.map(query, cfg['queries']))
    (run / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (run / 'manifest.json').write_text(json.dumps(dict(
        status='probe_complete', backtest_ready=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
