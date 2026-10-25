#!/usr/bin/env python3
"""抓取二级行业历史归属，作为冻结行情的独立附加版本，不修改原快照。"""
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
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
import ts_client


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = ROOT / 'configs/left_side_rules_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    m = cfg['membership']
    probe = PROJECT / cfg['probe_run']
    run = ROOT / 'data/sector_supplements' / datetime.now(timezone.utc).strftime('sw_l2_%Y%m%dT%H%M%S_%fZ')
    (run / 'raw').mkdir(parents=True)
    print('SUPPLEMENT=' + str(run), flush=True)
    catalogs = []
    for name in ['sw_2014_l2', 'sw_2021_l2']:
        d = pd.read_parquet(probe / (name + '.parquet'))
        catalogs.append(d)
    catalog = pd.concat(catalogs, ignore_index=True)
    catalog.to_parquet(run / 'catalog.parquet', index=False)
    client = ts_client.get_pro()
    client._DataApi__timeout = m['timeout_seconds']

    def query(code):
        row = dict(index_code=code, retrieved_at=datetime.now(timezone.utc).isoformat())
        for attempt in range(m['retries']):
            try:
                df = client.query('index_member', index_code=code)
                row.update(rows=len(df), status='ok', attempts=attempt + 1)
                if len(df):
                    required = {'index_code', 'con_code', 'in_date', 'out_date'}
                    if not required <= set(df) or not df.index_code.eq(code).all():
                        raise ValueError('invalid_schema_or_code')
                df.to_parquet(run / 'raw' / (code + '.parquet'), index=False)
                return row
            except Exception as error:
                row.update(status='error', error_type=type(error).__name__, attempts=attempt + 1)
        return row

    codes = sorted(catalog.index_code.unique())
    rows = []
    with ThreadPoolExecutor(max_workers=m['workers']) as pool:
        for i, row in enumerate(pool.map(query, codes), 1):
            rows.append(row)
            if i % 20 == 0 or row['status'] != 'ok':
                print('PROGRESS', i, len(codes), row, flush=True)
    frames = [pd.read_parquet(run / 'raw' / (r['index_code'] + '.parquet'))
              for r in rows if r['status'] == 'ok' and r['rows']]
    if not frames:
        raise RuntimeError('无可用的历史细分行业归属')
    all_members = pd.concat(frames, ignore_index=True).drop_duplicates()
    all_members.to_parquet(run / 'members.parquet', index=False)
    (run / 'requests.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (run / config.name).write_text(config.read_text())
    payload = dict(status='complete', historical_coverage_certified=False,
                   board_level='SW_L2', retrieved_at=datetime.now(timezone.utc).isoformat(),
                   original_market_snapshot='selection_v1_20260920',
                   rows=len(all_members), stocks=all_members.con_code.nunique(),
                   sectors=all_members.index_code.nunique(), requested_sectors=len(codes),
                   failed_sectors=[r['index_code'] for r in rows if r['status'] != 'ok'],
                   source_sha256=sha(Path(__file__)),
                   hashes={str(p.relative_to(run)): sha(p) for p in run.rglob('*') if p.is_file()})
    (run / 'manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != 'hashes'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
