#!/usr/bin/env python3
"""六次固定历史报表请求，检验原始/调整报表可否替代无版本指标快照。"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    out = parser.parse_args().run.resolve() / 'version_api_probe'
    out.mkdir(exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('禁止覆盖固定探针')
    from ts_client import get_pro
    queries = [(endpoint, kind) for endpoint in ['income', 'cashflow'] for kind in ['1', '4', '5']]
    (out / 'registration.json').write_text(json.dumps(dict(ts_code='600518.SH', period='20181231',
        announcement_range=['20180101', '20201231'], queries=queries, max_workers=2,
        intent='原始/重述版本可用性探针；不训练，不选择收益', future_data_persisted=False), ensure_ascii=False, indent=2))
    def request(query):
        endpoint, kind = query
        identity = f'{endpoint}_type{kind}'
        fields = ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'update_flag']
        fields += ['revenue', 'total_revenue', 'basic_eps'] if endpoint == 'income' else ['n_cashflow_act']
        try:
            cache = out / f'{identity}.parquet'
            recovered = cache.exists()
            frame = pd.read_parquet(cache) if recovered else getattr(get_pro(), endpoint)(
                ts_code='600518.SH', period='20181231', start_date='20180101',
                end_date='20201231', report_type=kind, fields=','.join(fields))
            if not isinstance(frame, pd.DataFrame):
                return dict(query=identity, status='unexpected_response_type')
            required = {'ts_code', 'ann_date', 'end_date', 'report_type'}
            if not required.issubset(frame.columns):
                return dict(query=identity, status='missing_identity_columns', rows=len(frame))
            raw_n = None if recovered else len(frame)
            valid = (frame.ts_code.eq('600518.SH') & frame.end_date.eq('20181231')
                     & frame.ann_date.between('20180101', '20201231'))
            if 'f_ann_date' in frame:
                valid &= frame.f_ann_date.isna() | frame.f_ann_date.le('20201231')
            frame = frame.loc[valid].copy()
            if not recovered:
                frame.to_parquet(cache, index=False)
            return dict(query=identity, status='received', rows=raw_n, historical_rows=len(frame),
                        recovered_from_saved_response=recovered,
                        returned_report_types=sorted(frame.report_type.dropna().astype(str).unique().tolist()),
                        columns=frame.columns.tolist(), requested_type_honored=bool(frame.report_type.astype(str).eq(kind).all()))
        except Exception as error:
            # 不打印代理异常字符串，避免其中携带认证信息。
            return dict(query=identity, status='request_failed', error_type=type(error).__name__)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, queries))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    manifest = dict(status='complete_probe_only', requests=results, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), files=files)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
