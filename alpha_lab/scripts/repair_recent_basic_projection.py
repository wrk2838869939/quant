"""仅恢复四个空daily_basic缓存日的两列，保存独立投影，不覆盖主面板。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
DATES = ['20240819', '20250120', '20250421', '20260129']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    out = args.run / 'data_repair'
    out.mkdir(exist_ok=True)
    from ts_client import get_pro

    def fetch(day):
        target = out / f'daily_basic_{day}.parquet'
        if target.exists():
            return
        try:
            frame = get_pro().daily_basic(trade_date=day)
            if frame is None or not 0 < len(frame) < 5800:
                raise ValueError('empty_or_truncated')
            assert frame.trade_date.astype(str).eq(day).all()
            assert not frame.ts_code.duplicated().any()
            frame.to_parquet(target, index=False)
            print(day, len(frame), 'rows downloaded', flush=True)
        except Exception as exc:
            # 不打印远端异常文本，避免凭据经URL或请求参数外泄。
            raise RuntimeError(f'{day}: {type(exc).__name__}') from None

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, DATES))
    columns = ['date', 'ts_code', 'hfq_close', 'total_mv_yuan', 'turnover_rate']
    raw = pd.read_parquet(ROOT / 'data/panel/daily_panel.parquet', columns=columns)
    assert raw.date.max() == pd.Timestamp('2026-09-08')
    repaired = raw.copy()
    changes, checks = [], []
    for day in DATES:
        frame = pd.read_parquet(out / f'daily_basic_{day}.parquet').set_index('ts_code')
        mask = raw.date.eq(pd.Timestamp(day))
        assert raw.loc[mask, ['total_mv_yuan', 'turnover_rate']].isna().all().all()
        codes = raw.loc[mask, 'ts_code']
        assert codes.isin(frame.index).all(), f'{day}: API股票覆盖不足'
        for column, source, scale in [('total_mv_yuan', 'total_mv', 10000.), ('turnover_rate', 'turnover_rate', 1.)]:
            values = codes.map(frame[source]) * scale
            assert np.isfinite(values).all() and (values >= 0).all()
            if column == 'total_mv_yuan':
                assert values.gt(0).all()
            repaired.loc[mask, column] = values
            changes.append(pd.DataFrame(dict(date=day, ts_code=codes, field=column,
                                             old_value=np.nan, new_value=values)))
        checks.append(dict(date=day, panel_rows=int(mask.sum()), response_rows=len(frame), complete=True))
    keep = ~raw.date.isin(pd.to_datetime(DATES))
    pd.testing.assert_frame_equal(raw.loc[keep], repaired.loc[keep], check_exact=True)
    pd.testing.assert_frame_equal(raw[['date', 'ts_code', 'hfq_close']], repaired[['date', 'ts_code', 'hfq_close']], check_exact=True)
    repaired.to_parquet(out / 'repaired_feature_projection.parquet', index=False)
    pd.concat(changes, ignore_index=True).to_csv(out / 'changed_cells.csv', index=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(status='complete', dates=DATES, checks=checks, changed_cells=sum(len(x) for x in changes),
        unchanged_rows=int(keep.sum()), raw_panel_modified=False, old_cache_modified=False,
        no_price_change=True, no_imputation=True, unit_conversion='total_mv（万元）乘10000得到total_mv_yuan；turnover_rate原值',
        source='Tushare daily_basic 按交易日重取，仅用于回填原先全缺的两列',
        artifacts={p.name: sha(p) for p in out.glob('*.parquet')}, source_sha256=sha(Path(__file__)))
    (out / 'repair_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
