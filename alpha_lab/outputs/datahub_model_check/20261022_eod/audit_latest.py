"""核验本次增量覆盖、停牌原因及旧行情不变，只保存审计产物。"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path('/root/autodl-tmp')
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'alpha_lab/experiments/sector_live_v1'
sys.path.insert(0, str(BASE))
from update_market import query

cfg = yaml.safe_load((BASE / 'config_20261022_eod.yaml').read_text())
update = ROOT / (BASE / 'LATEST_UPDATE').read_text().strip()
manifest = json.loads((update / 'manifest.json').read_text())
assert manifest['status'] == 'complete' and manifest['asof'] == cfg['asof']
for name, h in manifest['artifacts'].items():
    assert hashlib.sha256((update / name).read_bytes()).hexdigest() == h, name
delta = pd.read_parquet(update / 'cn_delta.parquet')
assert delta.date.eq(pd.Timestamp(cfg['asof'])).all()
fields = ['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'hfq_close', 'adj_factor', 'turnover_rate', 'float_share', 'total_mv_yuan', 'circ_mv_yuan', 'vol_share', 'amount_yuan', 'net_mf_amount_yuan']
coverage = pd.DataFrame([dict(field=f, missing=int(delta[f].isna().sum()), coverage=float(delta[f].notna().mean())) for f in fields])
coverage.to_csv(OUT / 'field_coverage.csv', index=False)
assert not coverage.missing.any()
api_rows = []
for api in cfg['apis']:
    raw = pd.read_parquet(update / 'raw' / f'{api}_20261022.parquet')
    missing = set(delta.ts_code) - set(raw.ts_code)
    assert not missing, (api, missing)
    api_rows.append(dict(api=api, raw_rows=len(raw), covered=len(set(delta.ts_code) & set(raw.ts_code)), missing=len(missing)))
pd.DataFrame(api_rows).to_csv(OUT / 'api_coverage.csv', index=False)
assert not delta.duplicated(['date', 'ts_code']).any()
assert (delta.high >= delta[['open', 'low', 'close']].max(axis=1)).all()
assert (delta.low <= delta[['open', 'high', 'close']].min(axis=1)).all()
assert np.allclose(delta.hfq_close, delta.close * delta.adj_factor)
raw = pd.read_parquet(update / 'raw/daily_20261022.parquet').set_index('ts_code').loc[delta.ts_code]
for col, source, mul in [('close', 'close', 1), ('amount_yuan', 'amount', 1000), ('vol_share', 'vol', 100)]:
    assert np.allclose(delta[col], raw[source] * mul)
suspensions = query('suspend_d', cfg, trade_date='20261022')
suspensions.to_parquet(OUT / 'suspend_d_20261022.parquet', index=False)
suspended = set(suspensions.loc[suspensions.suspend_type.eq('S'), 'ts_code'])
master = pd.read_parquet(ROOT / 'datahub/datasets/meta/stock_basic.parquet')
active = master[master.list_status.eq('L')]
missing = active[~active.ts_code.isin(delta.ts_code)][['ts_code', 'name', 'list_status', 'delist_date']].copy()
missing['reason'] = missing.ts_code.map(lambda c: '全日停牌' if c in suspended else '已终止上市、主档状态滞后' if c in ['601059.SH', '601198.SH'] else '待核查')
missing.to_csv(OUT / 'missing_daily_reconciled.csv', index=False)
assert not missing.reason.eq('待核查').any(), missing.to_dict('records')
old = pd.read_parquet(ROOT / cfg['old_release'] / 'datasets/cn_daily/year=2026/data.parquet')
new = pd.read_parquet(ROOT / cfg['release'] / 'datasets/cn_daily/year=2026/data.parquet')
pd.testing.assert_frame_equal(old.sort_values(['ts_code', 'date']).reset_index(drop=True), new[new.date.le(cfg['previous_end'])].sort_values(['ts_code', 'date']).reset_index(drop=True))
result = dict(asof=cfg['asof'], daily_rows=len(delta), active_master_rows=len(active), suspended=int(missing.ts_code.isin(suspended).sum()), stale_delisted=int(missing.reason.eq('已终止上市、主档状态滞后').sum()), missing_critical_fields=0, duplicate_keys=0, ohlc_invalid=0, hfq_mismatch=0, raw_volume_amount_close_match=True, prior_2026_prefix_exact=True, new_2026_rows=len(new), update_artifacts_verified=len(manifest['artifacts']))
(OUT / 'latest_quality.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
print(missing.to_string(index=False))
