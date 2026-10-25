#!/usr/bin/env python3
"""冻结PIT机会，补官方限价后构建共同120日政策单位路径标签。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from patient_trend import review_dates, trend_state
from policy_path_labels import policy_path_labels


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def prepare(run, cfg):
    """只读日期谓词内行情；保留原全A池归一化特征及末尾未成熟候选。"""
    if (run / 'label_preparation.json').exists():
        raise ValueError('已有准备产物不可覆盖')
    fields = ['date', 'ts_code', 'trend', 'monthly'] + cfg['features']
    sources = [ROOT / cfg['source_panel'], ROOT / cfg['execution_features']]
    parts = []
    for index, path in enumerate(sources):
        frame = pd.read_parquet(path, columns=fields)
        assert frame.date.max() < pd.Timestamp('2022-01-01' if index == 0 else '2024-01-01')
        frame = frame[frame.monthly & frame.ts_code.str.match(cfg['mainboard_pattern'])].copy()
        parts.append(frame)
    features = pd.concat(parts, ignore_index=True).sort_values(['date', 'ts_code']).reset_index(drop=True)
    assert not features.duplicated(['date', 'ts_code']).any()
    features.to_parquet(run / 'policy_features.parquet', index=False)
    lo, hi = map(pd.Timestamp, cfg['history'])
    assert hi < pd.Timestamp('2024-01-01')
    codes = pd.Index(sorted(features.ts_code.unique()))
    fields = ['date', 'ts_code', 'hfq_open', 'hfq_close', 'open', 'vol_share']
    filters = [('date', '>=', lo), ('date', '<=', hi), ('ts_code', 'in', list(codes))]
    raw = pd.read_parquet(ROOT / 'data/panel/daily_panel.parquet', columns=fields, filters=filters)
    assert raw.date.between(lo, hi).all()
    metadata = dict(raw_rows=len(raw), raw_columns=fields, raw_filters=filters,
        filtered_market_sha256=hashlib.sha256(pd.util.hash_pandas_object(raw, index=False).values.tobytes()).hexdigest())
    close = raw.pivot(index='date', columns='ts_code', values='hfq_close').reindex(columns=codes).sort_index()
    dates = close.index
    arrays = {field: raw.pivot(index='date', columns='ts_code', values=field).reindex_like(close).to_numpy()
        for field in ['hfq_open', 'open', 'vol_share']}
    arrays['hfq_close'] = close.to_numpy()
    del raw
    spec = cfg['trend']
    _, trend = trend_state(close, spec)
    observable = close.gt(0) & close.shift(spec['medium_window']).gt(0) & close.shift(spec['long_window']).gt(0)
    if spec['require_above_long_ma']:
        observable &= close.where(close.gt(0)).rolling(spec['long_window'], min_periods=spec['long_window']).mean().notna()
    arrays.update(trend=trend.to_numpy(), observable=observable.to_numpy())
    for end in ['2018-12-31', '2021-12-31']:
        _, prefix = trend_state(close.loc[:end], spec)
        pd.testing.assert_frame_equal(prefix, trend.loc[:end])
    reviews = review_dates(dates, 'monthly')
    arrays['review'] = dates.isin(reviews)
    names_path = ROOT / 'data/raw/namechange_all.parquet'
    names = pd.read_parquet(names_path)
    names = names[names.ts_code.isin(codes) & names.name.str.contains('ST', case=False, na=False)].copy()
    names['start'] = pd.to_datetime(names.start_date)
    names['end'] = pd.to_datetime(names.end_date)
    st = np.zeros(close.shape, dtype=bool)
    for row in names.itertuples():
        st[:, codes.get_loc(row.ts_code)] |= (dates >= row.start) & (dates <= row.end if pd.notna(row.end) else True)
    arrays['st'] = st
    basic_path = ROOT / 'data/raw/stock_basic_all.parquet'
    basic = pd.read_parquet(basic_path, columns=['ts_code', 'delist_date']).set_index('ts_code')
    death = pd.to_datetime(basic.delist_date).reindex(codes)
    death = death.where(death.le(hi))
    arrays['delist_day'] = np.where(death.notna(), dates.searchsorted(death.fillna(pd.Timestamp.max)), np.inf)
    arrays['dates'] = dates.to_numpy(dtype='datetime64[ns]')
    arrays['codes'] = codes.to_numpy(dtype=str)
    np.savez_compressed(run / 'policy_market.npz', **arrays)
    metadata.update(status='prepared', test_market_data_used=False, feature_rows=len(features),
        feature_dates=features.date.nunique(), codes=len(codes), prefix_checks=2,
        sources={str(p.relative_to(ROOT)): sha(p) for p in sources + [names_path, basic_path]},
        outputs={name: sha(run / name) for name in ['policy_features.parquet', 'policy_market.npz']})
    dump(run / 'label_preparation.json', metadata)
    print(f'prepared {len(features)} opportunities / {len(codes)} stocks', flush=True)


def build(run, cfg):
    """许可缺失显式保留，停牌不混同未知；所有标签等共同终点成熟。"""
    if (run / 'policy_labels.parquet').exists():
        raise ValueError('已有标签不可覆盖')
    download = json.loads((run / 'limits_manifest.json').read_text())
    assert download['status'] == 'complete'
    prep = json.loads((run / 'label_preparation.json').read_text())
    for name, expected in prep['outputs'].items():
        assert sha(run / name) == expected
    with np.load(run / 'policy_market.npz', allow_pickle=False) as compressed:
        a = {name: compressed[name] for name in compressed.files}
    dates, codes = pd.DatetimeIndex(a['dates']), pd.Index(a['codes'])
    shape = a['open'].shape
    up, down = np.full(shape, np.nan), np.full(shape, np.nan)
    download_rows = pd.read_csv(run / 'limits_download.csv')
    download_hashes = dict(zip(download_rows.path, download_rows.sha256))
    for i, date in enumerate(dates):
        path = ROOT / 'data/raw/policy_limits_by_date' / f'{date:%Y%m%d}.parquet'
        assert sha(path) == download_hashes[str(path.relative_to(ROOT))]
        frame = pd.read_parquet(path)
        assert pd.to_datetime(frame.trade_date).eq(date).all() and not frame.ts_code.duplicated().any()
        frame = frame.set_index('ts_code').reindex(codes)
        up[i], down[i] = frame.up_limit.to_numpy(), frame.down_limit.to_numpy()
    quote = np.isfinite(a['open']) & (a['open'] > 0) & np.isfinite(a['vol_share']) & (a['vol_share'] > 0)
    bounds = np.isfinite(up) & np.isfinite(down) & (up > 0) & (down > 0)
    parent_cfg = yaml.safe_load((ROOT / cfg['parent'] / 'execution/execution_config.yaml').read_text())
    half_tick = parent_cfg['portfolio']['price_tick'] / 2
    sell = np.where(~quote, 0., np.where(bounds, (a['open'] > down + half_tick).astype(float), np.nan))
    buy = np.where(~quote, 0., np.where(bounds, (a['open'] < up - half_tick).astype(float), np.nan))
    features = pd.read_parquet(run / 'policy_features.parquet')
    h = cfg['forecast']['horizon']
    all_rows, counts = [], []
    for date, group in features.groupby('date', sort=True):
        i = dates.get_loc(date)
        ix = codes.get_indexer(group.ts_code)
        assert (ix >= 0).all() and i+1 < len(dates)
        observed = min(h, len(dates)-i-2)
        start, stop = i+1, i+1+observed
        sl = np.ix_(np.arange(start, stop+1), ix)
        clsl = np.ix_(np.arange(start, stop), ix)
        op, cl = a['hfq_open'][sl], a['hfq_close'][clsl]
        review = np.broadcast_to(a['review'][start:stop, None], cl.shape)
        risk_open = np.zeros(op.shape, bool)
        if observed:
            risk_open[1:] = review & a['st'][np.ix_(np.arange(start+1, stop+1), ix)]
        entry = buy[start, ix].copy()
        entry[a['st'][start, ix] | a['st'][i, ix]] = 0.
        frame = group.copy().reset_index(drop=True)
        frame['label_entry_policy_in_scope'] = frame.trend
        frame['label_entry_date'] = dates[start]
        frame['label_end_120'] = dates[stop] if observed == h else pd.NaT
        unknown = np.zeros(len(frame), bool)
        no_entry = None
        extras = {}
        for target in cfg['targets']:
            kwargs = dict(observation_horizon=h, confirmations=cfg['exit_confirmations'], risk_exit_open=risk_open)
            if target == 'patient120':
                kwargs.update(policy='monthly_two_failures')
            else:
                kwargs.update(policy='fixed', fixed_exit_step=60 if target == 'fixed60' else 120)
            result = policy_path_labels(op, cl, review, a['trend'][clsl], a['observable'][clsl],
                entry, sell[sl], a['delist_day'][ix]-start, **kwargs)
            labels = result.labels
            extras[target+'_return'] = labels.label_return
            extras[target+'_mae'] = labels.label_mae
            for key in labels.columns.difference(['label_return', 'label_mae']):
                extras[target+'__'+key] = labels[key]
            extras[target+'__mfe'] = result.marked_nav.max(axis=0)-1
            if no_entry is not None:
                assert np.array_equal(no_entry, labels.no_entry.to_numpy())
            no_entry = labels.no_entry.to_numpy()
            unknown |= labels.entry_execution_unknown | labels.execution_unknown_sell_days.gt(0)
            counts.append(dict(date=date, target=target, rows=len(frame), mature=observed == h,
                entered=int(labels.entered.sum()), no_entry=int(labels.no_entry.sum()),
                settled=int((labels.entered & ~labels.open_at_cutoff).sum()),
                open_at_cutoff=int(labels.open_at_cutoff.sum()), pending=int(labels.pending_exit.sum()),
                delayed_sells=int(labels.blocked_sell_days.gt(0).sum()),
                unknown=int((labels.entry_execution_unknown | labels.execution_unknown_sell_days.gt(0)).sum()),
                missing_marks=int(labels.missing_hold_marks.gt(0).sum()),
                delisted=int(labels.delist_writeoff_step.notna().sum())))
        frame['entry_missing_120'] = no_entry
        frame['execution_unknown_120'] = unknown
        frame['unknown_reason_120'] = np.where(unknown, 'entry_or_requested_exit_execution_evidence_missing', '')
        frame = pd.concat([frame, pd.DataFrame(extras, index=frame.index)], axis=1)
        all_rows.append(frame)
        if len(all_rows) % 24 == 0:
            print(f'labels {len(all_rows)} review dates', flush=True)
    labels = pd.concat(all_rows, ignore_index=True)
    labels.to_parquet(run / 'policy_labels.parquet', index=False)
    pd.DataFrame(counts).to_csv(run / 'policy_label_coverage.csv', index=False)
    sources = [Path(__file__), ROOT / 'src/policy_path_labels.py', ROOT / 'src/patient_trend.py', run / 'config.yaml']
    for path in sources:
        target = run / 'label_sources' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    dump(run / 'policy_labels_manifest.json', dict(status='complete', test_market_data_used=False,
        fresh_holdout=False, rows=len(labels), review_dates=labels.date.nunique(),
        official_limit_missing_quoted_cells=int((quote & ~bounds).sum()),
        entry_rule='one next-open attempt; ST on signal/fill excluded; official limits and raw quote required',
        exit_rule='monthly trend 2 failures; unknown resets; monthly-review next-open ST; fixed deadlines retry',
        shared_maturity='T+121 open; no maturity acceleration after early exit',
        monetary_scope='gross adjusted unit proxy; excludes portfolio caps/lots/fees/cash dividend accounting',
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
        input_hashes={name: sha(run / name) for name in ['policy_features.parquet', 'policy_market.npz', 'limits_download.csv']},
        output_hashes={name: sha(run / name) for name in ['policy_labels.parquet', 'policy_label_coverage.csv']}))
    print(f'labels complete: {len(labels)} rows', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    if not (run / 'label_preparation.json').exists():
        prepare(run, cfg)
    if not args.prepare_only:
        build(run, cfg)


if __name__ == '__main__':
    main()
