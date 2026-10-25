#!/usr/bin/env python3
"""复用同一批板块事件和评分，单独检验两项换手硬门槛的机会损耗。"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'alpha_lab/src'), str(ROOT / 'alpha_lab/scripts')]
from datahub import DataHub
from left_side_rules import stock_shape
from sector_entry_window import trend_at_date
from research_sector_entry_window import save_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'alpha_lab/configs/sector_turnover_gate_ablation_v1.yaml')
    args = parser.parse_args()
    study = yaml.safe_load(args.config.read_text())
    parent = ROOT / study['parent_run']
    cfg = yaml.safe_load((parent / 'config.yaml').read_text())
    assert json.loads((parent / 'manifest.json').read_text())['status'] == 'complete'
    hub = DataHub()
    assert hub.cfg.root.resolve() == (ROOT / cfg['release']).resolve()
    out = ROOT / study['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    shutil.copy2(args.config, out / 'study_config.yaml')
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    save_json(out / 'manifest.json', dict(status='running', parent=str(parent),
                                         content_id=os.environ.get('DATAHUB_CONTENT_ID')))
    print('RUN', out, flush=True)
    days = pd.read_parquet(parent / 'event_days.parquet')
    membership = pd.read_parquet(ROOT / cfg['dataset'] / 'membership_daily.parquet')
    sectors = json.loads((ROOT / cfg['dataset'] / 'sector_codes.json').read_text())
    rules = yaml.safe_load((ROOT / cfg['stock_rules']).read_text())
    calendar = hub.calendar.cn_days(start=str(membership.index.min().date()), end=cfg['asof'])
    day_members = {(r.date, r.sector_code): membership.columns[membership.loc[r.date].eq(sectors.index(r.sector_code))].tolist()
                   for r in days.itertuples()}
    codes = sorted({code for members in day_members.values() for code in members})
    fields = ['hfq_close', 'turnover_rate', 'close', 'vol_share']
    raw = hub.cn.daily(codes=codes, start=str(calendar.min().date()), end=cfg['asof'], fields=fields)
    wide = {k: raw.pivot(index='date', columns='ts_code', values=k).reindex(index=calendar, columns=codes) for k in fields}
    del raw
    shape = stock_shape(wide, rules['stock'], rules['stock'])
    daily_return = wide['hfq_close'].pct_change(fill_method=None)
    sc = rules['stock']
    price_gate = (shape['early_return'].le(-sc['minimum_early_fall'])
                  & shape['slowdown'].le(sc['slowdown_ratio'])
                  & shape['late_return'].le(sc['maximum_recent_gain'])
                  & daily_return.ge(-sc['maximum_current_daily_fall']))
    masks = dict(original=shape['shape'],
                 no_expansion_gate=price_gate & shape['contraction'].le(sc['turnover_contraction']),
                 no_turnover_gates=price_gate)
    assert not (masks['original'] & ~masks['no_expansion_gate']).any().any()
    assert not (masks['no_expansion_gate'] & ~masks['no_turnover_gates']).any().any()
    names = pd.read_parquet(hub.cfg.meta_dir / 'namechange.parquet')
    for key in ['start_date', 'end_date']:
        names[key] = pd.to_datetime(names[key], format='%Y%m%d', errors='coerce')
    master = pd.read_parquet(hub.cfg.meta_dir / 'stock_basic.parquet').set_index('ts_code')
    listed = pd.to_datetime(master.list_date, format='%Y%m%d', errors='coerce')
    delisted = pd.to_datetime(master.delist_date, format='%Y%m%d', errors='coerce')
    rows = {key: [] for key in study['variants']}
    for number, (date, group) in enumerate(days.groupby('date', sort=True)):
        active_codes = sorted({c for r in group.itertuples() for c in day_members[(date, r.sector_code)]})
        trend = trend_at_date(wide['hfq_close'].loc[:date, active_codes].tail(study['stock_history_sessions']), date, rules['trend'])
        active_names = names[names.ts_code.isin(active_codes) & names.start_date.le(date)
                             & (names.end_date.isna() | names.end_date.ge(date))]
        count = active_names.groupby('ts_code').size()
        normal = active_names.assign(normal=~active_names['name'].str.upper().str.contains('ST|退', regex=True)).groupby('ts_code').normal.all()
        normal = (normal & count.eq(1)).reindex(active_codes, fill_value=False)
        qualification = (normal & listed.reindex(active_codes).le(date)
                         & (delisted.reindex(active_codes).isna() | delisted.reindex(active_codes).gt(date))
                         & wide['close'].loc[date, active_codes].gt(0) & wide['vol_share'].loc[date, active_codes].gt(0))
        for variant in study['variants']:
            passed = qualification & trend & masks[variant].loc[date, active_codes]
            for event in group.itertuples():
                members = day_members[(date, event.sector_code)]
                for code in passed.loc[members].index[passed.loc[members]]:
                    record = event._asdict()
                    record.pop('Index', None)
                    record.update(ts_code=code, turnover_expansion=shape['expansion'].at[date, code],
                                  turnover_contraction=shape['contraction'].at[date, code], slowdown=shape['slowdown'].at[date, code])
                    rows[variant].append(record)
        if number % study['progress_every_dates'] == 0:
            print('DATE', number, date.date(), {key: len(value) for key, value in rows.items()}, flush=True)
    frames = {key: pd.DataFrame(value) for key, value in rows.items()}
    old = pd.read_parquet(parent / 'stock_candidates_before_sector_gate.parquet')
    pd.testing.assert_frame_equal(frames['original'][old.columns].reset_index(drop=True), old.reset_index(drop=True))
    save_json(out / 'original_parity.json', dict(original_candidate_rows=len(old), exact=True))
    observe_codes = sorted({code for frame in frames.values() for code in frame.ts_code})
    ofields = ['hfq_open', 'hfq_close', 'open', 'close', 'vol_share', 'adj_factor']
    raw = hub.cn.daily(codes=observe_codes, start=str(calendar.min().date()), end=cfg['asof'], fields=ofields)
    observe = {k: raw.pivot(index='date', columns='ts_code', values=k).reindex(index=calendar, columns=observe_codes) for k in ofields}
    for variant, frame in frames.items():
        folder = out / variant
        folder.mkdir()
        vcfg = {**cfg, 'stock_gate_ablation': variant}
        (folder / 'config.yaml').write_text(yaml.safe_dump(vcfg, allow_unicode=True, sort_keys=False))
        for filename in ['event_days.parquet', 'event_scores.parquet', 'events.parquet', 'events_complete.json', 'models_complete.json',
                         'chronological_fits.json', 'model_timing_audit.json']:
            shutil.copy2(parent / filename, folder / filename)
        frame.to_parquet(folder / 'stock_candidates_before_sector_gate.parquet', index=False)
        for key, data in observe.items():
            data[sorted(frame.ts_code.unique())].to_parquet(folder / f'observation_{key}.parquet')
        save_json(folder / 'stocks_complete.json', dict(status='complete', stock_gate_ablation=variant, source_sha256=digest(__file__)))
        save_json(folder / 'manifest.json', dict(status='running', parent=str(parent),
                                                content_id=os.environ.get('DATAHUB_CONTENT_ID'), unbiased_oos=False))
    save_json(out / 'manifest.json', dict(status='candidates_complete', parent=str(parent),
                                         content_id=os.environ.get('DATAHUB_CONTENT_ID'), original_exact=True,
                                         sources={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), args.config,
                                                  ROOT / 'alpha_lab/src/sector_entry_window.py', ROOT / 'alpha_lab/src/left_side_rules.py']}))
    print('COMPLETE_CANDIDATES', out, flush=True)


if __name__ == '__main__':
    main()
