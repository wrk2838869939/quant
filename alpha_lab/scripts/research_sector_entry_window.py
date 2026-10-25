#!/usr/bin/env python3
"""冻结快照上的五日资格对照；保存事件重放、候选及时间推进评分。"""
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
sys.path[:0] = [str(ROOT / 'alpha_lab/src'), str(ROOT / 'alpha_lab/experiments/sector_fuzzy_v1')]
from datahub import DataHub
from sector_shape_events import identify_events
from sector_shape_expansion import stage_events
from sector_multiscale_events import merge_signals
from sector_entry_window import entry_window_rows, trend_at_date
from left_side_rules import stock_shape
from left_side_trend import trend_pullback


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_events(cfg, out):
    """重放五个窗口，并与已有331个确认逐条比对。"""
    panel = pd.read_parquet(ROOT / cfg['dataset'] / 'daily.parquet')
    source = ROOT / cfg['event_source']
    base = yaml.safe_load((source / 'sector_shape_events_v1.yaml').read_text())
    expansion = yaml.safe_load((source / 'sector_shape_expansion_v1.yaml').read_text())
    multi = yaml.safe_load((source / 'sector_multiscale_events_v1.yaml').read_text())
    params = {**base, **expansion['variants']['expanded']}
    states, signals = [], []
    columns = ['date', 'sector_code', 'event_id', 'segment_id', 'peak_date', 'pullback_gate', 'near_reference', 'confirmed']
    for priority, window in enumerate(multi['platform_windows']):
        obs, _ = identify_events(panel, {**params, 'platform_window': window})
        staged = stage_events(obs, expansion['confirmation_memory_days'])
        staged['window'], staged['priority'] = window, priority
        signals.append(staged.loc[staged.first_confirmed].copy())
        states.append(staged[columns + ['window']].copy())
        print('EVENT_WINDOW', window, 'states', len(staged), 'first_confirmed', len(signals[-1]), flush=True)
    merged = merge_signals(pd.concat(signals, ignore_index=True))
    accepted = merged[merged.accepted].copy()
    prior = pd.read_parquet(ROOT / cfg['scan'] / 'all_confirmations.parquet')
    keys = ['merged_event_id', 'date', 'sector_code', 'window', 'event_id', 'peak_date']
    pd.testing.assert_frame_equal(accepted[keys].reset_index(drop=True), prior[keys].reset_index(drop=True))
    accepted.to_parquet(out / 'events.parquet', index=False)
    full = pd.concat(states, ignore_index=True)
    representative = full.groupby(['date', 'sector_code']).confirmed.any().to_dict()
    groups = {key: part.set_index('date') for key, part in full.groupby(['sector_code', 'window'])}
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    rows = []
    for event in accepted.itertuples():
        i = calendar.get_loc(event.date)
        dates = calendar[i:i + cfg['signal_sessions']]
        rows.extend(entry_window_rows(event, dates, groups[(event.sector_code, event.window)], representative))
    status = pd.DataFrame(rows)
    first = status.age_sessions.eq(0)
    assert status.loc[first, cfg['variants']].all().all()
    status.to_parquet(out / 'event_days.parquet', index=False)
    save_json(out / 'event_replay.json', dict(events=len(accepted), reference_events_exact=True,
                                             event_days=len(status), cutoff=str(calendar.max().date())))


def score_events(cfg, out):
    """冻结随机测试子集与年初仅使用成熟标签的FNN对照，严格区分解释范围。"""
    import torch
    from model import features_at_events, TrainScaler, fit_network, predict_network, sample_weights
    torch.set_num_threads(1)
    source = ROOT / cfg['model_source']
    mcfg = yaml.safe_load((source / 'random_config.yaml').read_text())
    events = pd.read_parquet(out / 'events.parquet')
    panel = pd.read_parquet(ROOT / cfg['dataset'] / 'daily.parquet')
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    signals = events.copy()
    signals['event_id'] = signals.merged_event_id
    features = features_at_events(signals, panel, mcfg)
    old_features = pd.read_parquet(source / 'causal_features.parquet').set_index('event_id')
    pd.testing.assert_frame_equal(features.set_index('event_id').loc[old_features.index, old_features.columns],
                                  old_features, check_dtype=False, atol=1e-10, rtol=1e-10)
    features.to_parquet(out / 'causal_features.parquet', index=False)
    test = pd.read_parquet(source / 'test_predictions_before_evaluation.parquet')
    test = test[test.model.eq('fnn') & test.retention.eq(.5)].set_index('event_id')
    scores = features[['event_id', 'date', 'sector_code']].copy()
    scores['all_events'] = True
    scores['random_test_selected'] = scores.event_id.map(test.selected).fillna(False).astype(bool)
    scores['chronological_available'] = False
    scores['chronological_selected'] = False
    labels = pd.read_parquet(ROOT / cfg['outcome_source'] / 'sector_event_outcomes.parquet')
    labels = labels[labels.horizon.eq(mcfg['target_horizon'])]
    data = features.merge(labels[['event_id', 'end_date', 'complete', 'end_return']], on='event_id', how='left', validate='one_to_one')
    model_cfg = cfg['chronological_model']
    fits = []
    (out / 'models').mkdir(exist_ok=True)
    for year in sorted(data.date.dt.year.unique()):
        start = pd.Timestamp(year=year, month=1, day=1)
        position = calendar.searchsorted(start) - model_cfg['embargo_sessions']
        if position < 0:
            continue
        cutoff = calendar[position]
        train = data[data.date.lt(start) & data.end_date.lt(cutoff) & data.complete.eq(True) & data.end_return.notna()]
        target = data[data.date.dt.year.eq(year)]
        if len(train) < model_cfg['minimum_train_events']:
            fits.append(dict(year=int(year), status='insufficient_training_events', train_events=len(train)))
            continue
        assert train.date.max() < start and train.end_date.max() < cutoff
        scaler = TrainScaler(mcfg).fit(train[mcfg['features']])
        tx, vx = scaler.transform(train[mcfg['features']]), scaler.transform(target[mcfg['features']])
        weights = sample_weights(train.date, start, None)
        selected = np.zeros(len(target), dtype=bool)
        thresholds = {}
        for rules in model_cfg['rules']:
            train_preds, target_preds = [], []
            for seed in model_cfg['seeds']:
                net, _, _, _ = fit_network(tx, train.end_return.to_numpy() / mcfg['target_scale'], weights,
                                           rules, model_cfg['learning_rate'], seed, mcfg['fnn'], epochs=model_cfg['epochs'])
                train_preds.append(predict_network(net, tx) * mcfg['target_scale'])
                target_preds.append(predict_network(net, vx) * mcfg['target_scale'])
                torch.save(dict(state_dict=net.state_dict(), median=scaler.median, mean=scaler.mean, scale=scaler.scale,
                                train_ids=train.event_id.tolist(), rules=rules, seed=seed, cutoff=str(cutoff)),
                           out / 'models' / f'{year}_{rules}_{seed}.pt')
            threshold = float(np.quantile(np.mean(train_preds, axis=0), 1 - model_cfg['retention']))
            pred = np.mean(target_preds, axis=0)
            selected |= pred >= threshold
            scores.loc[target.index, f'chronological_score_{rules}'] = pred
            scores.loc[target.index, f'chronological_threshold_{rules}'] = threshold
            thresholds[rules] = threshold
        scores.loc[target.index, 'chronological_available'] = True
        scores.loc[target.index, 'chronological_selected'] = selected
        fits.append(dict(year=int(year), status='fit', train_events=len(train), test_events=len(target),
                         selected=int(selected.sum()), cutoff=cutoff, max_label_end=train.end_date.max(),
                         train_ids=train.event_id.tolist(), thresholds=thresholds))
        print('ANNUAL_MODEL', year, 'train', len(train), 'selected', int(selected.sum()), '/', len(target), flush=True)
    scores.to_parquet(out / 'event_scores.parquet', index=False)
    save_json(out / 'chronological_fits.json', fits)


def replay_stocks(cfg, out):
    """历史时点行业成员、历史名称及原个股形态的逐日期回放。"""
    hub = DataHub()
    assert hub.cfg.root.resolve() == (ROOT / cfg['release']).resolve()
    days = pd.read_parquet(out / 'event_days.parquet')
    membership = pd.read_parquet(ROOT / cfg['dataset'] / 'membership_daily.parquet')
    sector_codes = json.loads((ROOT / cfg['dataset'] / 'sector_codes.json').read_text())
    rules = yaml.safe_load((ROOT / cfg['stock_rules']).read_text())
    calendar = hub.calendar.cn_days(start=str(membership.index.min().date()), end=cfg['asof'])
    codes = set()
    day_members = {}
    for row in days.itertuples():
        members = membership.columns[membership.loc[row.date].eq(sector_codes.index(row.sector_code))].tolist()
        day_members[(row.date, row.sector_code)] = members
        codes.update(members)
    codes = sorted(codes)
    fields = ['hfq_close', 'hfq_open', 'close', 'open', 'vol_share', 'turnover_rate', 'adj_factor']
    raw = hub.cn.daily(codes=codes, start=str(calendar.min().date()), end=cfg['asof'], fields=fields)
    wide = {field: raw.pivot(index='date', columns='ts_code', values=field).reindex(index=calendar, columns=codes) for field in fields}
    del raw
    print('STOCK_PRICES', len(calendar), len(codes), flush=True)
    shape = stock_shape(wide, rules['stock'], rules['stock'])
    names = pd.read_parquet(hub.cfg.meta_dir / 'namechange.parquet')
    for key in ['start_date', 'end_date', 'ann_date']:
        names[key] = pd.to_datetime(names[key], format='%Y%m%d', errors='coerce')
    masters = pd.read_parquet(hub.cfg.meta_dir / 'stock_basic.parquet').set_index('ts_code')
    listed = pd.to_datetime(masters.list_date, format='%Y%m%d', errors='coerce')
    delisted = pd.to_datetime(masters.delist_date, format='%Y%m%d', errors='coerce')
    # 每个信号日只计算一次，只对相关板块成员求趋势。
    results, diagnostics, parity = [], [], []
    for number, (date, group) in enumerate(days.groupby('date', sort=True)):
        active_codes = sorted({code for row in group.itertuples() for code in day_members[(date, row.sector_code)]})
        hist = wide['hfq_close'].loc[:date, active_codes].tail(400)
        trend = trend_at_date(hist, date, rules['trend'])
        # 不同年代及行情缺口上的原函数对照，不以实现自身为测试预期。
        if number % 100 == 0:
            original = trend_pullback(hist, rules['trend'], rules['trend']['stock_minimum_advance'], rules['trend']['stock_minimum_pullback'])['passes'].iloc[-1].eq(1)
            np.testing.assert_array_equal(trend, original)
            parity.append(dict(date=date, stocks=len(active_codes), exact=True))
        active_names = names[names.ts_code.isin(active_codes) & names.start_date.le(date)
                             & (names.end_date.isna() | names.end_date.ge(date))]
        name_counts = active_names.groupby('ts_code').size()
        normal = active_names.assign(normal=~active_names['name'].str.upper().str.contains('ST|退', regex=True)).groupby('ts_code').normal.all()
        normal = normal & name_counts.eq(1)
        listing_ok = listed.reindex(active_codes).le(date) & (delisted.reindex(active_codes).isna() | delisted.reindex(active_codes).gt(date))
        quote_ok = wide['close'].loc[date, active_codes].gt(0) & wide['vol_share'].loc[date, active_codes].gt(0)
        qualified = quote_ok & normal.reindex(active_codes, fill_value=False) & listing_ok
        shape_ok = shape['shape'].loc[date, active_codes]
        passed = qualified & trend & shape_ok
        for row in group.itertuples():
            members = day_members[(date, row.sector_code)]
            diagnostics.append(dict(event_id=row.event_id, date=date, sector_code=row.sector_code,
                                    members=len(members), qualified=int(qualified.loc[members].sum()),
                                    trend=int(trend.loc[members].sum()), shape=int(shape_ok.loc[members].sum()),
                                    stock_pass=int(passed.loc[members].sum())))
            for code in passed.loc[members].index[passed.loc[members]]:
                rec = row._asdict()
                rec.pop('Index', None)
                rec.update(ts_code=code, turnover_expansion=shape['expansion'].at[date, code],
                           turnover_contraction=shape['contraction'].at[date, code], slowdown=shape['slowdown'].at[date, code])
                results.append(rec)
        if number % 100 == 0:
            print('STOCK_DATE', number, str(date.date()), 'candidates_before_sector_gate', len(results), flush=True)
    candidates = pd.DataFrame(results, columns=None if results else list(days.columns) + ['ts_code'])
    candidates.to_parquet(out / 'stock_candidates_before_sector_gate.parquet', index=False)
    pd.DataFrame(diagnostics).to_parquet(out / 'stock_funnel.parquet', index=False)
    save_json(out / 'trend_parity.json', parity)
    # 只保留有原始个股规则信号的证券，用于独立的未来开盘观察阶段。
    observed_codes = sorted(candidates.ts_code.unique())
    for key in ['hfq_open', 'hfq_close', 'open', 'close', 'vol_share', 'adj_factor']:
        wide[key][observed_codes].to_parquet(out / f'observation_{key}.parquet')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'alpha_lab/configs/sector_entry_window_v1.yaml')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--stage', choices=['events', 'models', 'stocks'], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = args.run or ROOT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'config.yaml').exists():
        shutil.copy2(args.config, out / 'config.yaml')
        save_json(out / 'manifest.json', dict(status='running', snapshot_id=os.environ.get('DATAHUB_SNAPSHOT_ID'),
                                             content_id=os.environ.get('DATAHUB_CONTENT_ID'), unbiased_oos=False))
    else:
        assert cfg == yaml.safe_load((out / 'config.yaml').read_text())
    print('RUN', out, 'STAGE', args.stage, flush=True)
    {'events': replay_events, 'models': score_events, 'stocks': replay_stocks}[args.stage](cfg, out)
    save_json(out / f'{args.stage}_complete.json', dict(status='complete', source_sha256=digest(__file__)))


if __name__ == '__main__':
    main()
