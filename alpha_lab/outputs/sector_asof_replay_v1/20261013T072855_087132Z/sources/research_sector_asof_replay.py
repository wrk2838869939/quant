#!/usr/bin/env python3
"""先冻结逐时点模型预测，再研究该条件下的个股筛选。"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'alpha_lab/src'), str(ROOT / 'alpha_lab/scripts'),
                str(ROOT / 'alpha_lab/experiments/sector_fuzzy_v1')]
from datahub import DataHub
from model import features_at_events
from sector_asof_replay import fit_asof, predict_asof, announced_normal_names
from sector_dataset import digest
from sector_shape_events import identify_events
from sector_shape_expansion import stage_events
from sector_multiscale_events import merge_signals
from sector_entry_window import trend_at_date
from left_side_rules import stock_shape


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def model_stage(cfg, out):
    """当前标签重新计算，所有选参只发生在历史内部。"""
    parent, dataset = ROOT / cfg['parent_run'], ROOT / cfg['dataset']
    events = pd.read_parquet(parent / 'events.parquet')
    panel = pd.read_parquet(dataset / 'daily.parquet')
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    mcfg = yaml.safe_load((ROOT / cfg['model_config']).read_text())
    signals = events.copy()
    signals['event_id'] = signals.merged_event_id
    features = features_at_events(signals, panel, mcfg)
    previous = pd.read_parquet(parent / 'causal_features.parquet')
    pd.testing.assert_frame_equal(features, previous)
    labels, groups = [], {k: v.set_index('date') for k, v in panel.groupby('sector_code')}
    for r in events.itertuples():
        i = calendar.get_loc(r.date)
        dates = calendar[i:i + mcfg['target_horizon'] + 1]
        path = groups[r.sector_code].reindex(dates)
        complete = (len(dates) == mcfg['target_horizon'] + 1 and path.index_valid.eq(True).all()
                    and path.segment_id.eq(r.segment_id).all() and path.index_close.gt(0).all())
        labels.append(dict(event_id=r.merged_event_id, complete=bool(complete),
                           end_date=dates[-1] if len(dates) == mcfg['target_horizon'] + 1 else pd.NaT,
                           target=path.index_close.iloc[-1] / path.index_close.iloc[0] - 1 if complete else np.nan))
    data = features.merge(pd.DataFrame(labels), on='event_id', validate='one_to_one')
    data.to_parquet(out / 'research_inputs_WITH_LABELS.parquet', index=False)
    predictions, fits, search = [], [], []
    (out / 'models').mkdir(exist_ok=True)
    for year in cfg['test_years']:
        target = features[features.date.dt.year.eq(year)].copy()
        start = pd.Timestamp(year=year, month=1, day=1)
        try:
            fitted = fit_asof(data, start, calendar, cfg['model'], mcfg)
        except ValueError as error:
            fits.append(dict(year=year, status='skipped', reason=str(error)))
            print('SKIP', year, str(error), flush=True)
            continue
        score, selected = predict_asof(fitted, target, mcfg)
        audit = fitted['audit']
        target['score'], target['selected'] = score, selected
        target['model_start'], target['threshold'] = start, audit['threshold']
        predictions.append(target[['event_id', 'date', 'sector_code', 'score', 'selected', 'model_start', 'threshold']])
        fits.append(dict(year=year, status='complete', selected=int(selected.sum()), test_events=len(target), **audit))
        search.extend(dict(year=year, **r) for r in fitted['search'])
        for seed, net in zip(cfg['model']['ensemble_seeds'], fitted['networks']):
            torch.save(dict(state_dict=net.state_dict(), feature_names=mcfg['features'],
                            median=fitted['scaler'].median, mean=fitted['scaler'].mean,
                            scale=fitted['scaler'].scale, audit=audit, seed=seed),
                       out / 'models' / f'{year}_{seed}.pt')
        # 只给拟合程序年初以前的记录，再次验证真实年度权重和评分不变。
        if year in [cfg['test_years'][0], cfg['test_years'][-1]]:
            prefix = fit_asof(data[data.date.lt(start)].copy(), start, calendar, cfg['model'], mcfg)
            np.testing.assert_array_equal(score, predict_asof(prefix, target, mcfg)[0])
            assert prefix['audit'] == audit
        print('MODEL', year, 'train', len(audit['train_ids']), 'valid', len(audit['valid_ids']),
              'rules', audit['rules'], 'epochs', audit['epochs'], 'selected', int(selected.sum()), '/', len(target), flush=True)
    pd.concat(predictions, ignore_index=True).to_parquet(out / 'predictions_BEFORE_EVALUATION.parquet', index=False)
    save_json(out / 'fits.json', fits)
    pd.DataFrame(search).to_csv(out / 'inner_search.csv', index=False)
    save_json(out / 'models_complete.json', dict(status='complete', no_random_frozen_models=True,
                                                actual_model_prefix_equality_years=[cfg['test_years'][0], cfg['test_years'][-1]]))


def prefix_stage(cfg, out):
    """截断行情重做状态机，确认历史信号及特征不受未来行影响。"""
    source = ROOT / cfg['event_source']
    base = yaml.safe_load((source / 'sector_shape_events_v1.yaml').read_text())
    expand = yaml.safe_load((source / 'sector_shape_expansion_v1.yaml').read_text())
    multi = yaml.safe_load((source / 'sector_multiscale_events_v1.yaml').read_text())
    panel = pd.read_parquet(ROOT / cfg['dataset'] / 'daily.parquet')
    mcfg = yaml.safe_load((ROOT / cfg['model_config']).read_text())
    checks = []
    for case in cfg['prefix_checks']:
        cut, code = pd.Timestamp(case['date']), case['sector']
        sector = panel[panel.sector_code.eq(code)]
        outputs = []
        for frame in [sector, sector[sector.date.le(cut)]]:
            signals = []
            for priority, window in enumerate(multi['platform_windows']):
                obs, _ = identify_events(frame, {**base, **expand['variants']['expanded'], 'platform_window': window})
                staged = stage_events(obs, expand['confirmation_memory_days'])
                staged['window'], staged['priority'] = window, priority
                signals.append(staged[staged.first_confirmed])
            combined = pd.concat(signals, ignore_index=True)
            merged = merge_signals(combined)
            accepted = merged[merged.accepted & merged.date.le(cut)].copy().reset_index(drop=True)
            accepted['event_id'] = accepted.merged_event_id
            outputs.append((accepted, features_at_events(accepted, frame, mcfg)))
        pd.testing.assert_frame_equal(outputs[0][0], outputs[1][0])
        pd.testing.assert_frame_equal(outputs[0][1], outputs[1][1])
        checks.append(dict(date=cut, sector=code, confirmed_events=len(outputs[0][0]), exact=True))
        print('PREFIX', code, cut.date(), len(outputs[0][0]), flush=True)
    save_json(out / 'event_prefix_audit.json', checks)


def stock_stage(cfg, out):
    """公告可知名称与每日成员筛选；此阶段不读取任何未来股票收益。"""
    hub = DataHub()
    assert hub.cfg.root.resolve() == (ROOT / cfg['release']).resolve()
    assert (out / 'models_complete.json').exists()
    pred = pd.read_parquet(out / 'predictions_BEFORE_EVALUATION.parquet')
    days = pd.read_parquet(ROOT / cfg['parent_run'] / 'event_days.parquet')
    days = days[days.event_id.isin(pred.loc[pred.selected, 'event_id']) & days[cfg['stock']['eligibility']]].copy()
    membership = pd.read_parquet(ROOT / cfg['dataset'] / 'membership_daily.parquet')
    sectors = json.loads((ROOT / cfg['dataset'] / 'sector_codes.json').read_text())
    members = {(r.date, r.sector_code): membership.columns[membership.loc[r.date].eq(sectors.index(r.sector_code))].tolist()
               for r in days.itertuples()}
    codes = sorted({s for stocks in members.values() for s in stocks})
    calendar = hub.calendar.cn_days(str(membership.index.min().date()), cfg['asof'])
    fields = ['hfq_close', 'turnover_rate', 'close', 'vol_share']
    raw = hub.cn.daily(codes=codes, start=str(calendar.min().date()), end=cfg['asof'], fields=fields)
    wide = {k: raw.pivot(index='date', columns='ts_code', values=k).reindex(index=calendar, columns=codes) for k in fields}
    del raw
    rules = yaml.safe_load((ROOT / cfg['stock_rules']).read_text())
    shapes = stock_shape(wide, rules['stock'], rules['stock'])
    sc = rules['stock']
    daily_ret = wide['hfq_close'].pct_change(fill_method=None)
    price = (shapes['early_return'].le(-sc['minimum_early_fall']) & shapes['slowdown'].le(sc['slowdown_ratio'])
             & shapes['late_return'].le(sc['maximum_recent_gain']) & daily_ret.ge(-sc['maximum_current_daily_fall']))
    strength = wide['hfq_close'] / wide['hfq_close'].shift(cfg['stock']['rank_window']) - 1
    volatility = daily_ret.rolling(cfg['stock']['volatility_window']).std()
    names = pd.read_parquet(hub.cfg.meta_dir / 'namechange.parquet')
    for key in ['start_date', 'end_date', 'ann_date']:
        names[key] = pd.to_datetime(names[key], format='%Y%m%d', errors='coerce')
    master = pd.read_parquet(hub.cfg.meta_dir / 'stock_basic.parquet').set_index('ts_code')
    listed = pd.to_datetime(master.list_date, format='%Y%m%d', errors='coerce')
    delisted = pd.to_datetime(master.delist_date, format='%Y%m%d', errors='coerce')
    records, diagnostics, prefix_tests = [], [], []
    for number, (date, group) in enumerate(days.groupby('date', sort=True)):
        active = sorted({c for r in group.itertuples() for c in members[(date, r.sector_code)]})
        trend = trend_at_date(wide['hfq_close'].loc[:date, active].tail(cfg['stock']['history_sessions']), date, rules['trend'])
        normal = announced_normal_names(names, date, active)
        qualified = (normal & listed.reindex(active).le(date) & (delisted.reindex(active).isna() | delisted.reindex(active).gt(date))
                     & wide['close'].loc[date, active].gt(0) & wide['vol_share'].loc[date, active].gt(0))
        if number % 50 == 0:
            prefix = {k: v.loc[:date, active] for k, v in wide.items()}
            partial = stock_shape(prefix, rules['stock'], rules['stock'])
            for key in shapes:
                pd.testing.assert_series_equal(partial[key].loc[date], shapes[key].loc[date, active])
            prefix_tests.append(dict(date=date, stocks=len(active), exact=True))
        for event in group.itertuples():
            stocks = members[(date, event.sector_code)]
            for code in stocks:
                basic, tr, pr = bool(qualified[code]), bool(trend[code]), bool(price.at[date, code])
                record = dict(event_id=event.event_id, date=date, confirmation_date=event.confirmation_date,
                              sector_code=event.sector_code, ts_code=code, qualified=basic, trend=tr, price=pr,
                              expansion=shapes['expansion'].at[date, code], contraction=shapes['contraction'].at[date, code],
                              slowdown=shapes['slowdown'].at[date, code], strength=strength.at[date, code],
                              volatility=volatility.at[date, code],
                              original=basic and tr and bool(shapes['shape'].at[date, code]),
                              no_expansion=basic and tr and pr and shapes['contraction'].at[date, code] <= sc['turnover_contraction'],
                              price_trend=basic and tr and pr, trend_only=basic and tr,
                              price_only=basic and pr, qualified_all=basic)
                records.append(record)
            diagnostics.append(dict(event_id=event.event_id, date=date, members=len(stocks),
                                     known_normal=int(normal.loc[stocks].sum()), qualified=int(qualified.loc[stocks].sum())))
        if number % 30 == 0:
            print('STOCK', number, date.date(), 'rows', len(records), flush=True)
    frame = pd.DataFrame(records)
    for key, column, ascending in [('price_trend_strength5', 'strength', False), ('price_trend_lowvol5', 'volatility', True)]:
        frame[key] = False
        eligible = frame[frame.price_trend & frame[column].notna()].sort_values(['event_id', 'date', column, 'ts_code'], ascending=[True, True, ascending, True])
        chosen = eligible.groupby(['event_id', 'date']).head(cfg['stock']['top_n']).index
        frame.loc[chosen, key] = True
    assert not (frame.original & ~frame.no_expansion).any()
    assert not (frame.no_expansion & ~frame.price_trend).any()
    frame.to_parquet(out / 'stock_decisions_BEFORE_EVALUATION.parquet', index=False)
    pd.DataFrame(diagnostics).to_csv(out / 'stock_qualification_audit.csv', index=False)
    save_json(out / 'stock_prefix_audit.json', prefix_tests)
    first = []
    for variant in cfg['stock']['variants']:
        choices = frame[frame[variant]].copy()
        choices = choices[choices.date.eq(choices.groupby('event_id').date.transform('min'))]
        choices['variant'] = variant
        first.append(choices)
    chosen = pd.concat(first, ignore_index=True)
    chosen.to_parquet(out / 'first_choices_BEFORE_EVALUATION.parquet', index=False)
    # 未来价格只在候选全部落盘后加载，不能改变选股集合。
    observe_codes = sorted(frame.loc[frame.qualified, 'ts_code'].unique())
    obsfields = ['hfq_open', 'open', 'vol_share', 'adj_factor']
    raw = hub.cn.daily(codes=observe_codes, start=str(days.date.min().date()), end=cfg['asof'], fields=obsfields)
    for field in obsfields:
        raw.pivot(index='date', columns='ts_code', values=field).reindex(index=calendar, columns=observe_codes).to_parquet(out / f'observation_{field}.parquet')
    save_json(out / 'stocks_complete.json', dict(status='complete', decision_rows=len(frame), name_ann_date_enforced=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'alpha_lab/configs/sector_asof_replay_v1.yaml')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--stage', choices=['models', 'prefix', 'stocks'], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = args.run or ROOT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=True)
    if args.run is None:
        (out / 'sources').mkdir()
        sources = [args.config, Path(__file__), ROOT / 'alpha_lab/src/sector_asof_replay.py',
                   ROOT / cfg['model_config'], ROOT / cfg['stock_rules'],
                   ROOT / 'alpha_lab/experiments/sector_fuzzy_v1/model.py',
                   ROOT / 'alpha_lab/docs/current_strategy_20261013/ASOF_REPLAY_PREREG.md']
        for p in sources:
            shutil.copy2(p, out / 'sources' / p.name)
        (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        save_json(out / 'manifest.json', dict(status='running', content_id=os.environ.get('DATAHUB_CONTENT_ID'),
                   data_vintage_certified=False, account_backtest=False,
                   sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
                   inputs={str(p.relative_to(ROOT)): digest(p) for p in [ROOT / cfg['dataset'] / 'daily.parquet',
                           ROOT / cfg['dataset'] / 'membership_daily.parquet', ROOT / cfg['parent_run'] / 'events.parquet',
                           ROOT / cfg['parent_run'] / 'event_days.parquet']}))
    else:
        assert yaml.safe_load((out / 'config.yaml').read_text()) == cfg
    print('RUN', out, flush=True)
    {'models': model_stage, 'prefix': prefix_stage, 'stocks': stock_stage}[args.stage](cfg, out)


if __name__ == '__main__':
    main()
