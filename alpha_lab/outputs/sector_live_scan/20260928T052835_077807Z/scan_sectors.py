#!/usr/bin/env python3
"""冻结事件规则的最新扫描；模型仅复用两轮历史训练结果，不用新结果调参。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys
import json
import shutil
import numpy as np
import pandas as pd
import torch
import yaml
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / 'alpha_lab/src'))
sys.path.insert(0, str(PROJECT / 'alpha_lab/experiments/sector_fuzzy_v1'))
from sector_dataset import SectorDataset, digest
from sector_shape_events import identify_events
from sector_shape_expansion import stage_events
from sector_multiscale_events import merge_signals
from model import features_at_events, FuzzyNetwork, predict_network


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=HERE / 'config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dataset = SectorDataset(args.dataset)
    panel = pd.read_parquet(dataset.root / 'daily.parquet')
    assert panel.date.max() == pd.Timestamp(cfg['asof'])
    old = pd.read_parquet(PROJECT / cfg['old_sector_dataset'] / 'daily.parquet')
    prefix = panel[panel.date <= cfg['previous_end']]
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), old.reset_index(drop=True), check_dtype=False, rtol=1e-10, atol=1e-10)
    source = PROJECT / 'alpha_lab/outputs/sector_multiscale_events/20260923T080028_978085Z'
    source_manifest = json.loads((source / 'manifest.json').read_text())
    for name, h in source_manifest['artifacts'].items():
        assert digest(source / name) == h, name
    expansion_path = PROJECT / 'alpha_lab/outputs/sector_shape_expansion/20260922T080121_809679Z'
    base = yaml.safe_load((expansion_path / 'sector_shape_events_v1.yaml').read_text())
    expansion = yaml.safe_load((expansion_path / 'sector_shape_expansion_v1.yaml').read_text())
    params = {**base, **expansion['variants']['expanded']}
    multiscale = yaml.safe_load((source / 'sector_multiscale_events_v1.yaml').read_text())
    out = PROJECT / 'alpha_lab/outputs/sector_live_scan' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    print('OUTPUT', out, flush=True)
    shutil.copy2(Path(__file__), out / 'scan_sectors.py')
    shutil.copy2(args.config, out / 'config.yaml')
    save_json(out / 'manifest.json', dict(status='building', dataset=str(dataset.root)))
    snapshots, signals = [], []
    for priority, window in enumerate(multiscale['platform_windows']):
        observations, _ = identify_events(panel, {**params, 'platform_window': window})
        staged = stage_events(observations, expansion['confirmation_memory_days'])
        staged['window'], staged['priority'] = window, priority
        snapshots.append(staged[staged.date.eq(pd.Timestamp(cfg['asof']))].copy())
        signals.append(staged[staged.first_confirmed].copy())
        print('WINDOW', window, 'today_states', len(snapshots[-1]), 'confirmed', len(signals[-1]), flush=True)
    latest = pd.concat(snapshots, ignore_index=True)
    merged = merge_signals(pd.concat(signals, ignore_index=True))
    accepted = merged[merged.accepted].copy()
    original = pd.read_parquet(source / 'confirmed_all_signals_with_dedup.parquet')
    expected = original[original.accepted][['merged_event_id', 'date', 'sector_code', 'window']].reset_index(drop=True)
    original_end = pd.read_parquet(PROJECT / 'alpha_lab/data/sector_datasets/sw_l2_v1_20260921T073401_889301Z/daily.parquet', columns=['date']).date.max()
    actual = accepted[accepted.date <= original_end][expected.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    if cfg.get('previous_scan'):
        prior = pd.read_parquet(PROJECT / cfg['previous_scan'] / 'all_confirmations.parquet')
        pd.testing.assert_frame_equal(accepted[accepted.date <= cfg['previous_end']][expected.columns].reset_index(drop=True), prior[expected.columns].reset_index(drop=True))
    accepted.to_parquet(out / 'all_confirmations.parquet', index=False)
    latest.to_parquet(out / 'latest_states_all_windows.parquet', index=False)
    catalog = pd.read_parquet(dataset.root / 'catalog_sw_l2.parquet')
    names = {(r.src, r.index_code): r.industry_name for r in catalog.itertuples()}
    latest['name'] = [names.get((r.classification_version, r.sector_code), r.sector_code) for r in latest.itertuples()]
    latest = latest.sort_values(['sector_code', 'confirmed', 'trigger', 'watch', 'priority'], ascending=[True, False, False, False, True])
    representative = latest.drop_duplicates('sector_code')
    representative.to_csv(out / 'sector_states_today.csv', index=False)
    watch = representative[representative.watch | representative.confirmed]
    watch.to_csv(out / 'watch_today.csv', index=False)
    # 保留近期历史确认，避免增量更新遗漏此前有效信号。
    recent_start = sorted(panel.date.unique())[-cfg['recent_signal_sessions']]
    new = accepted[(accepted.date > cfg['previous_end']) | (accepted.date >= recent_start)].copy()
    new['source_event_id'] = new.event_id
    new['event_id'] = new.merged_event_id
    new['name'] = [names.get((r.classification_version, r.sector_code), r.sector_code) for r in new.itertuples()]
    old_run = PROJECT / 'alpha_lab/outputs/sector_fuzzy_random_halves/20260923T111853_561924Z'
    model_manifest = json.loads((old_run / 'manifest.json').read_text())
    for name, h in model_manifest['artifacts'].items():
        assert digest(old_run / name) == h, name
    model_cfg = yaml.safe_load((old_run / 'random_config.yaml').read_text())
    scores = []
    if len(new):
        features = features_at_events(new, panel, model_cfg)
        features.to_parquet(out / 'new_confirmation_features.parquet', index=False)
        old_features = pd.read_parquet(old_run / 'causal_features.parquet').set_index('event_id')
        thresholds = pd.read_parquet(old_run / 'test_predictions_before_evaluation.parquet')
        torch.set_num_threads(1)
        for fold in [1, 2]:
            info = json.loads((old_run / f'round_{fold}_fit.json').read_text())
            predictions = []
            for seed in model_cfg['fnn']['ensemble_seeds']:
                saved = torch.load(old_run / 'models' / f'{fold}_fnn_{seed}.pt', map_location='cpu')
                def transform(frame):
                    a = frame[model_cfg['features']].to_numpy(float)
                    return np.clip((np.where(np.isfinite(a), a, saved['median']) - saved['mean']) / saved['scale'],
                                   -model_cfg['feature_clip'], model_cfg['feature_clip']).astype('float32')
                tx = transform(old_features.loc[info['train_ids']])
                network = FuzzyNetwork(tx, saved['rules'], seed, model_cfg['fnn'])
                network.load_state_dict(saved['state_dict'])
                predictions.append(predict_network(network, transform(features)) * model_cfg['target_scale'])
            threshold = thresholds[(thresholds['round'] == fold) & thresholds.model.eq('fnn') & thresholds.retention.eq(.5)].threshold.iloc[0]
            new[f'round_{fold}_score'] = np.mean(predictions, axis=0)
            new[f'round_{fold}_threshold'] = threshold
            new[f'round_{fold}_selected'] = new[f'round_{fold}_score'] >= threshold
        new['model_pass_count'] = new.round_1_selected.astype(int) + new.round_2_selected.astype(int)
        # 今日仍在同一窗口、同一上涨回落事件中满足条件，单独记录，不把历史确认当今日新信号。
        current = pd.concat(snapshots, ignore_index=True)[['sector_code', 'window', 'event_id', 'confirmed', 'watch', 'trigger', 'index_close']]
        current = current.rename(columns={'event_id': 'source_event_id', 'confirmed': 'confirmed_today',
            'watch': 'watch_today', 'trigger': 'trigger_today', 'index_close': 'index_close_today'})
        new = new.merge(current, on=['sector_code', 'window', 'source_event_id'], how='left', validate='many_to_one')
        new['return_since_confirmation'] = new.index_close_today / new.index_close - 1
    new.to_csv(out / 'recent_confirmations_scored.csv', index=False)
    new = new[new.date > cfg['previous_end']].copy()
    new.to_csv(out / 'new_confirmations_scored.csv', index=False)
    summary = dict(requested_date=cfg.get('requested_date', cfg['asof']), asof=cfg['asof'], previous_confirmations_unchanged=True, valid_sectors=int(panel[panel.date.eq(pd.Timestamp(cfg['asof']))].index_valid.sum()),
        active_structure_sectors=len(representative), watch_sectors=len(watch),
        trigger_sectors=int(representative.trigger.sum()), confirmed_today_sectors=int(representative.confirmed.sum()),
        new_confirmations=len(new), new_confirmations_today=int(new.date.eq(pd.Timestamp(cfg['asof'])).sum()),
        old_sector_prefix_unchanged=True, old_330_confirmations_unchanged=True,
        model_note='复用两轮独立模型及各自50%训练阈值；分数不是收益承诺，未新增训练或改变规则')
    save_json(out / 'summary.json', summary)
    save_json(out / 'manifest.json', dict(status='complete', config=cfg, dataset=str(dataset.root),
        dataset_content_id=dataset.manifest['content_id'], summary=summary,
        artifacts={p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}))
    (HERE / 'LATEST_SCAN').write_text(str(out.relative_to(PROJECT)) + '\n')
    print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)
    if len(new):
        print(new[['date', 'name', 'sector_code', 'model_pass_count', 'confirmed_today']].to_string(index=False), flush=True)
    print('WATCH', watch[['name', 'sector_code', 'watch', 'trigger', 'confirmed', 'rally_return', 'drawdown', 'reference_distance']].to_string(index=False), flush=True)
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
