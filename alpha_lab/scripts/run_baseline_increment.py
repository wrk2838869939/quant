#!/usr/bin/env python3
"""冻结账户上的基线增量监督学习与直接预测配对验证。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import traceback
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from datahub import DataHub
from baseline_combination import baseline_scores, blend_scores
from baseline_increment import (model_grid, prepare_data, predict_pair, compose_score,
                                select_paths, amplitude_neighbors)
from learning_search import build_labels
from rule_search import config_id, decision_dates
from selection_baselines import validate_entry
import run_baseline_combination as engine
from run_rule_strategy_search import write_result
from run_classic_strategy_smoke import digest, save_json

G = {}


def packages(year, end):
    """每次交付的原始数据物理截断，宿主资格与年参数独立于训练预测。"""
    allowed = G['eligible'].loc[:end]
    past = {k: v.loc[:end] for k, v in G['wide'].items()}
    entries = list(G['registry']['baselines'].values())
    params = [e['annual_parameters'][year]['params'] for e in entries]
    scores = baseline_scores(past, allowed, G['calendar'], params)
    labels = build_labels(past, allowed.index, G['plan']['training']['horizons'])
    result = {}
    for host in G['plan']['hosts']:
        if host == 'equal_weight':
            policy = G['parent_choices'][str(year)]['equal_weight']['params']
            raw = blend_scores(scores, policy['weights'])
        else:
            at = list(G['registry']['baselines']).index(host)
            policy, raw = params[at], scores[at]
        result[host] = dict(raw=raw, policy=policy,
                            data=prepare_data(scores, raw, labels, G['plan']['training']['min_label_stocks']))
    return result


def make_signals(package, params, predictions, start, end):
    raw, policy = package['raw'], package['policy']
    days = raw.index[(raw.index >= pd.Timestamp(start)) & (raw.index <= pd.Timestamp(end))]
    if params is None:
        signal = pd.DataFrame(np.nan, index=[days[0]], columns=raw.columns)
        return signal, pd.Series(1, index=signal.index, name='top_n')
    kind = params['kind']
    score = compose_score(raw, None if kind == 'baseline' else predictions[kind], kind,
                          params.get('amplitude', 1.), G['plan']['increment_clip'])
    decision = decision_dates(days, policy['frequency'], G['calendar'])
    return score.loc[decision], pd.Series(policy['top_n'], index=decision, name='top_n')


def evaluate_account(host, params, predictions):
    plan, year = G['plan'], G['year']
    package = G['inner_packages'][host]
    cid = config_id(params)
    out = G['run'] / 'search' / str(year) / host / cid
    signals, schedule = make_signals(package, params, predictions, plan['inner_start'], G['inner_end'])
    result = engine.execute(signals, schedule, plan['inner_start'], G['inner_end'])
    write_result(out, result, signals)
    schedule.to_frame().to_parquet(out / 'top_n_schedule.parquet')
    save_json(out / 'audit.json', engine.audit(result))
    row = dict(id=cid, host=host, outer_year=year, params=params, status='ok', metrics=engine.metrics(result))
    save_json(out / 'trial.json', row)
    return row


def model_job(task):
    host, spec = task
    try:
        if spec is None:
            return dict(status='ok', rows=[evaluate_account(host, dict(kind='baseline', amplitude=0.), None)], fit_pairs=0)
        folder = G['run'] / 'inner_models' / str(G['year']) / host / config_id(spec)
        predictions, fits = predict_pair(G['inner_packages'][host]['data'], spec, G['plan'],
                                         G['plan']['inner_start'], G['inner_end'], folder)
        folder.mkdir(parents=True, exist_ok=True)
        save_json(folder / 'fits.json', fits)
        for mode, frame in predictions.items():
            frame.to_parquet(folder / f'{mode}_predictions.parquet')
        rows = [evaluate_account(host, dict(kind='direct', model_spec=spec), predictions)]
        for amplitude in G['plan']['amplitudes']:
            if amplitude > 0:
                rows.append(evaluate_account(host, dict(kind='residual', amplitude=amplitude, model_spec=spec), predictions))
        return dict(status='ok', rows=rows, fit_pairs=sum(r['status'] == 'ok' for r in fits))
    except Exception as exc:
        return dict(status='error', host=host, spec=spec, error=repr(exc), traceback=traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/baseline_increment_v1.yaml')
    args = parser.parse_args()
    plan = yaml.safe_load(args.config.read_text())
    parent = PROJECT / plan['parent_run']
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    assert parent_manifest['status'] == 'complete'
    assert os.environ.get('DATAHUB_CONTENT_ID') == parent_manifest['dataset']['content_id']
    assert all(digest(PROJECT / p) == h for p, h in parent_manifest['source_hashes'].items())
    registry = yaml.safe_load((PROJECT / plan['registry']).read_text())
    for entry in registry['baselines'].values():
        validate_entry(entry, plan['outer_years'])
    account = yaml.safe_load((PROJECT / registry['account_config']).read_text())
    assert plan['search']['max_drawdown'] == account['risk']['max_drawdown']
    base = PROJECT / parent_manifest['source_run']
    hashes = json.loads((parent / 'input_hashes.json').read_text())
    assert all(digest(base / 'inputs' / p) == h for p, h in hashes.items())
    parent_choices = json.loads((parent / 'all_locked_choices.json').read_text())
    run = ROOT / 'outputs/baseline_increment' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    print('RUN=' + str(run), flush=True)
    files = [Path(__file__), ROOT / 'src/baseline_increment.py', args.config,
             ROOT / 'tests/test_baseline_increment.py', ROOT / 'docs/BASELINE_INCREMENT_PREREG_20260920.md']
    source_hashes = {**parent_manifest['source_hashes'], **{str(p.resolve().relative_to(PROJECT)): digest(p) for p in files}}
    (run / 'sources').mkdir()
    for p in source_hashes:
        shutil.copy2(PROJECT / p, run / 'sources' / Path(p).name)
    save_json(run / 'input_hashes.json', hashes)
    manifest = dict(status='running', study=plan['study'], dataset=account['dataset'], source_run=str(base.relative_to(PROJECT)),
                    parent_run=plan['parent_run'], parent_manifest_sha256=digest(parent / 'manifest.json'),
                    parent_choices_sha256=digest(parent / 'all_locked_choices.json'), source_hashes=source_hashes,
                    started_at=datetime.now(timezone.utc).isoformat(), unbiased_oos=False)
    save_json(run / 'manifest.json', manifest)
    try:
        frames = {k: pd.read_parquet(base / 'inputs' / f'{k}.parquet') for k in ['market', 'basics', 'st', 'limits', 'actions']}
        eligible = pd.read_parquet(base / 'inputs/eligible.parquet')
        history = pd.read_parquet(base / 'inputs/rule_history.parquet')
        calendar = DataHub().calendar.cn_days()
        grid = calendar[(calendar >= pd.Timestamp('2016-01-01')) & (calendar <= eligible.index.max())]
        wide = {k: history.pivot(index='date', columns='ts_code', values=k).reindex(index=grid, columns=eligible.columns)
                for k in history if k not in ['date', 'ts_code']}
        smoke = yaml.safe_load((base / 'sources/classic_strategy_smoke.yaml').read_text())
        portfolio = yaml.safe_load((base / 'sources/monthly_portfolio_audit.yaml').read_text())['portfolio']
        portfolio.update(capital=account['account']['capital'], top_n=5, band_mult=1.,
                         **{k: smoke[k] for k in ['buy_cost', 'sell_cost', 'min_fee', 'cash_dividend_haircut', 'annual_days']})
        rules = yaml.safe_load((base / 'sources/corporate_action_components.yaml').read_text())
        engine.G.update(frames=frames, portfolio=portfolio, smoke=smoke, rules=rules, plan=plan)
        G.update(plan=plan, registry=registry, parent_choices=parent_choices, eligible=eligible, wide=wide,
                 calendar=calendar, run=run)
        choices, all_rows, curves = {}, [], []
        fit_pairs = 0
        for year in plan['outer_years']:
            inner_end = f'{year-1}-12-31'
            G.update(year=year, inner_end=inner_end, inner_packages=packages(year, inner_end))
            jobs = [(host, spec) for host in plan['hosts'] for spec in [None, *model_grid(plan)]]
            rows, errors = [], []
            with mp.get_context('fork').Pool(plan['workers']) as pool:
                for at, result in enumerate(pool.imap_unordered(model_job, jobs), 1):
                    if result['status'] == 'ok':
                        rows.extend(result['rows']); fit_pairs += result['fit_pairs']
                    else:
                        errors.append(result)
                    if at % 6 == 0 or at == len(jobs):
                        print(f'INNER {year} jobs={at}/{len(jobs)} accounts={len(rows)} errors={len(errors)}', flush=True)
            save_json(run / f'errors_{year}.json', errors)
            save_json(run / f'all_trials_{year}.json', rows)
            if errors:
                raise RuntimeError(f'训练/回测错误不得丢弃：{errors[0]["error"]}')
            choices[year] = {}
            for host in plan['hosts']:
                host_rows = sorted([r for r in rows if r['host'] == host], key=lambda r: r['id'])
                assert len(host_rows) == 1 + len(model_grid(plan)) * len(plan['amplitudes'])
                curves.extend(dict(outer_year=year, host=host, **r) for r in amplitude_neighbors(host_rows, plan))
                choices[year][host] = select_paths(host_rows, plan)
                print('LOCKED ' + str(year) + ' ' + host + ' ' + json.dumps({k: r['params'] if r else None for k, r in choices[year][host].items()}), flush=True)
            all_rows.extend(rows)
            save_json(run / f'all_trials_{year}.json', sorted(rows, key=lambda r: (r['host'], r['id'])))
            save_json(run / f'locked_choices_{year}.json', dict(inner_end=inner_end,
                                                              locked_at=datetime.now(timezone.utc).isoformat(), choices=choices[year]))
        save_json(run / 'all_locked_choices.json', choices)
        lock_hash = digest(run / 'all_locked_choices.json')
        pd.DataFrame(curves).to_csv(run / 'amplitude_triplets.csv', index=False)
        flat = [dict(outer_year=r['outer_year'], host=r['host'], id=r['id'], **r['params'],
                     **{k: v for k, v in r['metrics'].items() if k != 'yearly' and k not in r['params']},
                     neighbor_count=r['neighbor_count'], neighbor_q25=r['neighbor_q25']) for r in all_rows]
        pd.DataFrame(flat).to_csv(run / 'parameter_surface.csv', index=False)
        package_cache, prediction_cache = {}, {}
        prediction_checks = []

        def outer_signal(host, method, cutoff=None):
            end = pd.Timestamp(cutoff or plan['outer_end'])
            parts, counts = [], []
            for year in plan['outer_years']:
                start = pd.Timestamp(f'{year}-01-01')
                if start > end:
                    continue
                last = min(pd.Timestamp(f'{year}-12-31'), end)
                key = (year, last)
                if key not in package_cache:
                    package_cache[key] = packages(year, last)
                package = package_cache[key][host]
                row = choices[year][host][method]
                params = row['params'] if row else None
                pred = None
                if params and params['kind'] != 'baseline':
                    spec = params['model_spec']
                    pkey = (host, year, config_id(spec), last)
                    if pkey not in prediction_cache:
                        full = last == pd.Timestamp(f'{year}-12-31')
                        folder = run / 'outer_models' / str(year) / host / config_id(spec) if full else None
                        predictions, fits = predict_pair(package['data'], spec, plan, start, last, folder)
                        prediction_cache[pkey] = predictions
                        if full:
                            folder.mkdir(parents=True, exist_ok=True)
                            save_json(folder / 'fits.json', fits)
                            for mode, frame in predictions.items():
                                frame.to_parquet(folder / f'{mode}_predictions.parquet')
                        else:
                            full_key = (host, year, config_id(spec), pd.Timestamp(f'{year}-12-31'))
                            for mode in predictions:
                                pd.testing.assert_frame_equal(predictions[mode], prediction_cache[full_key][mode].loc[:last])
                            prediction_checks.append(dict(host=host, year=year, spec=spec, cutoff=str(last.date()), paired_predictions=True))
                    pred = prediction_cache[pkey]
                signals, schedule = make_signals(package, params, pred, start, last)
                parts.append(signals); counts.append(schedule)
            return pd.concat(parts).sort_index(), pd.concat(counts).sort_index()

        summary, yearly, prefix_checks = [], [], []
        for host in plan['hosts']:
            for method in plan['methods']:
                signals, schedule = outer_signal(host, method)
                result = engine.execute(signals, schedule, plan['outer_start'], plan['outer_end'])
                folder = run / 'outer' / host / method
                write_result(folder, result, signals)
                schedule.to_frame().to_parquet(folder / 'top_n_schedule.parquet')
                save_json(folder / 'audit.json', engine.audit(result))
                if method == 'baseline':
                    reference = parent / 'outer' / host
                    pd.testing.assert_frame_equal(signals, pd.read_parquet(reference / 'signals.parquet'))
                    pd.testing.assert_frame_equal(schedule.to_frame(), pd.read_parquet(reference / 'top_n_schedule.parquet'))
                    for table in engine.TABLES:
                        pd.testing.assert_frame_equal(getattr(result, table), pd.read_parquet(reference / f'{table}.parquet'))
                for cut in plan['checks']['prefix_dates']:
                    short, n = outer_signal(host, method, cut)
                    pd.testing.assert_frame_equal(short, signals.loc[:cut])
                    pd.testing.assert_series_equal(n, schedule.loc[:cut])
                    short_result = engine.execute(short, n, plan['outer_start'], cut)
                    engine.compare_prefix(short_result, result, pd.Timestamp(cut))
                    prefix_checks.append(dict(host=host, method=method, cutoff=cut, raw_signals=True, account_tables=True))
                value = engine.metrics(result)
                summary.append(dict(host=host, method=method,
                                    historical_acceptance=bool(value['total'] > 0 and value['mdd'] >= -plan['search']['max_drawdown']),
                                    **{k: v for k, v in value.items() if k != 'yearly'}))
                yearly.extend(dict(host=host, method=method, **r) for r in value['yearly'])
                print(f'OUTER {host} {method} total={value["total"]:.6f} mdd={value["mdd"]:.6f}', flush=True)
        pd.DataFrame(summary).to_csv(run / 'summary.csv', index=False)
        pd.DataFrame(yearly).to_csv(run / 'yearly.csv', index=False)
        save_json(run / 'prefix_checks.json', prefix_checks)
        save_json(run / 'prediction_prefix_checks.json', prediction_checks)
        all_fits = [r for p in run.glob('*_models/*/*/*/fits.json') for r in json.loads(p.read_text())]
        assert all(pd.Timestamp(r['max_label_end']) < pd.Timestamp(r['fit_date']) and r['reload_verified'] for r in all_fits if r['status'] == 'ok')
        assert digest(run / 'all_locked_choices.json') == lock_hash
        assert digest(parent / 'all_locked_choices.json') == manifest['parent_choices_sha256']
        assert digest(parent / 'manifest.json') == manifest['parent_manifest_sha256']
        assert all(digest(PROJECT / p) == h for p, h in source_hashes.items())
        assert all(digest(base / 'inputs' / p) == h for p, h in hashes.items())
        manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
                        inner_candidates=len(all_rows), inner_errors=0, inner_fit_pairs=fit_pairs,
                        all_saved_fit_pairs=sum(r['status'] == 'ok' for r in all_fits),
                        insufficient_fit_pairs=sum(r['status'] != 'ok' for r in all_fits),
                        outer_accounts=len(summary), host_reference_replays=len(plan['hosts']),
                        raw_account_prefix_checks=len(prefix_checks), paired_prediction_prefix_checks=len(prediction_checks),
                        nonmonotonic_triplets=sum(r['nonmonotonic'] for r in curves), all_triplets=len(curves),
                        choices_locked_before_outer=True, frozen_sources_inputs_unchanged=True)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        save_json(run / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
