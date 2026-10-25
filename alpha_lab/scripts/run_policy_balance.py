#!/usr/bin/env python3
"""冻结数据上的窗口、代数、重训间隔联合滚动研究。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from datahub import DataHub
from initialized_policy import NeuralPolicy, make_package
from policy_balance import RollingPolicy, stable_update, select_balance, quarter_excess
from policy_evolution import candidates
from run_classic_strategy_smoke import digest, save_json
import run_policy_convergence as E
R, G = E.R, E.G


def setup(config, run):
    cfg = yaml.safe_load(config.read_text())
    registry = yaml.safe_load((PROJECT / cfg['registry']).read_text())
    account = yaml.safe_load((PROJECT / registry['account_config']).read_text())
    parent = json.loads((PROJECT / cfg['parent_manifest']).read_text())
    assert parent['status'] == 'complete'
    assert all(digest(PROJECT / p) == h for p, h in parent['source_hashes'].items())
    assert os.environ.get('DATAHUB_CONTENT_ID') == account['dataset']['content_id']
    assert cfg['search']['max_drawdown'] == cfg['selection']['max_drawdown'] == account['risk']['max_drawdown']
    base = PROJECT / registry['source_run']
    assert all(digest(base / 'inputs' / p) == h for p, h in parent['input_hashes'].items())
    frames = {k: pd.read_parquet(base / 'inputs' / f'{k}.parquet') for k in ['market', 'basics', 'st', 'limits', 'actions']}
    eligible = pd.read_parquet(base / 'inputs/eligible.parquet')
    history = pd.read_parquet(base / 'inputs/rule_history.parquet')
    calendar = DataHub().calendar.cn_days()
    grid = calendar[(calendar >= history.date.min()) & (calendar <= eligible.index.max())]
    wide = {k: history.pivot(index='date', columns='ts_code', values=k).reindex(index=grid, columns=eligible.columns)
            for k in history if k not in ['date', 'ts_code']}
    smoke = yaml.safe_load((base / 'sources/classic_strategy_smoke.yaml').read_text())
    portfolio = yaml.safe_load((base / 'sources/monthly_portfolio_audit.yaml').read_text())['portfolio']
    portfolio.update(capital=account['account']['capital'], top_n=5, band_mult=1.,
                     **{k: smoke[k] for k in ['buy_cost', 'sell_cost', 'min_fee', 'cash_dividend_haircut', 'annual_days']})
    rules = yaml.safe_load((base / 'sources/corporate_action_components.yaml').read_text())
    entries = [v['annual_parameters'][cfg['anchor_year']] for v in registry['baselines'].values()]
    assert all(pd.Timestamp(e['inner_end']) < pd.Timestamp(cfg['cohort_start']) for e in entries)
    G.update(plan=cfg, run=run, portfolio=portfolio, frames=frames, smoke=smoke, rules=rules,
             eligible=eligible, wide=wide, calendar=calendar, prepared={},
             specs=yaml.safe_load((PROJECT / cfg['features']).read_text())['features'],
             anchors=[e['params'] for e in entries])
    R.original.G.update(frames=frames, portfolio=portfolio, smoke=smoke, rules=rules, plan=cfg)
    return cfg, parent, account


def package_until(end):
    cfg = G['plan']
    return make_package({k: v.loc[:end] for k, v in G['wide'].items()}, G['eligible'].loc[:end],
                        G['calendar'], G['specs'], G['anchors'], cfg['equal_params'], cfg['feature_start'], end)


def origins(start, end, months):
    return list(pd.date_range(start, end, freq=pd.DateOffset(months=months)))


def key_date(date):
    return str(pd.Timestamp(date).date())


def task_folder(origin, window):
    return G['run'] / 'training' / key_date(origin) / f'w{window}'


def theta_path(origin, window, generation):
    return task_folder(origin, window) / f'theta_g{generation:03d}.npy'


def initial(package):
    cfg = G['plan']
    return NeuralPolicy.initialize(package, cfg['hidden'], cfg['host_index'], cfg['seed'])


def cash_theta(package):
    cfg = G['plan']
    shapes = NeuralPolicy.shapes(package['features'].shape[2], cfg['hidden'])
    theta = np.zeros(sum(int(np.prod(shape)) for _, shape in shapes))
    theta[sum(int(np.prod(shape)) for _, shape in shapes[:-1]) + cfg['hidden'] * 3] = 1.
    return theta


def train_task(task):
    origin, window = task
    cfg, folder = G['plan'], task_folder(origin, window)
    if (folder / 'complete.json').exists():
        return json.loads((folder / 'complete.json').read_text())
    folder.mkdir(parents=True, exist_ok=True)
    start, end = origin - pd.DateOffset(months=window), origin - pd.Timedelta(days=1)
    p = R.window_package(G['packages'][key_date(origin)], start, end)
    assert p['dates'].max() < origin
    assert start >= pd.Timestamp(cfg['feature_start'])
    theta = initial(p)
    rng = np.random.default_rng(cfg['seed'] + origin.year * 100 + origin.month)
    scales = NeuralPolicy.scales(p['features'].shape[2], cfg['hidden'])
    center, multiplier = theta.copy(), 1.
    trials, updates = [], []
    started = time.monotonic()

    def evaluate(value, generation, candidate):
        policy = NeuralPolicy(p, value, cfg['hidden'], G['portfolio']['capital'], trace=False)
        result = E.execute_cached(policy, start, end)
        R.original.audit(result)
        row = dict(generation=generation, candidate=candidate, metrics=R.metric(result),
                   parameter_sha256=hashlib.sha256(value.tobytes()).hexdigest())
        trials.append(row)
        return row

    champion = evaluate(theta, 0, 0)
    for generation in range(max(cfg['checkpoints']) + 1):
        if generation:
            evaluated = [(value, evaluate(value, generation, j))
                         for j, (value, _, _, _) in enumerate(candidates(center, theta, scales, multiplier, rng, cfg['search']))]
            center, theta, champion, multiplier, success, _ = stable_update(evaluated, theta, champion, multiplier, cfg['search'])
            updates.append(dict(generation=generation, champion=champion, multiplier=multiplier, success=success))
        np.save(folder / f'center_g{generation:03d}.npy', center)
        np.save(theta_path(origin, window, generation), theta)
        save_json(folder / f'state_g{generation:03d}.json', dict(rng=rng.bit_generator.state, multiplier=multiplier, champion=champion))
        if generation in [0, *cfg['checkpoints']]:
            policy = NeuralPolicy(p, theta, cfg['hidden'], G['portfolio']['capital'])
            result = E.execute_cached(policy, start, end)
            R.save_account(folder / f'checkpoint_g{generation:03d}', result, policy)
            print(f'TRAIN {key_date(origin)} W={window} K={generation} return={champion["metrics"]["total"]:.4f}', flush=True)
    save_json(folder / 'trials.json', trials)
    save_json(folder / 'updates.json', updates)
    row = dict(origin=key_date(origin), window_months=window, train_start=key_date(start), train_end=key_date(end),
               evaluations=len(trials), seconds=time.monotonic() - started, status='complete')
    save_json(folder / 'complete.json', row)
    return row


def schedule(start, end, window, generation, interval):
    result = []
    for date in origins(start, end, interval):
        path = theta_path(date, window, generation)
        assert pd.Timestamp(json.loads((path.parent / 'complete.json').read_text())['train_end']) < date
        result.append(dict(date=key_date(date), theta=str(path.relative_to(G['run']))))
    return result


def rollout(entries, start, end, folder=None, package=None, original=False):
    cfg = G['plan']
    p = R.window_package(package or G['full_package'], start, end)
    models = {}
    for entry in entries:
        if pd.Timestamp(entry['date']) > pd.Timestamp(end):
            continue
        value = initial(p) if entry['theta'] == 'baseline' else cash_theta(p) if entry['theta'] == 'cash' else np.load(G['run'] / entry['theta'])
        models[entry['date']] = NeuralPolicy(p, value, cfg['hidden'], G['portfolio']['capital'], trace=folder is not None)
    policy = RollingPolicy(models)
    result = (R.execute if original else E.execute_cached)(policy, start, end)
    audit = R.original.audit(result)
    if folder:
        R.save_account(folder, result, list(models.values()), detailed=True)
        save_json(folder / 'schedule.json', entries)
    return result, audit


def configurations():
    cfg = G['plan']
    return list(itertools.product(cfg['training_months'], cfg['checkpoints'], cfg['retrain_months']))


def select_at(date):
    cfg = G['plan']
    start = date - pd.DateOffset(months=cfg['validation_months'])
    end = date - pd.Timedelta(days=1)
    folder = G['run'] / 'selection' / key_date(date)
    baseline, _ = rollout([dict(date=key_date(start), theta='baseline')], start, end, folder / 'baseline')
    rows = []
    for window, generation, interval in configurations():
        cid = f'w{window}_k{generation}_r{interval}'
        entries = schedule(start, end, window, generation, interval)
        assert all(pd.Timestamp(e['date']) < date for e in entries)
        result, _ = rollout(entries, start, end, folder / cid)
        q = quarter_excess(result.daily, baseline.daily)
        m = R.metric(result)
        row = dict(id=cid, window_months=window, generation=generation, retrain_months=interval,
                   evaluation_start=key_date(start), evaluation_end=key_date(end), quarters=len(q),
                   log_excess=float(q.sum()), score=float(q.mean() - cfg['selection']['dispersion_penalty'] * q.std(ddof=0)),
                   quarterly_excess={str(k): float(v) for k, v in q.items()}, metrics=m)
        rows.append(row)
    selected = select_balance(rows, date, cfg['selection'])
    fallback = 'baseline' if R.metric(baseline)['feasible'] else 'cash'
    if selected:
        interval = selected['retrain_months']
        path = theta_path(date, selected['window_months'], selected['generation'])
        assert pd.Timestamp(json.loads((path.parent / 'complete.json').read_text())['train_end']) < date
        entry = dict(date=key_date(date), theta=str(path.relative_to(G['run'])))
    else:
        interval = cfg['selection']['fallback_review_months']
        entry = dict(date=key_date(date), theta=fallback)
    lock = dict(asof=key_date(date), selected=selected, fallback=None if selected else fallback,
                next_review=key_date(date + pd.DateOffset(months=interval)), entry=entry,
                evaluation_end=key_date(end), candidates=rows)
    save_json(folder / 'locked.json', lock)
    print('LOCK ' + key_date(date) + ' ' + (selected['id'] if selected else fallback), flush=True)
    return lock


def outer_accounts():
    cfg = G['plan']
    start, end = cfg['outer_start'], cfg['end']
    date, locks = pd.Timestamp(start), []
    while date <= pd.Timestamp(end):
        lock = select_at(date)
        locks.append(lock)
        date = pd.Timestamp(lock['next_review'])
    save_json(G['run'] / 'locked_choices.json', locks)
    lock_hash = digest(G['run'] / 'locked_choices.json')
    dynamic = [r['entry'] for r in locks]
    paths = {'adaptive': dynamic, 'baseline': [dict(date=start, theta='baseline')]}
    paths.update({f'w{w}_k{k}_r{r}': schedule(start, end, w, k, r) for w, k, r in configurations()})
    summaries, results = [], {}
    for name, entries in paths.items():
        result, _ = rollout(entries, start, end, G['run'] / 'outer' / name)
        results[name] = result
        summaries.append(dict(strategy=name, **R.metric(result)))
    pd.DataFrame(summaries).drop(columns='yearly').to_csv(G['run'] / 'outer_summary.csv', index=False)
    checks = []
    for cutoff in cfg['prefix_dates']:
        short_package = package_until(cutoff)
        n = len(short_package['dates'])
        np.testing.assert_array_equal(short_package['features'], G['full_package']['features'][:n])
        np.testing.assert_array_equal(short_package['schedules'], G['full_package']['schedules'][:n])
        short, _ = rollout(dynamic, start, cutoff, package=short_package)
        R.original.compare_prefix(short, results['adaptive'], pd.Timestamp(cutoff))
        checks.append(dict(cutoff=cutoff, feature_and_account_prefix=True))
    original, _ = rollout(dynamic, start, end, original=True)
    for table in R.original.TABLES:
        pd.testing.assert_frame_equal(getattr(original, table), getattr(results['adaptive'], table), check_exact=True)
    assert digest(G['run'] / 'locked_choices.json') == lock_hash
    save_json(G['run'] / 'checks.json', dict(prefix=checks, original_engine_exact_four_tables=True, locked_choices_sha256=lock_hash))
    return summaries, locks


def age_diagnostics():
    cfg, rows = G['plan'], []
    limit = pd.Timestamp(cfg['end'])
    # 各批次从 10 万开始，冻结参数连续持有至 12 月；同历基线单独连续回放。
    for origin in G['origins']:
        end = min(limit, origin + pd.DateOffset(months=cfg['max_age_months']) - pd.Timedelta(days=1))
        baseline, _ = rollout([dict(date=key_date(origin), theta='baseline')], origin, end)
        for window, generation in itertools.product(cfg['training_months'], cfg['checkpoints']):
            entries = [dict(date=key_date(origin), theta=str(theta_path(origin, window, generation).relative_to(G['run'])))]
            old, _ = rollout(entries, origin, end, G['run'] / 'age' / key_date(origin) / f'w{window}_k{generation}')
            for age in range(0, cfg['max_age_months'], cfg['age_step_months']):
                a = origin + pd.DateOffset(months=age)
                b = origin + pd.DateOffset(months=age + cfg['age_step_months']) - pd.Timedelta(days=1)
                if b > limit:
                    continue
                fresh_entries = [dict(date=key_date(a), theta=str(theta_path(a, window, generation).relative_to(G['run'])))]
                fresh, _ = rollout(fresh_entries, a, b)
                lr = lambda d: float(np.log1p(d.loc[a:b, 'ret']).sum())
                old_log, base_log, fresh_log = lr(old.daily), lr(baseline.daily), lr(fresh.daily)
                rows.append(dict(origin=key_date(origin), window_months=window, generation=generation,
                                 age_months=age, calendar_start=key_date(a), calendar_end=key_date(b),
                                 net_return=float(np.expm1(old_log)), baseline_return=float(np.expm1(base_log)),
                                 log_excess=old_log-base_log, log_old_minus_fresh=old_log-fresh_log))
    frame = pd.DataFrame(rows)
    frame.to_csv(G['run'] / 'age_details.csv', index=False)
    frame.groupby(['window_months', 'generation', 'age_months']).agg(
        cohorts=('log_excess', 'size'), mean_log_excess=('log_excess', 'mean'),
        median_log_excess=('log_excess', 'median'), positive_fraction=('log_excess', lambda s: (s > 0).mean()),
        mean_log_old_minus_fresh=('log_old_minus_fresh', 'mean')).reset_index().to_csv(G['run'] / 'age_summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/policy_balance_v1.yaml')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    run = args.resume.resolve() if args.resume else ROOT / 'outputs/policy_balance' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=True)
    print('RUN=' + str(run), flush=True)
    cfg, parent, account = setup(args.config, run)
    files = [Path(__file__), args.config, ROOT / 'src/policy_balance.py', ROOT / 'src/policy_evolution.py',
             ROOT / 'src/policy_portfolio_cached.py', ROOT / 'scripts/run_policy_convergence.py',
             ROOT / 'tests/test_policy_balance.py', ROOT / 'docs/POLICY_BALANCE_PREREG_20260921.md']
    sources = {**parent['source_hashes'], **{str(p.relative_to(PROJECT)): digest(p) for p in files}}
    manifest = dict(status='running', study=cfg['study'], source_hashes=sources, input_hashes=parent['input_hashes'],
                    dataset=account['dataset'], unbiased_oos=False, started_at=datetime.now(timezone.utc).isoformat())
    if args.resume:
        old = json.loads((run / 'manifest.json').read_text())
        assert old['source_hashes'] == sources and old['status'] != 'complete'
    (run / 'sources').mkdir(exist_ok=True)
    for p in sources:
        shutil.copy2(PROJECT / p, run / 'sources' / Path(p).name)
    save_json(run / 'manifest.json', manifest)
    try:
        G['origins'] = origins(cfg['cohort_start'], cfg['end'], cfg['origin_step_months'])
        G['full_package'] = package_until(cfg['end'])
        G['packages'] = {}
        for origin in G['origins']:
            p = package_until(origin - pd.Timedelta(days=1))
            n = len(p['dates'])
            np.testing.assert_array_equal(p['features'], G['full_package']['features'][:n])
            G['packages'][key_date(origin)] = p
        tasks = list(itertools.product(G['origins'], cfg['training_months']))
        with mp.get_context('fork').Pool(cfg['workers']) as pool:
            trained = []
            for row in pool.imap_unordered(train_task, tasks):
                trained.append(row)
                save_json(run / 'training_progress.json', trained)
        summaries, locks = outer_accounts()
        age_diagnostics()
        assert all(digest(PROJECT / p) == h for p, h in sources.items())
        manifest.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(),
                        training_evaluations=sum(r['evaluations'] for r in trained),
                        training_tasks=len(trained), outer_accounts=len(summaries), decisions=len(locks))
        save_json(run / 'manifest.json', manifest)
        save_json(run / 'artifact_hashes.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
        print('COMPLETE ' + str(run), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        save_json(run / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
