#!/usr/bin/env python3
"""基线编译、全参数回报演化及同预算随机初值对照。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import traceback
import time
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from datahub import DataHub
from initialized_policy import NeuralPolicy, AnnualPolicy, make_package
from policy_portfolio import run_backtest
from rule_search import decision_dates, inner_metrics
from run_classic_strategy_smoke import digest, save_json
from run_rule_strategy_search import write_result
import run_baseline_combination as original

G = {}


def window_package(package, start, end):
    p = dict(package)
    p['schedules'] = package['schedules'].copy()
    dates = package['dates']
    sub = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    for j, frequency in enumerate(['daily', 'weekly', 'monthly']):
        p['schedules'][dates.get_indexer(sub), j] = np.where(sub.isin(decision_dates(sub, frequency, G['calendar'])), 1., -1.)
    return p


def get_package(year, end=None):
    end = end or f'{year}-12-31'
    past = {k: v.loc[:end] for k, v in G['wide'].items()}
    eligible = G['eligible'].loc[:end]
    params = [v['annual_parameters'][year]['params'] for v in G['registry']['baselines'].values()]
    equal = G['parent_choices'][str(year)]['equal_weight']['params']
    return make_package(past, eligible, G['calendar'], G['specs'], params, equal, G['plan']['train_start'], end)


def execute(policy, start, end):
    f = G['frames']
    signals = pd.DataFrame(index=pd.DatetimeIndex([], name='date'), columns=G['eligible'].columns, dtype=float)
    return run_backtest(signals, f['market'], f['basics'], f['st'], f['limits'], f['actions'], start, end,
                        G['portfolio'], G['rules'].get('rights_issues', []), policy=policy,
                        slippage_rate=G['smoke']['slippage_rate'])


def metric(result):
    m = inner_metrics(result, G['portfolio']['annual_days'], G['portfolio']['capital'])
    m['feasible'] = bool(m['buys'] > 0 and min(m['mdd'], m['worst_block_mdd']) >= -G['plan']['search']['max_drawdown'])
    return m


def key(row):
    m = row['metrics']
    violation = max(0., -min(m['mdd'], m['worst_block_mdd']) - G['plan']['search']['max_drawdown'])
    return (int(m['feasible']), -violation, m['total'], -m['turnover'])


def policy_tables(policies):
    if not isinstance(policies, list):
        policies = [policies]
    records = sorted([r for p in policies for r in p.records], key=lambda r: r['date'])
    actions = sorted([a for p in policies for a in p.actions], key=lambda a: a[0])
    signals = pd.DataFrame({d: a.scores for d, a in actions}).T
    signals.index.name = 'date'
    return pd.DataFrame(records), actions, signals


def save_account(folder, result, policies, detailed=False):
    folder.mkdir(parents=True, exist_ok=True)
    records, actions, signals = policy_tables(policies)
    write_result(folder, result, signals)
    records.to_parquet(folder / 'observations_actions.parquet', index=False)
    save_json(folder / 'audit.json', original.audit(result))
    if detailed:
        save_json(folder / 'action_intents.json', [dict(date=str(d.date()), top_n=a.top_n, cash_fraction=a.cash_fraction,
                  keep=sorted(a.keep), exit=sorted(a.exit), reduce=a.reduce, weights=a.weights) for d, a in actions])


def evaluate(theta, package, start, end, folder, extra):
    policy = NeuralPolicy(package, theta, G['plan']['hidden'], G['portfolio']['capital'])
    result = execute(policy, start, end)
    save_account(folder, result, policy)
    np.save(folder / 'theta.npy', theta)
    row = dict(**extra, metrics=metric(result), folder=str(folder.relative_to(G['run'])),
               parameter_sha256=digest(folder / 'theta.npy'), status='ok', independent_audit=True)
    save_json(folder / 'trial.json', row)
    return row


def train_job(task):
    host, initialization = task
    try:
        year, plan, package = G['year'], G['plan'], G['training_package']
        h = plan['hosts'].index(host)
        # 两组使用完全相同的扰动流；初始化才决定初始区域。
        seed = plan['seed'] + year * 10 + h
        rng = np.random.default_rng(seed)
        theta0 = NeuralPolicy.initialize(package, plan['hidden'], h, seed, initialization == 'random')
        scales = NeuralPolicy.scales(package['features'].shape[2], plan['hidden'])
        folder = G['run'] / 'search' / str(year) / host / initialization
        cfg = plan['search']
        common = dict(outer_year=year, host=host, initialization=initialization)
        train_end, val_start, val_end = f'{year-2}-12-31', f'{year-1}-01-01', f'{year-1}-12-31'
        initial = evaluate(theta0, package, plan['train_start'], train_end, folder / 'train/g00_initial', dict(**common, generation=0, candidate=0))
        champion, center, champion_theta = initial, theta0.copy(), theta0.copy()
        archive, rows, multiplier, updates = [(theta0.copy(), initial)], [initial], 1., []
        for gen in range(1, cfg['generations'] + 1):
            evaluated, candidates = [], []
            for pair in range(cfg['population'] // 2):
                sigma = cfg['scales'][pair % len(cfg['scales'])] * multiplier
                anchor = center if pair % 2 == 0 else champion_theta
                noise = rng.normal(size=len(theta0)) * scales * sigma
                for direction in [1, -1]:
                    candidates.append((anchor + direction * noise, sigma, pair, direction))
            for j, (theta, sigma, pair, direction) in enumerate(candidates):
                row = evaluate(theta, package, plan['train_start'], train_end, folder / f'train/g{gen:02d}_c{j:02d}',
                               dict(**common, generation=gen, candidate=j, sigma=sigma, pair=pair, direction=direction,
                                    distance_from_initial=float(np.linalg.norm((theta-theta0)/scales))))
                evaluated.append((theta, row)); rows.append(row)
            success = np.mean([key(row) > key(champion) for _, row in evaluated])
            elite = sorted(evaluated + [(champion_theta, champion)], key=lambda pair: key(pair[1]), reverse=True)[:cfg['elites']]
            weights = np.log(cfg['elites'] + .5) - np.log(np.arange(1, cfg['elites'] + 1))
            weights /= weights.sum()
            center = sum(w * theta for w, (theta, _) in zip(weights, elite))
            champion_theta, champion = elite[0][0].copy(), elite[0][1]
            multiplier = float(np.clip(multiplier * (cfg['adaptation_up'] if success > cfg['success_target'] else cfg['adaptation_down']),
                                       cfg['min_scale_multiplier'], cfg['max_scale_multiplier']))
            archive.append((champion_theta.copy(), champion))
            np.save(folder / f'center_g{gen:02d}.npy', center)
            updates.append(dict(generation=gen, success_rate=float(success), scale_multiplier=multiplier,
                                champion=champion['folder'], elite=[r['folder'] for _, r in elite],
                                center_distance=float(np.linalg.norm((center-theta0)/scales))))
            save_json(folder / 'updates.json', updates)
            print(f'GEN {year} {host} {initialization} {gen}/{cfg["generations"]} train={champion["metrics"]["total"]:.4f}', flush=True)
        validation = []
        for gen, (theta, source) in enumerate(archive):
            row = evaluate(theta, G['validation_package'], val_start, val_end, folder / f'validation/g{gen:02d}',
                           dict(**common, generation=gen, source=source['folder']))
            validation.append(row)
        feasible = [r for r in validation if r['metrics']['feasible']]
        chosen = max(feasible, key=key) if feasible else None
        selected = dict(**common, train_end=train_end, validation_end=val_end, choice=chosen,
                        initial_train=initial['metrics'], final_train=champion['metrics'],
                        train_evaluations=len(rows), validation_evaluations=len(validation), parameter_dimension=len(theta0))
        save_json(folder / 'locked_choice.json', selected)
        return dict(status='ok', selected=selected)
    except Exception as exc:
        return dict(status='error', task=task, error=repr(exc), traceback=traceback.format_exc())


def baseline_checks(packages):
    rows = []
    for h, host in enumerate(G['plan']['hosts']):
        models = {}
        for year, package in packages.items():
            p = window_package(package, f'{year}-01-01', f'{year}-12-31')
            theta = NeuralPolicy.initialize(p, G['plan']['hidden'], h, G['plan']['seed'])
            models[year] = NeuralPolicy(p, theta, G['plan']['hidden'], G['portfolio']['capital'])
        result = execute(AnnualPolicy(models), G['plan']['outer_start'], G['plan']['outer_end'])
        ref = G['parent'] / 'outer' / host
        for table in original.TABLES:
            pd.testing.assert_frame_equal(getattr(result, table), pd.read_parquet(ref / f'{table}.parquet'))
        save_account(G['run'] / 'initialization' / host, result, list(models.values()), True)
        rows.append(dict(host=host, exact_four_account_tables=True, **result.summary))
    save_json(G['run'] / 'initialization_checks.json', rows)
    print('BASELINE PARITY 4/4 exact', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/baseline_initialized_policy_v1.yaml')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    plan = yaml.safe_load(args.config.read_text())
    registry = yaml.safe_load((PROJECT / plan['registry']).read_text())
    account = yaml.safe_load((PROJECT / registry['account_config']).read_text())
    parent = PROJECT / plan['parent_run']
    pm = json.loads((parent / 'manifest.json').read_text())
    assert os.environ.get('DATAHUB_CONTENT_ID') == account['dataset']['content_id']
    assert all(digest(PROJECT / p) == h for p, h in registry['protected_source_hashes'].items())
    assert pm['status'] == 'complete'
    assert all(digest(PROJECT / p) == h for p, h in pm['source_hashes'].items())
    base = PROJECT / registry['source_run']
    hashes = json.loads((parent / 'input_hashes.json').read_text())
    assert all(digest(base / 'inputs' / p) == h for p, h in hashes.items())
    assert digest(base / 'manifest.json') == registry['source_manifest_sha256']
    assert digest(base / 'all_locked_choices.json') == registry['source_choices_sha256']
    assert plan['search']['max_drawdown'] == account['risk']['max_drawdown']
    run = ROOT / 'outputs/baseline_initialized_policy' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    print('RUN=' + str(run), flush=True)
    files = [Path(__file__), args.config, ROOT / 'src/initialized_policy.py', ROOT / 'src/policy_portfolio.py',
             ROOT / 'tests/test_initialized_policy.py', ROOT / 'docs/BASELINE_INITIALIZED_POLICY_PREREG_20260920.md',
             PROJECT / plan['features'], ROOT / 'tests/test_portfolio.py']
    sources = {**pm['source_hashes'], **{str(p.relative_to(PROJECT)): digest(p) for p in files}}
    (run / 'sources').mkdir()
    for p in sources:
        shutil.copy2(PROJECT / p, run / 'sources' / Path(p).name)
    manifest = dict(status='running', study=plan['study'], dataset=account['dataset'], source_hashes=sources,
                    input_hashes=hashes, parent_manifest_sha256=digest(parent / 'manifest.json'),
                    parent_choices_sha256=digest(parent / 'all_locked_choices.json'),
                    unbiased_oos=False, verify_only=args.verify_only, started_at=datetime.now(timezone.utc).isoformat())
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
        original.G.update(frames=frames, portfolio=portfolio, smoke=smoke, rules=rules, plan=plan)
        G.update(plan=plan, registry=registry, parent=parent, run=run, portfolio=portfolio,
                 frames=frames, smoke=smoke, rules=rules, eligible=eligible, wide=wide, calendar=calendar,
                 specs=yaml.safe_load((PROJECT / plan['features']).read_text())['features'],
                 parent_choices=json.loads((parent / 'all_locked_choices.json').read_text()))
        # 等价检查属于工程验收；本轮选择不得读其外层成绩。
        packages = {year: get_package(year) for year in plan['outer_years']}
        baseline_checks(packages)
        dimension = len(NeuralPolicy.initialize(packages[2021], plan['hidden'], 0, plan['seed']))
        save_json(run / 'architecture.json', dict(parameter_dimension=dimension, trainable_parameter_dimension=dimension,
                  shapes=NeuralPolicy.shapes(packages[2021]['features'].shape[2], plan['hidden']),
                  feature_names=packages[2021]['feature_names'], stock_state=NeuralPolicy.STOCK_STATE,
                  global_state=NeuralPolicy.GLOBAL_STATE))
        if args.verify_only:
            manifest.update(status='verification_complete', parameter_dimension=dimension)
            save_json(run / 'manifest.json', manifest)
            return
        choices = []
        for year in plan['outer_years']:
            # 训练/验证市场特征各自在原始数据物理截断后计算。
            training_package = get_package(year, f'{year-2}-12-31')
            validation_package = window_package(get_package(year, f'{year-1}-12-31'), f'{year-1}-01-01', f'{year-1}-12-31')
            G.update(year=year, training_package=training_package, validation_package=validation_package)
            tasks = [(host, init) for host in plan['hosts'] for init in plan['initializations']]
            with mp.get_context('fork').Pool(plan['workers']) as pool:
                results = list(pool.imap_unordered(train_job, tasks))
            save_json(run / f'job_results_{year}.json', results)
            errors = [r for r in results if r['status'] != 'ok']
            if errors:
                raise RuntimeError(errors[0]['traceback'])
            choices.extend(r['selected'] for r in results)
            save_json(run / f'locked_choices_{year}.json', dict(locked_at=datetime.now(timezone.utc).isoformat(), choices=choices[-len(tasks):]))
            print(f'LOCKED {year} all 8 paths', flush=True)
        save_json(run / 'all_locked_choices.json', choices)
        lock_hash = digest(run / 'all_locked_choices.json')
        summaries, checks = [], []
        for host in plan['hosts']:
            for init in plan['initializations']:
                models, thetas = {}, {}
                for year, package in packages.items():
                    choice = next(r for r in choices if r['host'] == host and r['initialization'] == init and r['outer_year'] == year)['choice']
                    p = window_package(package, f'{year}-01-01', f'{year}-12-31')
                    if choice:
                        theta = np.load(run / choice['folder'] / 'theta.npy')
                    else:
                        theta = np.zeros(dimension)  # 调仓头=0：无新买单，既有仓需显式退出。
                        shapes = NeuralPolicy.shapes(p['features'].shape[2], plan['hidden'])
                        offset = sum(int(np.prod(s)) for _, s in shapes[:-1])
                        theta[offset + plan['hidden'] * 3] = 1.  # 每日提交空排名清仓，含失败重试。
                    thetas[year] = theta
                    models[year] = NeuralPolicy(p, theta, plan['hidden'], portfolio['capital'])
                policy = AnnualPolicy(models)
                result = execute(policy, plan['outer_start'], plan['outer_end'])
                folder = run / 'outer' / host / init
                save_account(folder, result, list(models.values()), True)
                for year, theta in thetas.items():
                    np.save(folder / f'theta_{year}.npy', theta)
                summaries.append(dict(host=host, initialization=init, **result.summary,
                                       accepted=bool(result.summary['total'] > 0 and result.summary['mdd'] >= -plan['search']['max_drawdown'])))
                for cutoff in plan['prefix_dates']:
                    prefix_models = {}
                    for year in plan['outer_years']:
                        if year > pd.Timestamp(cutoff).year:
                            continue
                        end = min(pd.Timestamp(cutoff), pd.Timestamp(f'{year}-12-31'))
                        p = window_package(get_package(year, str(end.date())), f'{year}-01-01', str(end.date()))
                        full = models[year].package
                        count = len(p['dates'])
                        np.testing.assert_array_equal(p['features'], full['features'][:count])
                        np.testing.assert_array_equal(p['schedules'], full['schedules'][:count])
                        prefix_models[year] = NeuralPolicy(p, thetas[year], plan['hidden'], portfolio['capital'])
                    short = execute(AnnualPolicy(prefix_models), plan['outer_start'], cutoff)
                    original.compare_prefix(short, result, pd.Timestamp(cutoff))
                    for year, m in prefix_models.items():
                        expected = pd.DataFrame(models[year].records)
                        expected = expected[expected.date <= pd.Timestamp(cutoff)].reset_index(drop=True)
                        pd.testing.assert_frame_equal(pd.DataFrame(m.records), expected)
                    original.audit(short)
                    checks.append(dict(host=host, initialization=init, cutoff=cutoff, raw_features_and_account_prefix=True))
                # 确定性检查使用独立新策略实例和完整回放。
                again = {year: NeuralPolicy(models[year].package, theta, plan['hidden'], portfolio['capital']) for year, theta in thetas.items()}
                repeat = execute(AnnualPolicy(again), plan['outer_start'], plan['outer_end'])
                for table in original.TABLES:
                    pd.testing.assert_frame_equal(getattr(repeat, table), getattr(result, table))
                print('OUTER ' + host + ' ' + init + ' ' + json.dumps(summaries[-1]), flush=True)
        assert digest(run / 'all_locked_choices.json') == lock_hash
        assert all(digest(PROJECT / p) == h for p, h in sources.items())
        pd.DataFrame(summaries).to_csv(run / 'outer_summary.csv', index=False)
        save_json(run / 'prefix_checks.json', checks)
        manifest.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(), parameter_dimension=dimension,
                        training_evaluations=sum(r['train_evaluations'] for r in choices),
                        validation_evaluations=sum(r['validation_evaluations'] for r in choices),
                        outer_accounts=len(summaries), exact_initializations=4, prefix_checks=len(checks), deterministic_replays=len(summaries),
                        locked_choices_sha256=lock_hash)
        save_json(run / 'manifest.json', manifest)
        save_json(run / 'artifact_hashes.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        save_json(run / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
