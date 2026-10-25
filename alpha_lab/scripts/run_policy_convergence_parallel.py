#!/usr/bin/env python3
"""同代候选并行回放；按原候选顺序更新，精确接续已冻结的搜索轨迹。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import multiprocessing as mp
import shutil
import sys
import traceback
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_policy_convergence as E

WORKERS = 8
ORIGINAL_POOL_RUN = E.pool_run


def evaluate_candidate(payload):
    """候选之间不共享可变策略或账户；进程返回顺序由 pool.map 保持。"""
    year, theta, folder, extra = payload
    return E.R.evaluate(theta, E.G['training'][year], E.G['plan']['train_start'],
                        f'{year-2}-12-31', folder, extra)


def generation_payload(task, generation, batch, theta0, scales, root):
    year, host, initialization = task
    payload = []
    for j, (theta, sigma, pair, direction) in enumerate(batch):
        folder = root / 'search' / str(year) / host / initialization / 'train' / f'g{generation:03d}_c{j:02d}'
        extra = dict(outer_year=year, host=host, initialization=initialization,
                     generation=generation, candidate=j, sigma=sigma, pair=pair,
                     direction=direction, artifact_root=str(root),
                     distance_from_initial=float(np.linalg.norm((theta-theta0)/scales)))
        payload.append((year, theta, folder, extra))
    return payload


def train_job_parallel(task_and_target):
    task, target = task_and_target
    task = tuple(task)
    year, host, initialization = task
    try:
        path = sorted(E.task_folder(task).glob('state_g*.json'))[-1]
        state = json.loads(path.read_text())
        start = state['generation']
        assert start < target
        center = np.load(path.parent / state['center'])
        theta = np.load(path.parent / state['theta'])
        champion, multiplier, archive = state['champion'], state['multiplier'], state['archive']
        rng = E.restore_rng(state['rng_state'])
        cfg = E.G['plan']['search']
        scales = E.NeuralPolicy.scales(E.G['training'][year]['features'].shape[2], E.G['plan']['hidden'])
        theta0 = np.load(E.resolve(archive[0]) / 'theta.npy')
        records = []
        with mp.get_context('fork').Pool(WORKERS) as pool:
            for generation in range(start + 1, target + 1):
                batch = E.candidates(center, theta, scales, multiplier, rng, cfg)
                payload = generation_payload(task, generation, batch, theta0, scales, E.G['run'])
                rows = pool.map(evaluate_candidate, payload)
                evaluated = [(item[0], row) for item, row in zip(batch, rows)]
                center, theta, champion, multiplier, success, elite = E.update(evaluated, theta, champion, multiplier, cfg)
                archive.append(champion)
                records.append(dict(generation=generation, success_rate=success, scale_multiplier=multiplier,
                                    champion=champion, elite=[r['folder'] for r in elite],
                                    center_distance=float(np.linalg.norm((center-theta0)/scales))))
                E.save_state(task, generation, center, theta, champion, multiplier, rng, archive)
                E.save_json(E.task_folder(task) / f'updates_{start+1:03d}_{target:03d}.json', records)
                if generation % 8 == 0:
                    print(f'PARALLEL GEN {year} {host} {initialization} {generation}/{target} return={champion["metrics"]["total"]:.5f}', flush=True)
        assessment = E.plateau(archive, target, E.G['extension']['convergence'])
        E.save_json(E.task_folder(task) / f'convergence_g{target:03d}.json', assessment)
        return dict(status='ok', task=task, generation=target, assessment=assessment)
    except Exception as exc:
        return dict(status='error', task=task, error=repr(exc), traceback=traceback.format_exc())


def dispatch(function, tasks, label):
    if function is not train_job_parallel:
        return ORIGINAL_POOL_RUN(function, tasks, label)
    results = []
    for task in tasks:
        row = train_job_parallel(task)
        results.append(row)
        E.save_json(E.G['run'] / f'{label}_parallel_results.json', results)
        if row['status'] != 'ok':
            raise RuntimeError(row['traceback'])
        print(label + ' PARALLEL FINISHED ' + str(row['task']), flush=True)
    return results


def verify(run, config):
    output = E.ROOT / 'outputs' / 'policy_parallel_verification' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True)
    E.setup(E.ROOT / 'configs/policy_convergence_v1.yaml', output)
    E.R.execute = E.execute_cached
    generation = config['verification_generation']
    checks = []
    for task in config['verification_tasks']:
        task = tuple(task)
        year, host, initialization = task
        old = run / 'search' / str(year) / host / initialization
        state = json.loads((old / f'state_g{generation-1:03d}.json').read_text())
        expected = json.loads((old / f'state_g{generation:03d}.json').read_text())
        center, theta = np.load(old / state['center']), np.load(old / state['theta'])
        scales = E.NeuralPolicy.scales(E.G['training'][year]['features'].shape[2], E.G['plan']['hidden'])
        rng = E.restore_rng(state['rng_state'])
        batch = E.candidates(center, theta, scales, state['multiplier'], rng, E.G['plan']['search'])
        theta0 = np.load(E.resolve(state['archive'][0]) / 'theta.npy')
        payload = generation_payload(task, generation, batch, theta0, scales, output)
        with mp.get_context('fork').Pool(WORKERS) as pool:
            rows = pool.map(evaluate_candidate, payload)
        for j, (candidate, row) in enumerate(zip(batch, rows)):
            reference = old / 'train' / f'g{generation:03d}_c{j:02d}'
            np.testing.assert_array_equal(candidate[0], np.load(reference / 'theta.npy'))
            prior = json.loads((reference / 'trial.json').read_text())
            assert row['metrics'] == prior['metrics']
            for table in [*E.R.original.TABLES, 'signals', 'observations_actions']:
                pd.testing.assert_frame_equal(pd.read_parquet(E.resolve(row) / f'{table}.parquet'),
                                              pd.read_parquet(reference / f'{table}.parquet'), check_exact=True)
        actual = E.update([(item[0], row) for item, row in zip(batch, rows)], theta,
                          state['champion'], state['multiplier'], E.G['plan']['search'])
        np.testing.assert_array_equal(actual[0], np.load(old / expected['center']))
        np.testing.assert_array_equal(actual[1], np.load(old / expected['theta']))
        assert actual[2]['parameter_sha256'] == expected['champion']['parameter_sha256']
        assert actual[3] == expected['multiplier']
        assert rng.bit_generator.state == expected['rng_state']
        checks.append(dict(task=task, generation=generation, candidates=len(rows),
                           six_tables_each_exact=True, parameters_exact=True, center_champion_rng_exact=True))
        print('PARALLEL PARITY ' + str(task) + ' passed', flush=True)
    E.save_json(output / 'verification.json', dict(status='passed', checks=checks,
                script_sha256=E.digest(Path(__file__)), config=config, candidate_workers=WORKERS))
    print('VERIFICATION=' + str(output), flush=True)


def main():
    global WORKERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verification', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    execution_config = E.ROOT / 'configs/policy_convergence_execution_v1.yaml'
    config = yaml.safe_load(execution_config.read_text())
    WORKERS = int(config['candidate_workers'])
    if args.verify_only:
        verify(run, config)
        return
    assert args.verification is not None
    proof = json.loads(args.verification.read_text())
    assert proof['status'] == 'passed' and proof['script_sha256'] == E.digest(Path(__file__))
    assert proof['config'] == config and sum(c['candidates'] for c in proof['checks']) == 24
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'running'
    sources = {str(p.relative_to(E.PROJECT)): E.digest(p) for p in [Path(__file__), execution_config]}
    checkpoints = []
    for folder in sorted((run / 'search').glob('*/*/*')):
        path = sorted(folder.glob('state_g*.json'))[-1]
        state = json.loads(path.read_text())
        checkpoints.append(dict(task=list(folder.parts[-3:]), generation=state['generation'],
                                state_sha256=E.digest(path), center_sha256=E.digest(folder/state['center']),
                                champion_sha256=E.digest(folder/state['theta'])))
    amendment = dict(reason='独立候选并行执行，消除串行回放导致的长尾；搜索数学定义与随机流不变',
                     timestamp=datetime.now(timezone.utc).isoformat(), source_hashes=sources,
                     candidate_workers=WORKERS, ordered_results=True, verification=str(args.verification.resolve()),
                     verification_sha256=E.digest(args.verification), resume_checkpoints=checkpoints)
    E.save_json(run/'execution_amendment.json', amendment)
    manifest['execution_amendment_sha256'] = E.digest(run/'execution_amendment.json')
    E.save_json(run/'manifest.json', manifest)
    for name in sources:
        shutil.copy2(E.PROJECT/name, run/'sources'/Path(name).name)
    E.train_job = train_job_parallel
    E.pool_run = dispatch
    sys.argv = [str(E.__file__), '--resume', str(run)]
    E.main()
    assert all(E.digest(E.PROJECT/p) == h for p, h in sources.items())


if __name__ == '__main__':
    main()
