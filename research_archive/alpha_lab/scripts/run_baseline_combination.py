# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""冻结数据上的完整基线端点、IC目标和账户目标配对学习诊断。"""
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
from baseline_combination import (baseline_scores, candidate_grid, candidate_signals,
                                  annotate_neighbors, select_methods, nonmonotonic_triplets)
from selection_baselines import build_signals, validate_entry
from learning_search import build_labels, daily_ic
from rule_search import config_id, inner_metrics
from portfolio import run_backtest
from classic_smoke import verify_ledger
from run_rule_strategy_search import write_result
from run_classic_strategy_smoke import digest, save_json

G = {}
TABLES = ['daily', 'trades', 'events', 'holdings']


def execute(signals, schedule, start, end):
    f = G['frames']
    return run_backtest(signals, f['market'], f['basics'], f['st'], f['limits'], f['actions'],
                        start, end, G['portfolio'], G['rules'].get('rights_issues', []),
                        slippage_rate=G['smoke']['slippage_rate'], top_n_schedule=schedule)


def audit(result):
    f, cfg, smoke = G['frames'], G['portfolio'], G['smoke']
    report = verify_ledger(result, cfg['capital'], f['market'], f['limits'], f['actions'],
                           cfg, smoke['slippage_rate'], smoke['audit']['money_atol'])
    assert result.daily.ret.abs().max() < smoke['audit']['extreme_daily_return']
    return report


def metrics(result):
    cfg = G['portfolio']
    value = inner_metrics(result, cfg['annual_days'], cfg['capital'])
    risk = G['plan']['search']['max_drawdown']
    value['feasible'] = bool(value['buys'] and value['mdd'] >= -risk and value['worst_block_mdd'] >= -risk)
    return value


def evaluate(params):
    plan, year = G['plan'], G['year']
    cid = config_id(params)
    folder = G['run'] / 'search' / str(year) / cid
    folder.mkdir(parents=True)
    row = dict(id=cid, outer_year=year, inner_end=G['inner_end'], params=params,
               status='running', max_label_end=None)
    try:
        signals, schedule = candidate_signals(G['inner_scores'], params, G['calendar'],
                                               plan['inner_start'], G['inner_end'])
        result = execute(signals, schedule, plan['inner_start'], G['inner_end'])
        write_result(folder, result, signals)
        schedule.to_frame().to_parquet(folder / 'top_n_schedule.parquet')
        report = audit(result)
        save_json(folder / 'audit.json', report)
        label = G['label']
        target = label['returns'].reindex_like(signals)
        ic = daily_ic({'score': signals}, target, plan['ic']['min_stocks'])['score']
        ic.to_frame().to_parquet(folder / 'daily_ic.parquet')
        ends = label['end'].reindex(ic.index[ic.notna()]).dropna()
        latest = ends.max() if len(ends) else pd.NaT
        assert pd.isna(latest) or latest < pd.Timestamp(f'{year}-01-01')
        row.update(status='ok', metrics=metrics(result), mean_ic=float(ic.mean()),
                   ic_dates=int(ic.notna().sum()), max_label_end=str(latest.date()) if pd.notna(latest) else None,
                   signal_dates=len(signals), signal_nonmissing=int(signals.notna().sum().sum()),
                   independent_audit=True)
        if params['endpoint'] is not None:
            entry = list(G['registry']['baselines'].values())[params['endpoint']]
            old = G['base'] / 'search' / str(year) / entry['source_family'] / entry['annual_parameters'][year]['selection_id']
            pd.testing.assert_frame_equal(signals, pd.read_parquet(old / 'signals.parquet'))
            for table in TABLES:
                pd.testing.assert_frame_equal(getattr(result, table), pd.read_parquet(old / f'{table}.parquet'))
            row['original_inner_endpoint_match'] = True
    except Exception as exc:
        row.update(status='error', error=repr(exc), traceback=traceback.format_exc())
    save_json(folder / 'trial.json', row)
    return row


def make_outer(method, choices, cutoff=None):
    """只使用已落盘的年度选择；截断原始行情后重算，而非截取已算好分数。"""
    plan = G['plan']
    end = min(pd.Timestamp(cutoff or plan['outer_end']), pd.Timestamp(plan['outer_end']))
    pieces, schedules = [], []
    for year in plan['outer_years']:
        start = pd.Timestamp(f'{year}-01-01')
        if start > end:
            continue
        last = min(pd.Timestamp(f'{year}-12-31'), end)
        eligible = G['eligible'].loc[:last]
        wide = {k: v.loc[:last] for k, v in G['wide'].items()}
        baseline_params = [entry['annual_parameters'][year]['params'] for entry in G['registry']['baselines'].values()]
        scores = baseline_scores(wide, eligible, G['calendar'], baseline_params)
        row = choices[year][method]
        sig, schedule = candidate_signals(scores, row['params'] if row else None, G['calendar'], start, last)
        pieces.append(sig); schedules.append(schedule)
    return pd.concat(pieces).sort_index(), pd.concat(schedules).sort_index()


def compare_prefix(short, full, cutoff):
    for table in TABLES:
        a, b = getattr(short, table), getattr(full, table)
        b = b.loc[a.index] if table == 'daily' else (b[b.date <= cutoff].reset_index(drop=True) if len(b) else b.copy())
        columns = sorted(set(a.columns) | set(b.columns))
        pd.testing.assert_frame_equal(a.reindex(columns=columns).reset_index(drop=True),
                                      b.reindex(columns=columns).reset_index(drop=True), check_dtype=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/baseline_combination_v1.yaml')
    args = parser.parse_args()
    plan = yaml.safe_load(args.config.read_text())
    registry_path = PROJECT / plan['registry']
    registry = yaml.safe_load(registry_path.read_text())
    base = PROJECT / registry['source_run']
    account_path = PROJECT / registry['account_config']
    account = yaml.safe_load(account_path.read_text())
    assert os.environ.get('DATAHUB_CONTENT_ID') == account['dataset']['content_id'], '必须使用冻结启动器'
    assert plan['search']['max_drawdown'] == account['risk']['max_drawdown']
    assert digest(base / 'manifest.json') == registry['source_manifest_sha256']
    assert digest(base / 'all_locked_choices.json') == registry['source_choices_sha256']
    assert all(digest(PROJECT / p) == h for p, h in registry['protected_source_hashes'].items())
    original_choices = json.loads((base / 'all_locked_choices.json').read_text())
    for entry in registry['baselines'].values():
        validate_entry(entry, plan['outer_years'])
        for year in plan['outer_years']:
            old = next(c for c in original_choices if c['outer_year'] == year and c['family'] == entry['source_family'])
            assert entry['annual_parameters'][year]['params'] == old['selected']['params']
            assert entry['annual_parameters'][year]['inner_end'] == old['inner_end']
    hashes = json.loads((base / 'inputs/frozen_manifest.json').read_text())
    hashes['rule_history.parquet'] = json.loads((base / 'inputs/rule_history_manifest.json').read_text())['sha256']
    assert all(digest(base / 'inputs' / p) == h for p, h in hashes.items())
    run = ROOT / 'outputs/baseline_combination' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    print('RUN=' + str(run), flush=True)
    files = [Path(__file__), ROOT / 'src/baseline_combination.py', ROOT / 'src/selection_baselines.py',
             ROOT / 'src/learning_search.py', ROOT / 'src/rule_search.py', ROOT / 'src/portfolio.py',
             ROOT / 'src/classic_smoke.py', ROOT / 'scripts/audit_monthly_portfolio.py',
             ROOT / 'scripts/run_rule_strategy_search.py', ROOT / 'scripts/run_classic_strategy_smoke.py',
             ROOT / 'tests/test_baseline_combination.py', ROOT / 'docs/BASELINE_COMBINATION_PREREG_20260920.md',
             args.config, registry_path, account_path]
    source_hashes = {str(p.resolve().relative_to(PROJECT)): digest(p) for p in files}
    (run / 'sources').mkdir()
    for p in files:
        shutil.copy2(p, run / 'sources' / p.name)
    save_json(run / 'input_hashes.json', hashes)
    manifest = dict(status='running', dataset=account['dataset'], source_hashes=source_hashes,
                    study=plan['study'], source_run=registry['source_run'],
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
        G.update(plan=plan, registry=registry, base=base, run=run, frames=frames, eligible=eligible,
                 wide=wide, calendar=calendar, smoke=smoke, portfolio=portfolio, rules=rules)
        choices, all_rows, triplets, checks = {}, [], [], []
        # 所有年度选择先锁定；训练数据物理截断后才交给因子与标签函数。
        for year in plan['outer_years']:
            inner_end = f'{year - 1}-12-31'
            past = {k: v.loc[:inner_end] for k, v in wide.items()}
            allowed = eligible.loc[:inner_end]
            baseline_params = [e['annual_parameters'][year]['params'] for e in registry['baselines'].values()]
            candidates = candidate_grid(baseline_params, plan['search']['simplex_denominator'], plan['search']['include_equal_weight'])
            G.update(year=year, inner_end=inner_end, inner_scores=baseline_scores(past, allowed, calendar, baseline_params),
                     label=build_labels(past, allowed.index, [plan['ic']['horizon']])[plan['ic']['horizon']])
            save_json(run / f'candidate_space_{year}.json', candidates)
            rows = []
            with mp.get_context('fork').Pool(plan['search']['workers']) as pool:
                for row in pool.imap_unordered(evaluate, candidates):
                    rows.append(row)
                    if len(rows) % 24 == 0 or len(rows) == len(candidates):
                        print(f'INNER {year} {len(rows)}/{len(candidates)} errors={sum(r["status"] != "ok" for r in rows)}', flush=True)
            rows.sort(key=lambda r: r['id'])
            save_json(run / f'all_trials_{year}.json', rows)
            errors = [r for r in rows if r['status'] != 'ok']
            if errors:
                raise RuntimeError(f'内层发生{len(errors)}个错误，禁止丢弃后继续选择：{errors[0]["error"]}')
            annotate_neighbors(rows, plan['search']['simplex_denominator'], plan['search']['neighbor_minimum'])
            choices[year] = select_methods(rows, plan['search']['stable_tie_band'])
            assert set(choices[year]) == set(plan['selection']['methods'])
            all_rows.extend(rows)
            triplets.extend(dict(outer_year=year, **r) for r in nonmonotonic_triplets(rows, plan['search']['simplex_denominator']))
            save_json(run / f'all_trials_{year}.json', rows)
            save_json(run / f'locked_choices_{year}.json', dict(outer_year=year, inner_end=inner_end,
                                                              locked_at=datetime.now(timezone.utc).isoformat(), choices=choices[year]))
            print('LOCKED ' + str(year) + ' ' + json.dumps({k: v['params'] if v else None for k, v in choices[year].items()}), flush=True)
        save_json(run / 'all_locked_choices.json', choices)
        lock_hash = digest(run / 'all_locked_choices.json')
        pd.DataFrame(triplets).to_csv(run / 'nonmonotonic_triplets.csv', index=False)
        flat = []
        for r in all_rows:
            flat.append(dict(outer_year=r['outer_year'], id=r['id'], **r['params'],
                             **{k: v for k, v in r['metrics'].items() if k != 'yearly' and k not in r['params']},
                             mean_ic=r['mean_ic'], ic_dates=r['ic_dates'], neighbor_count=r['neighbor_count'], neighbor_q25=r['neighbor_q25']))
        pd.DataFrame(flat).to_csv(run / 'parameter_surface.csv', index=False)
        rows, yearly = [], []
        methods = list(plan['selection']['methods']) + list(registry['baselines'])
        for method in methods:
            is_baseline = method in registry['baselines']
            if is_baseline:
                signals, schedule = build_signals(registry['baselines'][method], wide, eligible, calendar,
                                                  plan['outer_start'], plan['outer_end'])
            else:
                signals, schedule = make_outer(method, choices)
            result = execute(signals, schedule, plan['outer_start'], plan['outer_end'])
            folder = run / 'outer' / method
            write_result(folder, result, signals)
            schedule.to_frame().to_parquet(folder / 'top_n_schedule.parquet')
            save_json(folder / 'audit.json', audit(result))
            if is_baseline:
                entry = registry['baselines'][method]
                reference = base / 'outer' / entry['source_family']
                assert all(digest(reference / p) == h for p, h in entry['reference_artifacts'].items())
                pd.testing.assert_frame_equal(signals, pd.read_parquet(reference / 'signals.parquet'))
                for table in TABLES:
                    pd.testing.assert_frame_equal(getattr(result, table), pd.read_parquet(reference / f'{table}.parquet'))
                # 通过同一组合入口逐年构造顶点，不能用独立基线重放冒充端点测试。
                at = list(registry['baselines']).index(method)
                endpoint_choices = {y: {method: next(r for r in all_rows if r['outer_year'] == y and r['params']['endpoint'] == at)}
                                    for y in plan['outer_years']}
                endpoint_signals, endpoint_schedule = make_outer(method, endpoint_choices)
                pd.testing.assert_frame_equal(endpoint_signals, signals)
                pd.testing.assert_series_equal(endpoint_schedule, schedule)
                endpoint_result = execute(endpoint_signals, endpoint_schedule, plan['outer_start'], plan['outer_end'])
                for table in TABLES:
                    pd.testing.assert_frame_equal(getattr(endpoint_result, table), getattr(result, table))
                checks.append(dict(method=method, endpoint_reference_match=True, tables=4))
            for cut in plan['checks']['prefix_dates']:
                cutoff = pd.Timestamp(cut)
                if is_baseline:
                    short, short_schedule = build_signals(registry['baselines'][method], {k: v.loc[:cutoff] for k, v in wide.items()},
                                                          eligible.loc[:cutoff], calendar, plan['outer_start'], cutoff)
                else:
                    short, short_schedule = make_outer(method, choices, cutoff)
                pd.testing.assert_frame_equal(short, signals.loc[:cutoff])
                pd.testing.assert_series_equal(short_schedule, schedule.loc[:cutoff])
                prefix_result = execute(short, short_schedule, plan['outer_start'], cutoff)
                compare_prefix(prefix_result, result, cutoff)
                checks.append(dict(method=method, cutoff=cut, raw_signal_prefix=True, account_tables=True))
            value = metrics(result)
            rows.append(dict(method=method, kind='baseline' if is_baseline else 'learner_or_control',
                             historical_acceptance=bool(value['total'] > 0 and value['mdd'] >= -plan['search']['max_drawdown']),
                             **{k: v for k, v in value.items() if k != 'yearly'}))
            yearly.extend(dict(method=method, **y) for y in value['yearly'])
            print(f'OUTER {method} total={value["total"]:.6f} mdd={value["mdd"]:.6f}', flush=True)
        pd.DataFrame(rows).to_csv(run / 'summary.csv', index=False)
        pd.DataFrame(yearly).to_csv(run / 'yearly.csv', index=False)
        save_json(run / 'checks.json', checks)
        assert digest(run / 'all_locked_choices.json') == lock_hash
        assert all(digest(PROJECT / p) == h for p, h in source_hashes.items()), '运行中代码/协议变化'
        assert all(digest(base / 'inputs' / p) == h for p, h in hashes.items())
        manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(), inner_trials=len(all_rows),
                        inner_errors=0, inner_endpoint_replays=9, outer_accounts=len(rows), outer_endpoint_replays=3,
                        prefix_checks=sum('cutoff' in c for c in checks),
                        nonmonotonic_triplets=sum(r['nonmonotonic'] for r in triplets),
                        all_triplets=len(triplets), choices_locked_before_outer=True, source_inputs_unchanged=True)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        save_json(run / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
