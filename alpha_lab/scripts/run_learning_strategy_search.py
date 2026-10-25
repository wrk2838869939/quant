#!/usr/bin/env python3
"""在冻结执行样本上训练、内层联合选参、连续外层评价；归档所有尝试与模型。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
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
from run_classic_strategy_smoke import digest, save_json
from rule_search import config_id, decision_dates, inner_metrics, select_trial
from learning_search import build_features, build_labels, prepare_arrays, predict_walk_forward, score_predictions, sample_parameters
from portfolio import run_backtest
from classic_smoke import verify_ledger
from run_rule_strategy_search import write_result

G = {}


def verify_basic(result):
    d, t = result.daily, result.trades
    assert np.isfinite(d[['cash', 'equity', 'stock_value', 'receivables']]).all().all()
    assert d.cash.min() >= -1e-6
    assert (d.equity - d.cash - d.stock_value - d.receivables).abs().max() < 1e-6
    if len(t):
        assert (t.signal_date < t.date).all()
        buy = t.side.eq('buy')
        fee = np.maximum(G['smoke']['min_fee'], t.gross * np.where(buy, G['smoke']['buy_cost'], G['smoke']['sell_cost']))
        fee = np.where(buy, fee, np.minimum(fee, t.gross)) + t.gross * G['smoke']['slippage_rate']
        np.testing.assert_allclose(t.fee, fee, atol=1e-6, rtol=1e-12)
    assert result.daily.ret.abs().max() < G['smoke']['audit']['extreme_daily_return'], '账户极端日收益须单独排查'


def account(signals, start, end, top_n=5, schedule=None):
    result = run_backtest(signals, G['frames']['market'], G['frames']['basics'], G['frames']['st'],
                          G['frames']['limits'], G['frames']['actions'], start, end,
                          dict(G['portfolio'], top_n=top_n), G['rules'].get('rights_issues', []),
                          slippage_rate=G['smoke']['slippage_rate'], top_n_schedule=schedule)
    verify_basic(result)
    return result


def audit(result, folder):
    checked = verify_ledger(result, G['portfolio']['capital'], G['frames']['market'], G['frames']['limits'],
                            G['frames']['actions'], G['portfolio'], G['smoke']['slippage_rate'],
                            G['smoke']['audit']['money_atol'])
    save_json(folder / 'audit.json', checked)


def predictions(params, start, end, context):
    key = {k: v for k, v in params.items() if k not in ['frequency', 'top_n', 'risk_aversion', 'risk_cap']}
    cid = config_id(dict(key, start=start, end=end))
    path = G['run'] / 'predictions' / context / params['family'] / cid
    if not (path / 'complete.json').exists():
        path.mkdir(parents=True, exist_ok=True)
        a, b, fits = predict_walk_forward(G['data'], params, G['plan']['training'], G['plan']['seed'], start, end, path / 'models')
        a.to_parquet(path / 'return_predictions.parquet')
        b.to_parquet(path / 'risk_predictions.parquet')
        save_json(path / 'fits.json', fits)
        save_json(path / 'complete.json', dict(model_key=key, start=start, end=end,
                                              fitted=sum(x['status'] == 'ok' for x in fits),
                                              skipped=sum(x['status'] != 'ok' for x in fits)))
    return (pd.read_parquet(path / 'return_predictions.parquet'),
            pd.read_parquet(path / 'risk_predictions.parquet'), path)


def family_search(family, year, budget):
    start, end = G['plan']['inner_start'], f'{year - 1}-12-31'
    folder = G['run'] / 'search' / str(year) / family
    folder.mkdir(parents=True, exist_ok=True)
    seed = int(config_id(dict(family=family, year=year, seed=G['plan']['seed']))[:8], 16)
    rows, seen = [], set()
    began = time.monotonic()

    def evaluate(candidates, phase):
        for p in candidates:
            cid = config_id(p)
            if cid in seen:
                continue
            seen.add(cid)
            out = folder / cid
            out.mkdir(exist_ok=True)
            row = dict(id=cid, family=family, outer_year=year, inner_start=start, inner_end=end,
                       params=p, phase=phase, order=len(rows), status='running')
            try:
                pred, risk, location = predictions(p, start, end, f'inner_{year}')
                score = score_predictions(pred, risk, p)
                signals = score.loc[decision_dates(score.index, p['frequency'], G['calendar'])]
                result = account(signals, start, end, p['top_n'])
                write_result(out, result, signals)
                metrics = inner_metrics(result, G['portfolio']['annual_days'], G['portfolio']['capital'])
                row.update(status='ok', metrics=metrics, predictions=str(location.relative_to(G['run'])), accounting_check='pass',
                           constant_score_days=int((score.nunique(axis=1) == 1).sum()),
                           no_score_days=int((score.notna().sum(axis=1) == 0).sum()))
            except Exception as exc:
                row.update(status='error', error=repr(exc), traceback=traceback.format_exc())
            save_json(out / 'trial.json', row)
            with (folder / 'trials.jsonl').open('a') as stream:
                stream.write(json.dumps(row, default=str) + '\n')
            rows.append(row)
            if len(rows) % 4 == 0:
                print(f"SEARCH {year} {family} {len(rows)}/{budget} feasible={sum(r.get('metrics', {}).get('feasible', False) for r in rows)} errors={sum(r['status']!='ok' for r in rows)} elapsed={time.monotonic()-began:.0f}s", flush=True)

    global_n = int(budget * G['plan']['search']['global_fraction'])
    local_n = int(budget * G['plan']['search']['local_fraction'])
    evaluate(sample_parameters(G['plan'], family, global_n, seed), 'global_joint')
    ordered = sorted([r for r in rows if r['status'] == 'ok'], key=lambda r: (not r['metrics']['feasible'], -r['metrics']['median_ann']))
    centers, groups = [], set()
    for row in ordered:
        p = row['params']
        group = (p['variant'], p['frequency'], p['top_n'])
        if group not in groups:
            groups.add(group)
            centers.append(p)
        if len(centers) == G['plan']['search']['distinct_centers']:
            break
    if centers:
        evaluate(sample_parameters(G['plan'], family, local_n, seed + 1, seen, centers), 'local_joint')
    evaluate(sample_parameters(G['plan'], family, budget - len(rows), seed + 2, seen), 'continued_global')
    selected = select_trial(rows, G['plan']['search']['tie_band'])
    choice = dict(family=family, outer_year=year, inner_start=start, inner_end=end, selected=selected,
                  attempted=len(rows), errors=sum(r['status'] != 'ok' for r in rows),
                  feasible=sum(r.get('metrics', {}).get('feasible', False) for r in rows),
                  locked_at=datetime.now(timezone.utc).isoformat(), no_feasible_action='cash')
    save_json(folder / 'selection_locked.json', choice)
    print(f"LOCKED {year} {family}: feasible={choice['feasible']} errors={choice['errors']}", flush=True)
    return choice


def outer_accounts(choices):
    rows = []
    for family in G['plan']['families']:
        parts, pair_parts, schedules = [], [], []
        for choice in sorted([x for x in choices if x['family'] == family], key=lambda x: x['outer_year']):
            year, selected = choice['outer_year'], choice['selected']
            dates = G['eligible'].index[G['eligible'].index.year == year]
            if selected:
                p = selected['params']
                a, b, _ = predictions(p, f'{year}-01-01', f'{year}-12-31', f'outer_{year}')
                days = decision_dates(dates, p['frequency'], G['calendar'])
                signal = score_predictions(a, b, p).loc[days]
                pair = a.loc[days]
                top_n = p['top_n']
            else:
                signal = pd.DataFrame(np.nan, index=[dates[0]], columns=G['eligible'].columns)
                pair, top_n = signal.copy(), 5
            parts.append(signal)
            pair_parts.append(pair)
            schedules.append(pd.Series(top_n, index=signal.index))
        signals, schedule = pd.concat(parts), pd.concat(schedules)
        result = account(signals, '2021-01-01', '2023-12-31', schedule=schedule)
        path = G['run'] / 'outer' / family
        write_result(path, result, signals)
        schedule.rename('top_n').to_frame().to_parquet(path / 'top_n_schedule.parquet')
        audit(result, path)
        prefix = []
        for year in [2021, 2022]:
            cutoff = f'{year}-12-31'
            short = signals.loc[:cutoff]
            replay = account(short, '2021-01-01', cutoff, schedule=schedule.loc[short.index])
            for name in ['daily', 'trades', 'events', 'holdings']:
                a, b = getattr(replay, name), getattr(result, name)
                b = b.loc[a.index] if name == 'daily' else (b[b.date <= cutoff].reset_index(drop=True) if len(b) else b.copy())
                columns = sorted(set(a.columns) | set(b.columns))
                pd.testing.assert_frame_equal(a.reindex(columns=columns).reset_index(drop=True),
                                              b.reindex(columns=columns).reset_index(drop=True), check_dtype=False)
            prefix.append(dict(through_year=year, passed=True))
        save_json(path / 'prefix_checks.json', prefix)
        available = sum(x['selected'] is not None for x in choices if x['family'] == family)
        metrics = inner_metrics(result, G['portfolio']['annual_days'], G['portfolio']['capital'])
        rows.append(dict(family=family, **result.summary, risk_pass=result.summary['mdd'] >= -G['plan']['search']['max_drawdown'],
                         selected_years=available, all_cash=available == 0, yearly=json.dumps(metrics['yearly'])))
        if family == 'dual':
            paired = account(pd.concat(pair_parts), '2021-01-01', '2023-12-31', schedule=schedule)
            paired_path = G['run'] / 'paired' / 'dual_without_risk'
            write_result(paired_path, paired, pd.concat(pair_parts))
            audit(paired, paired_path)
            pd.DataFrame([dict(strategy='dual_without_risk', **paired.summary)]).to_csv(G['run'] / 'paired_summary.csv', index=False)
        print(f"OUTER {family}: return={result.summary['total']:.4f} mdd={result.summary['mdd']:.4f} selected_years={available}", flush=True)
    pd.DataFrame(rows).to_csv(G['run'] / 'outer_summary.csv', index=False)


def controls():
    rows = []
    dates = G['eligible'].loc['2021':'2023'].index
    equal = pd.DataFrame(G['data']['ranks'].mean(axis=2), index=G['eligible'].index, columns=G['eligible'].columns).where(G['data']['allowed'])
    scores = {'equal_factor': equal, 'constant_score': equal * 0.,
              **{k: G['features'][k].where(G['data']['allowed']) for k in G['plan']['controls']['fixed_features']}}
    for name, score in scores.items():
        for frequency in G['plan']['controls']['frequencies']:
            signals = score.loc[decision_dates(dates, frequency, G['calendar'])]
            result = account(signals, '2021-01-01', '2023-12-31', G['plan']['controls']['top_n'])
            tag = name + '_' + frequency
            path = G['run'] / 'controls' / tag
            write_result(path, result, signals)
            audit(result, path)
            rows.append(dict(strategy=tag, **result.summary))
    pd.DataFrame(rows).to_csv(G['run'] / 'control_summary.csv', index=False)


def real_prediction_prefix(choices):
    """真实预测重放：截断所有数组并作废截止日未成熟标签，重新拟合后比对。"""
    cutoff = pd.Timestamp('2020-06-15')
    size = int((G['data']['dates'] <= cutoff).sum())
    short = {k: (v[:size].copy() if k in ['dates', 'x', 'ranks', 'allowed'] else v)
             for k, v in G['data'].items() if k != 'labels'}
    short['labels'] = {}
    for horizon, label in G['data']['labels'].items():
        value = {k: v[:size].copy() for k, v in label.items()}
        unknown = pd.isna(value['end']) | (value['end'] > np.datetime64(cutoff))
        for key in ['returns', 'risk', 'ic']:
            value[key][unknown] = np.nan
        value['end'][unknown] = np.datetime64('NaT')
        short['labels'][horizon] = value
    checks = []
    for family in G['plan']['families']:
        items = [c for c in choices if c['family'] == family and c['selected'] is not None]
        if not items:
            # 工程核验独立于绩效选择；无可行参数时取首个成功执行的候选。
            paths = sorted((G['run'] / 'search').glob(f'*/{family}/trials.jsonl'))
            records = [json.loads(line) for line in paths[0].read_text().splitlines()]
            candidate = next(x for x in records if x['status'] == 'ok')
        else:
            candidate = min(items, key=lambda c: c['outer_year'])['selected']
        p = candidate['params']
        location = G['run'] / candidate['predictions']
        reference = pd.read_parquet(location / 'return_predictions.parquet').loc[:cutoff]
        reference_risk = pd.read_parquet(location / 'risk_predictions.parquet').loc[:cutoff]
        a, b, _ = predict_walk_forward(short, p, G['plan']['training'], G['plan']['seed'],
                                       G['plan']['inner_start'], cutoff)
        pd.testing.assert_frame_equal(reference, a)
        pd.testing.assert_frame_equal(reference_risk, b)
        checks.append(dict(family=family, params=p, cutoff=str(cutoff.date()), passed=True))
    save_json(G['run'] / 'real_prediction_prefix_checks.json', checks)


def setup(plan, run):
    base = PROJECT / plan['source_run']
    manifest = json.loads((base / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name in ['src/portfolio.py', 'src/classic_smoke.py', 'src/rule_search.py', 'configs/classic_strategy_smoke.yaml',
                 'configs/monthly_portfolio_audit.yaml', 'configs/corporate_action_components.yaml']:
        assert digest(ROOT / name) == manifest['source_hashes']['alpha_lab/' + name], name
    account_cfg = yaml.safe_load((PROJECT / plan['account_config']).read_text())
    assert os.environ.get('DATAHUB_CONTENT_ID') == account_cfg['dataset']['content_id'], '必须通过冻结启动器执行'
    hashes = json.loads((base / 'inputs/frozen_manifest.json').read_text())
    history = json.loads((base / 'inputs/rule_history_manifest.json').read_text())
    hashes['rule_history.parquet'] = history['sha256']
    assert all(digest(base / 'inputs' / p) == h for p, h in hashes.items())
    shutil.copytree(base / 'inputs', run / 'inputs')
    save_json(run / 'input_hashes.json', hashes)
    frames = {k: pd.read_parquet(run / 'inputs' / f'{k}.parquet') for k in ['market', 'basics', 'st', 'limits', 'actions']}
    eligible = pd.read_parquet(run / 'inputs/eligible.parquet')
    h = pd.read_parquet(run / 'inputs/rule_history.parquet')
    calendar = pd.DatetimeIndex(sorted(h.date.unique()))
    wide = {k: h.pivot(index='date', columns='ts_code', values=k).reindex(index=calendar, columns=eligible.columns)
            for k in h if k not in ['date', 'ts_code']}
    features = build_features(wide, eligible, calendar, plan['features'])
    labels = build_labels(wide, eligible.index, plan['common_axes']['horizon'])
    data = prepare_arrays(features, eligible, labels, plan['training'])
    factor = pd.concat({k: v.stack(future_stack=True) for k, v in features.items()}, axis=1)
    factor.index.names = ['date', 'ts_code']
    factor.to_parquet(run / 'inputs/learning_features.parquet')
    for horizon, label in labels.items():
        for key in ['returns', 'risk']:
            label[key].to_parquet(run / 'inputs' / f'label_{key}_{horizon}.parquet')
    pd.DataFrame({str(h): label['end'] for h, label in labels.items()}).to_parquet(run / 'inputs/label_end_dates.parquet')
    hashes.update({p.name: digest(p) for p in (run / 'inputs').glob('label_*.parquet')})
    hashes['learning_features.parquet'] = digest(run / 'inputs/learning_features.parquet')
    save_json(run / 'input_hashes.json', hashes)
    smoke = yaml.safe_load((base / 'sources/classic_strategy_smoke.yaml').read_text())
    portfolio = yaml.safe_load((base / 'sources/monthly_portfolio_audit.yaml').read_text())['portfolio']
    portfolio.update(capital=account_cfg['account']['capital'], top_n=5, band_mult=1.,
                     **{k: smoke[k] for k in ['buy_cost', 'sell_cost', 'min_fee', 'cash_dividend_haircut', 'annual_days']})
    rules = yaml.safe_load((base / 'sources/corporate_action_components.yaml').read_text())
    G.update(plan=plan, run=run, frames=frames, eligible=eligible, calendar=calendar, features=features,
             labels=labels, wide=wide, data=data, smoke=smoke, portfolio=portfolio, rules=rules)
    # 真实数据先截断再重建特征/标签，验证可得标签不会因未来数据改变。
    cutoff = pd.Timestamp('2020-12-31')
    short_eligible = eligible.loc[:cutoff]
    short_wide = {k: v.loc[:cutoff] for k, v in wide.items()}
    short_f = build_features(short_wide, short_eligible, calendar, plan['features'])
    for k in features:
        pd.testing.assert_frame_equal(features[k].loc[:cutoff], short_f[k])
    short_l = build_labels(short_wide, short_eligible.index, plan['common_axes']['horizon'])
    for horizon, label in short_l.items():
        valid = label['end'].notna()
        for key in ['returns', 'risk']:
            pd.testing.assert_frame_equal(labels[horizon][key].loc[valid.index[valid]], label[key].loc[valid])
    save_json(run / 'real_prefix_checks.json', dict(features=len(features), label_horizons=len(labels), passed=True))
    return account_cfg['dataset']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, default=ROOT / 'configs/learning_strategy_search_v1.yaml')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    plan = yaml.safe_load(args.plan.read_text())
    run = ROOT / 'outputs/learning_strategy_search' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    print(f'RUN={run}', flush=True)
    sources = [Path(__file__), ROOT / 'src/learning_search.py', ROOT / 'src/rule_search.py', ROOT / 'src/portfolio.py',
               ROOT / 'src/classic_smoke.py', ROOT / 'scripts/run_rule_strategy_search.py', ROOT / 'scripts/run_classic_strategy_smoke.py',
               ROOT / 'scripts/audit_monthly_portfolio.py', args.plan, PROJECT / plan['account_config'],
               ROOT / 'configs/classic_strategy_smoke.yaml', ROOT / 'configs/monthly_portfolio_audit.yaml', ROOT / 'configs/corporate_action_components.yaml']
    (run / 'sources').mkdir()
    hashes = {str(p.relative_to(PROJECT)): digest(p) for p in sources}
    for p in sources:
        shutil.copy2(p, run / 'sources' / p.name)
    import sklearn, lightgbm
    manifest = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(), pilot=args.pilot,
                    source_hashes=hashes, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__,
                    budget=4 if args.pilot else plan['budget_per_family_fold'])
    save_json(run / 'manifest.json', manifest)
    try:
        manifest['dataset'] = setup(plan, run)
        choices = []
        years = plan['outer_years'][:1] if args.pilot else plan['outer_years']
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
            futures = [pool.submit(family_search, family, year, manifest['budget'])
                       for year in years for family in plan['families']]
            for future in as_completed(futures):
                choices.append(future.result())
                save_json(run / 'all_locked_choices.json', choices)
        real_prediction_prefix(choices)
        if not args.pilot:
            outer_accounts(choices)
            controls()
        assert all(digest(PROJECT / p) == h for p, h in hashes.items()), '运行中代码/配置改变'
        assert all(digest(run / 'inputs' / p) == h for p, h in json.loads((run / 'input_hashes.json').read_text()).items())
        errors = sum(x['errors'] for x in choices)
        manifest.update(status='complete' if errors == 0 else 'completed_with_errors',
                        finished_at=datetime.now(timezone.utc).isoformat(), trials=sum(x['attempted'] for x in choices),
                        errors=errors, no_feasible_folds=sum(x['selected'] is None for x in choices))
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save_json(run / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'source_hashes'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
