# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""固定数据与账本上的规则联合搜索；完整保存内层试验及连续外层账户。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
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
from datahub import DataHub
from portfolio import run_backtest, daily_metrics
from classic_smoke import verify_ledger, make_signals
from run_classic_strategy_smoke import load_prepared, digest, save_json
from rule_search import (RuleSignals, config_id, sample_configs, all_configs, local_configs,
                         select_trial, inner_metrics, decision_dates)

G = {}


def write_result(folder, result, signals):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ['daily', 'trades', 'events', 'holdings']:
        frame = getattr(result, name)
        frame.to_parquet(folder / f'{name}.parquet', index=name == 'daily')
    # 每次实际决策的全部非空分数保留，可重建排名与未成交的候选。
    signals.to_parquet(folder / 'signals.parquet')


def cheap_audit(result):
    d, t, h = result.daily, result.trades, result.holdings
    assert np.isfinite(d[['cash', 'equity', 'stock_value', 'receivables']]).all().all()
    assert d.cash.min() >= -1e-6
    value = h.groupby('date').value.sum().reindex(d.index, fill_value=0) if len(h) else 0
    assert (d.stock_value - value).abs().max() <= 1e-6
    assert (d.equity - d.cash - d.stock_value - d.receivables).abs().max() <= 1e-6
    if len(t):
        assert (t.signal_date < t.date).all()
        assert (t.shares > 0).all() and (t.shares % 1 == 0).all()
        buy = t.side.eq('buy')
        fee = np.maximum(G['cfg']['min_fee'], t.gross * np.where(buy, G['cfg']['buy_cost'], G['cfg']['sell_cost']))
        fee = np.where(buy, fee, np.minimum(fee, t.gross)) + t.gross * G['cfg']['slippage_rate']
        np.testing.assert_allclose(t.fee, fee, rtol=1e-12, atol=1e-6)


def reference_configs(family):
    plan = G['plan']
    if family in plan['family_spaces']:
        space = plan['family_spaces'][family]
        reference = {k: v[len(v) // 2] for k, v in space.items() if k != 'variant'}
        if family == 'S06':
            reference.update(lookback=60, skip=5)
        if family == 'S07':
            reference.update(short_window=60, long_window=120, ma_window=120)
        if 'window' in reference and 20 in space['window']:
            reference['window'] = 20
        return [dict(reference, family=family, variant=v, frequency='monthly', top_n=5)
                for v in space['variant']]
    reference = {'momentum': {'window': 20}, 'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                 'moving_average': {'fast': 20, 'slow': 60}}[family]
    return [dict(reference, family=family, frequency='monthly', top_n=5)]


def distinct_centers(rows, count):
    """优先可行解，但全不可行时仍按表现探索，不因第一轮差而停止家族。"""
    usable = [r for r in rows if r['status'] == 'ok']
    usable.sort(key=lambda r: (not r['metrics']['feasible'], -r['metrics']['median_ann']))
    chosen, groups = [], set()
    for row in usable:
        p = row['params']
        key = (p.get('variant'), p['frequency'], p['top_n'])
        if key not in groups:
            chosen.append(p)
            groups.add(key)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        used = {config_id(p) for p in chosen}
        chosen.extend(r['params'] for r in usable if r['id'] not in used)
    return chosen[:count]


def family_search(family, fold):
    plan = G['plan']
    outer = fold['outer_year']
    end = pd.Timestamp(f"{max(fold['inner_years'])}-12-31")
    folder = G['run'] / 'search' / str(outer) / family
    folder.mkdir(parents=True, exist_ok=True)
    history = {k: v.loc[:end] for k, v in G['wide'].items()}
    # 子进程中只向信号器交付内层截止日以前数据。
    model = RuleSignals(history, G['eligible'].loc[:end], G['calendar'])
    market = G['market'].loc[G['market'].date.le(end)]
    limits = G['limits'].loc[G['limits'].date.le(end)]
    rows, seen = [], set()
    maximum = G['budget']
    seed = int(config_id({'family': family, 'outer': outer, 'seed': plan['search']['seed']})[:8], 16)
    started = time.monotonic()

    def evaluate(configs, stage):
        for params in configs:
            cid = config_id(params)
            if cid in seen:
                continue
            seen.add(cid)
            trial = folder / cid
            trial.mkdir(exist_ok=True)
            record = dict(id=cid, family=family, outer_year=outer, inner_end=str(end.date()),
                          params=params, phase=stage, order=len(rows), status='running')
            try:
                signals = model.signals(params)
                cfg = dict(G['portfolio'], top_n=params['top_n'])
                result = run_backtest(signals, market, G['basics'], G['st'], limits, G['actions'],
                                      G['cfg']['start'], end, cfg, G['rules'].get('rights_issues', []),
                                      slippage_rate=G['cfg']['slippage_rate'])
                cheap_audit(result)
                write_result(trial, result, signals)
                record.update(status='ok', metrics=inner_metrics(result, cfg['annual_days'], cfg['capital']),
                              signal_nonmissing=int(signals.notna().sum().sum()),
                              signal_dates=len(signals), accounting_check='pass')
            except Exception as exc:
                record.update(status='error', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            save_json(trial / 'trial.json', record)
            rows.append(record)
            with (folder / 'trials.jsonl').open('a') as stream:
                stream.write(json.dumps(record, default=str, allow_nan=True) + '\n')
            if len(rows) % 32 == 0:
                print(f"SEARCH {outer} {family} {len(rows)}/{maximum} feasible={sum(r.get('metrics', {}).get('feasible', False) for r in rows)} elapsed={time.monotonic()-started:.0f}s", flush=True)

    exhaustive = all_configs(plan, family, maximum)
    if exhaustive is not None:
        evaluate(exhaustive, 'exhaustive')
    else:
        global_count = maximum // 2
        local_count = maximum // 4
        refs = reference_configs(family)[:global_count]
        evaluate(refs, 'reference')
        evaluate(sample_configs(plan, family, global_count - len(rows), seed, seen), 'global_joint')
        centers = distinct_centers(rows, plan['search']['retain_distinct_regions'])
        evaluate(local_configs(plan, family, centers, local_count, seed + 1, seen), 'local_joint')
        evaluate(sample_configs(plan, family, maximum - len(rows), seed + 2, seen), 'continued_global')
    best = select_trial(rows, plan['selection']['tie_band_annualized_return'], plan.get('variant_complexity'))
    choice = dict(family=family, outer_year=outer, inner_end=str(end.date()),
                  locked_at=datetime.now(timezone.utc).isoformat(), selected=best,
                  attempted=len(rows), errors=sum(r['status'] != 'ok' for r in rows),
                  feasible=sum(r.get('metrics', {}).get('feasible', False) for r in rows),
                  seconds=time.monotonic() - started,
                  no_feasible_action='cash_until_next_parameter_review' if best is None else None)
    save_json(folder / 'selection_locked.json', choice)
    print(f"LOCKED {outer} {family}: trials={len(rows)} feasible={choice['feasible']} errors={choice['errors']}", flush=True)
    return choice


def outer_accounts(choices):
    """只消费已锁定选择，拼接信号而非重置现金后的净值。"""
    first_year = min(f['outer_year'] for f in G['plan']['validation']['engineering_folds'])
    last_year = max(f['outer_year'] for f in G['plan']['validation']['engineering_folds'])
    start, end = f'{first_year}-01-01', f'{last_year}-12-31'
    model = RuleSignals(G['wide'], G['eligible'], G['calendar'])
    reports = []
    for family in sorted({x['family'] for x in choices}):
        parts, schedules, transitions = [], [], []
        for choice in sorted([x for x in choices if x['family'] == family], key=lambda x: x['outer_year']):
            year = choice['outer_year']
            dates = G['eligible'].index[G['eligible'].index.year == year]
            selected = choice['selected']
            if selected is None:
                signals = pd.DataFrame(np.nan, index=[dates[0]], columns=G['eligible'].columns)
                top_n = 5
            else:
                params = selected['params']
                # 换参数发生在新年份第一个收盘，下一交易日执行；之前旧仓仍连续计价。
                score = model.score(params).loc[dates]
                signals = score.loc[decision_dates(dates, params['frequency'], G['calendar'])]
                top_n = params['top_n']
            parts.append(signals)
            schedules.append(pd.Series(top_n, index=signals.index))
            transitions.append(dict(year=year, decision_date=dates[0], params=selected['params'] if selected else None))
        signals, schedule = pd.concat(parts).sort_index(), pd.concat(schedules).sort_index()
        result = run_backtest(signals, G['market'], G['basics'], G['st'], G['limits'], G['actions'],
                              start, end, dict(G['portfolio'], top_n=5), G['rules'].get('rights_issues', []),
                              slippage_rate=G['cfg']['slippage_rate'], top_n_schedule=schedule)
        cheap_audit(result)
        audit = verify_ledger(result, G['portfolio']['capital'], G['market'], G['limits'], G['actions'],
                              G['portfolio'], G['cfg']['slippage_rate'], G['cfg']['audit']['money_atol'])
        folder = G['run'] / 'outer' / family
        write_result(folder, result, signals)
        schedule.rename('top_n').to_frame().to_parquet(folder / 'top_n_schedule.parquet')
        save_json(folder / 'transitions.json', transitions)
        save_json(folder / 'audit.json', audit)
        # 截至每个年份重新跑同一前缀，确认以后换参数不会改变过去的账本。
        prefix_checks = []
        for year in range(first_year, last_year):
            cut = pd.Timestamp(f'{year}-12-31')
            short = signals.loc[:cut]
            replay = run_backtest(short, G['market'], G['basics'], G['st'], G['limits'], G['actions'],
                                  start, cut, dict(G['portfolio'], top_n=5), G['rules'].get('rights_issues', []),
                                  slippage_rate=G['cfg']['slippage_rate'], top_n_schedule=schedule.loc[short.index])
            pd.testing.assert_frame_equal(replay.daily, result.daily.loc[replay.daily.index])
            prefix_checks.append({'through_year': year, 'pass': True})
        save_json(folder / 'prefix_checks.json', prefix_checks)
        available = sum(x['selected'] is not None for x in choices if x['family'] == family)
        metrics = inner_metrics(result, G['portfolio']['annual_days'], G['portfolio']['capital'])
        reports.append(dict(family=family, **result.summary, risk_pass=result.summary['mdd'] >= -.30,
                            selected_years=available, all_cash_due_to_no_candidate=available == 0,
                            complete_selection_path=available == len(transitions), yearly=json.dumps(metrics['yearly'])))
        print(f"OUTER {family}: return={result.summary['total']:.4f} mdd={result.summary['mdd']:.4f} selected_years={available}", flush=True)
    pd.DataFrame(reports).to_csv(G['run'] / 'outer_summary.csv', index=False)


def controls():
    """固定基准保持原公式和参数；随机种子事先固定，不进入选参竞争。"""
    reports = []
    for algorithm in ['cash', 'momentum', 'macd', 'moving_average', 'random']:
        seeds = G['plan']['controls']['random_seeds'] if algorithm == 'random' else [G['cfg']['random_seed']]
        frequencies = ['daily', 'weekly', 'monthly'] if algorithm != 'cash' else ['daily']
        for frequency in frequencies:
            for seed in seeds:
                tag = f'{algorithm}_{frequency}' + (f'_{seed}' if algorithm == 'random' else '')
                cfg = dict(G['cfg'], random_seed=seed)
                full = (make_signals(G['old_features'], G['eligible'], cfg, algorithm, 'daily')
                        if algorithm != 'cash' else pd.DataFrame(index=G['eligible'].index, columns=G['eligible'].columns, dtype=float))
                dates = G['eligible'].loc['2021':'2023'].index
                signals = full.reindex(dates).loc[decision_dates(dates, frequency, G['calendar'])]
                result = run_backtest(signals, G['market'], G['basics'], G['st'], G['limits'], G['actions'],
                                      '2021-01-01', '2023-12-31', G['portfolio'], G['rules'].get('rights_issues', []),
                                      slippage_rate=G['cfg']['slippage_rate'])
                cheap_audit(result)
                folder = G['run'] / 'controls' / tag
                write_result(folder, result, signals)
                audit = verify_ledger(result, G['portfolio']['capital'], G['market'], G['limits'], G['actions'],
                                      G['portfolio'], G['cfg']['slippage_rate'], G['cfg']['audit']['money_atol'])
                save_json(folder / 'audit.json', audit)
                reports.append(dict(strategy=tag, **result.summary))
    pd.DataFrame(reports).to_csv(G['run'] / 'control_summary.csv', index=False)


def setup(args, run):
    plan = yaml.safe_load(args.plan.read_text())
    cfg = yaml.safe_load((PROJECT / plan['fixed_execution']['inherit']).read_text())
    account = yaml.safe_load((PROJECT / plan['account_config']).read_text())
    if os.environ.get('DATAHUB_CONTENT_ID') != account['dataset']['content_id']:
        raise ValueError('必须通过冻结启动器执行')
    portfolio = yaml.safe_load((ROOT / cfg['portfolio_config']).read_text())['portfolio']
    portfolio.update(capital=account['account']['capital'], top_n=5, band_mult=1.,
                     **{k: cfg[k] for k in ['buy_cost', 'sell_cost', 'min_fee', 'cash_dividend_haircut', 'annual_days']})
    values = load_prepared(PROJECT / plan['snapshot']['prepared_execution'], cfg, account, run)
    market, basics, st, limits, actions, eligible, features, rules, index, preparation = values
    hub = DataHub()
    fields = ['hfq_close', 'hfq_high', 'hfq_low', 'open', 'close', 'high', 'low', 'pre_close',
              'vol_share', 'amount_yuan', 'turnover_rate', 'pe_ttm', 'pb', 'circ_mv_yuan',
              'buy_lg_amount_yuan', 'sell_lg_amount_yuan', 'buy_elg_amount_yuan', 'sell_elg_amount_yuan']
    # 同一冻结数据中一次派生基础宽表，所有参数使用相同起点和市场日历。
    history = hub.cn.daily(list(eligible.columns), start='2016-01-01', end=cfg['end'], fields=fields)
    history.to_parquet(run / 'inputs/rule_history.parquet', index=False)
    calendar = hub.calendar.cn_days()
    dates = calendar[(calendar >= pd.Timestamp('2016-01-01')) & (calendar <= pd.Timestamp(cfg['end']))]
    wide = {k: history.pivot(index='date', columns='ts_code', values=k).reindex(index=dates, columns=eligible.columns) for k in fields}
    coverage = pd.DataFrame({k: v.reindex(eligible.index).notna().mean(axis=1) for k, v in wide.items()})
    coverage.to_csv(run / 'input_field_coverage.csv')
    save_json(run / 'inputs/rule_history_manifest.json', {'sha256': digest(run / 'inputs/rule_history.parquet'),
              'source_content_id': account['dataset']['content_id'], 'warmup_start': '2016-01-01',
              'missing_policy': 'market_session_grid_no_price_forward_fill_full_rolling_windows', 'fields': fields})
    G.update(plan=plan, cfg=cfg, portfolio=portfolio, account=account, market=market, basics=basics,
             st=st, limits=limits, actions=actions, eligible=eligible, old_features=features,
             rules=rules, calendar=calendar, wide=wide, run=run, budget=args.budget)
    return preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'configs/rule_strategy_search_v1.yaml')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--families', nargs='*')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--run', type=Path, help='继续同一计划和代码版本的运行，仅复用已完成家族')
    args = parser.parse_args()
    if args.budget < 8 or args.budget > 256:
        parser.error('每家族每折预算须为8至256')
    run = args.run or ROOT / 'outputs/rule_strategy_search' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if args.run:
        raise ValueError('暂不支持未校验的续跑；使用新输出目录')
    run.mkdir(parents=True)
    print(f'RUN={run}', flush=True)
    sources = [Path(__file__), ROOT / 'src/rule_search.py', ROOT / 'src/portfolio.py',
               ROOT / 'src/classic_smoke.py', ROOT / 'scripts/run_classic_strategy_smoke.py',
               ROOT / 'scripts/audit_monthly_portfolio.py', args.plan, PROJECT / 'alpha_lab/configs/classic_strategy_smoke.yaml',
               ROOT / 'configs/monthly_portfolio_audit.yaml', ROOT / 'configs/corporate_action_components.yaml',
               ROOT / 'configs/selection_account_v1.yaml']
    (run / 'sources').mkdir()
    hashes = {str(p.relative_to(PROJECT)): digest(p) for p in sources}
    for p in sources:
        shutil.copy2(p, run / 'sources' / p.name)
    manifest = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                    source_hashes=hashes, budget=args.budget, pilot=args.pilot,
                    purpose='固定93股工程样本的规则搜索；不是全市场新鲜样本外证据')
    save_json(run / 'manifest.json', manifest)
    try:
        manifest['preparation'] = setup(args, run)
        manifest['snapshot'] = G['account']['dataset']
        families = args.families or list(G['plan']['family_spaces']) + list(G['plan']['classic_tunable_variants'])
        manifest['families'] = families
        folds = G['plan']['validation']['engineering_folds']
        if args.pilot:
            folds = folds[:1]
        choices = []
        for fold in folds:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
                pending = {pool.submit(family_search, family, fold): family for family in families}
                for future in as_completed(pending):
                    choices.append(future.result())
            save_json(run / 'all_locked_choices.json', choices)
        if not args.pilot:
            outer_accounts(choices)
            controls()
        assert all(digest(PROJECT / p) == value for p, value in hashes.items()), '运行中代码/配置改变'
        input_manifest = json.loads((run / 'inputs/frozen_manifest.json').read_text())
        assert all(digest(run / 'inputs' / p) == value for p, value in input_manifest.items()), '执行输入改变'
        history_manifest = json.loads((run / 'inputs/rule_history_manifest.json').read_text())
        assert digest(run / 'inputs/rule_history.parquet') == history_manifest['sha256'], '规则基础行情改变'
        manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
                        trials=sum(c['attempted'] for c in choices), errors=sum(c['errors'] for c in choices),
                        families_with_no_feasible_fold=sum(c['selected'] is None for c in choices))
        if manifest['errors']:
            manifest['status'] = 'completed_with_errors'
    except Exception as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save_json(run / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['source_hashes', 'preparation']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
