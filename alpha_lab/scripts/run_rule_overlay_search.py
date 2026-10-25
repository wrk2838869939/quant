#!/usr/bin/env python3
"""首批固定规则覆盖层：完整联合枚举、一次锁参及同宿主成对对照。"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rule_search import RuleSignals, config_id, decision_dates, inner_metrics, select_trial
from portfolio import run_backtest, daily_metrics
from classic_smoke import verify_ledger


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/rule_overlay_search_v1.yaml')
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    base = PROJECT / config['base_run']
    run = base / 'overlay_study'
    run.mkdir(exist_ok=False)
    inputs = base / 'inputs'
    source_files = [args.config, Path(__file__), ROOT / 'src/portfolio.py', ROOT / 'src/rule_search.py', ROOT / 'src/classic_smoke.py']
    hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (run / 'sources').mkdir()
    for p in source_files:
        shutil.copy2(p, run / 'sources' / p.name)
    # 核对共享运行器代码版本，补充实验不能暗中使用不同账本或规则。
    base_manifest = json.loads((base / 'manifest.json').read_text())
    for p, value in hashes.items():
        if p in base_manifest['source_hashes']:
            assert value == base_manifest['source_hashes'][p]
    manifest = dict(status='running', preregistered_at=datetime.now(timezone.utc).isoformat(),
                    config=config, source_hashes=hashes, base_run=str(base))
    save(run / 'manifest.json', manifest)
    h = pd.read_parquet(inputs / 'rule_history.parquet')
    eligible = pd.read_parquet(inputs / 'eligible.parquet')
    calendar = pd.DatetimeIndex(sorted(h.date.unique()))
    wide = {k: h.pivot(index='date', columns='ts_code', values=k).reindex(index=calendar, columns=eligible.columns)
            for k in h if k not in ['date', 'ts_code']}
    frames = {name: pd.read_parquet(inputs / f'{name}.parquet') for name in ['market', 'basics', 'st', 'limits', 'actions']}
    smoke = yaml.safe_load((base / 'sources/classic_strategy_smoke.yaml').read_text())
    portfolio = yaml.safe_load((base / 'sources/monthly_portfolio_audit.yaml').read_text())['portfolio']
    portfolio.update(capital=config['capital'], top_n=config['top_n'], band_mult=1.,
                     **{k: smoke[k] for k in ['buy_cost', 'sell_cost', 'min_fee', 'cash_dividend_haircut', 'annual_days']})
    rules = yaml.safe_load((base / 'sources/corporate_action_components.yaml').read_text())
    inner_end = pd.Timestamp(config['inner'][1])
    builders = {'inner': RuleSignals({k: v.loc[:inner_end] for k, v in wide.items()}, eligible.loc[:inner_end], calendar),
                'outer': RuleSignals(wide, eligible, calendar)}
    score_cache = {}

    def scores(host, module, p, phase):
        model = builders[phase]
        key = host, phase
        if key not in score_cache:
            score_cache[key] = model.score(config['hosts'][host])
        score = score_cache[key]
        w = model.w
        if module == 'O01':
            spec = config['modules']['O01']
            history, q = spec['history_window'], spec['reference_quantile']
            shadow = (w['high'] - np.maximum(w['open'], w['close'])) / w['pre_close'].where(w['pre_close'] > 0)
            overnight = w['open'] / w['pre_close'].where(w['pre_close'] > 0) - 1
            vol = model.volatility(p['vol_window'])
            ratio = w['amount_yuan'] / w['amount_yuan'].rolling(spec['amount_window'], min_periods=spec['amount_window']).mean().where(lambda x: x > 0)
            shadow_ref = shadow.rolling(history, min_periods=history).quantile(q).shift()
            gap_ref = overnight.rolling(history, min_periods=history).quantile(q).shift()
            vol_ref = vol.rolling(history, min_periods=history).quantile(q).shift()
            valid = shadow.notna() & shadow_ref.notna() & overnight.notna() & gap_ref.notna() & vol.notna() & vol_ref.notna() & ratio.notna()
            count = shadow.le(shadow_ref).astype(int) + overnight.ge(gap_ref).astype(int) + vol.le(vol_ref).astype(int) + ratio.le(p['amount_ratio_max']).astype(int)
            gate = valid & count.ge(p['count_required'])
            score = score.where(gate.reindex_like(score).fillna(False))
        elif module == 'O02':
            score = score.rank(axis=1, pct=True).fillna(0).rolling(p['smooth_window'], min_periods=p['smooth_window']).mean().where(score.notna())
        period = config[phase]
        frame = score.loc[period[0]:period[1]]
        return frame.loc[decision_dates(frame.index, p['frequency'], calendar)]

    def evaluate(host, module, p, phase, folder, audit=False):
        signals = scores(host, module, p, phase)
        reserve = config['capital'] * p.get('cash_fraction', 0.)
        cfg = dict(portfolio, capital=config['capital'] - reserve, band_mult=p.get('band_mult', 1.))
        start, end = config[phase]
        market = frames['market'].loc[frames['market'].date.between(start, end)]
        limits = frames['limits'].loc[frames['limits'].date.between(start, end)]
        result = run_backtest(signals, market, frames['basics'], frames['st'], limits, frames['actions'],
                              start, end, cfg, rules.get('rights_issues', []), slippage_rate=smoke['slippage_rate'])
        if reserve:
            # 保留现金始终属于同一个10万元账户；账户中的现金/总资产与成交后现金同步回加。
            result.daily['cash'] += reserve
            result.daily['equity'] += reserve
            if not result.trades.empty:
                result.trades['cash_after'] += reserve
            metrics, returns = daily_metrics(result.daily.equity, config['capital'], portfolio['annual_days'])
            result.daily['ret'] = returns
            result.daily['nav'] = result.daily.equity / config['capital']
            result.summary.update(metrics, capital=config['capital'], mean_invested=float((result.daily.stock_value / result.daily.equity).mean()))
        assert (result.daily.equity - result.daily.cash - result.daily.stock_value - result.daily.receivables).abs().max() < 1e-6
        assert result.daily.cash.min() >= -1e-6
        folder.mkdir(parents=True, exist_ok=True)
        signals.to_parquet(folder / 'signals.parquet')
        for name in ['daily', 'trades', 'events', 'holdings']:
            getattr(result, name).to_parquet(folder / f'{name}.parquet', index=name == 'daily')
        if audit:
            checked = verify_ledger(result, config['capital'], frames['market'], frames['limits'], frames['actions'],
                                    portfolio, smoke['slippage_rate'], smoke['audit']['money_atol'])
            save(folder / 'audit.json', checked)
        return result

    choices, records = [], []
    try:
        for module, spec in config['modules'].items():
            dimensions = {k: v for k, v in spec.items() if isinstance(v, list) and k != 'conditions'}
            dimensions['frequency'] = config['frequencies']
            for host in config['hosts']:
                rows = []
                for values in itertools.product(*dimensions.values()):
                    params = dict(zip(dimensions, values), family=host, module=module, top_n=config['top_n'])
                    cid = config_id(params)
                    folder = run / 'inner' / module / host / cid
                    result = evaluate(host, module, params, 'inner', folder)
                    metrics = inner_metrics(result, portfolio['annual_days'], config['capital'])
                    row = dict(id=cid, status='ok', params=params, metrics=metrics)
                    save(folder / 'trial.json', row)
                    rows.append(row)
                    records.append(dict(module=module, host=host, id=cid, params=json.dumps(params), **{k: v for k, v in metrics.items() if k != 'yearly'}))
                best = select_trial(rows, config['tie_band'])
                choice = dict(module=module, host=host, inner_end=config['inner'][1], selected=best,
                              locked_at=datetime.now(timezone.utc).isoformat(), trials=len(rows))
                choices.append(choice)
                save(run / f'locked_{module}_{host}.json', choice)
                print(f"OVERLAY LOCKED {module} {host}: {len(rows)} trials; feasible={sum(r['metrics']['feasible'] for r in rows)}", flush=True)
        # 全部锁定后才访问外层账户；没有可行解时只记录，不凭外层重新选择。
        output = []
        for choice in choices:
            best, module, host = choice['selected'], choice['module'], choice['host']
            if best is None:
                output.append(dict(module=module, host=host, selected=False))
                continue
            params = best['params']
            candidate = evaluate(host, module, params, 'outer', run / 'outer' / module / host, audit=True)
            paired = evaluate(host, 'none', {'frequency': params['frequency']}, 'outer', run / 'paired' / module / host, audit=True)
            output.append(dict(module=module, host=host, selected=True, params=json.dumps(params),
                               total=candidate.summary['total'], ann=candidate.summary['ann'], mdd=candidate.summary['mdd'],
                               risk_pass=candidate.summary['mdd'] >= -config['max_drawdown'],
                               fees=candidate.summary['total_fees'], exposure=candidate.summary['mean_invested'],
                               paired_total=paired.summary['total'], paired_mdd=paired.summary['mdd'],
                               paired_fees=paired.summary['total_fees'], paired_exposure=paired.summary['mean_invested']))
        pd.DataFrame(records).to_csv(run / 'all_trials.csv', index=False)
        pd.DataFrame(output).to_csv(run / 'outer_summary.csv', index=False)
        save(run / 'locked_choices.json', choices)
        assert all(hashlib.sha256((PROJECT / p).read_bytes()).hexdigest() == value for p, value in hashes.items())
        manifest.update(status='complete', trials=len(records), locked_choices=len(choices), finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        save(run / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
