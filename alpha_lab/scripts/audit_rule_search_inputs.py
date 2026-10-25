#!/usr/bin/env python3
"""规则搜索的真实数据前缀检查与相同交易日窗口定义的固定基准。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rule_search import RuleSignals, decision_dates
from portfolio import run_backtest
from classic_smoke import verify_ledger


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run, inputs = args.run, args.run / 'inputs'
    h = pd.read_parquet(inputs / 'rule_history.parquet')
    eligible = pd.read_parquet(inputs / 'eligible.parquet')
    dates = pd.DatetimeIndex(sorted(h.date.unique()))
    wide = {k: h.pivot(index='date', columns='ts_code', values=k).reindex(index=dates, columns=eligible.columns)
            for k in h if k not in ['date', 'ts_code']}
    full = RuleSignals(wide, eligible, dates)
    cut = pd.Timestamp('2020-12-31')
    short = RuleSignals({k: v.loc[:cut] for k, v in wide.items()}, eligible.loc[:cut], dates)
    checks = []
    for path in sorted((run / 'search/2021').glob('*/trials.jsonl')):
        item = json.loads(path.read_text().splitlines()[0])
        p = item['params']
        a, b = full.score(p), short.score(p)
        pd.testing.assert_frame_equal(a.loc[:cut], b)
        checks.append(dict(family=p['family'], config_id=item['id'], cutoff=str(cut.date()), prefix_equal=True))
    pd.DataFrame(checks).to_csv(run / 'real_input_signal_prefix_checks.csv', index=False)
    market, basics, st, limits, actions = [pd.read_parquet(inputs / f'{name}.parquet')
                                          for name in ['market', 'basics', 'st', 'limits', 'actions']]
    cfg = yaml.safe_load((run / 'sources/classic_strategy_smoke.yaml').read_text())
    portfolio = yaml.safe_load((run / 'sources/monthly_portfolio_audit.yaml').read_text())['portfolio']
    portfolio.update(capital=100000., top_n=5, band_mult=1.,
                     **{k: cfg[k] for k in ['buy_cost', 'sell_cost', 'min_fee', 'cash_dividend_haircut', 'annual_days']})
    rules = yaml.safe_load((run / 'sources/corporate_action_components.yaml').read_text())
    old = pd.read_parquet(inputs / 'features.parquet')
    old_features = {k: old.pivot(index='date', columns='ts_code', values=k).reindex_like(eligible)
                    for k in old if k not in ['date', 'ts_code']}
    controls = [dict(family='momentum', window=20), dict(family='macd', fast=12, slow=26, signal=9),
                dict(family='moving_average', fast=20, slow=60)]
    rows, differences = [], []
    for params in controls:
        score = full.score(params)
        if params['family'] == 'momentum':
            original = old_features['ret20'].where(old_features['ret20'] > 0)
        elif params['family'] == 'macd':
            dif, dea = old_features['macd_dif'], old_features['macd_dea']
            original = ((dif - dea) / old_features['hfq_close']).where((dif > dea) & (dif > 0))
        else:
            fast, slow = old_features['ma20'], old_features['ma60']
            original = (fast / slow - 1).where((fast > slow) & (old_features['hfq_close'] > fast))
        original = original.where(eligible)
        difference = ~np.isclose(score, original, equal_nan=True, rtol=1e-10, atol=1e-10)
        differences.append(dict(family=params['family'], different_cells=int(difference.sum()),
                                total_cells=int(difference.size), reason='market_session_windows_vs_observed_quote_windows'))
        for frequency in ['daily', 'weekly', 'monthly']:
            index = eligible.loc['2021':'2023'].index
            signals = score.loc[decision_dates(index, frequency, dates)]
            result = run_backtest(signals, market, basics, st, limits, actions, '2021-01-01', '2023-12-31',
                                  portfolio, rules.get('rights_issues', []), slippage_rate=cfg['slippage_rate'])
            audit = verify_ledger(result, portfolio['capital'], market, limits, actions, portfolio,
                                  cfg['slippage_rate'], cfg['audit']['money_atol'])
            tag = params['family'] + '_' + frequency
            out = run / 'matched_controls' / tag
            out.mkdir(parents=True, exist_ok=True)
            signals.to_parquet(out / 'signals.parquet')
            for name in ['daily', 'trades', 'events', 'holdings']:
                getattr(result, name).to_parquet(out / f'{name}.parquet', index=name == 'daily')
            (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
            rows.append(dict(strategy=tag, **result.summary))
    pd.DataFrame(rows).to_csv(run / 'matched_control_summary.csv', index=False)
    pd.DataFrame(differences).to_csv(run / 'fixed_formula_calendar_differences.csv', index=False)
    print(json.dumps(dict(prefix_checks=len(checks), matched_controls=len(rows), differences=differences), ensure_ascii=False))


if __name__ == '__main__':
    main()
