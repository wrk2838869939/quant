# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""固定等权交易政策后，顺序拆分交易安排、资格交集与组合排名。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from datahub import DataHub
from baseline_combination import baseline_scores, blend_scores
import run_baseline_combination as engine
from run_classic_strategy_smoke import digest, save_json
from run_rule_strategy_search import write_result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/baseline_combination_ablation_v1.yaml')
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    parent = PROJECT / config['parent_run']
    manifest = json.loads((parent / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert os.environ.get('DATAHUB_CONTENT_ID') == manifest['dataset']['content_id']
    assert all(digest(PROJECT / p) == h for p, h in manifest['source_hashes'].items())
    plan = yaml.safe_load((parent / 'sources/baseline_combination_v1.yaml').read_text())
    registry = yaml.safe_load((parent / 'sources/selection_baselines_v1.yaml').read_text())
    account = yaml.safe_load((parent / 'sources/selection_account_v1.yaml').read_text())
    base = PROJECT / manifest['source_run']
    input_hashes = json.loads((parent / 'input_hashes.json').read_text())
    assert all(digest(base / 'inputs' / p) == h for p, h in input_hashes.items())
    output = parent / 'posthoc_ablation'
    output.mkdir()
    files = [Path(__file__), args.config, ROOT / 'docs/BASELINE_COMBINATION_ABLATION_20260920.md']
    (output / 'sources').mkdir()
    for p in files:
        shutil.copy2(p, output / 'sources' / p.name)
    record = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(), parameter_search=False,
                  parent_manifest_sha256=digest(parent / 'manifest.json'),
                  source_hashes={str(p.relative_to(PROJECT)): digest(p) for p in files})
    save_json(output / 'manifest.json', record)
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
    original = parent / 'outer' / config['source_policy']
    reference_signals = pd.read_parquet(original / 'signals.parquet')
    reference_schedule = pd.read_parquet(original / 'top_n_schedule.parquet')['top_n']

    def build(method, cutoff):
        parts = []
        for year in plan['outer_years']:
            end = min(pd.Timestamp(f'{year}-12-31'), pd.Timestamp(cutoff))
            dates = reference_signals.index[(reference_signals.index.year == year) & (reference_signals.index <= end)]
            if not len(dates):
                continue
            params = [e['annual_parameters'][year]['params'] for e in registry['baselines'].values()]
            scores = baseline_scores({k: v.loc[:end] for k, v in wide.items()}, eligible.loc[:end], calendar, params)
            if method == 'equal_weight_replay':
                score = blend_scores(scores, [1 / 3] * 3)
            else:
                score = scores[0].copy()
                if method == 'low_turnover_with_intersection':
                    score = score.where(scores[1].notna() & scores[2].notna())
                elif method != 'low_turnover':
                    raise ValueError(method)
            parts.append(score.loc[dates])
        signals = pd.concat(parts).sort_index()
        return signals, reference_schedule.loc[signals.index]

    try:
        rows, checks = [], []
        for method in config['score_controls']:
            signals, schedule = build(method, plan['outer_end'])
            result = engine.execute(signals, schedule, plan['outer_start'], plan['outer_end'])
            folder = output / method
            write_result(folder, result, signals)
            schedule.to_frame().to_parquet(folder / 'top_n_schedule.parquet')
            save_json(folder / 'audit.json', engine.audit(result))
            if method == 'equal_weight_replay':
                pd.testing.assert_frame_equal(signals, reference_signals)
                pd.testing.assert_series_equal(schedule, reference_schedule)
                for table in engine.TABLES:
                    pd.testing.assert_frame_equal(getattr(result, table), pd.read_parquet(original / f'{table}.parquet'))
            for cutoff in config['prefix_dates']:
                short, n = build(method, cutoff)
                pd.testing.assert_frame_equal(short, signals.loc[:cutoff])
                pd.testing.assert_series_equal(n, schedule.loc[:cutoff])
                replay = engine.execute(short, n, plan['outer_start'], cutoff)
                engine.compare_prefix(replay, result, pd.Timestamp(cutoff))
                checks.append(dict(method=method, cutoff=cutoff, passed=True))
            row = dict(method=method, **result.summary)
            rows.append(row)
            print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(output / 'summary.csv', index=False)
        save_json(output / 'checks.json', checks)
        assert all(digest(PROJECT / p) == h for p, h in manifest['source_hashes'].items())
        assert all(digest(base / 'inputs' / p) == h for p, h in input_hashes.items())
        assert digest(parent / 'manifest.json') == record['parent_manifest_sha256']
        record.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(), accounts=3,
                      prefix_checks=len(checks), equal_weight_account_exact_match=True)
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        save_json(output / 'manifest.json', record)


if __name__ == '__main__':
    main()
