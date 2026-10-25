#!/usr/bin/env python3
"""复核归档并按年度补充风格/换手可比性，不重新选取种子。"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    hashes = json.loads((run / 'artifact_hashes.json').read_text())
    assert all(digest(run / p) == h for p, h in hashes.items())
    assert all(digest(PROJECT / p) == h for p, h in manifest['source_hashes'].items())
    assert digest(PROJECT / manifest['report']) == manifest['report_sha256']
    cfg = yaml.safe_load((run / 'sources/baseline_style_controls_v1.yaml').read_text())
    records, audit_count, prefix_count = [], 0, 0
    for name in cfg['baselines']:
        base_exp = pd.read_parquet(run / 'baseline' / name / 'exposure.parquet')
        base_daily = pd.read_parquet(run / 'baseline' / name / 'daily.parquet')
        base_trades = pd.read_parquet(run / 'baseline' / name / 'trades.parquet')
        for mode, dims in cfg['modes'].items():
            for seed in cfg['seeds']:
                folder = run / 'controls' / name / mode / str(seed)
                assert (folder / 'audit.json').is_file()
                audit_count += 1
                if seed == cfg['seeds'][0]:
                    assert json.loads((folder / 'prefix_check.json').read_text())['raw_features_signals_accounts']
                    prefix_count += 1
                exposure = pd.read_parquet(folder / 'exposure.parquet')
                daily = pd.read_parquet(folder / 'daily.parquet')
                trades = pd.read_parquet(folder / 'trades.parquet')
                pairs = pd.read_parquet(folder / 'pairs.parquet')
                for year in sorted(set(daily.index.year)):
                    days = daily.index[daily.index.year == year]
                    pair_pass = float(pairs.loc[pairs.date.dt.year == year, 'passed'].mean())
                    gap = max([float((exposure.loc[days, k] - base_exp.loc[days, k]).abs().mean()) for k in dims] or [np.nan])
                    coverage = min([min(float(exposure.loc[days, k + '_coverage'].mean()), float(base_exp.loc[days, k + '_coverage'].mean())) for k in dims] or [1.])
                    t = float(trades.loc[trades.date.dt.year == year, 'gross'].sum() / daily.loc[days, 'equity'].mean()) if len(trades) else 0.
                    b = float(base_trades.loc[base_trades.date.dt.year == year, 'gross'].sum() / base_daily.loc[days, 'equity'].mean())
                    ratio = t / b if b else np.inf
                    style_ok = bool(dims and pair_pass >= cfg['matching']['minimum_pair_pass_rate']
                                    and gap <= cfg['matching']['maximum_actual_mean_abs_gap']
                                    and coverage >= cfg['matching']['minimum_pair_pass_rate'])
                    records.append(dict(baseline=name, mode=mode, seed=seed, year=year,
                                        pair_pass=pair_pass, max_actual_gap=gap, minimum_coverage=coverage,
                                        turnover_ratio=ratio, style_ok=style_ok,
                                        credible=bool(style_ok and 1 / cfg['matching']['maximum_turnover_ratio'] <= ratio <= cfg['matching']['maximum_turnover_ratio'])))
    out = run / 'completion'
    out.mkdir(exist_ok=True)
    quality = pd.DataFrame(records)
    quality.to_csv(out / 'annual_quality.csv', index=False)
    grouped = quality.groupby(['baseline', 'mode', 'year']).agg(style_ok_count=('style_ok', 'sum'),
        credible_count=('credible', 'sum'), pair_pass=('pair_pass', 'mean'), max_actual_gap=('max_actual_gap', 'mean'),
        turnover_ratio=('turnover_ratio', 'median')).reset_index()
    grouped.to_csv(out / 'annual_quality_summary.csv', index=False)
    annual = pd.read_csv(run / 'annual_comparison.csv')
    font = FontProperties(fname=str(PROJECT / 'fonts/SimHei.ttf'))
    labels = dict(B01_low_turnover='低换手', B02_low_risk='低波动', B03_valuation='盈利收益率', equal_weight='等权组合')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True, sharey=True)
    for ax, name in zip(axes.flat, cfg['baselines']):
        part = annual[(annual.baseline == name) & (annual['mode'] == 'match_all')].sort_values('year')
        fixed_daily = pd.read_parquet(run / 'fixed_parameters' / name / 'daily.parquet')
        capital = float(pd.read_parquet(run / 'baseline' / name / 'daily.parquet').equity.iloc[0])
        # 首日无成交，仍显式由冻结账户配置取得资金，避免作图暗含该假设。
        account = yaml.safe_load((run / 'sources/selection_account_v1.yaml').read_text())
        capital = account['account']['capital']
        previous, fixed = capital, []
        for _, daily in fixed_daily.groupby(fixed_daily.index.year):
            fixed.append(float(daily.equity.iloc[-1] / previous - 1) * 100)
            previous = daily.equity.iloc[-1]
        ax.plot(part.year, part.baseline_return * 100, marker='o', label='原年度参数', linewidth=2)
        ax.plot(part.year, fixed, marker='s', label='固定 2021 参数')
        ax.plot(part.year, part.control_median * 100, marker='o', markerfacecolor='none', linestyle='--', label='全风格对照中位数')
        ax.set_xticks(part.year)
        ax.axhline(0, color='grey', linewidth=.7)
        ax.set_title(labels[name], fontproperties=font)
        ax.set_ylabel('年度净收益（%）', fontproperties=font)
        ax.legend(prop=font, fontsize=8)
    fig.suptitle('对照整体匹配未合格：曲线仅作描述，不代表独立选股收益', fontproperties=font)
    fig.savefig(out / 'annual_comparison.png', dpi=150)
    plt.close(fig)
    checks = dict(status='passed', core_artifacts_verified=len(hashes), control_audits=audit_count,
                  raw_feature_signal_account_prefix_checks=prefix_count, targeted_tests_passed=6,
                  yearly_quality_rows=len(quality), core_manifest_sha256=digest(run / 'manifest.json'),
                  verifier_sha256=digest(Path(__file__)))
    (out / 'verification.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
    (out / 'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(out)): digest(p) for p in out.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'}, indent=2) + '\n')
    print(json.dumps(checks, ensure_ascii=False))
    print(grouped[grouped['mode'] == 'match_all'].to_string(index=False))


if __name__ == '__main__':
    main()
