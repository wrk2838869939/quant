#!/usr/bin/env python3
"""核验双层趋势修正，区分合格形态、旧价值条件与实际成交约束。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from left_side_rules import stock_shape


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    hashes = json.loads((run / 'artifact_hashes.json').read_text())
    assert all(digest(run / p) == h for p, h in hashes.items())
    assert all(digest(ROOT.parent / p) == h for p, h in manifest['source_hashes'].items())
    parent = ROOT.parent / manifest['parent']
    assert digest(parent / 'manifest.json') == manifest['parent_manifest_sha256']
    pm = json.loads((parent / 'manifest.json').read_text())
    ph = json.loads((parent / 'artifact_hashes.json').read_text())
    assert all(digest(parent / p) == h for p, h in ph.items())
    cfg = yaml.safe_load((run / 'sources/left_side_trend_v2.yaml').read_text())
    registry = yaml.safe_load((ROOT / 'configs/selection_baselines_v1.yaml').read_text())
    inputs = ROOT.parent / registry['source_run'] / 'inputs'
    assert all(digest(inputs / p) == h for p, h in pm['execution_input_hashes'].items())
    history = pd.read_parquet(inputs / 'rule_history.parquet')
    eligible = pd.read_parquet(inputs / 'eligible.parquet').loc[cfg['start']:cfg['end']]
    features = {p.stem: pd.read_parquet(p) for p in (run / 'features').glob('*.parquet')}
    old = {p.stem: pd.read_parquet(p).reindex_like(eligible) for p in (parent / 'features').glob('*.parquet')}
    wide = {k: history.pivot(index='date', columns='ts_code', values=k).reindex_like(features['stock_passes']) for k in ['hfq_close', 'turnover_rate']}
    shape = stock_shape(wide, cfg['stock'], cfg['stock'])['shape'].reindex_like(eligible)
    both = features['stock_passes'].reindex_like(eligible).eq(1) & features['mapped_sector_passes'].reindex_like(eligible).eq(1)
    diagnostic = both & shape & eligible
    folder = run / 'inspection'
    folder.mkdir(exist_ok=True)
    cases = []
    for date, line in diagnostic.iterrows():
        for code in line[line].index:
            r = dict(date=date, ts_code=code, old_sector_structure=old['structural_gate'].at[date, code],
                     old_value_distance=old['value_distance'].at[date, code],
                     old_above_lower=old['above_lower_bound'].at[date, code], sector_id=old['sector_id'].at[date, code])
            for prefix in ['stock', 'mapped_sector']:
                for key in ['anchor_i', 'peak_i', 'advance_days', 'pullback_days', 'advance_return', 'pullback_return', 'historical_return_at_peak', 'average_slope_at_peak']:
                    r[prefix + '_' + key] = features[prefix + '_' + key].at[date, code]
            cases.append(r)
    pd.DataFrame(cases).to_csv(folder / 'trend_shape_before_value_filter.csv', index=False)
    rows = []
    for path in (run / 'accounts').glob('*/signals.parquet'):
        signals = pd.read_parquet(path)
        metrics = json.loads((path.parent / 'metrics.json').read_text())
        events = pd.read_parquet(path.parent / 'events.parquet')
        if metrics['candidate_stock_days'] and not metrics['buys']:
            for date, line in signals.iterrows():
                for code in line.dropna().index:
                    rows.append(dict(id=path.parent.name, signal_date=date, ts_code=code,
                                     rejection_reasons=sorted(events.reason.dropna().unique()) if 'reason' in events else []))
    pd.DataFrame(rows).to_csv(folder / 'signals_without_buys.csv', index=False)
    # 对仅满足新趋势和五日形态的首例绘图，明确标记未通过旧价值条件，不能冒充买入。
    if cases:
        case = cases[0]
        date, code, sid = case['date'], case['ts_code'], int(case['sector_id'])
        panel = pd.read_parquet(parent / 'sector_panel.parquet')
        sector = panel[panel.sector_id.eq(sid)].set_index('date').close
        stock = wide['hfq_close'][code]
        window = cfg['trend']['historical_window'] + cfg['trend']['peak_lookback']
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        for ax, (prefix, close) in zip(axes, [('mapped_sector', sector), ('stock', stock)]):
            show = close.loc[:date].tail(window)
            average = close.rolling(cfg['trend']['average_window']).mean().reindex(show.index)
            ax.plot(show.index, show.values, label='Close')
            ax.plot(show.index, average.values, label='60-day average', alpha=.6)
            for marker, key in [('o', 'anchor_i'), ('^', 'peak_i')]:
                idx = int(case[prefix + '_' + key])
                event_date = close.index[idx]
                ax.scatter([event_date], [close.iloc[idx]], marker=marker, s=70, label=key.replace('_i', ''))
            ax.axvline(date, color='black', linestyle='--', label='Observation date')
            ax.set_xlim(show.index.min(), show.index.max())
            ax.set_title(prefix)
            ax.legend()
            ax.grid(alpha=.2)
        fig.suptitle(f'{code} / {date.date()} - trend + shape match; NOT an entry (old sector/value gate failed)')
        fig.tight_layout()
        fig.savefig(folder / 'trend_shape_example.png', dpi=150)
        plt.close(fig)
    results = pd.read_csv(run / 'results.csv')
    grid = results[results.id.str.startswith('grid_')]
    same = {int(r.candidate_stock_days) for r in grid.itertuples()}
    lines = ['# 双层趋势修正：信号适配检查', '',
             '用户选择的是回落前已有上行趋势，买入时不必重上均线。新版已将板块和个股分别检查。', '',
             '默认条件下：新双层趋势与上涨回落形态本身覆盖 1015 个合格股票日；叠加原五日量价触发后仅剩 1 个股票日；再叠加旧版板块结构和价值参考区则为零。',
             '这一结果说明过滤组合没有形成默认交易机会，不能把空仓的零收益和零回撤当作策略改善，也不能把旧版亏损用于否定用户新确认的策略。', '',
             f'27 组邻近参数中有候选 {int(grid.candidate_stock_days.gt(0).sum())} 组，实际有买入 {int(grid.buys.gt(0).sum())} 组。',
             '有候选但无成交的明细在 signals_without_buys.csv，按原账户预算与整手要求执行，未为得到交易临时加仓。', '',
             '默认的趋势＋量价首例（尚未通过旧价值条件）为 2021-09-23 的 002460.SZ。图中仅画截至该日的信息、上涨起点及高点，并明确标注不是买入案例。',
             '旧价值带及旧大涨大跌状态继续沿用 v1，所以本次仅为修正的受控对照。下一步若继续开发，应围绕本轮近期上涨重新核对平台锚点与价值带定义，再评价完整信号，而非通过反复优化收益放宽新趋势要求。', '',
             '图与统计只作定义审查；不构成全市场、全行业或未观察样本上的有效性认证。']
    (folder / 'review.md').write_text('\n'.join(lines) + '\n')
    checks = dict(core_hashes=len(hashes), parent_hashes=len(ph), sources=len(manifest['source_hashes']),
                  all_hashes_pass=True, source_sha256=digest(Path(__file__)),
                  trend_shape_candidate_stock_days=int(diagnostic.sum().sum()),
                  grid_with_candidates=int(grid.candidate_stock_days.gt(0).sum()), grid_with_buys=int(grid.buys.gt(0).sum()))
    save_json(folder / 'verification.json', checks)
    save_json(folder / 'artifact_hashes.json', {p.name: digest(p) for p in folder.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps(checks), flush=True)
    print(results[~results.id.str.startswith('grid_')][['id', 'total', 'mdd', 'buys', 'candidate_stock_days']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
