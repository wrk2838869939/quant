#!/usr/bin/env python3
"""核验左侧研究产物，绘制净值及参数邻域，保存信号筛选损失。"""
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
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from left_side_rules import stock_shape


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    supplement = ROOT.parent / manifest['supplement']
    assert digest(supplement / 'manifest.json') == manifest['supplement_manifest_sha256']
    sh = json.loads((supplement / 'manifest.json').read_text())['hashes']
    assert all(digest(supplement / p) == h for p, h in sh.items())
    folder = run / 'inspection'
    folder.mkdir(exist_ok=True)
    cfg = yaml.safe_load((run / 'sources/left_side_rules_v1.yaml').read_text())
    registry = yaml.safe_load((ROOT / 'configs/selection_baselines_v1.yaml').read_text())
    inputs = ROOT.parent / registry['source_run'] / 'inputs'
    assert all(digest(inputs / p) == h for p, h in manifest['execution_input_hashes'].items())
    history = pd.read_parquet(inputs / 'rule_history.parquet')
    eligible = pd.read_parquet(inputs / 'eligible.parquet').loc[cfg['start']:cfg['end']]
    feature = {p.stem: pd.read_parquet(p).reindex_like(eligible) for p in (run / 'features').glob('*.parquet')}
    all_dates = pd.read_parquet(run / 'features/sector_id.parquet').index
    wide = {f: history.pivot(index='date', columns='ts_code', values=f).reindex(index=all_dates, columns=eligible.columns)
            for f in ['hfq_close', 'turnover_rate']}
    shape = stock_shape(wide, cfg['stock'], cfg['stock'])['shape'].reindex_like(eligible)
    structural = feature['structural_gate'].eq(1)
    value = feature['above_lower_bound'].eq(1) & feature['value_distance'].le(cfg['entry']['value_tolerance'])
    gates = {'eligible': eligible, 'known_sector': eligible & feature['sector_id'].ge(0),
             'sector_structure': eligible & structural, 'sector_value': eligible & structural & value,
             'stock_shape_alone': eligible & shape, 'all_conditions': eligible & structural & value & shape}
    funnel = pd.DataFrame([dict(stage=k, stock_days=int(v.sum().sum()), days=int(v.any(axis=1).sum()), stocks=int(v.any().sum())) for k, v in gates.items()])
    funnel.to_csv(folder / 'stock_funnel.csv', index=False)
    results = pd.read_csv(run / 'results.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in ['default', 'no_leader', 'stock_only']:
        d = pd.read_parquet(run / 'accounts' / name / 'daily.parquet')
        axes[0].plot(d.index, d.nav, label=name)
    axes[0].set_title('Fixed rules / engineering universe')
    axes[0].set_ylabel('Net asset value (initial = 1)')
    axes[0].legend()
    axes[0].grid(alpha=.2)
    grid = results[results.id.str.startswith('grid_')]
    selected = grid[grid.shape_window.eq(cfg['stock']['shape_window']) & grid.value_tolerance.eq(cfg['entry']['value_tolerance'])]
    pivot = selected.pivot(index='turnover_contraction', columns='turnover_expansion', values='total')
    bound = max(abs(pivot.to_numpy()).max() * 100, 1e-6)
    image = axes[1].imshow(pivot.to_numpy() * 100, cmap='RdYlGn', aspect='auto',
                           norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    axes[1].set_xticks(range(len(pivot.columns)), pivot.columns)
    axes[1].set_yticks(range(len(pivot.index)), pivot.index)
    axes[1].set_xlabel('Early turnover expansion')
    axes[1].set_ylabel('Late / early turnover ceiling')
    axes[1].set_title('5-day shape, 12% value tolerance')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            axes[1].text(j, i, f'{pivot.iloc[i, j]:+.1%}', ha='center', va='center')
    fig.colorbar(image, ax=axes[1], label='Total net return (%)')
    fig.tight_layout()
    fig.savefig(folder / 'overview.png', dpi=150)
    plt.close(fig)
    trades = pd.read_parquet(run / 'accounts/default/trades.parquet')
    if len(trades):
        trades.head(20).to_csv(folder / 'first_trades.csv', index=False)
        states = pd.read_parquet(run / 'sector_states.parquet')
        diagnostics = pd.read_parquet(run / 'accounts/default/candidate_diagnostics.parquet')
        sectors = json.loads((run / 'sector_ids.json').read_text())
        catalog = pd.read_parquet(supplement / 'catalog.parquet')
        examples = []
        # 按交易时间展示前三次买入的信号前历史，不按盈亏挑案例。
        for number, trade in enumerate(trades[trades.side.eq('buy')].head(3).itertuples(), 1):
            signal = pd.Timestamp(trade.signal_date)
            code = trade.ts_code
            detail = diagnostics[(diagnostics.date == signal) & diagnostics.ts_code.eq(code)].iloc[0]
            sector = sectors[int(detail.sector_id)]
            version = cfg['membership']['older_version'] if signal < pd.Timestamp(cfg['membership']['version_change']) else cfg['membership']['newer_version']
            sector_name = catalog.loc[catalog.index_code.eq(sector) & catalog.src.eq(version), 'industry_name'].iloc[0]
            timeline = states[(states.sector == sector) & (states.date <= signal)].tail(300)
            stock = history[(history.ts_code == code) & (history.date <= signal)].sort_values('date').tail(35)
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            axes[0].plot(timeline.date, timeline.close, label='Self-calculated sector')
            last = timeline.iloc[-1]
            axes[0].axhspan(last.platform_low, last.value_high, color='orange', alpha=.2, label='Historical reference band')
            axes[0].set_xlim(timeline.date.min(), timeline.date.max())
            axes[0].set_title(f'{code} / {sector} / signal {signal.date()} -> next open')
            axes[0].legend()
            x = mdates.date2num(stock.date.to_numpy())
            adjusted_open = stock.open * stock.hfq_close / stock.close
            for day, op, hi, lo, cl in zip(x, adjusted_open, stock.hfq_high, stock.hfq_low, stock.hfq_close):
                color = 'tab:red' if cl >= op else 'tab:green'
                axes[1].vlines(day, lo, hi, color=color, linewidth=1)
                axes[1].add_patch(plt.Rectangle((day - .3, min(op, cl)), .6, max(abs(cl - op), cl * .0001), facecolor=color))
            axes[1].xaxis_date()
            axes[1].set_xlim(stock.date.min() - pd.Timedelta(days=1), stock.date.max() + pd.Timedelta(days=1))
            axes[1].set_ylabel('Adjusted daily candles')
            axes[2].bar(stock.date, stock.turnover_rate, label='Turnover (%)')
            axes[2].axvspan(stock.date.iloc[-cfg['stock']['shape_window']], signal, color='orange', alpha=.2)
            axes[2].legend()
            for ax in axes:
                ax.grid(alpha=.2)
            fig.tight_layout()
            filename = f'entry_example_{number}.png'
            fig.savefig(folder / filename, dpi=150)
            plt.close(fig)
            examples.append(dict(ts_code=code, signal_date=str(signal.date()), sector=sector, sector_name=sector_name, figure=filename,
                                 expansion=float(detail.expansion), contraction=float(detail.contraction),
                                 early_return=float(detail.early_return), late_return=float(detail.late_return)))
        (folder / 'entry_examples.json').write_text(json.dumps(examples, indent=2))
    check = dict(core_artifacts=len(hashes), source_files=len(manifest['source_hashes']), execution_inputs=len(manifest['execution_input_hashes']),
                 supplement_artifacts=len(sh), all_hashes_pass=True,
                 inspection_source_sha256=digest(Path(__file__)))
    (folder / 'verification.json').write_text(json.dumps(check, indent=2))
    (folder / 'artifact_hashes.json').write_text(json.dumps({p.name: digest(p) for p in folder.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'}, indent=2))
    print(json.dumps(check), flush=True)
    print(funnel.to_string(index=False), flush=True)
    print(results[results.id.isin(['default', 'no_leader', 'stock_only'])][['id', 'total', 'mdd', 'buys', 'candidate_stock_days', 'mean_invested']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
