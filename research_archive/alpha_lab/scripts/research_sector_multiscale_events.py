# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""保持涨跌幅和确认门槛，检验多窗口识别对事件覆盖的增量。"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import sys
import unittest
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from sector_dataset import SectorDataset, digest
from sector_shape_events import identify_events
from sector_shape_expansion import stage_events
from sector_multiscale_events import merge_signals


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def main():
    config = ROOT / 'configs/sector_multiscale_events_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    source = PROJECT / cfg['source']
    source_manifest = json.loads((source / 'manifest.json').read_text())
    for name, expected in source_manifest['artifacts'].items():
        assert digest(source / name) == expected, name
    expansion = yaml.safe_load((source / 'sector_shape_expansion_v1.yaml').read_text())
    base = yaml.safe_load((source / 'sector_shape_events_v1.yaml').read_text())
    params = {**base, **expansion['variants']['expanded']}
    ds = SectorDataset(PROJECT / params['dataset'])
    assert source_manifest['dataset_content_id'] == ds.manifest['content_id']
    assert os.environ.get('DATAHUB_CONTENT_ID') == ds.manifest['market_content_id']
    tests = unittest.TestSuite()
    for pattern in ['test_sector_shape_events.py', 'test_sector_shape_expansion.py', 'test_sector_multiscale_events.py']:
        tests.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=1).run(tests)
    assert result.wasSuccessful()
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    save_json(out / 'manifest.json', dict(status='building', config=cfg))
    panel = pd.read_parquet(ds.root / 'daily.parquet')
    old = pd.read_parquet(source / 'expanded_observations.parquet')
    signals, coverage, cumulative, checks = [], [], [], []
    for priority, window in enumerate(cfg['platform_windows']):
        if window == params['platform_window']:
            staged = old.copy()
        else:
            observations, _ = identify_events(panel, {**params, 'platform_window': window})
            staged = stage_events(observations, expansion['confirmation_memory_days'])
        if window == cfg['platform_windows'][1]:
            # 新窗口真实截断复算，不能因未来是否完成上涨改变历史快照。
            cut = pd.Timestamp(params['prefix_check_end'])
            partial, _ = identify_events(panel[panel.date <= cut], {**params, 'platform_window': window})
            partial = stage_events(partial, expansion['confirmation_memory_days'])
            pd.testing.assert_frame_equal(partial, staged[staged.date <= cut].reset_index(drop=True))
            checks.append(dict(window=window, upstream_prefix_exact=True))
        staged['window'] = window
        staged['priority'] = priority
        first = staged[staged.first_watch | staged.first_trigger | staged.first_confirmed].copy()
        first.to_parquet(out / f'window_{window}_first_stages.parquet', index=False)
        signals.append(first)
        for stage in ['watch', 'trigger', 'confirmed']:
            hit = first[first['first_' + stage]]
            coverage.append(dict(window=window, stage=stage, events=len(hit), dates=hit.date.nunique(), sectors=hit.sector_code.nunique()))
            combined = pd.concat([s[s['first_' + stage]] for s in signals], ignore_index=True)
            merged = merge_signals(combined)
            accepted = merged[merged.accepted]
            cumulative.append(dict(added_window=window, stage=stage, raw_signals=len(combined), events=len(accepted),
                dates=accepted.date.nunique(), sectors=accepted.sector_code.nunique(), duplicates=int((~merged.accepted).sum())))
        print('WINDOW', window, json.dumps(cumulative[-3:], ensure_ascii=False), flush=True)
    all_signals = pd.concat(signals, ignore_index=True)
    pd.DataFrame(coverage).to_csv(out / 'single_window_coverage.csv', index=False)
    pd.DataFrame(cumulative).to_csv(out / 'cumulative_union_coverage.csv', index=False)
    merged_stages = {}
    for stage in ['watch', 'trigger', 'confirmed']:
        hits = all_signals[all_signals['first_' + stage]]
        merged = merge_signals(hits)
        cut = pd.Timestamp(params['prefix_check_end'])
        pd.testing.assert_frame_equal(merge_signals(hits[hits.date <= cut]), merged[merged.date <= cut].reset_index(drop=True))
        merged.to_parquet(out / f'{stage}_all_signals_with_dedup.parquet', index=False)
        merged[merged.accepted].to_csv(out / f'{stage}_events.csv', index=False)
        merged_stages[stage] = merged
    confirmed = merged_stages['confirmed']
    accepted = confirmed[confirmed.accepted].copy()
    baseline_window = params['platform_window']
    # 下列组归属只供事后覆盖诊断；不作为信号特征。
    groups = confirmed.groupby('merged_event_id').agg(
        earliest=('date', 'min'), latest=('date', 'max'), signal_count=('event_id', 'size'),
        windows=('window', lambda x: '|'.join(map(str, sorted(x.unique())))),
        contains_baseline=('window', lambda x: x.eq(baseline_window).any()))
    groups.to_csv(out / 'group_coverage_RETROSPECTIVE.csv')
    new_groups = groups.index[~groups.contains_baseline]
    accepted['baseline_group_absent_RETROSPECTIVE'] = accepted.merged_event_id.isin(new_groups)
    old_active = old[['sector_code', 'date', 'event_id', 'rally_confirmed', 'trend_at_peak', 'pullback_gate', 'near_reference', 'slowed']].rename(columns={'event_id': 'old_active_event_id'})
    accepted = accepted.merge(old_active, on=['sector_code', 'date'], how='left', suffixes=('', '_old'), validate='one_to_one')
    accepted.to_parquet(out / 'confirmed_coverage_diagnostics.parquet', index=False)
    new = accepted[accepted.baseline_group_absent_RETROSPECTIVE]
    blockers = new.old_active_event_id.notna().value_counts().rename_axis('old_event_active').rename('events')
    blockers.to_csv(out / 'new_events_old_state_presence.csv')
    annual = accepted.groupby(accepted.date.dt.year).agg(events=('merged_event_id', 'size'), dates=('date', 'nunique'), sectors=('sector_code', 'nunique'))
    annual.to_csv(out / 'annual_confirmed.csv')
    baseline = old[old.first_confirmed]
    catalog = pd.read_parquet(ds.root / 'catalog_sw_l2.parquet')
    names = {(r.src, r.index_code): r.industry_name for r in catalog.itertuples()}
    font_manager.fontManager.addfont(str(PROJECT / params['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / params['font']).get_name(), 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    cv = pd.DataFrame(cumulative)
    for stage, label in [('watch', '观察池'), ('trigger', '价格区触发'), ('confirmed', '确认')]:
        g = cv[cv.stage.eq(stage)]
        axes[0].plot(range(len(g)), g.events, marker='o', label=label)
    axes[0].set_xticks(range(len(cfg['platform_windows'])), [f'累计加入{w}日' for w in cfg['platform_windows']])
    axes[0].set(ylabel='去重事件数', title='窗口按预设次序加入，重复回落只计一次')
    axes[0].legend()
    old_year = baseline.groupby(baseline.date.dt.year).size().reindex(annual.index, fill_value=0)
    x = np.arange(len(annual))
    axes[1].bar(x - .2, old_year, width=.4, label='原15日窗口')
    axes[1].bar(x + .2, annual.events, width=.4, label='多窗口去重')
    axes[1].set_xticks(x, annual.index.astype(str), rotation=45)
    axes[1].set(ylabel='确认事件数', title='年度覆盖')
    axes[1].legend()
    fig.savefig(out / 'coverage.png', dpi=params['dpi'])
    plt.close(fig)
    # 只按覆盖差异与日期抽例，不读取未来收益；组是否含旧版只用于复核选样。
    pool = new.sort_values(['date', 'sector_code'])
    indices = np.unique(np.linspace(0, len(pool)-1, min(cfg['review_cases'], len(pool))).astype(int)) if len(pool) else []
    pages = []
    (out / 'cases').mkdir()
    with PdfPages(out / 'additional_cases.pdf') as book:
        for r in pool.iloc[indices].itertuples():
            history = panel[(panel.sector_code == r.sector_code) & (panel.segment_id == r.segment_id) & panel.index_valid & (panel.date <= r.date)].sort_values('date')
            start = max(0, int((history.date < r.platform_start).sum()) - cfg['review_context_days'])
            history = history.iloc[start:]
            fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout='constrained', gridspec_kw={'height_ratios': [3, 1]})
            fig.get_layout_engine().set(rect=(0, .06, 1, .94))
            ax = axes[0]
            ax.plot(history.date, history.index_close, color='#286596', label='板块收盘指数')
            ax.axvspan(r.platform_start, r.platform_end, color='#78a985', alpha=.22, label=f'{r.window}日平台候选')
            ax.axhspan(r.platform_low, r.reference_high, color='#dba969', alpha=.20, label='平台至上涨初期参照')
            ax.axhline(r.support_at_peak, color='#976bb0', ls=':', label='峰前支撑')
            ax.scatter([r.peak_date, r.date], [r.peak_price, r.index_close], color=['#d0872c', '#b02e39'], zorder=4)
            ax.set_ylabel('板块点位')
            ax.legend(ncol=2, fontsize=9)
            axes[1].plot(history.date, history.turnover_pct, color='#80629d')
            axes[1].set_ylabel('换手率（%）')
            for ax in axes:
                ax.axvline(r.date, color='#b02e39', ls='--')
                ax.set_xlim(history.date.min(), history.date.max() + pd.Timedelta(days=2))
                ax.grid(alpha=.15)
            axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
            name = names.get((r.classification_version, r.sector_code), r.sector_code)
            number = len(pages) + 1
            fig.suptitle(f'{number:02d} {name} {r.date:%Y-%m-%d} · 新覆盖回落／{r.window}日窗口\n先涨{r.rally_return:.1%}，后跌{r.drawdown:.1%}，峰后{r.pullback_days}日', fontsize=14)
            fig.text(.025, .02, '只绘确认日及以前行情；绿色区仍是待人工核对的平台候选，可能含斜坡或V形。', fontsize=10)
            filename = f'cases/{number:02d}_{r.merged_event_id}.png'
            fig.savefig(out / filename, dpi=params['dpi'])
            book.savefig(fig)
            plt.close(fig)
            pages.append(dict(number=number, name=name, date=str(r.date.date()), event_id=r.merged_event_id, file=filename))
    pd.DataFrame(pages).to_csv(out / 'review_cases.csv', index=False)
    (out / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>多窗口新增案例</title><style>body{max-width:1200px;margin:auto;font-family:system-ui}img{width:100%}</style><h1>多窗口新增案例</h1>' + ''.join(f'<h2>{r["number"]} {r["name"]} {r["date"]}</h2><img loading="lazy" src="{r["file"]}">' for r in pages))
    summary = dict(baseline_events=len(baseline), baseline_dates=baseline.date.nunique(),
        merged_events=len(accepted), merged_dates=accepted.date.nunique(), merged_sectors=accepted.sector_code.nunique(),
        raw_confirmation_signals=len(confirmed), merged_duplicates=int((~confirmed.accepted).sum()),
        groups_without_baseline=len(new), new_groups_old_event_active=int(new.old_active_event_id.notna().sum()),
        groups_with_baseline=int(groups.contains_baseline.sum()))
    save_json(out / 'summary.json', summary)
    lines = ['# 多平台窗口的事件覆盖诊断', '',
        '保持现有上涨15%、回落10%、10日内快跌7%、历史上行、参考区和5日顺序确认条件。只并行使用10／15／20／30／40日平台窗口，每个窗口独立维护已有状态机。未计算后续收益。', '',
        '## 去重口径', '',
        '- 每个窗口先取本事件首次信号，再按时间合并。',
        '- 同板块同连续段，若新信号的峰值不晚于前一个已保留确认日，归为重复回落；只有前次确认后形成新峰值，才允许再计新事件。',
        '- 同日冲突按配置顺序15、10、20、30、40日保留代表，不按未来走势或更晚信号挑选。观察池、触发、确认各层独立执行相同去重。',
        '- 去重偏保守，可能合并部分相近回落；每板块各窗口仍各自只维护一个事件，不等于已经支持所有嵌套平台。', '',
        '## 规模', '', *[f'- {k}: {v}' for k, v in summary.items()], '',
        '| 累计加入窗口 | 观察池 | 价格区触发 | 确认 | 确认日期 |', '|---|---:|---:|---:|---:|']
    for window in cfg['platform_windows']:
        sub = cv[cv.added_window.eq(window)].set_index('stage')
        lines.append(f'| {window}日 | {sub.loc["watch", "events"]} | {sub.loc["trigger", "events"]} | {sub.loc["confirmed", "events"]} | {sub.loc["confirmed", "dates"]} |')
    lines += ['', '## 如何解释', '',
        '- 多窗口新增数量说明固定窗口存在覆盖限制；不代表这些案例都比旧版更符合主观形态，也不代表收益提高。',
        '- groups_without_baseline 是最终归组后没有15日窗口信号的组数，只能用于事后覆盖诊断；它不等同于总数的简单差值。信号代表可能提前，也可能被另一窗口归并。',
        '- new_groups_old_event_active 表示新增组确认当日，原窗口存在另一个活动事件。此状态是查找锚点占用或条件不匹配的线索，不单独证明早期事件阻挡了信号。',
        '- 平台范围20%、首末变化10%的启发式未改，仍可能把V形或倾斜段当平台；图册标为平台候选，需继续复核。',
        '- 原230事件收益报告仍只适用于原样本，不能直接套到多窗口集合。', '',
        '## 文件', '',
        '- confirmed_events.csv：按时间可产生的去重确认信号；trigger_events.csv和watch_events.csv为另两层。',
        '- *_all_signals_with_dedup.parquet：每条来源信号及去重归属，保留审计痕迹。',
        '- confirmed_coverage_diagnostics.parquet、group_coverage_RETROSPECTIVE.csv：含事后分组诊断，不能直接作为信号特征。',
        '- additional_cases.pdf、review.html：按日期等距抽取新增覆盖案例，只展示确认时已知行情。',
        '- single_window_coverage.csv、cumulative_union_coverage.csv、annual_confirmed.csv：单窗口、合并及年度覆盖。',
        f'- {result.testsRun}项测试通过；10日窗口真实截断重算一致；三层合并真实前缀一致；源文件哈希核验通过。', '']
    (out / 'README.md').write_text('\n'.join(lines))
    save_json(out / 'validation.json', dict(tests=result.testsRun, upstream_checks=checks, merged_prefix_exact=True, source_hashes=True))
    for path in [config, Path(__file__), ROOT / 'src/sector_multiscale_events.py', ROOT / 'src/sector_shape_events.py',
                 ROOT / 'src/sector_shape_expansion.py', ROOT / 'tests/test_sector_multiscale_events.py',
                 source / 'sector_shape_expansion_v1.yaml', source / 'sector_shape_events_v1.yaml']:
        shutil.copy2(path, out / path.name)
    save_json(out / 'manifest.json', dict(status='complete', config=cfg, dataset_content_id=ds.manifest['content_id'],
        source_manifest_sha256=digest(source / 'manifest.json'),
        artifacts={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print(json.dumps(dict(output=str(out), **summary), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
