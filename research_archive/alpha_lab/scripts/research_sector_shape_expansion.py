# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""事件门槛与顺序确认的覆盖诊断，输出仅截至观察日的案例图册。"""
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


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def main():
    config = ROOT / 'configs/sector_shape_expansion_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    base_path = PROJECT / cfg['base_config']
    base = yaml.safe_load(base_path.read_text())
    ds = SectorDataset(PROJECT / base['dataset'])
    assert os.environ.get('DATAHUB_CONTENT_ID') == ds.manifest['market_content_id']
    source = PROJECT / cfg['source_events']
    source_manifest = json.loads((source / 'manifest.json').read_text())
    for name, expected in source_manifest['artifacts'].items():
        assert digest(source / name) == expected, name
    suite = unittest.TestSuite()
    for pattern in ['test_sector_shape_events.py', 'test_sector_shape_expansion.py']:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern))
    tested = unittest.TextTestRunner(verbosity=1).run(suite)
    assert tested.wasSuccessful()
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    print('OUTPUT', out, flush=True)
    save_json(out / 'manifest.json', {'status': 'building'})
    panel = pd.read_parquet(ds.root / 'daily.parquet')
    old = pd.read_parquet(source / 'observations.parquet')
    catalog = pd.read_parquet(ds.root / 'catalog_sw_l2.parquet')
    names = {(r.src, r.index_code): r.industry_name for r in catalog.itertuples()}
    rows, full = [], {}
    for variant, changes in cfg['variants'].items():
        params = {**base, **changes}
        if variant == 'original':
            obs = old.copy()
            episodes = pd.read_parquet(source / 'episodes.parquet')
        else:
            obs, episodes = identify_events(panel, params)
        staged = stage_events(obs, cfg['confirmation_memory_days'])
        full[variant] = staged
        assert (~staged.candidate | staged.confirmed).all()
        confirmed = staged[staged.confirmed]
        assert confirmed.trend_at_peak.all()
        assert (confirmed.rally_return >= params['minimum_rally']).all()
        assert (confirmed.drawdown <= -params['minimum_pullback']).all()
        assert (confirmed.platform_end < confirmed.breakout_date).all()
        assert (confirmed.peak_date < confirmed.date).all()
        for stage, mask in [('平台突破', np.ones(len(staged), dtype=bool)),
                ('上涨达标', staged.rally_confirmed),
                ('上涨且历史趋势达标', staged.rally_confirmed & staged.trend_at_peak),
                ('结构回落', staged.pullback_gate), ('观察池', staged.watch),
                ('价格区触发', staged.trigger), ('同日确认', staged.candidate),
                ('五日顺序确认', staged.confirmed)]:
            selected = staged.loc[mask]
            first = selected.drop_duplicates('event_id')
            rows.append(dict(variant=variant, stage=stage, events=len(first),
                dates=first.date.nunique(), sectors=first.sector_code.nunique(), days=len(selected)))
        staged.to_parquet(out / f'{variant}_observations.parquet', index=False)
        episodes.to_parquet(out / f'{variant}_episodes_RETROSPECTIVE.parquet', index=False)
        hits = staged[staged.first_watch | staged.first_trigger | staged.first_confirmed].copy()
        hits['name'] = [names.get((r.classification_version, r.sector_code), r.sector_code) for r in hits.itertuples()]
        hits.to_csv(out / f'{variant}_first_stages.csv', index=False)
        print(variant, json.dumps(rows[-3:], ensure_ascii=False), flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'coverage.csv', index=False)
    # 真实截断重算同时验证新增结构和顺序确认；中间态完全逐值一致。
    cut = pd.Timestamp(base['prefix_check_end'])
    params = {**base, **cfg['variants']['expanded']}
    short, _ = identify_events(panel[panel.date <= cut], params)
    short = stage_events(short, cfg['confirmation_memory_days'])
    expected = full['expanded'][full['expanded'].date <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(short, expected)
    # 新阈值也必须拒绝纯下跌，不能用增大样本量替代形态语义。
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_sector_shape_events import panel as synthetic_panel
    declining, _ = identify_events(synthetic_panel(np.linspace(200, 50, 300)), params)
    assert declining.empty
    expanded = full['expanded']
    yearly = []
    for variant, staged in full.items():
        for stage in ['watch', 'trigger', 'confirmed']:
            first = staged[staged['first_' + stage]]
            for year, part in first.groupby(first.date.dt.year):
                yearly.append(dict(variant=variant, stage=stage, year=year, events=len(part), dates=part.date.nunique()))
    pd.DataFrame(yearly).to_csv(out / 'annual_coverage.csv', index=False)
    # 仅用于标注图册差异；使用旧版截至同一观察日已经确认的同峰事件。
    old_peaks = old[old.candidate].groupby(['sector_code', 'segment_id', 'peak_date']).date.min().to_dict()
    expanded['old_same_peak_confirmed'] = [old_peaks.get((r.sector_code, r.segment_id, r.peak_date), pd.Timestamp.max) <= r.date for r in expanded.itertuples()]
    groups = [
        ('新版确认／旧版同峰已确认', expanded[expanded.first_confirmed & expanded.old_same_peak_confirmed]),
        ('新版确认／旧版同峰尚未确认', expanded[expanded.first_confirmed & ~expanded.old_same_peak_confirmed]),
        ('首次价格区触发／当日未确认', expanded[expanded.first_trigger & ~expanded.confirmed]),
        ('首次进入观察池／尚未接近价格区', expanded[expanded.first_watch & ~expanded.trigger])]
    font_manager.fontManager.addfont(str(PROJECT / base['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / base['font']).get_name(),
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    cases = out / 'cases'
    cases.mkdir()
    pages = []
    with PdfPages(out / 'trigger_confirmation_review.pdf') as pdf:
        for category, pool in groups:
            pool = pool.sort_values(['date', 'sector_code'])
            indices = np.unique(np.linspace(0, len(pool) - 1, min(cfg['review_per_group'], len(pool))).astype(int)) if len(pool) else []
            for r in pool.iloc[indices].itertuples():
                history = panel[(panel.sector_code == r.sector_code) & (panel.segment_id == r.segment_id) & panel.index_valid & (panel.date <= r.date)].sort_values('date')
                start = max(0, int((history.date < r.platform_start).sum()) - cfg['review_context_days'])
                history = history.iloc[start:].tail(cfg['review_max_prior_days'])
                fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout='constrained', gridspec_kw={'height_ratios': [3, 1, 1]})
                ax = axes[0]
                ax.plot(history.date, history.index_close, color='#286596', label='板块指数')
                ax.axvspan(r.platform_start, r.platform_end, color='#78a985', alpha=.22, label='上涨前平台（时间）')
                ax.axhspan(r.platform_low, r.reference_high, color='#dba969', alpha=.20, label='平台—上涨初期参照（价格）')
                ax.axhline(r.reference_high * (1 + params['reference_upper_tolerance']), color='#b98839', ls='--', label='参考区上容差')
                ax.axhline(r.support_at_peak, color='#976bb0', ls=':', label='峰前支撑')
                ax.scatter([r.peak_date, r.date], [r.peak_price, r.index_close], color=['#d0872c', '#b02e39'], zorder=4)
                ax.legend(fontsize=8, ncol=3)
                ax.set_ylabel('板块点位')
                axes[1].plot(history.date, history.up_share, color='#368467')
                axes[1].set(ylabel='上涨成员占比', ylim=(0, 1))
                axes[2].plot(history.date, history.turnover_pct, color='#80629d')
                axes[2].set_ylabel('换手率（%）')
                for ax in axes:
                    ax.axvline(r.date, color='#b02e39', ls='--')
                    ax.set_xlim(history.date.min(), history.date.max() + pd.Timedelta(days=2))
                    ax.grid(alpha=.15)
                axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
                number = len(pages) + 1
                name = names.get((r.classification_version, r.sector_code), r.sector_code)
                fig.suptitle(f'{number:02d} {name} {r.date:%Y-%m-%d} · {category}\n先涨 {r.rally_return:.1%} → 回落 {r.drawdown:.1%}；峰后 {r.pullback_days} 日', fontsize=13)
                fig.text(.01, .003, f'仅绘截至观察日行情。快跌已见={r.fast_fall_seen}；当前破支撑={r.support_broken}；5日内破支撑={r.support_recent}；当前减速={r.slowed}。', fontsize=9)
                filename = f'{number:02d}_{r.event_id}.png'
                fig.savefig(cases / filename, dpi=base['dpi'])
                pdf.savefig(fig)
                plt.close(fig)
                pages.append(dict(page=number, category=category, name=name, event_id=r.event_id, date=str(r.date.date()), file='cases/' + filename))
    pd.DataFrame(pages).to_csv(out / 'review_pages.csv', index=False)
    (out / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>触发与确认案例</title><style>body{max-width:1200px;margin:auto;font-family:system-ui}img{width:100%}</style><h1>触发与确认案例复核</h1><p>每组按日期均匀抽取；只展示观察日及以前行情。</p>' + ''.join(f'<h2>{p["page"]} {p["name"]} {p["date"]} {p["category"]}</h2><img loading="lazy" src="{p["file"]}">' for p in pages))
    labels = {'original': '原规则', 'amplitude': '仅调整涨跌幅', 'platform': '仅调整平台', 'trend': '仅调整趋势窗口', 'expanded': '组合试行规则'}
    table = summary.pivot(index='variant', columns='stage', values='events').reindex(cfg['variants'])
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    stages = ['结构回落', '观察池', '价格区触发', '同日确认', '五日顺序确认']
    table[stages].rename(index=labels).plot.bar(ax=ax, rot=0)
    ax.set(ylabel='独立事件数（各规则内部去重）', xlabel='', title='触发与确认的覆盖比较')
    fig.savefig(out / 'coverage.png', dpi=base['dpi'])
    plt.close(fig)
    lines = ['# 板块触发与确认条件：覆盖扩展试行版', '',
        '本轮围绕事件定义与案例覆盖做比较。各版本参数事先写入配置，未计算后续收益，也未据收益选择条件。组合版供形态复核，尚未认定为最优规则。', '',
        '## 分层条件', '',
        '- 结构回落：同轮平台后上涨、峰值时历史上行、随后回落。',
        '- 观察池：结构回落且峰后曾快速下跌，尚不要求已到价格参照区或减速。',
        '- 价格区触发：观察池进入本轮平台／上涨初期参照附近。',
        '- 确认：仍在回落和参考区，当前跌速放缓；最近5日内曾触发且曾跌破支撑。支持先后出现，新峰值后记忆清零。',
        '- 同一事件各层首次只计一次；多日停留不增加事件数。观察池是待跟踪清单，不能当成买入确认。', '',
        '## 分项比较', '',
        '| 版本 | 结构回落 | 观察池 | 价格区触发 | 同日确认 | 五日顺序确认 |',
        '|---|---:|---:|---:|---:|---:|']
    for variant, row in table.iterrows():
        lines.append('| ' + labels[variant] + ' | ' + ' | '.join(str(int(row[s])) for s in stages) + ' |')
    lines += ['', '## 改动与限制', '',
        '- 涨跌幅组：上涨20%→15%，回落15%→10%，10日内快跌10%→7%。',
        '- 平台组：窗口20→15日，收盘范围15%→20%，首末绝对涨跌8%→10%，突破幅度2%→1%。更容易纳入倾斜或V形平台，需人工复核。',
        '- 趋势组：历史涨幅窗口120→60日，仍须上涨≥5%；均线60→40日，均线抬升间隔20→10日。当前仍无需站上均线。',
        '- 组合版同时应用上述三组改动。原来的支撑定义、价格参照容差、减速比例、峰后最长40日等保持原值。',
        '- 单项和组合变化会改变事件起点、占用状态和超时路径，因此效果不保证单调或相加。不同版本事件数不能相加。',
        '- 原状态机每板块每行情段同时只维护一个平台事件，尚未覆盖嵌套／重叠平台；本次没有通过重复计数扩大案例。',
        '- 新版并非旧版案例的严格超集。图册以同板块、同段、同峰值且截至当前已确认对照旧版，不能把“旧版同峰尚未确认”一律解释为新增加的独立行情。',
        '- 仍使用当前冻结申万二级数据集，历史归属缺失和分类粒度问题未改变。', '',
        '## 查看与复现', '',
        '- trigger_confirmation_review.pdf / review.html：四类案例各最多6页，按时间等距抽取，不看后续走势。',
        '- coverage.csv：各层事件数、首次日期数、板块数和观察天数；annual_coverage.csv：分年度覆盖。',
        '- expanded_first_stages.csv：完整分层首次清单；各版本 observations.parquet 保存当时状态。',
        '- *_episodes_RETROSPECTIVE.parquet 为最终事后目录，不能作为早期预测特征。',
        '- 7项测试通过；组合版真实数据物理截断重算逐值一致；纯下跌不能触发；所有旧同日确认均被各自版本顺序确认包含。',
        '- 本轮通过覆盖和形态复核寻找更合适的定义，增加数量本身不是有效性的证据。', '']
    (out / 'README.md').write_text('\n'.join(lines))
    save_json(out / 'validation.json', dict(tests=tested.testsRun, real_prefix_exact=True,
        pure_decline_rejected=True, same_day_included=True, sources_verified=True))
    for path in [config, base_path, Path(__file__), ROOT / 'src/sector_shape_expansion.py', ROOT / 'src/sector_shape_events.py',
                 ROOT / 'tests/test_sector_shape_expansion.py', ROOT / 'tests/test_sector_shape_events.py']:
        shutil.copy2(path, out / path.name)
    save_json(out / 'manifest.json', dict(status='complete', dataset_content_id=ds.manifest['content_id'],
        source_manifest_sha256=digest(source / 'manifest.json'), config=cfg,
        artifacts={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print(table.to_string(), flush=True)
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
