#!/usr/bin/env python3
"""从已冻结的观察快照重绘触发／确认图册，修正页脚与日期重叠。"""
from pathlib import Path
import json
import shutil
import sys
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
sys.path.insert(0, str(ROOT / 'src'))
from sector_dataset import digest


def main():
    source = Path(sys.argv[1]).resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    for name, expected in manifest['artifacts'].items():
        assert digest(source / name) == expected, name
    cfg = yaml.safe_load((source / 'sector_shape_expansion_v1.yaml').read_text())
    base = yaml.safe_load((source / 'sector_shape_events_v1.yaml').read_text())
    params = {**base, **cfg['variants']['expanded']}
    panel = pd.read_parquet(ROOT.parent / base['dataset'] / 'daily.parquet')
    observations = pd.read_parquet(source / 'expanded_observations.parquet').set_index(['event_id', 'date'])
    pages = pd.read_csv(source / 'review_pages.csv', parse_dates=['date'])
    out = source / 'review_v2'
    out.mkdir(exist_ok=False)
    (out / 'cases').mkdir()
    font = ROOT.parent / base['font']
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    with PdfPages(out / 'trigger_confirmation_review.pdf') as book:
        for page in pages.itertuples():
            r = observations.loc[(page.event_id, page.date)]
            history = panel[(panel.sector_code == r.sector_code) & (panel.segment_id == r.segment_id) &
                            panel.index_valid & (panel.date <= page.date)].sort_values('date')
            start = max(0, int((history.date < r.platform_start).sum()) - cfg['review_context_days'])
            history = history.iloc[start:].tail(cfg['review_max_prior_days'])
            fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                gridspec_kw={'height_ratios': [3, 1, 1]}, layout='constrained')
            fig.get_layout_engine().set(rect=(0, .06, 1, .94))
            ax = axes[0]
            ax.plot(history.date, history.index_close, color='#286596', label='板块指数')
            ax.axvspan(r.platform_start, r.platform_end, color='#78a985', alpha=.22, label='上涨前平台（时间）')
            ax.axhspan(r.platform_low, r.reference_high, color='#dba969', alpha=.2, label='平台—上涨初期参照（价格）')
            ax.axhline(r.reference_high * (1 + params['reference_upper_tolerance']), color='#b98839', ls='--', label='参考区上容差')
            ax.axhline(r.support_at_peak, color='#976bb0', ls=':', label='峰前支撑')
            ax.scatter([r.peak_date, page.date], [r.peak_price, r.index_close], color=['#d0872c', '#b02e39'], zorder=4)
            ax.legend(fontsize=8, ncol=3)
            ax.set_ylabel('板块点位')
            axes[1].plot(history.date, history.up_share, color='#368467')
            axes[1].set(ylabel='上涨成员占比', ylim=(0, 1))
            axes[2].plot(history.date, history.turnover_pct, color='#80629d')
            axes[2].set_ylabel('换手率（%）')
            for ax in axes:
                ax.axvline(page.date, color='#b02e39', ls='--')
                ax.set_xlim(history.date.min(), history.date.max() + pd.Timedelta(days=2))
                ax.grid(alpha=.15)
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
            fig.suptitle(f'{page.page:02d} {page.name} {page.date:%Y-%m-%d} · {page.category}\n先涨 {r.rally_return:.1%} → 回落 {r.drawdown:.1%}；峰后 {r.pullback_days} 日', fontsize=13)
            yes = lambda value: '是' if value else '否'
            fig.text(.025, .02, f'仅绘截至观察日行情。曾快跌：{yes(r.fast_fall_seen)}；当前破支撑：{yes(r.support_broken)}；5日内破支撑：{yes(r.support_recent)}；当前减速：{yes(r.slowed)}。', fontsize=10)
            fig.savefig(out / page.file, dpi=base['dpi'])
            book.savefig(fig)
            plt.close(fig)
    pages.to_csv(out / 'review_pages.csv', index=False)
    (out / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>触发与确认复核</title><style>body{max-width:1200px;margin:auto;font-family:system-ui}img{width:100%}</style><h1>触发与确认复核</h1><p>与原版相同的24个案例，只绘当时已知行情；已修正页脚布局。</p>' + ''.join(f'<h2>{r.page} {r.name} {r.date:%Y-%m-%d} {r.category}</h2><img loading="lazy" src="{r.file}">' for r in pages.itertuples()))
    (out / 'README.md').write_text('''# 触发与确认图册复核版

确认事件由原版69个（35个日期、53个板块）扩展为试行版230个（115个日期、101个板块）；观察池532个，价格区触发340个。各版本内部按事件去重，不等于相互独立的行情样本。

图册仍是原先按时间抽取的24个案例，没有根据后续涨跌重新选样。每页只绘观察日及以前行情。此次修正页脚与日期标签重叠，事件识别和计数均未改变。

## 形态抽查发现

- 第7页营销传播：存在先涨后落，但绿色“平台”包含明显的上升段，平台平整度仍需改进。
- 第10页电网设备：绿色窗口含深V反弹，不能直接认定为横向整理平台；这是放宽平台范围后的重要误识别候选。
- 第12页通信设备：上升前区间相对平稳，上涨再快速回落的轮廓较清楚。
- 第16页厨卫电器：已到价格区，但当日尚未跌速放缓或破支撑，保留为触发案例而未确认为完整候选。

抽查只说明具体案例的问题，不能代表全部230个事件的通过率。下一轮应优化平台锚点，区分横向整理、倾斜整理和V形反弹，并保留观察／触发／确认分层；无需用后续收益决定形态定义。
''')
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    result = dict(status='complete', source_manifest_sha256=digest(source / 'manifest.json'), pages=len(pages),
        artifacts={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()})
    (out / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(out, flush=True)


if __name__ == '__main__':
    main()
