#!/usr/bin/env python3
"""板块间独立比较页：每个具体板块占一行，不混入个股分布。"""
from pathlib import Path
from datetime import datetime, timezone
import sys
import json
import shutil
import html
import numpy as np
import pandas as pd
import yaml
import pymupdf
from matplotlib.ticker import MaxNLocator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import PROJECT, HERE, save_json, digest, plt, font_manager, PdfPages


def main():
    cfg = yaml.safe_load((HERE / 'between_config.yaml').read_text())
    source = PROJECT / cfg['source']
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, expected in manifest['artifacts'].items():
        assert digest(source / name) == expected, name
    data = pd.read_csv(source / 'named_sector_summary.csv')
    key = ['sector_code', 'classification_version']
    assert not data.duplicated(key + ['horizon']).any()
    boards = data[key + ['name']].drop_duplicates().sort_values(key)
    assert len(boards) == 65
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    save_json(out / 'manifest.json', dict(status='building'))
    print('OUTPUT', out, flush=True)
    for name in ['between_config.yaml', 'report_between.py']:
        shutil.copy2(HERE / name, out / name)
    data.to_csv(out / 'between_sector_values.csv', index=False)
    font_manager.fontManager.addfont(str(PROJECT / cfg['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / cfg['font']).get_name(),
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    metrics = ['mean', 'path_maximum', 'path_minimum']
    labels = ['期末平均收益', '期间最高收益', '期间最低收益']
    colors = ['#367aa3', '#438c65', '#b85457']
    pages, histogram_rows, drawn_rows = [], [], []
    with PdfPages(out / 'between_sectors_report.pdf') as book:
        def save(fig, name, title):
            fig.savefig(out / f'{name}.png', dpi=cfg['dpi'])
            book.savefig(fig)
            plt.close(fig)
            pages.append(dict(page=len(pages) + 1, title=title, image=f'{name}.png'))

        fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout='constrained')
        for j, h in enumerate(cfg['horizons']):
            group = data[data.horizon.eq(h) & data.complete_events.gt(0)]
            for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
                ax = axes[i, j]
                counts, edges, _ = ax.hist(group[metric] * 100, bins=cfg['plot_bins'], color=color, alpha=.8,
                                           edgecolor='white', linewidth=.4)
                assert int(counts.sum()) == len(group)
                histogram_rows.extend(dict(horizon=h, metric=metric, left=edges[k], right=edges[k + 1], boards=int(v)) for k, v in enumerate(counts))
                ax.axvline(0, color='#777', ls='--', lw=.8)
                ax.set(title=f'{h} 日 · 各板块{label} · n={len(group)}', xlabel='收益（%）', ylabel='板块-分类版本数')
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.suptitle('板块之间的收益分布：每个板块-分类版本只贡献一个数值、等权统计\n上排为各板块历次事件均值；中/下排为该板块所有完整事件期间的最高/最低值', fontsize=14)
        save(fig, 'between_sector_histograms', '板块间分布直方图（每板块一个数值）')

        step = cfg['boards_per_page']
        for start in range(0, len(boards), step):
            batch = boards.iloc[start:start + step]
            fig, axes = plt.subplots(1, 3, figsize=(22, 13), layout='constrained')
            for ax, h in zip(axes, cfg['horizons']):
                g = batch.merge(data[data.horizon.eq(h)], on=key + ['name'], validate='one_to_one')
                labels_y = []
                for index, r in enumerate(g.itertuples()):
                    labels_y.append(f'{r.name} {r.sector_code[:6]} · {r.classification_version}\n[{r.complete_events}/{r.selected_events}]')
                    if r.complete_events == 0:
                        ax.text(0, index, ' 无完整窗口', color='#888', va='center', fontsize=8)
                for offset, metric, label, color in zip([-.25, 0, .25], metrics, labels, colors):
                    values = g[metric].to_numpy() * 100
                    y = np.arange(len(g)) + offset
                    valid = np.isfinite(values)
                    ax.barh(y[valid], values[valid], height=.22, color=color, alpha=.85, label=label)
                    for position, value in zip(y[valid], values[valid]):
                        ax.text(value, position, f' {value:.1f}' if value >= 0 else f'{value:.1f} ',
                                va='center', ha='left' if value >= 0 else 'right', fontsize=6)
                # 同一期限所有分页统一坐标范围，支持跨页比较。
                full = data[data.horizon.eq(h)][metrics].to_numpy() * 100
                low, high = min(0, np.nanmin(full)), max(0, np.nanmax(full))
                padding = max((high - low) * .12, 1)
                ax.set_xlim(low - padding, high + padding)
                ax.set_yticks(np.arange(len(g)), labels_y, fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color='#777', lw=.8)
                ax.grid(axis='x', alpha=.15)
                ax.legend(loc='lower right', fontsize=8)
                ax.set(title=f'{h} 日：同一组板块并列比较', xlabel='收益（%）；数字标注为柱长')
                drawn_rows.extend(g[key + ['horizon']].to_dict('records'))
            title = f'具体板块并列比较 {start + 1}-{min(start + step, len(boards))} / {len(boards)}'
            fig.suptitle(title + '\n蓝：期末收益均值；绿/红：全部完整事件期间最高/最低值（不是均值）', fontsize=14)
            fig.supxlabel('按板块代码排列，不按事后收益排名；[完整/选中事件数]；样本少或不同时点的板块不能据此认定稳定优劣', fontsize=10)
            save(fig, f'named_sector_comparison_{start // step + 1:02d}', title)
    drawn = pd.DataFrame(drawn_rows)
    assert len(drawn) == len(data) and not drawn.duplicated(key + ['horizon']).any()
    pd.testing.assert_frame_equal(drawn.sort_values(key + ['horizon']).reset_index(drop=True),
        data[key + ['horizon']].sort_values(key + ['horizon']).reset_index(drop=True))
    pd.DataFrame(histogram_rows).to_csv(out / 'histogram_bins.csv', index=False)
    pd.DataFrame(pages).to_csv(out / 'page_index.csv', index=False)
    short = pymupdf.open(out / 'between_sectors_report.pdf')
    short.set_toc([[1, p['title'], p['page']] for p in pages])
    short.saveIncr()
    combined = pymupdf.open()
    combined.insert_pdf(short)
    old = pymupdf.open(source / 'confirmation_returns_report.pdf')
    combined.insert_pdf(old)
    combined.set_toc([[1, '板块间对比（新增）', 1]] + [[2, p['title'], p['page']] for p in pages]
        + [[1, '板块内分布及逐板块详情', len(short) + 1]]
        + [[level + 1, title, page + len(short)] for level, title, page in old.get_toc()])
    combined.save(out / 'confirmation_returns_report.pdf', garbage=4, deflate=True)
    full_pages = len(combined)
    short.close(); old.close(); combined.close()
    (out / 'overview.html').write_text('<!doctype html><meta charset="utf-8"><title>板块之间收益对比</title><style>body{max-width:1800px;margin:auto;font-family:system-ui}img{width:100%}</style><h1>板块之间收益对比</h1><p>每个板块一行，列为3/10/20日；原有板块内部详情已保留在完整版 PDF 后部。</p>' + ''.join(f'<h2>{html.escape(p["title"])}</h2><img src="{p["image"]}">' for p in pages))
    (out / 'README.md').write_text(f'''# 板块之间收益比较补充

新增 {len(pages)} 页：第 1 页为板块间直方图，第 2-4 页是 65 个具体板块-分类版本的并列柱状图。
每个板块一行，横向 3/10/20 日，标明名称、编号、分类版本及完整/选中事件数；缺失不置零。

蓝柱为该板块所有完整事件期末收益的均值；绿柱为所有事件期间最高收益的最大值；
红柱为所有事件期间最低收益的最小值。绿色和红色是跨事件极值，不是极值均值。
直方图同样每板块-分类版本只贡献一个数值，纵轴是板块数，不是事件数或个股数。
板块代码顺序固定；同一期限不同分页使用相同横轴范围。没有收益曲线或累计分布曲线。

板块-分类版本共 65 组（61 个代码），3/10/20 日有完整数据的组数分别 61/60/57；
对应完整事件数仍为 77/72/68。不同板块事件发生时点不同，且多数仅少量事件，不能把此图当作稳定排名。

- between_sectors_report.pdf：只看新增的板块间对比，4 页。
- confirmation_returns_report.pdf：把板块间对比前置、保留旧详情的完整版，{full_pages} 页，含书签。
- between_sector_values.csv：全部柱形数值及期末最小/最大值。
- histogram_bins.csv：板块间直方图每个分箱的边界及板块数量。

展示全部 195 个板块-版本-期限组合，柱形数据与原报告一致，无重训或重新筛选。
''')
    save_json(out / 'manifest.json', dict(status='complete', config=cfg,
        source_manifest_sha256=digest(source / 'manifest.json'), compared_groups=len(boards),
        combinations=len(drawn), new_pages=len(pages), full_pages=full_pages,
        artifacts={p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}))
    print('COMPLETE', out, 'NEW', len(pages), 'FULL', full_pages, flush=True)


if __name__ == '__main__':
    main()
