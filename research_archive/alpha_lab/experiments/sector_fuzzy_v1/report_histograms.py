# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""参照历史收益报告的分面直方图，增加全部具体板块的命名详情页。"""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import PROJECT, HERE, save_json, digest, plt, font_manager, PdfPages
from report_distribution import weights


def histogram(ax, values, bins, color, w=None, label=None):
    """实心分箱柱形；不连线、不平滑、不画累计分布。"""
    x = np.asarray(values, float) * 100
    if not len(x):
        ax.text(.5, .5, '无完整收益窗口', ha='center', va='center', transform=ax.transAxes)
        return
    heights, _, _ = ax.hist(x, bins=bins, weights=w, color=color, alpha=.72,
                            edgecolor='white', linewidth=.35, label=label)
    np.testing.assert_allclose(heights.sum(), len(x) if w is None else np.sum(w), atol=1e-7)
    ax.axvline(0, color='#777', ls='--', lw=.8)
    ax.grid(axis='y', alpha=.12)


def main():
    cfg = yaml.safe_load((HERE / 'histogram_config.yaml').read_text())
    for key in ['experiment', 'returns', 'distribution_report']:
        path = PROJECT / cfg[key]
        manifest = json.loads((path / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        for name, expected in manifest['artifacts'].items():
            assert digest(path / name) == expected, name
    p = pd.read_parquet(PROJECT / cfg['experiment'] / 'test_predictions_before_evaluation.parquet')
    p = p[(p.model == cfg['model']) & (p.retention == cfg['retention'])]
    assert p.event_id.is_unique and p.role.eq('test').all()
    selected = set(p[p.selected].event_id)
    b = pd.read_parquet(PROJECT / cfg['returns'] / 'sector_event_outcomes.parquet')
    b = b[b.event_id.isin(selected)].copy()
    m = pd.read_parquet(PROJECT / cfg['returns'] / 'member_outcomes.parquet')
    m = m[m.event_id.isin(selected) & m.board_complete & m.complete].copy()
    inside = pd.read_csv(PROJECT / cfg['distribution_report'] / 'within_event.csv')
    inside = inside[inside['sample'] == 'selected']
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    save_json(out / 'manifest.json', dict(status='building'))
    print('OUTPUT', out, flush=True)
    for name in ['histogram_config.yaml', 'report_histograms.py']:
        shutil.copy2(HERE / name, out / name)
    for name in ['distribution_summary.csv', 'by_sector.csv', 'within_event.csv', 'coverage.csv', 'extreme_cases.csv']:
        shutil.copy2(PROJECT / cfg['distribution_report'] / name, out / name)
    font_manager.fontManager.addfont(str(PROJECT / cfg['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / cfg['font']).get_name(),
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    metrics = ['end_return', 'max_return', 'min_return']
    labels = dict(end_return='期末收益', max_return='期间最高收盘收益', min_return='期间最低收盘收益')
    colors = dict(end_return='#367aa3', max_return='#438c65', min_return='#b85457')
    toc, pages, rows = [], [], []
    pdf_path = out / 'confirmation_returns_report.pdf'
    with PdfPages(pdf_path) as book:
        def save(fig, filename, title, code='', version=''):
            number = len(pages) + 1
            fig.savefig(out / f'{filename}.png', dpi=cfg['dpi'])
            book.savefig(fig)
            plt.close(fig)
            pages.append(dict(page=number, title=title, image=f'{filename}.png', sector_code=code, classification_version=version))
            toc.append([1, title, number])

        for level, frame in [('sector', b[b.complete]), ('member', m)]:
            fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout='constrained')
            for j, h in enumerate(cfg['horizons']):
                f = frame[frame.horizon.eq(h)]
                for i, metric in enumerate(metrics):
                    ax = axes[i, j]
                    w = None if level == 'sector' else weights(f, 'member') * 100
                    histogram(ax, f[metric], cfg['plot_bins'], colors[metric], w=w)
                    if level == 'sector':
                        ax.axvline(f[metric].median() * 100, color='#222', ls=':', label='中位数')
                        ax.legend(fontsize=8)
                    ax.set(title=f'{h} 日 · {labels[metric]} · n={len(f)}', xlabel='收益（%）',
                           ylabel='事件数' if level == 'sector' else '事件等权后的观测占比（%）')
            title = '板块事件收益分布' if level == 'sector' else '板块内部个股收益分布'
            fig.suptitle(title + '：神经网络选中测试事件；期间极值取未来每日收盘', fontsize=15)
            save(fig, f'overall_{level}_histograms', title)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
        for j, h in enumerate(cfg['horizons']):
            f = b[b.horizon.eq(h) & b.complete]
            medians = f.groupby(['classification_version', 'sector_code']).end_return.median()
            histogram(axes[0, j], medians, cfg['plot_bins'], '#367aa3')
            axes[0, j].set(title=f'{h} 日：各板块事件收益中位数', xlabel='收益（%）', ylabel='板块-分类版本数')
            f = inside[inside.horizon.eq(h)]
            histogram(axes[1, j], f.spread_q90_q10, cfg['plot_bins'], '#8c69a5')
            axes[1, j].set(title=f'{h} 日：板块内个股 q90 - q10', xlabel='收益差（百分点）', ylabel='事件数')
        fig.suptitle('板块之间分布与板块内部个股分化；具体名称和编号见后续逐板块页面', fontsize=15)
        save(fig, 'between_and_within_histograms', '板块之间与板块内部的分化')

        groups = list(b.groupby(['sector_code', 'classification_version'], sort=True))
        for k, ((code, version), group) in enumerate(groups):
            name = group.sort_values('date')['name'].iloc[-1]
            all_events = group.event_id.nunique()
            dates = group[['event_id', 'date']].drop_duplicates().sort_values('date')
            fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout='constrained')
            for j, h in enumerate(cfg['horizons']):
                board = group[group.horizon.eq(h) & group.complete]
                stocks = m[m.horizon.eq(h) & m.event_id.isin(board.event_id)]
                n = len(board)
                # 分箱由该板块当前期限的数据确定，不截去任何极端值。
                ax = axes[0, j]
                histogram(ax, board.end_return, cfg['detail_bins'], colors['end_return'])
                ax.set(title=f'{h} 日：板块期末收益 [{n}/{all_events} 事件]', xlabel='收益（%）', ylabel='事件数')
                if n:
                    ax.text(.98, .96, f'期末均值 {board.end_return.mean():.2%}\n期末最小/最大 {board.end_return.min():.2%} / {board.end_return.max():.2%}\n期间最低/最高 {board.min_return.min():.2%} / {board.max_return.max():.2%}',
                            transform=ax.transAxes, ha='right', va='top', fontsize=8,
                            bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
                ax = axes[1, j]
                w = weights(stocks, 'member') * 100 if len(stocks) else None
                histogram(ax, stocks.end_return, cfg['detail_bins'], colors['end_return'], w=w)
                ax.set(title=f'{h} 日：板块内个股期末收益 [{len(stocks)} 观测]', xlabel='收益（%）', ylabel='事件等权后的观测占比（%）')
                ax = axes[2, j]
                if len(stocks):
                    edges = np.histogram_bin_edges(np.r_[stocks.min_return, stocks.max_return] * 100, bins=cfg['detail_bins'])
                    histogram(ax, stocks.min_return, edges, colors['min_return'], w=w, label='期间最低收盘收益')
                    histogram(ax, stocks.max_return, edges, colors['max_return'], w=w, label='期间最高收盘收益')
                    ax.legend(fontsize=8)
                else:
                    histogram(ax, [], cfg['detail_bins'], colors['min_return'])
                ax.set(title=f'{h} 日：板块内个股的期间最低/最高收益', xlabel='收益（%）', ylabel='事件等权后的观测占比（%）')
                rows.append(dict(sector_code=code, name=name, classification_version=version, horizon=h,
                    selected_events=all_events, complete_events=n, member_observations=len(stocks),
                    mean=board.end_return.mean(), median=board.end_return.median(),
                    minimum=board.end_return.min(), maximum=board.end_return.max(),
                    path_minimum=board.min_return.min(), path_maximum=board.max_return.max()))
            title = f'{name} {code} · {version}'
            dates_text = '、'.join(dates.date.dt.strftime('%Y-%m-%d'))
            fig.suptitle(f'{title}：选中 {all_events} 次事件\n确认日期：{dates_text}', fontsize=14)
            fig.supxlabel('方括号为完整/选中事件数；同一事件的个股等权、事件之间等权；样本少的板块仅作个案观察', fontsize=10)
            save(fig, f'sector_{code[:6]}_{version}', title, code, version)
            if (k + 1) % 10 == 0:
                print('SECTORS', k + 1, '/', len(groups), flush=True)
    # 写入书签：在 PDF 阅读器中直接按板块名称跳转。
    document = pymupdf.open(pdf_path)
    assert len(document) == len(pages)
    document.set_toc(toc)
    document.set_metadata({'title': '神经网络选中事件：板块内外收益直方图', 'subject': '3/10/20日收益；逐板块详情'})
    document.saveIncr()
    document.close()
    pd.DataFrame(pages).to_csv(out / 'page_index.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'named_sector_summary.csv', index=False)
    index = ['<!doctype html><meta charset="utf-8"><title>具体板块收益直方图</title>',
        '<style>body{max-width:1500px;margin:24px auto;font-family:system-ui}img{width:100%}nav{columns:3}section{padding-top:20px}</style>',
        '<h1>具体板块收益直方图</h1><p>采用指定 PDF 的多面板直方图格式；仅神经网络选中的两轮测试事件。点击名称跳转。</p><nav>']
    index += [f'<p><a href="#page{r["page"]}">第{r["page"]}页 {html.escape(r["title"])}</a></p>' for r in pages]
    index += ['</nav>']
    index += [f'<section id="page{r["page"]}"><h2>第{r["page"]}页 {html.escape(r["title"])}</h2><img loading="lazy" src="{r["image"]}"></section>' for r in pages]
    (out / 'overview.html').write_text('\n'.join(index))
    (out / 'README.md').write_text(f'''# 具体板块收益：直方图版

参照用户指定的 confirmation_returns_report.pdf，取消累计分布曲线和收益路径曲线。
保持 3/10/20 日横向排列、分箱直方图、期末/期间最高/最低收益的展示方式。

- 第 1 页：板块事件收益，3×3 直方图。
- 第 2 页：板块内个股收益，3×3 直方图。
- 第 3 页：板块之间收益中位数与板块内部 q90-q10 差距的直方图。
- 第 4 页起：每个板块-分类版本一页，共 {len(groups)} 页，覆盖 {b.sector_code.nunique()} 个板块代码、{len(selected)} 个选中事件。
- 逐板块页上排为该板块历次事件期末收益，中排为个股期末收益，下排为个股期间最低/最高收益；所有图均为分箱柱形。
- PDF 内含按名称/编号跳转的书签；overview.html 可点击板块名跳转；page_index.csv 保存页码。

没有重训或变更样本；3/10/20 日完整板块事件分别 77/72/68。未完整事件仍列在对应板块页并注明。
个股成员固定在确认日，只统计板块与个股都完整的窗口；个股先在事件内等权，再让事件等权。
灰虚线为零收益，首页黑点线为中位数；蓝色期末收益、绿色期间最高、红色期间最低。
期间极值按每日收盘计算，不是日内极值或账户最大回撤。
分类版本 SW2014/SW2021 分页，避免同编号的历史口径混在一起。
每个板块的事件数很少，直方图可能仅有一根柱；这不代表稳定收益分布。
逐板块的分箱随数据范围变化，比较时应读横轴，不按柱子位置直接比较。

数据与此前分布报告一致，详细数值及极值身份见 CSV；本次仅改变展示。
''')
    # 核对数据完整覆盖，并保存参考版式文件哈希。
    assert len(rows) == len(groups) * len(cfg['horizons'])
    assert set(b.event_id) == selected
    save_json(out / 'manifest.json', dict(status='complete', config=cfg, pages=len(pages),
        sector_codes=b.sector_code.nunique(), sector_versions=len(groups), selected_events=len(selected),
        reference_pdf_sha256=digest(PROJECT / cfg['reference_pdf']),
        artifacts={str(path.relative_to(out)): digest(path) for path in out.iterdir() if path.is_file() and path.name != 'manifest.json'}))
    print('COMPLETE', out, 'PAGES', len(pages), flush=True)


if __name__ == '__main__':
    main()
