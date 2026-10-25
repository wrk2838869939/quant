#!/usr/bin/env python3
"""绘制完整板块数量分布与日收益相关矩阵，并保存样本量和可交互网页。"""
from datetime import datetime, timezone
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
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from sector_dataset import SectorDataset, digest


def save_fig(fig, out, name, dpi):
    fig.savefig(out / (name + '.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_heatmap(corr, labels, out, name, title, cfg):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(26, 24))
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('#d8dce2')
    plot = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap=cm, interpolation='nearest')
    ax.set_xticks(range(n), labels, rotation=90, fontsize=6.3)
    ax.set_yticks(range(n), labels, fontsize=6.3)
    ax.tick_params(length=0, pad=2)
    ax.set_title(title + f'\nPearson 日收益相关系数；共同有效样本 ≥ {cfg["minimum_pair_days"]} 日；灰色 = 无法估计',
                 fontsize=18, pad=16)
    fig.colorbar(plot, ax=ax, fraction=.025, pad=.02, label='相关系数')
    fig.text(.13, .015, '采用分段内日收益；不填补缺失值。全目录的不同配对可能覆盖不同年份；不代表相同市场阶段。', fontsize=11)
    save_fig(fig, out, name, cfg['dpi'])


def html_page(payload, out):
    # 单文件 Canvas 图，悬浮同时展示系数和样本数；无需网络或额外绘图库。
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>板块日收益相关性</title>
<style>body{font-family:system-ui;margin:24px;color:#172033}header{max-width:1100px}select,input{font-size:16px;padding:6px;margin:8px}#wrap{overflow:auto;height:78vh;border:1px solid #ddd}canvas{display:block}#tip{position:fixed;display:none;background:#172033;color:white;padding:12px;border-radius:6px;pointer-events:none;z-index:2;white-space:pre-line}small{color:#586174}#status{margin:10px 0}</style>
<header><h2>所有板块的日收益相关系数</h2><p>收盘后数据；Pearson 相关；至少 60 个共同有效交易日。灰色表示无法估计，不表示零相关。悬浮查看板块名称、编号、共同样本数与实际共同起止日期。</p>
<label>范围 <select id="scope"></select></label><label>单元格大小 <input id="size" type="range" min="5" max="25" value="10"></label><label>定位板块 <input id="search" placeholder="名称或编号"></label><div id="status"></div><small>不同配对的共同日期可能有缺口；全历史合并图混合两代分类，研究时优先查看分类版本图。名称带 * 表示历史分类编号。</small></header>
<div id="wrap"><canvas id="chart"></canvas></div><div id="tip"></div><script>
const data=__PAYLOAD__,select=document.getElementById('scope'),slider=document.getElementById('size'),canvas=document.getElementById('chart'),ctx=canvas.getContext('2d'),tip=document.getElementById('tip');
for(const [key,d] of Object.entries(data)){let o=document.createElement('option');o.value=key;o.textContent=d.title;select.appendChild(o)}
let d,cell,left=190,top=200;
function color(x){if(x===null)return '#d8dce2';let a=Math.abs(x),base=x>=0?[178,24,43]:[33,102,172];return `rgb(${base.map(v=>Math.round(247+(v-247)*a)).join(',')})`}
function draw(){d=data[select.value];cell=+slider.value;let n=d.labels.length;canvas.width=left+n*cell+20;canvas.height=top+n*cell+20;ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.font='10px sans-serif';
for(let i=0;i<n;i++){for(let j=0;j<n;j++){ctx.fillStyle=color(d.corr[i][j]);ctx.fillRect(left+j*cell,top+i*cell,cell,cell)}ctx.fillStyle='#172033';ctx.textAlign='right';ctx.fillText(d.labels[i],left-5,top+(i+.5)*cell+3);ctx.save();ctx.translate(left+(i+.5)*cell,top-5);ctx.rotate(-Math.PI/2);ctx.textAlign='left';ctx.fillText(d.labels[i],0,3);ctx.restore()}
document.getElementById('status').textContent=`${n} 个板块，${d.start} 至 ${d.end}。红色为正相关，蓝色为负相关；固定色域 [-1,1]。`;highlight()}
function highlight(){let q=document.getElementById('search').value.trim();if(!q)return;let i=d.labels.findIndex(s=>s.includes(q));if(i<0)return;ctx.strokeStyle='#00a67e';ctx.lineWidth=2;ctx.strokeRect(left,top+i*cell,d.labels.length*cell,cell);ctx.strokeRect(left+i*cell,top,cell,d.labels.length*cell)}
canvas.onmousemove=e=>{let box=canvas.getBoundingClientRect(),j=Math.floor((e.clientX-box.left-left)/cell),i=Math.floor((e.clientY-box.top-top)/cell);if(i<0||j<0||i>=d.labels.length||j>=d.labels.length){tip.style.display='none';return}let r=d.corr[i][j];tip.textContent=`${d.labels[i]}\n${d.labels[j]}\n相关系数：${r===null?'无法估计':r.toFixed(4)}\n共同有效日：${d.n[i][j]}\n共同范围：${d.first[i][j]||'无'} 至 ${d.last[i][j]||'无'}`;tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-360)+'px';tip.style.top=Math.min(e.clientY+14,innerHeight-170)+'px'};
canvas.onmouseleave=()=>tip.style.display='none';select.onchange=draw;slider.oninput=draw;document.getElementById('search').oninput=draw;draw();</script></html>'''
    (out / 'correlation_explorer.html').write_text(template.replace('__PAYLOAD__', json.dumps(payload, ensure_ascii=False, allow_nan=False)))


def main():
    config = ROOT / 'configs/sector_dataset_plots_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    dataset = PROJECT / cfg['dataset']
    ds = SectorDataset(dataset)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = PROJECT / cfg['output_root'] / stamp
    out.mkdir(parents=True, exist_ok=False)
    font_manager.fontManager.addfont(str(PROJECT / cfg['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / cfg['font']).get_name(),
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    daily = pd.read_parquet(dataset / 'daily.parquet')
    catalog = pd.read_parquet(dataset / 'catalog_sw_l2.parquet')
    names = catalog.drop_duplicates('index_code', keep='last').set_index('index_code').industry_name.to_dict()
    codes = sorted(names)
    last = daily.sort_values('date').groupby('sector_code').tail(1).set_index('sector_code').reindex(codes)
    last['name'] = pd.Series(names)
    last['current'] = last.classification_version.eq('SW2021')
    last['quantity_status'] = np.where(last.known_members.gt(0), 'observed', 'unknown_or_empty_source')
    last.to_csv(out / 'member_counts_all.csv')
    # 两代分类分别在最后有效分类日做直方图，避免混合日期误读为当前分布。
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, version in zip(axes, ['SW2014', 'SW2021']):
        scope = daily[daily.classification_version.eq(version)]
        snapshot = scope[scope.date.eq(scope.date.max())]
        values = snapshot.loc[snapshot.known_members.gt(0), 'known_members']
        bins = cfg['histogram_bins']
        if len(values) and values.max() > bins[-1]:
            bins = bins + [int(values.max()) + 1]
        counts, edges, patches = ax.hist(values, bins=bins, color='#367bb5', edgecolor='white')
        for count, patch in zip(counts, patches):
            if count:
                ax.text(patch.get_x() + patch.get_width()/2, count + .25, str(int(count)), ha='center')
        ax.set(title=f'{version} · {scope.date.max():%Y-%m-%d}\n{len(values)} 个有成员记录；{len(snapshot)-len(values)} 个未知/空来源',
               xlabel='每个板块的已知成分股数量', ylabel='板块数量')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('板块成分股数量分布（不将缺失成员记录当作真实零成分）', fontsize=15)
    save_fig(fig, out, 'member_count_distribution', cfg['dpi'])
    # 全部编号均显示；旧版专属编号使用其最后分类日，不假装当前仍有效。
    ordered = last.sort_values(['known_members', 'name'], ascending=[False, True])
    fig, axes = plt.subplots(1, 3, figsize=(23, 19), layout='constrained')
    for ax, positions in zip(axes, np.array_split(np.arange(len(ordered)), 3)):
        part = ordered.iloc[positions]
        y = np.arange(len(part))
        colors = np.where(part.current, '#367bb5', '#8a70ac')
        ax.barh(y, part.known_members, color=colors)
        ax.set_yticks(y, [f'{r["name"]} {code.replace(".SI", "")}' + ('' if r.current else ' *')
                         for code, r in part.iterrows()], fontsize=8.2)
        for yy, (_, r) in zip(y, part.iterrows()):
            ax.text(r.known_members + 2, yy, str(int(r.known_members)) if r.known_members > 0 else '未知', va='center', fontsize=8)
        ax.set_xlim(0, max(ordered.known_members) * 1.14)
        ax.invert_yaxis()
        ax.set_xlabel('已知成分股数量')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('全部 170 个板块的成分股数量\n现行分类取 2026-09-08；带 * 的旧版专属编号取 2021-12-10', fontsize=18)
    fig.legend(handles=[Patch(color='#367bb5', label='现行 SW2021'), Patch(color='#8a70ac', label='旧版专属编号 *')], loc='outside lower center', ncol=2)
    save_fig(fig, out, 'member_counts_all', cfg['dpi'])
    print('COUNTS_COMPLETE', flush=True)
    payload, summaries = {}, {}
    for key, version in [('all', None), ('sw2021', 'SW2021'), ('sw2014', 'SW2014')]:
        part = daily if version is None else daily[daily.classification_version.eq(version)]
        selected_catalog = catalog if version is None else catalog[catalog.src.eq(version)]
        selected_names = selected_catalog.drop_duplicates('index_code', keep='last').set_index('index_code').industry_name
        order = sorted(selected_names.index)
        returns = part.pivot(index='date', columns='sector_code', values=cfg['return_column']).reindex(columns=order)
        corr = returns.corr(method=cfg['correlation_method'], min_periods=cfg['minimum_pair_days'])
        valid = returns.notna().to_numpy(dtype=np.int32)
        overlap = pd.DataFrame(valid.T @ valid, index=order, columns=order)
        first = np.full((len(order), len(order)), '', object)
        last_date = first.copy()
        pairs = []
        for i, a in enumerate(order):
            for j in range(i, len(order)):
                b = order[j]
                positions = np.flatnonzero((valid[:, i] > 0) & (valid[:, j] > 0))
                if len(positions):
                    first[i, j] = first[j, i] = str(returns.index[positions[0]].date())
                    last_date[i, j] = last_date[j, i] = str(returns.index[positions[-1]].date())
                if i != j:
                    pairs.append(dict(sector_a=a, name_a=selected_names[a], sector_b=b, name_b=selected_names[b],
                                      correlation=corr.iloc[i, j], common_days=int(overlap.iloc[i, j]),
                                      first_common_date=first[i, j], last_common_date=last_date[i, j]))
        pairs = pd.DataFrame(pairs)
        corr.to_csv(out / f'correlation_{key}.csv')
        overlap.to_csv(out / f'pair_days_{key}.csv')
        pairs.to_csv(out / f'pairs_{key}.csv', index=False)
        returns.to_parquet(out / f'returns_{key}.parquet')
        assert np.allclose(corr, corr.T, equal_nan=True)
        assert np.array_equal(overlap, overlap.T)
        assert not np.isfinite(corr.to_numpy()[overlap.to_numpy() < cfg['minimum_pair_days']]).any()
        assert (np.abs(corr.to_numpy()[np.isfinite(corr.to_numpy())]) <= 1 + 1e-12).all()
        if len(pairs[pairs.correlation.notna()]):
            p = pairs[pairs.correlation.notna()].iloc[len(pairs[pairs.correlation.notna()]) // 2]
            aligned = returns[[p.sector_a, p.sector_b]].dropna()
            np.testing.assert_allclose(np.corrcoef(aligned.to_numpy().T)[0, 1], p.correlation, atol=1e-12)
        title = f'{"全历史合并分类" if version is None else version} · {len(order)} 个板块 · {part.date.min():%Y-%m-%d}—{part.date.max():%Y-%m-%d}'
        labels = [f'{selected_names[c]} {c.replace(".SI", "")}' + (' *' if version is None and not last.at[c, 'current'] else '') for c in order]
        plot_heatmap(corr, labels, out, f'correlation_{key}', title, cfg)
        payload[key] = dict(title=title, labels=labels, start=str(part.date.min().date()), end=str(part.date.max().date()),
                            corr=corr.astype(object).where(corr.notna(), None).values.tolist(), n=overlap.values.tolist(),
                            first=first.tolist(), last=last_date.tolist())
        summaries[key] = dict(sectors=len(order), pairs=len(pairs), estimable_pairs=int(pairs.correlation.notna().sum()),
                              missing_pairs=int(pairs.correlation.isna().sum()), median_correlation=float(pairs.correlation.median()))
        print(f'HEATMAP_COMPLETE {key}', flush=True)
    html_page(payload, out)
    text = ['# 板块数量和相关性图', '',
            f'来源版本：`{ds.manifest["dataset_id"]}`，内容哈希 `{ds.manifest["content_id"]}`。', '',
            '- 数量指板块已知成分股数，不是交易所板块的真实完整成分数。无成员记录标为未知，不视为零成分。',
            '- 全部 170 个编号的数量图：现行 134 个使用 2026-09-08；旧版专属 36 个使用 2021-12-10，并带 *。两代分类直方图分别统计。',
            '- 日线相关采用分段内加权指数日收益 chain_return 的 Pearson 系数，不对指数点位做相关；缺失不填零，指数重启日不跨段计算。',
            f'- 每对至少 {cfg["minimum_pair_days"]} 个共同有效交易日，否则显示灰色。CSV 和交互图均提供每对实际样本量及起止日期；该区间可能存在缺口。',
            '- 全目录合并矩阵覆盖全部 170 个编号；分类变化可能改变同编号的含义，不同配对的年份也不同。SW2021 / SW2014 两份矩阵分别保留各自全部 134 / 104 个板块，更适合同版本分析。',
            '- 相关矩阵包含共同市场涨跌，不是剔除市场因素后的残差相关；配对缺失计算的矩阵不保证半正定，不能未经处理直接作为组合优化协方差。', '',
            '| 范围 | 板块 | 不重复配对 | 可估计配对 | 样本不足/无波动配对 | 相关中位数 |', '|---|---:|---:|---:|---:|---:|']
    for key, row in summaries.items():
        text.append(f'| {key} | {row["sectors"]} | {row["pairs"]} | {row["estimable_pairs"]} | {row["missing_pairs"]} | {row["median_correlation"]:.3f} |')
    text += ['', 'PNG 用于浏览，PDF 可放大阅读全部标签；correlation_explorer.html 可离线悬浮查询。', '',
             '已检查编号完整性、矩阵对称性、系数范围、最少共同天数屏蔽，并用 NumPy 独立复核每份矩阵的一组相关系数。', '']
    (out / 'README.md').write_text('\n'.join(text))
    assert len(last) == 170
    shutil.copy2(config, out / config.name)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    files = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(dict(dataset_content_id=ds.manifest['content_id'], config=cfg,
                                                    summary=summaries, artifacts=files), ensure_ascii=False, indent=2))
    print(json.dumps(dict(output=str(out), summary=summaries), ensure_ascii=False))


if __name__ == '__main__':
    main()
