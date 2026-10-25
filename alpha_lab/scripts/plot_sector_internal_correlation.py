#!/usr/bin/env python3
"""统计历史同时归属同板块的个股日收益相关性；输出全配对及分布图。"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from sector_dataset import SectorDataset, digest
from datahub import DataHub


def savefig(fig, out, name, dpi):
    fig.savefig(out / (name + '.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def draw_distributions(stats, samples, out, key, title, cfg):
    ordered = stats.sort_values(['median_correlation', 'sector_code'], ascending=[False, True], na_position='last')
    fig, axes = plt.subplots(1, 3, figsize=(25, 19), layout='constrained')
    for ax, positions in zip(axes, np.array_split(np.arange(len(ordered)), 3)):
        part = ordered.iloc[positions]
        labels = []
        for y, r in enumerate(part.itertuples()):
            labels.append(f'{r.name} {r.sector_code.replace(".SI", "")}')
            if r.valid_pairs:
                ax.bxp([dict(med=r.median_correlation, q1=r.q25_correlation, q3=r.q75_correlation,
                             whislo=r.q10_correlation, whishi=r.q90_correlation, fliers=[])],
                       positions=[y], vert=False, widths=.6, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor='#b8d9ec', edgecolor='#367bb5'),
                       medianprops=dict(color='#143e63', linewidth=1.4),
                       whiskerprops=dict(color='#367bb5'), capprops=dict(color='#367bb5'))
                ax.scatter(r.mean_correlation, y, s=13, color='#d56839', zorder=3)
                ax.text(1.03, y, f'{r.valid_pairs:,}', fontsize=7.5, va='center')
            else:
                ax.text(.0, y, '数据不足', color='#888888', fontsize=8, va='center')
        ax.set_yticks(np.arange(len(part)), labels, fontsize=8)
        ax.set_ylim(len(part)-.3, -.7)
        ax.set_xlim(-.4, 1.23)
        ax.set_xticks([-.4, -.2, 0, .2, .4, .6, .8, 1])
        ax.axvline(0, color='#aaaaaa', lw=.7)
        ax.grid(axis='x', alpha=.17)
        ax.set_xlabel('个股两两日收益相关系数                  有效配对数', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title + '\n箱体：25%—75%分位；横线：10%—90%分位；深线：中位数；橙点：均值（不是置信区间）', fontsize=17)
    savefig(fig, out, f'internal_distribution_{key}', cfg['dpi'])


def main():
    config = ROOT / 'configs/sector_internal_correlation_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    dataset = PROJECT / cfg['dataset']
    ds = SectorDataset(dataset)
    if os.environ.get('DATAHUB_CONTENT_ID') != ds.manifest['market_content_id']:
        raise RuntimeError('请通过绑定版本的 frozen_research.py run 启动')
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    font_manager.fontManager.addfont(str(PROJECT / cfg['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / cfg['font']).get_name(),
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    mapping = pd.read_parquet(dataset / 'membership_daily.parquet')
    sectors = json.loads((dataset / 'sector_codes.json').read_text())
    catalog = pd.read_parquet(dataset / 'catalog_sw_l2.parquet')
    daily = pd.read_parquet(dataset / 'daily.parquet')
    hub = DataHub()
    if hub.cfg.root.resolve() != (PROJECT / ds.manifest['config']['release']).resolve():
        raise RuntimeError('行情目录不符')
    print('LOAD_PRICES', flush=True)
    raw = hub.cn.daily(start=str(mapping.index.min().date()), end=str(mapping.index.max().date()), fields=['hfq_close'])
    assert not raw.duplicated(['date', 'ts_code']).any()
    close = raw.pivot(index='date', columns='ts_code', values='hfq_close').reindex(index=mapping.index, columns=mapping.columns)
    del raw
    close = close.where(close > 0)
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    del close
    # 当日及前日均属同一已知行业，排除刚进入归属/上市首日和停牌缺口。
    unchanged = mapping.eq(mapping.shift(1)) & mapping.ge(0)
    returns = returns.where(unchanged)
    # 分类切换日不跨版本使用前日归属，即使编号相同。
    switch = pd.Timestamp(ds.manifest['config']['version_change'])
    returns.loc[returns.index == switch] = np.nan
    versions = [('recent', 'SW2021', mapping.index[-cfg['recent_sessions']:]),
                ('sw2021', 'SW2021', mapping.index[mapping.index >= switch]),
                ('sw2014', 'SW2014', mapping.index[mapping.index < switch])]
    summaries, all_stats, all_samples, check = {}, {}, {}, []
    for key, version, dates in versions:
        current_catalog = catalog[catalog.src.eq(version)].drop_duplicates('index_code').sort_values('index_code')
        membership = mapping.loc[dates].to_numpy()
        rvalues = returns.loc[dates].to_numpy()
        scope = out / key
        scope.mkdir()
        results, samples, all_pairs = [], {}, []
        for pos, r in enumerate(current_catalog.itertuples()):
            sector = r.index_code
            code_id = sectors.index(sector)
            mask = membership == code_id
            ever_member = mask.any(axis=0)
            stocks = mapping.columns[ever_member]
            frame = pd.DataFrame(np.where(mask[:, ever_member], rvalues[:, ever_member], np.nan), index=dates, columns=stocks)
            eligible = frame.notna().sum() >= cfg['minimum_pair_days']
            frame = frame.loc[:, eligible]
            corr = frame.corr(method=cfg['correlation_method'], min_periods=cfg['minimum_pair_days'])
            good = frame.notna().to_numpy(np.int32)
            common = good.T @ good
            ii, jj = np.triu_indices(len(frame.columns), k=1)
            rho = corr.to_numpy()[ii, jj]
            days = common[ii, jj]
            enough = np.isfinite(rho)
            values = rho[enough]
            samples[sector] = values
            pairs = pd.DataFrame(dict(sector_code=sector, stock_a=frame.columns.to_numpy()[ii],
                                      stock_b=frame.columns.to_numpy()[jj], correlation=rho, common_days=days))
            all_pairs.append(pairs)
            corr.to_parquet(scope / f'{sector}_correlation.parquet')
            pd.DataFrame(common, index=frame.columns, columns=frame.columns).to_parquet(scope / f'{sector}_pair_days.parquet')
            record = dict(sector_code=sector, name=r.industry_name, classification_version=version,
                          start=str(dates.min().date()), end=str(dates.max().date()),
                          window_sessions=len(dates), ever_known_members=int(ever_member.sum()),
                          eligible_stocks=len(frame.columns), possible_pairs=len(rho), valid_pairs=len(values),
                          pair_coverage=len(values)/len(rho) if len(rho) else np.nan,
                          mean_correlation=float(values.mean()) if len(values) else np.nan,
                          median_correlation=float(np.median(values)) if len(values) else np.nan,
                          q10_correlation=float(np.quantile(values, .1)) if len(values) else np.nan,
                          q25_correlation=float(np.quantile(values, .25)) if len(values) else np.nan,
                          q75_correlation=float(np.quantile(values, .75)) if len(values) else np.nan,
                          q90_correlation=float(np.quantile(values, .9)) if len(values) else np.nan,
                          negative_pair_fraction=float((values < 0).mean()) if len(values) else np.nan,
                          median_common_days=float(np.median(days[enough])) if len(values) else np.nan)
            results.append(record)
            assert np.allclose(corr, corr.T, equal_nan=True)
            assert not np.isfinite(corr.to_numpy()[common < cfg['minimum_pair_days']]).any()
            assert ((values >= -1-1e-12) & (values <= 1+1e-12)).all()
            # 每个可统计板块独立复核一对，与 pandas 算法和聚合数据分开验证。
            if len(values):
                pair = pairs[pairs.correlation.notna()].iloc[0]
                aligned = frame[[pair.stock_a, pair.stock_b]].dropna()
                np.testing.assert_allclose(np.corrcoef(aligned.to_numpy().T)[0, 1], pair.correlation, atol=1e-12)
                assert len(aligned) == pair.common_days
                assert mapping.loc[aligned.index, [pair.stock_a, pair.stock_b]].eq(code_id).all().all()
                check.append(dict(scope=key, sector_code=sector, days=len(aligned)))
            if pos % 30 == 0:
                print(f'{key} {pos}/{len(current_catalog)}', flush=True)
        stats = pd.DataFrame(results)
        stats.to_csv(out / f'summary_{key}.csv', index=False)
        pd.concat(all_pairs, ignore_index=True).to_parquet(out / f'pairs_{key}.parquet', index=False)
        all_stats[key], all_samples[key] = stats, samples
        title = f'{version} 板块内部相关性 · {len(stats)} 个板块 · {dates.min():%Y-%m-%d}—{dates.max():%Y-%m-%d}'
        draw_distributions(stats, samples, out, key, title, cfg)
        estimable = stats[stats.valid_pairs.gt(0)]
        summaries[key] = dict(start=str(dates.min().date()), end=str(dates.max().date()), sectors=len(stats),
                              estimable_sectors=len(estimable), unavailable_sectors=int(stats.valid_pairs.eq(0).sum()),
                              valid_pairs=int(stats.valid_pairs.sum()),
                              sector_median_of_medians=float(estimable.median_correlation.median()),
                              sector_mean_of_means=float(estimable.mean_correlation.mean()))
    recent = all_stats['recent']
    full = all_stats['sw2021'].set_index('sector_code')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    x = recent.median_correlation.dropna()
    axes[0].hist(x, bins=np.linspace(-.1, 1, 23), color='#367bb5', edgecolor='white')
    axes[0].axvline(x.median(), color='#d56839', label=f'中位数 {x.median():.3f}')
    axes[0].legend()
    axes[0].set(xlabel='每个板块的内部相关中位数', ylabel='板块数量', title=f'最近 {cfg["recent_sessions"]} 个交易日 · {len(x)} 个可统计板块')
    merged = recent.set_index('sector_code').join(full[['median_correlation']], rsuffix='_full')
    axes[1].scatter(merged.median_correlation_full, merged.median_correlation, color='#367bb5', alpha=.7)
    axes[1].plot([0, 1], [0, 1], '--', color='#aaa', lw=1)
    axes[1].set(xlabel='SW2021 全历史内部相关中位数', ylabel='最近窗口内部相关中位数', title='同一板块的长短窗口对照')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    savefig(fig, out, 'internal_overview', cfg['dpi'])
    lines = ['# 板块内部日收益相关性', '', f'来源：`{ds.manifest["dataset_id"]}`；行情内容 `{ds.manifest["market_content_id"]}`。', '',
             '- 研究对象是同板块内不同股票两两日收益，不包含自己与自己；每对等权汇总，板块总体比较每个板块等权。',
             '- 个股使用复权收盘的相邻交易日收益，缺报价不前填；要求当日和前日归属相同，且两只股票在统计日同时属于该板块。排除分类切换日。',
             f'- 每对至少 {cfg["minimum_pair_days"]} 个共同有效日。不足的配对不进入均值/分位数；无有效配对的板块保留并标为数据不足。',
             '- eligible_stocks 是窗口内至少有 60 个有效归属收益日的股票。possible_pairs 是它们两两组合数；pair_coverage 衡量其中实际满足共同日要求且非零方差的比例，不是官方行业成员覆盖率。',
             '- 同一板块内各配对的共同日期可能不同，可能包含间断，也可能因成员轮换从未同时存在。统计只覆盖已知历史成员，不以今天的成员倒推历史。',
             '- 箱体为配对相关系数分布的四分位区间，须线为 10%—90% 分位，橙点为均值；不是置信区间。配对之间共享股票，并非相互独立样本。',
             '- 这些是包含共同市场因素的原始收益相关，不能直接解释为板块特有联动或预测能力。', '',
             '| 窗口 | 起止 | 目录板块 | 可统计 | 有效个股配对 | 板块中位数的中位数 |',
             '|---|---|---:|---:|---:|---:|']
    for key, s in summaries.items():
        lines.append(f'| {key} | {s["start"]}—{s["end"]} | {s["sectors"]} | {s["estimable_sectors"]} | {s["valid_pairs"]:,} | {s["sector_median_of_medians"]:.3f} |')
    for label, part in [('最近窗口内部相关较高的板块', recent[recent.valid_pairs.gt(0)].nlargest(10, 'median_correlation')),
                        ('最近窗口内部相关较低的板块', recent[recent.valid_pairs.gt(0)].nsmallest(10, 'median_correlation'))]:
        lines += ['', f'## {label}', '', '| 板块 | 中位数 | 均值 | 有效配对 | 共同日中位数 |', '|---|---:|---:|---:|---:|']
        for r in part.itertuples():
            lines.append(f'| {r.name} {r.sector_code} | {r.median_correlation:.3f} | {r.mean_correlation:.3f} | {r.valid_pairs:,} | {r.median_common_days:.0f} |')
    lines += ['', '## 文件', '', '- internal_overview.png/pdf：总体分布与长短窗口对照。',
              '- internal_distribution_recent/sw2021/sw2014.png/pdf：各窗口全部板块内部配对分布。',
              '- summary_*.csv：板块均值、中位数、各分位、股票数、配对数和共同日统计。',
              '- pairs_*.parquet：所有候选股票配对（含无效配对的空值）、相关系数及共同有效日数。',
              '- 各窗口子目录：各板块完整股票相关矩阵与样本量矩阵。', '',
              f'核验：源版本和产物哈希、矩阵对称与系数范围、样本门槛；对 {len(check)} 个板块/窗口各独立复核一对相关及历史成员归属。', '']
    (out / 'README.md').write_text('\n'.join(lines))
    (out / 'validation.json').write_text(json.dumps(dict(independent_pair_checks=check, minimum_days=True, symmetry=True), indent=2))
    shutil.copy2(config, out / config.name)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    files = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(dict(dataset_content_id=ds.manifest['content_id'], config=cfg, summary=summaries, artifacts=files), ensure_ascii=False, indent=2))
    print(json.dumps(dict(output=str(out), summary=summaries), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
