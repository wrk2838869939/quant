#!/usr/bin/env python3
"""固定230个首次确认事件的3／10／20日收益及板块内外分布。"""
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
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
from datahub import DataHub
from sector_dataset import SectorDataset, digest
from sector_confirmation_returns import window_metrics, cumulative_paths


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def describe(values):
    """只描述可观察结果，样本不足时仍保留实际计数。"""
    x = pd.Series(values).dropna()
    if x.empty:
        return dict(n=0, mean=np.nan, median=np.nan, q10=np.nan, q25=np.nan,
                    q75=np.nan, q90=np.nan, minimum=np.nan, maximum=np.nan, positive=np.nan)
    return dict(n=len(x), mean=x.mean(), median=x.median(), q10=x.quantile(.1),
        q25=x.quantile(.25), q75=x.quantile(.75), q90=x.quantile(.9),
        minimum=x.min(), maximum=x.max(), positive=x.gt(0).mean())


def main():
    config = ROOT / 'configs/sector_confirmation_returns_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    source, dataset = PROJECT / cfg['events'], PROJECT / cfg['dataset']
    ds = SectorDataset(dataset)
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['dataset_content_id'] == ds.manifest['content_id']
    for name, expected in manifest['artifacts'].items():
        assert digest(source / name) == expected, name
    assert os.environ.get('DATAHUB_CONTENT_ID') == ds.manifest['market_content_id']
    hub = DataHub()
    assert hub.cfg.root.resolve() == (PROJECT / ds.manifest['config']['release']).resolve()
    tests = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(
        str(ROOT / 'tests'), pattern='test_sector_confirmation_returns.py'))
    assert tests.wasSuccessful()
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'manifest.json', dict(status='building', config=cfg))
    print('OUTPUT', out, flush=True)
    origins = pd.read_parquet(source / 'expanded_observations.parquet',
        columns=['event_id', 'date', 'sector_code', 'classification_version', 'segment_id', 'index_close', 'known_members', 'first_confirmed'],
        filters=[('first_confirmed', '==', True)]).sort_values(['date', 'sector_code']).reset_index(drop=True)
    assert len(origins) == origins.event_id.nunique() == cfg['expected_events']
    catalog = pd.read_parquet(dataset / 'catalog_sw_l2.parquet')
    names = {(r.src, r.index_code): r.industry_name for r in catalog.itertuples()}
    origins['name'] = [names.get((r.classification_version, r.sector_code), r.sector_code) for r in origins.itertuples()]
    origins.to_csv(out / 'confirmation_origins.csv', index=False)
    panel = pd.read_parquet(dataset / 'daily.parquet', columns=['date', 'sector_code', 'segment_id', 'index_valid', 'index_close'])
    prices = panel.set_index(['sector_code', 'date'])
    calendar = hub.calendar.cn_days(ds.manifest['config']['start'], ds.manifest['config']['end'])
    mapping = pd.read_parquet(dataset / 'membership_daily.parquet')
    codes = json.loads((dataset / 'sector_codes.json').read_text())
    member_lists = {r.event_id: mapping.columns[mapping.loc[r.date].eq(codes.index(r.sector_code))].tolist() for r in origins.itertuples()}
    for r in origins.itertuples():
        assert len(member_lists[r.event_id]) == r.known_members
    universe = sorted(set(s for members in member_lists.values() for s in members))
    print('LOAD_MEMBERS', len(universe), flush=True)
    raw = hub.cn.daily(codes=universe, start=str(origins.date.min().date()),
        end=str(calendar.max().date()), fields=['hfq_close'])
    assert not raw.duplicated(['date', 'ts_code']).any()
    stock = raw.pivot(index='date', columns='ts_code', values='hfq_close').reindex(index=calendar, columns=universe)
    del raw
    # 仅限制历史存续日期，保留确认后退出板块的原成员，不筛当前上市状态。
    basic = pd.read_parquet(hub.cfg.meta_dir / 'stock_basic.parquet')
    for r in basic.itertuples():
        if r.ts_code not in stock:
            continue
        start = pd.to_datetime(r.list_date, format='%Y%m%d', errors='coerce')
        end = pd.to_datetime(r.delist_date, format='%Y%m%d', errors='coerce')
        if pd.notna(start):
            stock.loc[stock.index < start, r.ts_code] = np.nan
        if pd.notna(end):
            stock.loc[stock.index >= end, r.ts_code] = np.nan
    quotes = pd.read_parquet(hub.cfg.root / 'datasets/index/cn_index_daily.parquet')
    benchmark = quotes[quotes.ts_code.eq(cfg['benchmark'])].set_index('date').close.reindex(calendar)
    board_rows, member_rows, paths, membership_rows = [], [], [], []
    horizon_max = max(cfg['horizons'])
    for r in origins.itertuples():
        start = calendar.get_loc(r.date)
        dates = calendar[start:start + horizon_max + 1]
        part = prices.reindex(pd.MultiIndex.from_product([[r.sector_code], dates], names=['sector_code', 'date']))
        board_values = np.full(horizon_max + 1, np.nan)
        board_values[:len(dates)] = part.index_close.where(part.index_valid.fillna(False) & part.segment_id.eq(r.segment_id)).to_numpy()
        assert np.isclose(board_values[0], r.index_close)
        members = member_lists[r.event_id]
        for code in members:
            membership_rows.append(dict(event_id=r.event_id, date=r.date, sector_code=r.sector_code, ts_code=code))
        values = np.full((horizon_max + 1, len(members)), np.nan)
        values[:len(dates)] = stock.reindex(index=dates, columns=members).to_numpy()
        bpath, mpath = cumulative_paths(board_values)[:, 0], cumulative_paths(values)
        identity = dict(event_id=r.event_id, date=r.date, sector_code=r.sector_code, name=r.name,
                        classification_version=r.classification_version, segment_id=r.segment_id)
        for offset in range(horizon_max + 1):
            observed = mpath[offset, np.isfinite(mpath[offset])]
            paths.append(dict(**identity, offset=offset, target_date=dates[offset] if offset < len(dates) else pd.NaT,
                board_return=bpath[offset], members_total=len(members), members_observed=len(observed),
                member_mean=observed.mean() if len(observed) else np.nan,
                member_q10=np.quantile(observed, .1) if len(observed) else np.nan,
                member_median=np.median(observed) if len(observed) else np.nan,
                member_q90=np.quantile(observed, .9) if len(observed) else np.nan))
        for horizon in cfg['horizons']:
            b = {k: v[0] for k, v in window_metrics(board_values, horizon).items()}
            expected_end = dates[horizon] if horizon < len(dates) else pd.NaT
            market = benchmark.get(expected_end, np.nan) / benchmark.get(r.date, np.nan) - 1
            status = 'complete' if b['complete'] else 'dataset_end' if horizon >= len(dates) else 'segment_or_quote_gap'
            board_rows.append(dict(**identity, horizon=horizon, end_date=expected_end, status=status,
                market_return=market, excess_return=b['end_return'] - market, **b))
            metrics = window_metrics(values, horizon)
            for i, code in enumerate(members):
                member_rows.append(dict(**identity, horizon=horizon, end_date=expected_end, ts_code=code,
                    board_complete=bool(b['complete']),
                    **{key: array[i] for key, array in metrics.items()}))
    board, member, path = pd.DataFrame(board_rows), pd.DataFrame(member_rows), pd.DataFrame(paths)
    assert not board.duplicated(['event_id', 'horizon']).any()
    assert not member.duplicated(['event_id', 'horizon', 'ts_code']).any()
    for frame in [board, member]:
        valid = frame[frame.complete]
        assert (valid.min_return <= valid.end_return + 1e-12).all()
        assert (valid.max_return >= valid.end_return - 1e-12).all()
    board.to_csv(out / 'sector_event_outcomes.csv', index=False)
    board.to_parquet(out / 'sector_event_outcomes.parquet', index=False)
    member.to_parquet(out / 'member_outcomes.parquet', index=False)
    path.to_parquet(out / 'daily_paths.parquet', index=False)
    pd.DataFrame(membership_rows).to_parquet(out / 'fixed_membership.parquet', index=False)
    print('LABELED', len(board), len(member), flush=True)
    metrics = ['end_return', 'max_return', 'min_return']
    pooled, by_sector, by_version, yearly = [], [], [], []
    for horizon, group in board.groupby('horizon'):
        for metric in metrics + ['excess_return']:
            for weight, values in [('event_equal', group[metric]),
                    ('date_equal', group.groupby('date')[metric].mean()),
                    ('sector_equal', group.groupby('sector_code')[metric].mean())]:
                pooled.append(dict(horizon=horizon, metric=metric, weighting=weight, **describe(values)))
        for code, g in group.groupby('sector_code'):
            for metric in metrics:
                by_sector.append(dict(sector_code=code, name=g.name.iloc[-1], horizon=horizon, metric=metric,
                    total_events=len(g), complete_dates=g.loc[g.complete, 'date'].nunique(),
                    versions='|'.join(sorted(g.classification_version.unique())), **describe(g[metric])))
        for (version, code), g in group.groupby(['classification_version', 'sector_code']):
            by_version.append(dict(classification_version=version, sector_code=code, name=g.name.iloc[-1], horizon=horizon, **describe(g.end_return)))
        for year, g in group.groupby(group.date.dt.year):
            yearly.append(dict(year=year, horizon=horizon, **describe(g.end_return)))
    pooled, sectors = pd.DataFrame(pooled), pd.DataFrame(by_sector)
    pooled.to_csv(out / 'overall_distribution.csv', index=False)
    sectors.to_csv(out / 'by_sector_distribution.csv', index=False)
    pd.DataFrame(by_version).to_csv(out / 'by_sector_version_distribution.csv', index=False)
    pd.DataFrame(yearly).to_csv(out / 'by_year_distribution.csv', index=False)
    board.groupby(['horizon', 'status']).size().rename('events').to_csv(out / 'censoring.csv')
    counts = origins.groupby('sector_code').size()
    counts.rename('events').to_csv(out / 'sector_event_counts.csv')
    date_counts = origins.groupby('date').size()
    date_counts.rename('events').to_csv(out / 'confirmation_date_counts.csv')
    member_event = []
    for (event_id, horizon), g in member.groupby(['event_id', 'horizon']):
        common = {key: g[key].iloc[0] for key in ['event_id', 'date', 'sector_code', 'name', 'board_complete']}
        complete = int(g.complete.sum())
        for metric in metrics:
            info = describe(g[metric])
            member_event.append(dict(**common, horizon=horizon, metric=metric, total_members=len(g),
                complete_members=complete, coverage=complete / len(g),
                sufficient_coverage=complete / len(g) >= cfg['minimum_member_coverage'],
                spread_q90_q10=info['q90'] - info['q10'], **info))
    member_event = pd.DataFrame(member_event)
    member_event.to_csv(out / 'within_event_member_distribution.csv', index=False)
    member_sector, member_summary = [], []
    for (horizon, metric), g in member_event.groupby(['horizon', 'metric']):
        for sample, sub in [('all_events', g), ('coverage_at_least_80pct', g[g.sufficient_coverage])]:
            for statistic in ['mean', 'median', 'q10', 'q90', 'positive', 'spread_q90_q10']:
                member_summary.append(dict(horizon=horizon, metric=metric, sample=sample, statistic=statistic, **describe(sub[statistic])))
        for code, sub in g.groupby('sector_code'):
            member_sector.append(dict(horizon=horizon, metric=metric, sector_code=code, name=sub.name.iloc[-1],
                events=len(sub), median_coverage=sub.coverage.median(),
                mean_of_event_member_means=sub['mean'].mean(), median_of_event_member_medians=sub['median'].median(),
                median_within_event_spread=sub.spread_q90_q10.median()))
    pd.DataFrame(member_summary).to_csv(out / 'member_event_equal_summary.csv', index=False)
    pd.DataFrame(member_sector).to_csv(out / 'member_by_sector_summary.csv', index=False)
    # 缺失随时间增加：同时提供逐日可观察样本和固定完整20日样本。
    common_ids = board.loc[board.horizon.eq(horizon_max) & board.complete, 'event_id']
    paths_summary = []
    for cohort, sub in [('available_each_day', path), ('complete_20d', path[path.event_id.isin(common_ids)])]:
        for offset, g in sub.groupby('offset'):
            paths_summary.append(dict(cohort=cohort, offset=offset,
                date_equal_mean=g.groupby('date').board_return.mean().mean(), **describe(g.board_return)))
    paths_summary = pd.DataFrame(paths_summary)
    paths_summary.to_csv(out / 'path_distribution.csv', index=False)
    font_manager.fontManager.addfont(str(PROJECT / cfg['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / cfg['font']).get_name(),
                         'axes.unicode_minus': False, 'pdf.fonttype': 42})
    charts = []
    def save(fig, name, book):
        fig.savefig(out / (name + '.png'), dpi=cfg['dpi'])
        book.savefig(fig)
        charts.append(name + '.png')
        plt.close(fig)
    colors = dict(end_return='#367aa3', max_return='#438c65', min_return='#b85457')
    labels = dict(end_return='期末收益', max_return='期间最高收盘收益', min_return='期间最低收盘收益')
    with PdfPages(out / 'confirmation_returns_report.pdf') as book:
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout='constrained')
        for j, horizon in enumerate(cfg['horizons']):
            g = board[board.horizon.eq(horizon) & board.complete]
            for i, metric in enumerate(metrics):
                ax = axes[i, j]
                ax.hist(g[metric] * 100, bins=cfg['plot_bins'], color=colors[metric], alpha=.85)
                ax.axvline(0, color='#777', ls='--')
                ax.axvline(g[metric].median() * 100, color='#222', ls=':', label='中位数')
                ax.set(title=f'{horizon}日 · {labels[metric]} · n={len(g)}', xlabel='收益（%）', ylabel='事件数')
                ax.legend(fontsize=8)
        fig.suptitle('确认后板块收益分布：每个首次确认事件等权；极值取未来每日收盘', fontsize=15)
        save(fig, 'overall_distributions', book)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
        g = paths_summary[paths_summary.cohort.eq('complete_20d')]
        ax = axes[0]
        ax.fill_between(g.offset, g.q10 * 100, g.q90 * 100, alpha=.15, label='事件分布10%—90%')
        ax.fill_between(g.offset, g.q25 * 100, g.q75 * 100, alpha=.25, label='事件分布25%—75%')
        ax.plot(g.offset, g['median'] * 100, label='事件中位数')
        ax.plot(g.offset, g['mean'] * 100, label='事件均值')
        ax.plot(g.offset, g.date_equal_mean * 100, label='日期等权均值', ls='--')
        ax.axhline(0, color='#777', ls=':')
        ax.set(xlabel='确认后交易日', ylabel='累计收盘收益（%）', title=f'固定完整20日样本，n={len(common_ids)}；阴影不是置信区间')
        ax.legend(fontsize=8)
        values = [counts.eq(1).sum(), counts.between(2, 4).sum(), counts.ge(5).sum()]
        axes[1].bar(['仅1次事件', '2—4次事件', '至少5次事件'], values, color='#bc8854')
        for i, value in enumerate(values):
            axes[1].text(i, value, str(value), ha='center', va='bottom')
        axes[1].set(ylabel='板块数', title='每板块可用事件数量（截尾前）')
        save(fig, 'return_paths_and_support', book)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
        for j, horizon in enumerate(cfg['horizons']):
            sub = sectors[sectors.horizon.eq(horizon) & sectors.metric.eq('end_return') & sectors.n.gt(0)]
            axes[0, j].hist(sub['median'] * 100, bins=cfg['plot_bins'], color='#367aa3')
            axes[0, j].set(title=f'{horizon}日：各板块事件收益中位数', xlabel='收益（%）', ylabel='板块数')
            sub = member_event[member_event.horizon.eq(horizon) & member_event.metric.eq('end_return')]
            axes[1, j].hist(sub.spread_q90_q10 * 100, bins=cfg['plot_bins'], color='#8c69a5')
            axes[1, j].set(title=f'{horizon}日：每事件成分股90%−10%收益差', xlabel='板块内部差距（百分点）', ylabel='事件数')
            axes[0, j].axvline(0, color='#777', ls='--')
        fig.suptitle('板块间分布与板块内成分股分化；没有按样本数筛除板块', fontsize=15)
        save(fig, 'between_sectors_and_member_dispersion', book)
        sector_list = sorted(origins.sector_code.unique())
        rng = np.random.default_rng(cfg['plot_seed'])
        for start in range(0, len(sector_list), 35):
            selected = sector_list[start:start + 35]
            fig, axes = plt.subplots(1, 3, figsize=(19, 13), layout='constrained')
            for ax, horizon in zip(axes, cfg['horizons']):
                ticks = []
                for i, code in enumerate(selected):
                    g = board[board.horizon.eq(horizon) & board.sector_code.eq(code)]
                    v = g.end_return.dropna() * 100
                    ticks.append(f'{g.name.iloc[-1]} {code[:6]} [{len(v)}/{len(g)}]')
                    if len(v):
                        ax.scatter(v, i + rng.uniform(-.10, .10, len(v)), s=14, alpha=.65, color='#367aa3')
                        ax.plot([v.quantile(.1), v.quantile(.9)], [i, i], lw=2, alpha=.5, color='#367aa3')
                        ax.scatter([v.median()], [i], marker='|', s=100, color='#b85457')
                ax.set_yticks(range(len(selected)), ticks, fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color='#888', ls='--')
                ax.grid(axis='x', alpha=.15)
                ax.set(title=f'{horizon}日；[完整/全部事件数]', xlabel='期末收益（%）')
            fig.suptitle('同一板块历次确认事件：点=每次收益，横线=10%—90%，红线=中位数\n板块按代码排列；不同板块的事件时点可能不同', fontsize=14)
            save(fig, f'within_sector_events_{start // 35 + 1:02d}', book)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout='constrained')
        for j, horizon in enumerate(cfg['horizons']):
            g = member[member.horizon.eq(horizon) & member.complete]
            sizes = g.groupby('event_id').ts_code.transform('size')
            weights = 1 / sizes
            for metric in metrics:
                axes[0, j].hist(g[metric] * 100, bins=cfg['plot_bins'], weights=weights, density=True,
                    histtype='step', lw=1.5, color=colors[metric], label=labels[metric])
            axes[0, j].legend(fontsize=8)
            axes[0, j].set(title=f'{horizon}日成分股收益：先事件等权、事件内等权', xlabel='收益（%）', ylabel='概率密度')
            sub = member_event[member_event.horizon.eq(horizon) & member_event.metric.eq('end_return')]
            axes[1, j].hist(sub.positive * 100, bins=cfg['plot_bins'], color='#438c65')
            axes[1, j].set(title=f'{horizon}日：各事件内部上涨成分股占比', xlabel='上涨占比（%）', ylabel='事件数')
        fig.suptitle('确认日冻结成分股；完整报价窗口单独统计，缺失见覆盖率表', fontsize=15)
        save(fig, 'within_event_members', book)
    fmt = lambda value: f'{value:.2%}' if pd.notna(value) else '缺失'
    lines = ['# 确认后3／10／20交易日收益与板块内外分布', '',
        f'固定230个首次确认事件，{origins.date.nunique()}个日期，{origins.sector_code.nunique()}个板块代码。未修改触发或确认条件。', '',
        '## 口径', '',
        '- 起点为首次确认日收盘。第3／10／20日为全市场交易日，不是下一条可用报价。',
        '- 期末收益=P(T+h)/P(T)−1；期间最高／最低=P(T+1)…P(T+h)相对P(T)的最高／最低收益，不含第0日。最高值可以为负，最低值可以为正。极值出现日保留在明细。',
        '- 使用每日收盘价。自算板块高低价是异步成分股边界，未用作真实盘中极值；期间最低收益也不是峰谷最大回撤。',
        '- 板块路径不得跨数据缺口、指数分段或分类切换。完整窗口必须有所有交易日价格；截尾窗口不填零、不跳日。',
        '- 成分股取确认日已知归属，冻结名单；后续调出仍跟踪，后续调入不加入。使用后复权收盘，不按当前上市状态筛选；已知退市后报价排除。缺失/停牌无报价不填充，不能把完整样本当成幸存者偏差已消除。',
        '- 板块为每日成员、前日市值限权加权代理；成分股分布是固定起点成员等权，两者不要求一致。',
        '- 这是确认后的走势观察，确认日收盘信息不代表可以同价成交；未计费用、成交限制、选股或账户收益。', '',
        '## 总体：事件等权', '',
        '| 窗口 | 完整/全部 | 期末均值 | 期末中位数 | 上涨占比 | 期末10%—90% | 期间最高均值 | 期间最低均值 |',
        '|---|---:|---:|---:|---:|---|---:|---:|']
    headline = []
    for horizon in cfg['horizons']:
        g = board[board.horizon.eq(horizon) & board.complete]
        d = describe(g.end_return)
        lines.append(f'| {horizon}日 | {len(g)}/230 | {fmt(d["mean"])} | {fmt(d["median"])} | {fmt(d["positive"])} | {fmt(d["q10"])}～{fmt(d["q90"])} | {fmt(g.max_return.mean())} | {fmt(g.min_return.mean())} |')
        headline.append(dict(horizon=horizon, complete=len(g), mean=d['mean'], median=d['median'], positive=d['positive'],
            q10=d['q10'], q90=d['q90'], mean_max=g.max_return.mean(), mean_min=g.min_return.mean(),
            date_equal_mean=g.groupby('date').end_return.mean().mean(), mean_excess=g.excess_return.mean()))
    lines += ['', '## 日期集中与板块差异', '',
        f'- 最大确认日为{date_counts.idxmax():%Y-%m-%d}，含{date_counts.max()}个事件。事件并非相互独立。',
        f'- 101个板块中，{counts.eq(1).sum()}个仅1次事件，{counts.between(2, 4).sum()}个有2—4次，{counts.ge(5).sum()}个至少5次；单样本板块不能据此判断稳定性。',
        '- 同一代码的跨版本事件可汇总分布，但不拼接价格。另附按分类版本拆分表；不同板块触发时点不同，均值差异不能直接解释为板块本身优劣。', '',
        '| 窗口 | 日期等权期末均值 | 板块等权期末均值 | 事件等权相对沪深300 |', '|---|---:|---:|---:|']
    for horizon in cfg['horizons']:
        g = board[board.horizon.eq(horizon) & board.complete]
        lines.append(f'| {horizon}日 | {fmt(g.groupby("date").end_return.mean().mean())} | {fmt(g.groupby("sector_code").end_return.mean().mean())} | {fmt(g.excess_return.mean())} |')
    lines += ['', '## 每次事件内部的成分股分布', '',
        '以下先统计每个事件的成员分布，再对事件等权汇总；成员较多的板块不会自动占据更大权重。分位数差取各事件的中位数。', '',
        '| 窗口 | 完整股票-事件/全部 | 成员覆盖率中位数 | 覆盖≥80%的事件 | 成员收益均值的事件均值 | 上涨成员占比均值 | 成员90%−10%收益差中位数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    member_headline = []
    for horizon in cfg['horizons']:
        g = member_event[member_event.horizon.eq(horizon) & member_event.metric.eq('end_return')]
        m = member[member.horizon.eq(horizon)]
        lines.append(f'| {horizon}日 | {m.complete.sum()}/{len(m)} | {fmt(g.coverage.median())} | {g.sufficient_coverage.sum()}/230 | {fmt(g["mean"].mean())} | {fmt(g.positive.mean())} | {fmt(g.spread_q90_q10.median())} |')
        member_headline.append(dict(horizon=horizon, total=len(m), complete=int(m.complete.sum()), coverage_median=g.coverage.median(),
            event_mean=g['mean'].mean(), positive_member_fraction=g.positive.mean(), spread_median=g.spread_q90_q10.median(),
            sufficient_events=int(g.sufficient_coverage.sum())))
    lines += ['', '## 查看文件', '',
        '- confirmation_returns_report.pdf：全部图表；overview.html：浏览版。',
        '- sector_event_outcomes.csv：230事件×3窗口，含收益、极值、极值发生日、完整性和缺失原因。',
        '- overall_distribution.csv：期末／最高／最低收益的均值、中位数、分位数、全样本最小最大值，含事件、日期、板块等权三种口径。日期／板块等权行描述的是组均值分布。',
        '- by_sector_distribution.csv：每板块每窗口各收益指标的历史事件分布；by_sector_version_distribution.csv按分类版本拆分；by_year_distribution.csv按年份。',
        '- within_event_member_distribution.csv：每事件每窗口的成员均值、分位数、上涨占比、分化程度和覆盖率；member_outcomes.parquet为每只股票的明细。',
        '- member_event_equal_summary.csv：成分股分布在事件间的汇总，附覆盖率≥80%子样本；member_by_sector_summary.csv：按板块汇总。',
        '- daily_paths.parquet：0至20日板块收益及成员分位数；path_distribution.csv同时提供逐日可观察队列和固定完整20日队列。',
        '- fixed_membership.parquet：确认日冻结名单；缺失不被静默移出分母。', '',
        '## 核验与边界', '',
        '- 四项测试覆盖极值与期末、起点不混入极值、缺口与截尾、窗口和资产之间独立处理。源数据和事件产物哈希已核验；确认日价格与原快照一致。',
        '- 所有收益完整窗口满足最低≤期末≤最高，事件及股票明细唯一键检查通过。',
        '- 历史行业归属不完整、近期平台识别误差仍保留在样本中。本报告是这版规则的描述性结果，不是独立样本外收益证明。', '']
    (out / 'README.md').write_text('\n'.join(lines))
    (out / 'overview.html').write_text('<!doctype html><meta charset="utf-8"><title>确认后收益分布</title><style>body{max-width:1500px;margin:auto;font-family:system-ui}img{width:100%}</style><h1>确认后3／10／20日收益及板块内外分布</h1><p>起点为首次确认日收盘；极值为随后每日收盘收益极值。详见同目录README。</p>' + ''.join(f'<h2>{name}</h2><img loading="lazy" src="{name}">' for name in charts))
    save_json(out / 'summary.json', dict(events=len(origins), dates=origins.date.nunique(), sectors=len(counts),
        single_event_sectors=int(counts.eq(1).sum()), board=headline, members=member_headline))
    save_json(out / 'validation.json', dict(tests=tests.testsRun, source_hashes=True, origin_price_exact=True,
        membership_counts_exact=True, unique_keys=True, extrema_order=True))
    for p in [config, Path(__file__), ROOT / 'src/sector_confirmation_returns.py', ROOT / 'tests/test_sector_confirmation_returns.py']:
        shutil.copy2(p, out / p.name)
    save_json(out / 'manifest.json', dict(status='complete', dataset_content_id=ds.manifest['content_id'],
        source_manifest_sha256=digest(source / 'manifest.json'), config=cfg,
        artifacts={p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}))
    print(json.dumps(dict(output=str(out), board=headline, members=member_headline), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
