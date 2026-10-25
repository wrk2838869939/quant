#!/usr/bin/env python3
"""既有随机测试评分的板块间/板块内收益报告，不重训或更改选择。"""
from pathlib import Path
from datetime import datetime, timezone
import sys
import os
import json
import shutil
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import PROJECT, HERE, save_json, digest, DataHub, plt, font_manager, PdfPages
from sector_confirmation_returns import window_metrics


def weights(frame, level):
    """板块事件等权；个股先在每次事件内等权，再让事件等权。"""
    w = np.ones(len(frame)) if level == 'sector' else 1 / frame.event_id.map(frame.event_id.value_counts()).to_numpy(float)
    return w / w.sum()


def distribution(values, w, weighted=True):
    """个股用加权经验分布逆；板块沿用既有报告的线性插值分位数。"""
    v = np.asarray(values, float)
    order = np.argsort(v)
    x, cdf = v[order], np.cumsum(w[order])
    return dict(mean=float(np.sum(v * w)), positive=float(w[v > 0].sum()),
                minimum=float(x[0]), maximum=float(x[-1]),
                **{k: float(x[min(np.searchsorted(cdf, q), len(x) - 1)] if weighted else np.quantile(v, q))
                   for k, q in [('q10', .1), ('median', .5), ('q90', .9)]})


def audit_extremes(extremes, cfg):
    """从冻结行情复算被报告的最极端事件，防止汇总掩盖串价或单位错误。"""
    hub = DataHub()
    assert hub.cfg.root.resolve() == (PROJECT / cfg['release']).resolve()
    dm = json.loads((PROJECT / cfg['dataset'] / 'manifest.json').read_text())
    assert os.environ['DATAHUB_CONTENT_ID'] == dm['market_content_id']
    calendar = hub.calendar.cn_days(dm['config']['start'], dm['config']['end'])
    panel = pd.read_parquet(PROJECT / cfg['dataset'] / 'daily.parquet',
                            columns=['date', 'sector_code', 'segment_id', 'index_valid', 'index_close'])
    stocks = extremes[extremes.level == 'member']
    quotes = hub.cn.daily(codes=sorted(stocks.ts_code.unique()), start=str(stocks.date.min().date()),
                          end=str(stocks.end_date.max().date()), fields=['hfq_close'])
    prices = quotes.pivot(index='date', columns='ts_code', values='hfq_close')
    checks = []
    for r in extremes.itertuples():
        i = calendar.get_loc(r.date)
        days = calendar[i:i + r.horizon + 1]
        if r.level == 'member':
            x = prices.reindex(days)[r.ts_code].to_numpy()
        else:
            a = panel[panel.sector_code.eq(r.sector_code)].set_index('date').reindex(days)
            x = a.index_close.where(a.index_valid & a.segment_id.eq(r.segment_id)).to_numpy()
        actual = window_metrics(x, r.horizon)
        assert actual['complete'][0]
        np.testing.assert_allclose(actual[r.metric][0], r.value, rtol=1e-10, atol=1e-10)
        checks.append(dict(level=r.level, horizon=r.horizon, metric=r.metric, side=r.side,
                           event_id=r.event_id, recomputed=float(actual[r.metric][0])))
    return checks


def main():
    cfg = yaml.safe_load((HERE / 'distribution_config.yaml').read_text())
    experiment, returns = [PROJECT / cfg[k] for k in ['experiment', 'returns']]
    for path in [experiment, returns]:
        manifest = json.loads((path / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        for file, expected in manifest['artifacts'].items():
            assert digest(path / file) == expected, file
    prediction = pd.read_parquet(experiment / 'test_predictions_before_evaluation.parquet')
    prediction = prediction[(prediction.model == cfg['model']) & (prediction.retention == cfg['retention'])]
    assert prediction.event_id.is_unique and prediction.role.eq('test').all()
    key = prediction[['event_id', 'selected', 'round']]
    board = pd.read_parquet(returns / 'sector_event_outcomes.parquet').merge(key, on='event_id', validate='many_to_one')
    member = pd.read_parquet(returns / 'member_outcomes.parquet').merge(key, on='event_id', validate='many_to_one')
    assert not board.duplicated(['event_id', 'horizon']).any()
    assert not member.duplicated(['event_id', 'horizon', 'ts_code']).any()
    # 简单合成例检查：含一只股票和含三只股票的事件各占一半。
    toy = pd.DataFrame({'event_id': ['a', 'b', 'b', 'b']})
    np.testing.assert_allclose(weights(toy, 'member'), [.5, 1/6, 1/6, 1/6])
    assert np.isclose(distribution([0., 1., 1., 1.], weights(toy, 'member'))['mean'], .5)
    out = PROJECT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    print('OUTPUT', out, flush=True)
    for name in ['report_distribution.py', 'distribution_config.yaml']:
        shutil.copy2(HERE / name, out / name)
    shutil.copy2(PROJECT / 'alpha_lab/src/sector_confirmation_returns.py', out / 'sector_confirmation_returns.py')
    save_json(out / 'manifest.json', dict(status='building'))
    summary, coverage, within, extremes, sectors = [], [], [], [], []
    for sample in ['baseline', 'selected']:
        for h in cfg['horizons']:
            b = board[board.horizon.eq(h) & board.complete & (board.selected if sample == 'selected' else True)].copy()
            all_m = member[member.horizon.eq(h) & member.event_id.isin(b.event_id)].copy()
            m = all_m[all_m.complete].copy()
            c = all_m.groupby('event_id').complete.agg(['sum', 'count'])
            c['coverage'] = c['sum'] / c['count']
            coverage.append(dict(sample=sample, horizon=h, events=len(b), sectors=b.sector_code.nunique(),
                member_observations=len(all_m), complete_members=len(m), unique_stocks=m.ts_code.nunique(),
                coverage_min=c.coverage.min(), coverage_median=c.coverage.median(),
                below_coverage_threshold=int(c.coverage.lt(cfg['coverage_sensitivity']).sum())))
            assert set(m.event_id) == set(b.event_id)
            for level, f in [('sector', b), ('member', m)]:
                w = weights(f, level)
                if level == 'member':
                    np.testing.assert_allclose(pd.Series(w, index=f.event_id).groupby(level=0).sum(), 1 / len(b))
                    np.testing.assert_allclose(np.average(f.end_return, weights=w), f.groupby('event_id').end_return.mean().mean())
                for metric in ['end_return', 'max_return', 'min_return']:
                    summary.append(dict(sample=sample, horizon=h, level=level, metric=metric,
                        events=len(b), observations=len(f), **distribution(f[metric], w, weighted=level == 'member')))
                    if sample == 'selected':
                        for side, index in [('minimum', f[metric].idxmin()), ('maximum', f[metric].idxmax())]:
                            r = f.loc[index]
                            extremes.append(dict(level=level, horizon=h, metric=metric, side=side, value=r[metric],
                                event_id=r.event_id, date=r.date, end_date=r.end_date, sector_code=r.sector_code,
                                name=r['name'], segment_id=r.segment_id, ts_code=r.get('ts_code', '')))
            for event, f in m.groupby('event_id'):
                q = f.end_return.quantile([.1, .5, .9])
                within.append(dict(sample=sample, horizon=h, event_id=event, sector_code=f.sector_code.iloc[0],
                    name=f['name'].iloc[0], date=f.date.iloc[0], n=len(f), coverage=c.loc[event, 'coverage'],
                    mean=f.end_return.mean(), median=q.loc[.5], q10=q.loc[.1], q90=q.loc[.9],
                    spread_q90_q10=q.loc[.9] - q.loc[.1], minimum=f.end_return.min(), maximum=f.end_return.max(),
                    positive=f.end_return.gt(0).mean(), path_low=f.min_return.min(), path_high=f.max_return.max()))
            for (version, code, name), f in b.groupby(['classification_version', 'sector_code', 'name']):
                sectors.append(dict(sample=sample, horizon=h, classification_version=version, sector_code=code, name=name,
                    events=len(f), mean=f.end_return.mean(), median=f.end_return.median(),
                    minimum=f.end_return.min(), maximum=f.end_return.max(), path_low=f.min_return.min(), path_high=f.max_return.max()))
    summary, coverage, within, extremes, sectors = [pd.DataFrame(a) for a in [summary, coverage, within, extremes, sectors]]
    for name, frame in [('distribution_summary', summary), ('coverage', coverage), ('within_event', within), ('extreme_cases', extremes), ('by_sector', sectors)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    checks = audit_extremes(extremes, cfg)
    save_json(out / 'extremes_recomputed.json', checks)
    inside = within.groupby(['sample', 'horizon']).agg(events=('event_id', 'size'),
        event_mean=('mean', 'mean'), typical_median=('median', 'median'), typical_q10=('q10', 'median'),
        typical_q90=('q90', 'median'), typical_spread=('spread_q90_q10', 'median'), positive_share=('positive', 'mean')).reset_index()
    inside.to_csv(out / 'within_typical_summary.csv', index=False)
    sensitivity = within[within.coverage.ge(cfg['coverage_sensitivity'])].groupby(['sample', 'horizon']).agg(events=('event_id', 'size'), event_mean=('mean', 'mean')).reset_index()
    sensitivity.to_csv(out / 'coverage_sensitivity.csv', index=False)
    charts(out, cfg, board, member, within)
    text_report(out, cfg, summary, coverage, inside, sensitivity)
    save_json(out / 'validation.json', dict(source_hashes_verified=True, event_equal_weights_checked=True,
        extreme_metrics_recomputed=len(checks), test_events=prediction.event_id.nunique(), selected_events=int(prediction.selected.sum())))
    save_json(out / 'manifest.json', dict(status='complete', config=cfg,
        experiment_manifest_sha256=digest(experiment / 'manifest.json'), returns_manifest_sha256=digest(returns / 'manifest.json'),
        artifacts={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print(summary[(summary['sample'] == 'selected') & summary.metric.eq('end_return')].to_string(index=False), flush=True)
    print(inside.to_string(index=False), flush=True)
    print('COMPLETE', out, flush=True)


def charts(out, cfg, board, member, within):
    font_manager.fontManager.addfont(str(PROJECT / cfg['font']))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=PROJECT / cfg['font']).get_name(), 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    with PdfPages(out / 'sector_member_returns.pdf') as pdf:
        for metric, title in [('end_return', '期末收益'), ('max_return', '期间最高收盘收益'), ('min_return', '期间最低收盘收益')]:
            fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
            for col, h in enumerate(cfg['horizons']):
                for row, level in enumerate(['sector', 'member']):
                    ax = axes[row, col]
                    for chosen, label, color in [(False, '未筛选测试集', '#777777'), (True, '神经网络选中', '#1565c0')]:
                        b = board[board.horizon.eq(h) & board.complete & (board.selected if chosen else True)]
                        f = b if level == 'sector' else member[member.horizon.eq(h) & member.complete & member.event_id.isin(b.event_id)]
                        order = np.argsort(f[metric].to_numpy())
                        ax.step(f[metric].to_numpy()[order] * 100, np.cumsum(weights(f, level)[order]), where='post',
                                label=f'{label}：{len(b)}事件 / {len(f)}观测', color=color)
                    ax.set(title=f'{h} 日 · ' + ('板块之间（事件等权）' if level == 'sector' else '板块内个股（先事件、后个股等权）'),
                           xlabel=title + '（%）', ylabel='累计比例')
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=8)
            fig.suptitle(title + '分布：随机两轮测试结果；成员固定在确认日', fontsize=15)
            fig.savefig(out / f'{metric}_ecdf.png', dpi=cfg['dpi']); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
        for ax, h in zip(axes, cfg['horizons']):
            for sample, label, color in [('baseline', '未筛选', '#777777'), ('selected', '神经网络选中', '#1565c0')]:
                x = within[(within['sample'] == sample) & within.horizon.eq(h)].spread_q90_q10.sort_values().to_numpy() * 100
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=color, label=f'{label}，中位数 {np.median(x):.2f}pp')
            ax.set(title=f'{h} 日板块内分化', xlabel='同次事件内个股 q90 - q10（百分点）', ylabel='事件累计比例')
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        pdf.savefig(fig); fig.savefig(out / 'within_dispersion.png', dpi=cfg['dpi']); plt.close(fig)


def text_report(out, cfg, summary, coverage, inside, sensitivity):
    lines = ['# 随机测试事件：板块之间与板块内部收益分布', '',
        '主样本为神经网络、训练评分 50% 保留阈值选中的两轮测试事件。未重训、未修改阈值。',
        '收益起点是确认日收盘；3/10/20 均为交易日。板块采用既有指数，个股采用后复权收盘；成员固定为确认日成员。', '',
        '## 期末收益分布', '',
        '| 样本 | 层次 | 天数 | 事件/观测 | 均值 | 中位数 | q10 | q90 | 最小 | 最大 | 正收益比例 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary[summary.metric.eq('end_return')].itertuples():
        lines.append(f'| {r.sample} | {r.level} | {r.horizon} | {r.events}/{r.observations} | ' + ' | '.join(f'{getattr(r,k):.2%}' for k in ['mean','median','q10','q90','minimum','maximum','positive']) + ' |')
    lines += ['', '## 期间最高/最低收盘收益', '',
              '| 样本 | 层次 | 天数 | 极值类型 | 极值均值 | 全样本最小 | 全样本最大 |', '|---|---|---:|---|---:|---:|---:|']
    for r in summary[~summary.metric.eq('end_return')].itertuples():
        lines.append(f'| {r.sample} | {r.level} | {r.horizon} | {r.metric} | {r.mean:.2%} | {r.minimum:.2%} | {r.maximum:.2%} |')
    lines += ['', '## 板块内分化', '', inside.to_csv(index=False), '', '## 覆盖', '', coverage.to_csv(index=False),
              '', '## 成员覆盖至少 80% 的敏感性', '', sensitivity.to_csv(index=False), '',
              '## 口径说明', '',
              '- baseline 为同两轮全部测试事件，selected 为神经网络选中；不是全部 330 个原始事件。',
              '- 板块间主分布按板块事件等权；同板块可在不同日期重复出现。by_sector.csv 另按分类版本/板块汇总事件数和收益，避免把重复事件当作独立板块。',
              '- 板块内先令事件等权，再令该事件内完整个股等权；股票大板块不会获得更高权重。个股观测为事件×股票，不是独立股票数量。',
              '- 板块与个股均使用板块完整的同一事件集合；个股缺日不填充，缺失成员在该期限剔除。原始缺失数量、覆盖及 80% 敏感性单列。',
              '- 个股混合分布的 weighted q10/median/q90 采用加权经验累计分布的逆，不插值；板块分布沿用上份报告的通常线性插值分位数。',
              '- within_event.csv 的每次板块内部分位数用通常线性插值，within_typical_summary.csv 再取事件间中位数描述典型板块。二者不能与混合个股分布的分位数混用。',
              '- 期间极值为未来第 1 至 h 日相对确认收盘的累计收益极值，不含第 0 日，不是日内高低、峰谷最大回撤或实际可兑现止盈。',
              '- 不同期限完整样本有变化；分位区间和期间极值不是同一组股票的可交易收益路径。',
              '- 随机集合有共同日期/行情冲击，历史板块成员覆盖仍有限；没有新增独立样本，不报告实盘收益或账户最大回撤。',
              '- 极端值已用冻结板块与个股收盘行情逐项复算；这验证计算一致，不代表已排除原始行情的一切质量问题。', '',
              '## 文件', '',
              '- sector_member_returns.pdf：期末、期间最高/最低收益、板块内分化的四页图。',
              '- distribution_summary.csv：全部口径；within_event.csv：逐事件个股分布；by_sector.csv：逐板块汇总。',
              '- extreme_cases.csv：极值所属板块、股票和日期；extremes_recomputed.json：冻结行情复算。', '']
    (out / 'README.md').write_text('\n'.join(lines))
    (out / 'overview.html').write_text('<!doctype html><meta charset="utf-8"><title>板块内外收益分布</title><style>body{max-width:1500px;margin:auto;font-family:system-ui}img{width:100%}</style><h1>板块内外收益分布</h1><p>仅两轮随机测试事件；蓝线选中，灰线未筛选。</p>' + ''.join(f'<img src="{p}">' for p in ['end_return_ecdf.png','max_return_ecdf.png','min_return_ecdf.png','within_dispersion.png']))


if __name__ == '__main__':
    main()
