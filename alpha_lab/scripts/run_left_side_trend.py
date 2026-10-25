#!/usr/bin/env python3
"""在已冻结板块和账本上验证双层上行趋势及近期上涨回落修正。"""
from datetime import datetime, timezone
from pathlib import Path
import itertools
import json
import multiprocessing as mp
import shutil
import sys
import traceback
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from left_side_trend import trend_pullback, project_sector_features, filter_scores
from left_side_rules import entry_scores, LeftSidePolicy
from run_classic_strategy_smoke import digest, save_json
from run_rule_strategy_search import write_result
import run_policy_balance as B
G = B.G


def features(params, cutoff=None):
    cfg = {**G['cfg']['trend'], **params}
    stock_close, sector_close, membership = G['wide']['hfq_close'], G['sector_close'], G['old_features']['sector_id']
    if cutoff:
        stock_close, sector_close, membership = stock_close.loc[:cutoff], sector_close.loc[:cutoff], membership.loc[:cutoff]
    stock = trend_pullback(stock_close, cfg, cfg['stock_minimum_advance'], cfg['stock_minimum_pullback'])
    sector = trend_pullback(sector_close, cfg, cfg['sector_minimum_advance'], cfg['sector_minimum_pullback'])
    mapped = project_sector_features(sector, membership)
    return stock, sector, mapped


def execute(name, params, mode, f):
    cfg = G['cfg']
    stock, sector, mapped = f
    scores = filter_scores(G['base_scores'], stock, mapped, mode).loc[cfg['start']:cfg['end']]
    policy = LeftSidePolicy(scores, G['wide'], cfg)
    result = B.R.execute(policy, cfg['start'], cfg['end'])
    audit = B.R.original.audit(result)
    folder = G['run'] / 'accounts' / name
    write_result(folder, result, scores)
    pd.DataFrame(policy.records).to_parquet(folder / 'position_decisions.parquet', index=False)
    save_json(folder / 'audit.json', audit)
    metrics = B.R.metric(result)
    row = dict(id=name, mode=mode, **params, **metrics,
               candidate_stock_days=int(scores.notna().sum().sum()), candidate_days=int(scores.notna().any(axis=1).sum()))
    save_json(folder / 'metrics.json', row)
    if name == 'legacy':
        for table in B.R.original.TABLES:
            actual = getattr(result, table)
            expected = pd.read_parquet(G['parent'] / 'accounts/default' / (table + '.parquet'))
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        save_json(folder / 'legacy_exact.json', dict(tables=B.R.original.TABLES, exact=True))
    if name in ['default', 'sector_only', 'stock_only']:
        records = []
        for date, line in scores.iterrows():
            for code, score in line.dropna().items():
                rec = dict(date=date, ts_code=code, score=score)
                for level, data in [('stock', stock), ('sector', mapped)]:
                    for k, frame in data.items():
                        rec[level + '_' + k] = frame.at[date, code]
                    dates = G['wide']['hfq_close'].index if level == 'stock' else G['sector_close'].index
                    for k in ['anchor_i', 'peak_i']:
                        value = rec[level + '_' + k]
                        rec[level + '_' + k.replace('_i', '_date')] = dates[int(value)] if np.isfinite(value) else pd.NaT
                records.append(rec)
        pd.DataFrame(records, columns=None if records else ['date', 'ts_code', 'score']).to_parquet(folder / 'candidate_diagnostics.parquet', index=False)
    if name == 'default':
        short_scores = scores.loc[:cfg['prefix_date']]
        short_policy = LeftSidePolicy(short_scores, {k: v.loc[:cfg['prefix_date']] for k, v in G['wide'].items()}, cfg)
        short = B.R.execute(short_policy, cfg['start'], cfg['prefix_date'])
        B.R.original.audit(short)
        B.R.original.compare_prefix(short, result, pd.Timestamp(cfg['prefix_date']))
        save_json(folder / 'account_prefix.json', dict(passed=True))
    print('ACCOUNT', name, 'return', round(metrics['total'], 6), 'buys', metrics['buys'], 'candidates', row['candidate_stock_days'], flush=True)
    return row


def grid_task(task):
    name, params = task
    return execute(name, params, 'both', features(params))


def report(rows):
    cfg, run = G['cfg'], G['run']
    frame = pd.DataFrame(rows)
    frame.to_csv(run / 'results.csv', index=False)
    grid = frame[frame.id.str.startswith('grid_')]
    triplets = []
    for parameter in cfg['search']:
        others = [k for k in cfg['search'] if k != parameter]
        for key, part in grid.groupby(others):
            part = part.sort_values(parameter)
            y = part.total.to_numpy()
            triplets.append(dict(parameter=parameter, fixed=json.dumps(dict(zip(others, key)), default=float),
                                 low=float(y[0]), middle=float(y[1]), high=float(y[2]),
                                 nonmonotonic=bool((y[1] - y[0]) * (y[2] - y[1]) < 0)))
    pd.DataFrame(triplets).to_csv(run / 'parameter_triplets.csv', index=False)
    lines = ['# 左侧规则 v2：双层上行趋势及上涨回落修正', '',
             f'运行：`{run.relative_to(PROJECT)}`。固定 10 万、2021—2023 连续账户、次日开盘执行。', '',
             '用户已确认：板块和个股在本轮回落前都应有明确中期上行趋势；买入时可以仍在下跌，不要求重新站上均线。约五日仍是量价触发窗口。', '',
             '本轮仅增加两层趋势和波段先后约束，保留原板块参考带、量价触发、龙头排序及退出，便于定位修正影响。', '',
             '| 版本 | 累计净收益 | 最大回撤 | 买入次数 | 候选股票日 | 平均投入 |', '|---|---:|---:|---:|---:|---:|']
    labels = {'legacy': '原 v1 逐表复现', 'sector_only': '仅增加板块上涨回落', 'stock_only': '仅增加个股上涨回落', 'default': '双层同时满足'}
    for name, label in labels.items():
        r = frame[frame.id.eq(name)].iloc[0]
        lines.append(f'| {label} | {r.total:+.2%} | {abs(r.mdd):.2%} | {int(r.buys)} | {int(r.candidate_stock_days)} | {r.mean_invested:.1%} |')
    lines += ['', '默认阈值：高点时 120 日收益≥5%、价格在60日均线上方且均线较20日前上行；高点前40日上涨段至少持续5日，板块涨幅≥10%、个股≥15%；高点后2—20日分别回落至少8%/4%。买入时无站上均线条件。所有阈值是工程假设。', '',
              f'27 个预定邻域组合中，有买入 {int(grid.buys.gt(0).sum())}/27，正收益且满足回撤限制 {int((grid.total.gt(0) & grid.feasible).sum())}/27。全组合收益范围 {grid.total.min():+.2%} 至 {grid.total.max():+.2%}。',
              f'相邻三点非单调 {sum(r["nonmonotonic"] for r in triplets)}/{len(triplets)}；无交易组合的零收益不是策略通过验收，feasible=false。参数组合只描述同一历史样本，不构成独立验证。', '',
              '## 验证与限制', '',
              '5项新增反例测试通过：纯下跌被拒绝，陈旧上涨被拒绝，缺任一层不成立，缺口不当作趋势，未来冲击不改写历史；正例允许买入价低于均线。',
              '原v1账户四表完全复现；默认新特征与信号从截断收盘重算通过前缀检查，默认账户截断回放通过；全部账户进行独立账本审计。',
              '原93股工程池、历史分类覆盖缺口、全期筛选偏差不变；板块历史价值带的旧锚点定义也未改变。本轮是纠正信号语义的受控对照，不能将零交易或亏损直接归结为用户完整主观策略失效。',
              '信号诊断保存起点/高点日期、涨幅、回撤、距高点天数，以及高点时的趋势指标。funnel.csv 保留每一层过滤数量。']
    path = ROOT / 'docs/LEFT_SIDE_TREND_RESULTS_20260921.md'
    path.write_text('\n'.join(lines) + '\n')
    return path


def main():
    config = ROOT / 'configs/left_side_trend_v2.yaml'
    cfg = yaml.safe_load(config.read_text())
    parent = PROJECT / cfg['parent_run']
    pm = json.loads((parent / 'manifest.json').read_text())
    assert pm['status'] == 'complete'
    hashes = json.loads((parent / 'artifact_hashes.json').read_text())
    assert all(digest(parent / p) == h for p, h in hashes.items())
    assert all(digest(PROJECT / p) == h for p, h in pm['source_hashes'].items())
    run = ROOT / 'outputs/left_side_trend' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    print('RUN=' + str(run), flush=True)
    _, _, account = B.setup(PROJECT / cfg['loader_config'], run)
    assert cfg['max_drawdown'] == account['risk']['max_drawdown']
    files = [Path(__file__), config, ROOT / 'src/left_side_trend.py', ROOT / 'tests/test_left_side_trend.py', ROOT / 'docs/LEFT_SIDE_TREND_PREREG_20260921.md']
    sources = {**pm['source_hashes'], **{str(p.relative_to(PROJECT)): digest(p) for p in files}}
    manifest = dict(status='running', dataset=account['dataset'], source_hashes=sources,
                    parent=str(parent.relative_to(PROJECT)), parent_manifest_sha256=digest(parent / 'manifest.json'), unbiased_oos=False)
    save_json(run / 'manifest.json', manifest)
    (run / 'sources').mkdir()
    for p in files:
        shutil.copy2(p, run / 'sources' / p.name)
    try:
        old_features = {p.stem: pd.read_parquet(p) for p in (parent / 'features').glob('*.parquet')}
        panel = pd.read_parquet(parent / 'sector_panel.parquet')
        sector_close = panel.pivot(index='date', columns='sector_id', values='close').sort_index(axis=1)
        assert list(sector_close.columns) == list(range(len(sector_close.columns)))
        G.update(cfg=cfg, parent=parent, old_features=old_features, sector_close=sector_close)
        base_params = {**{k: cfg['stock'][k] for k in ['shape_window', 'turnover_expansion', 'turnover_contraction']}, 'value_tolerance': cfg['entry']['value_tolerance']}
        scores, _, _ = entry_scores(G['wide'], G['eligible'], old_features, cfg, base_params)
        G['base_scores'] = scores
        pd.testing.assert_frame_equal(scores.loc[cfg['start']:cfg['end']], pd.read_parquet(parent / 'accounts/default/signals.parquet'), check_exact=True)
        defaults = {k: cfg['trend'][k] for k in cfg['search']}
        print('DEFAULT_TREND_FEATURES', flush=True)
        full = features(defaults)
        (run / 'features').mkdir()
        for prefix, part in zip(['stock', 'sector', 'mapped_sector'], full):
            for k, f in part.items():
                f.to_parquet(run / 'features' / (prefix + '_' + k + '.parquet'))
        short = features(defaults, cfg['prefix_date'])
        for a, b in zip(short, full):
            for k in a:
                pd.testing.assert_frame_equal(a[k], b[k].loc[:cfg['prefix_date']], check_exact=True)
        filtered = filter_scores(scores, full[0], full[2])
        pd.testing.assert_frame_equal(filter_scores(scores.loc[:cfg['prefix_date']], short[0], short[2]), filtered.loc[:cfg['prefix_date']], check_exact=True)
        save_json(run / 'feature_prefix.json', dict(passed=True, cutoff=cfg['prefix_date']))
        eligible = G['eligible'].loc[cfg['start']:cfg['end']]
        masks = {'eligible': eligible, 'old_candidates': scores.reindex_like(eligible).notna(),
                 'stock_trend_and_cycle': full[0]['passes'].reindex_like(eligible).eq(1) & eligible,
                 'sector_trend_and_cycle': full[2]['passes'].reindex_like(eligible).eq(1) & eligible,
                 'both_trend_and_cycle': full[0]['passes'].reindex_like(eligible).eq(1) & full[2]['passes'].reindex_like(eligible).eq(1) & eligible}
        for mode in ['stock', 'sector', 'both']:
            masks['old_plus_' + mode] = filter_scores(scores, full[0], full[2], mode).reindex_like(eligible).notna()
        pd.DataFrame([dict(stage=k, stock_days=int(v.sum().sum()), days=int(v.any(axis=1).sum()), stocks=int(v.any().sum())) for k, v in masks.items()]).to_csv(run / 'funnel.csv', index=False)
        rows = [execute(name, defaults, mode, full) for name, mode in [('legacy', 'legacy'), ('sector_only', 'sector'), ('stock_only', 'stock'), ('default', 'both')]]
        keys = list(cfg['search'])
        tasks = [(f'grid_{i:03d}', dict(zip(keys, values))) for i, values in enumerate(itertools.product(*[cfg['search'][k] for k in keys]))]
        save_json(run / 'tasks.json', tasks)
        with mp.get_context('fork').Pool(cfg['workers']) as pool:
            rows.extend(pool.imap_unordered(grid_task, tasks))
        path = report(sorted(rows, key=lambda r: r['id']))
        assert all(digest(PROJECT / p) == h for p, h in sources.items())
        save_json(run / 'artifact_hashes.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file() and p.name != 'manifest.json'})
        manifest.update(status='complete', accounts=len(rows), report=str(path.relative_to(PROJECT)), report_sha256=digest(path))
        save_json(run / 'manifest.json', manifest)
        print('COMPLETE', path, flush=True)
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc())
        save_json(run / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
