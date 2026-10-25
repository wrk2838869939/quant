#!/usr/bin/env python3
"""细分行业自算板块及左侧规则账户回放，完整保留组合参数与过程。"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
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
from datahub import DataHub
from left_side_rules import membership_matrix, sector_panel, sector_states, stock_sector_features, entry_scores, LeftSidePolicy
from run_classic_strategy_smoke import digest, save_json
from run_rule_strategy_search import write_result
import run_policy_balance as B
G = B.G


def default_params(cfg):
    return {**{k: cfg['stock'][k] for k in ['shape_window', 'turnover_expansion', 'turnover_contraction']},
            'value_tolerance': cfg['entry']['value_tolerance']}


def run_account(task):
    name, params, mode = task
    cfg = G['left_cfg']
    scores, shape, sector_ok = entry_scores(G['wide'], G['eligible'], G['sector_features'], cfg, params, mode)
    scores = scores.loc[cfg['start']:cfg['end']]
    policy = LeftSidePolicy(scores, G['wide'], cfg)
    result = B.R.execute(policy, cfg['start'], cfg['end'])
    audit = B.R.original.audit(result)
    folder = G['run'] / 'accounts' / name
    write_result(folder, result, scores)
    actions = pd.DataFrame([dict(date=d, keep=sorted(a.keep), exit=sorted(a.exit)) for d, a in policy.actions])
    actions.to_parquet(folder / 'actions.parquet', index=False)
    pd.DataFrame(policy.records).to_parquet(folder / 'position_decisions.parquet', index=False)
    save_json(folder / 'audit.json', audit)
    metric = B.R.metric(result)
    row = dict(id=name, mode=mode, **params, **metric,
               candidate_stock_days=int(scores.notna().sum().sum()),
               candidate_days=int(scores.notna().any(axis=1).sum()))
    save_json(folder / 'metrics.json', row)
    if name == 'default':
        records = []
        for date, line in scores.iterrows():
            for code in line.dropna().index:
                r = dict(date=date, ts_code=code, score=line[code],
                         circ_mv_yuan=G['wide']['circ_mv_yuan'].at[date, code])
                for k, f in G['sector_features'].items():
                    r[k] = f.at[date, code]
                for k, f in shape.items():
                    r[k] = f.at[date, code]
                records.append(r)
        pd.DataFrame(records).to_parquet(folder / 'candidate_diagnostics.parquet', index=False)
        cutoff = cfg['prefix_date']
        prefix_scores = scores.loc[:cutoff]
        short = LeftSidePolicy(prefix_scores, {k: v.loc[:cutoff] for k, v in G['wide'].items()}, cfg)
        replay = B.R.execute(short, cfg['start'], cutoff)
        B.R.original.audit(replay)
        B.R.original.compare_prefix(replay, result, pd.Timestamp(cutoff))
        save_json(folder / 'prefix_check.json', dict(account=True, cutoff=cutoff))
    return row


def build_features(hub, supplement, cfg, run):
    print('LOAD_FULL_MARKET', flush=True)
    data = hub.cn.daily(start=cfg['history_start'], end=cfg['end'],
                        fields=['hfq_close', 'total_mv_yuan'])
    assert not data.duplicated(['date', 'ts_code']).any()
    dates = G['calendar'][(G['calendar'] >= cfg['history_start']) & (G['calendar'] <= cfg['end'])]
    close = data.pivot(index='date', columns='ts_code', values='hfq_close').reindex(dates)
    cap = data.pivot(index='date', columns='ts_code', values='total_mv_yuan').reindex_like(close)
    del data
    members = pd.read_parquet(supplement / 'members.parquet')
    catalog = pd.read_parquet(supplement / 'catalog.parquet')
    mapping, sectors = membership_matrix(members, catalog, dates, close.columns, cfg['membership'])
    observed = close.notna().to_numpy()
    coverage = pd.DataFrame(dict(date=dates, quoted_stocks=observed.sum(axis=1),
                                assigned=((mapping >= 0) & observed).sum(axis=1),
                                missing=((mapping == -1) & observed).sum(axis=1),
                                conflicting=((mapping == -2) & observed).sum(axis=1)))
    subset = close.columns.get_indexer(G['eligible'].columns)
    coverage['engineering_assigned'] = (mapping[:, subset] >= 0).sum(axis=1)
    coverage.to_csv(run / 'membership_coverage.csv', index=False)
    print('BUILD_SECTOR_PANEL', flush=True)
    panel = sector_panel(close, cap, mapping, sectors, cfg['sector'])
    panel.to_parquet(run / 'sector_panel.parquet', index=False)
    print('BUILD_SECTOR_STATES', flush=True)
    states = sector_states(panel, cfg['sector'])
    states.to_parquet(run / 'sector_states.parquet', index=False)
    print('BUILD_LEADER_HISTORY', flush=True)
    features = stock_sector_features(states, close, mapping, sectors, G['eligible'].columns, cfg['stock'])
    (run / 'features').mkdir()
    for name, frame in features.items():
        frame.to_parquet(run / 'features' / (name + '.parquet'))
    save_json(run / 'sector_ids.json', sectors)
    # 从原始截断输入重新生成板块及状态，不能只把全样本特征切片当作前缀检查。
    print('PREFIX_CHECK_RAW_INPUTS', flush=True)
    take = dates <= cfg['prefix_date']
    short_map, _ = membership_matrix(members, catalog, dates[take], close.columns, cfg['membership'])
    np.testing.assert_array_equal(short_map, mapping[take])
    short_panel = sector_panel(close.loc[take], cap.loc[take], short_map, sectors, cfg['sector'])
    pd.testing.assert_frame_equal(short_panel, panel[panel.date <= cfg['prefix_date']].reset_index(drop=True), check_exact=True)
    short_states = sector_states(short_panel, cfg['sector'])
    pd.testing.assert_frame_equal(short_states, states[states.date <= cfg['prefix_date']].reset_index(drop=True), check_exact=True)
    short_features = stock_sector_features(short_states, close.loc[take], short_map, sectors, G['eligible'].columns, cfg['stock'])
    for k in features:
        pd.testing.assert_frame_equal(short_features[k], features[k].loc[:cfg['prefix_date']], check_exact=True)
    short_scores, _, _ = entry_scores({k: v.loc[:cfg['prefix_date']] for k, v in G['wide'].items()},
                                    G['eligible'].loc[:cfg['prefix_date']], short_features, cfg, default_params(cfg))
    full_scores, _, _ = entry_scores(G['wide'], G['eligible'], features, cfg, default_params(cfg))
    pd.testing.assert_frame_equal(short_scores, full_scores.loc[:cfg['prefix_date']], check_exact=True)
    save_json(run / 'feature_prefix_check.json', dict(membership=True, panel=True, states=True, leaders=True, scores=True))
    return features


def report(run, cfg, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(run / 'results.csv', index=False)
    grid = frame[frame.id.str.startswith('grid_')]
    triplets = []
    for parameter in cfg['search']:
        others = [k for k in cfg['search'] if k != parameter]
        for key, part in grid.groupby(others):
            part = part.sort_values(parameter)
            if len(part) == 3:
                y = part.total.to_numpy()
                triplets.append(dict(parameter=parameter, fixed=json.dumps(dict(zip(others, key)), default=float),
                                     low=float(y[0]), middle=float(y[1]), high=float(y[2]),
                                     nonmonotonic=bool((y[1] - y[0]) * (y[2] - y[1]) < 0)))
    pd.DataFrame(triplets).to_csv(run / 'parameter_triplets.csv', index=False)
    coverage = pd.read_csv(run / 'membership_coverage.csv')
    coverage = coverage[coverage.date.between(cfg['start'], cfg['end'])]
    sector = pd.read_parquet(run / 'sector_states.parquet')
    sector = sector[sector.date.between(cfg['start'], cfg['end'])]
    funnel = {k: int(sector[k].fillna(False).sum()) for k in ['rallied', 'drawn_down', 'fast_fall', 'broken', 'slowed', 'structural_gate']}
    funnel['sector_entry_days'] = int((sector.structural_gate & sector.above_lower_bound & sector.value_distance.le(cfg['entry']['value_tolerance'])).sum())
    save_json(run / 'sector_funnel.json', funnel)
    annual = []
    for row in rows:
        daily = pd.read_parquet(run / 'accounts' / row['id'] / 'daily.parquet')
        previous = G['portfolio']['capital']
        for year, part in daily.groupby(daily.index.year):
            annual.append(dict(id=row['id'], year=int(year), total=float(part.equity.iloc[-1] / previous - 1)))
            previous = part.equity.iloc[-1]
    pd.DataFrame(annual).to_csv(run / 'annual_returns.csv', index=False)
    chosen = frame[frame.id.isin(['default', 'no_leader', 'stock_only'])]
    lines = ['# 左侧板块—个股量价规则：首轮结果', '',
             f'运行：`{run.relative_to(PROJECT)}`。2021—2023 年，10 万连续账户，收盘信号、次日开盘执行。', '',
             '使用申万二级历史归属和冻结全市场个股数据自算板块；不是东方财富官方指数。原始细分行业资料独立封存，原行情快照和账户引擎未改。', '',
             '## 预定默认与机制对照', '', '| 版本 | 累计净收益 | 最大回撤 | 买入次数 | 候选股日 | 平均投入 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in chosen.to_dict('records'):
        lines.append(f'| {r["id"]} | {r["total"]:+.2%} | {abs(r["mdd"]):.2%} | {int(r["buys"])} | {r["candidate_stock_days"]} | {r["mean_invested"]:.1%} |')
    lines += ['', 'default 保留板块条件、量价形态和历史龙头排序；no_leader 仅取消龙头排序；stock_only 仅保留个股量价形态。市值和广度作为过程变量记录，未擅自确定用户的大小盘偏好。', '',
              '## 参数组合', '',
              f'预先声明的 {len(grid)} 个完整组合：窗口 4/5/7 日，放量倍数 1.1/1.4/1.8，缩量比例 0.5/0.7/0.9，板块参考区上沿容忍 5%/12%/20%。',
              f'组合收益范围 {grid.total.min():+.2%} 至 {grid.total.max():+.2%}，中位 {grid.total.median():+.2%}；有买入的组合 {int(grid.buys.gt(0).sum())}/{len(grid)}。',
              f'收益为正且符合回撤限制的组合 {int((grid.total.gt(0) & grid.feasible).sum())}/{len(grid)}。相邻三点非单调 {sum(x["nonmonotonic"] for x in triplets)}/{len(triplets)}。',
              '所有结果都披露；本轮不按全窗口最高收益选部署参数。多个组合来自同一段已观察历史，不是独立样本。', '',
              '## 数据与识别限制', '',
              f'行情有报价股票中，历史二级归属可分配比例日均 {(coverage.assigned/coverage.quoted_stocks).mean():.1%}；冲突股票日 {int(coverage.conflicting.sum())}。原 93 股平均可分配 {coverage.engineering_assigned.mean():.1f} 只。',
              '归属未知和冲突不回填；资料中的进出日期不代表历史覆盖已完整认证。分类版次使用 SW2014/SW2021 分段，供应商的回溯修订仍可能影响历史口径。',
              '自算指数采用昨日成员和昨日总市值权重，单股权重上限 20%，有效成员至少 5 只，已知成员报价覆盖至少 80%。不足时断开序列，重新热身。覆盖率分母是已知成员，不代表真实全板块完整度。',
              '行业合计市值会受成员变动影响，本轮仅记录；历史价值入场用价格指数平台参考带，尚未完成市值一致成员比较。',
              '执行仅覆盖已有 93 股工程样本，具有全窗口质量筛选偏差，并不代表全市场历史龙头可买集合。',
              '平台长度、上涨回撤阈值、支撑与止盈参数均为研究假设，不能声称完整复现主观判断。形态代码中的放量采用换手率，历史龙头采用此前上涨段收益排名；没有强制等待当前龙头启动。', '',
              '## 验证', '',
              '原始截断输入重新计算成员、板块、状态、历史龙头和入场信号，逐值通过前缀检查；默认账户通过截断回放；每个账户通过独立现金、股数、费用、涨跌停、T+1 与估值核对。',
              '5 项针对性测试覆盖成员进出/冲突、权重上限、量价顺序及允许继续创新低、历史锚点因果性、止盈线不下降及退出请求不可撤销。', '',
              '过程见 sector_panel.parquet、sector_states.parquet、sector_funnel.json、accounts/default/candidate_diagnostics.parquet，以及每个账户的四张账本与持仓决策。']
    path = ROOT / 'docs/LEFT_SIDE_RULES_RESULTS_20260921.md'
    path.write_text('\n'.join(lines) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supplement', type=Path, required=True)
    args = parser.parse_args()
    config = ROOT / 'configs/left_side_rules_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    supplement = args.supplement.resolve()
    s = json.loads((supplement / 'manifest.json').read_text())
    assert s['status'] == 'complete'
    assert all(digest(supplement / p) == h for p, h in s['hashes'].items())
    run = ROOT / 'outputs/left_side_rules' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    print('RUN=' + str(run), flush=True)
    _, parent, account = B.setup(PROJECT / cfg['loader_config'], run)
    assert cfg['max_drawdown'] == account['risk']['max_drawdown']
    G.update(left_cfg=cfg)
    files = [Path(__file__), config, ROOT / 'src/left_side_rules.py', ROOT / 'tests/test_left_side_rules.py', ROOT / 'docs/LEFT_SIDE_RULES_DESIGN_20260921.md',
             ROOT / 'scripts/run_policy_balance.py', ROOT / 'scripts/run_policy_convergence.py', ROOT / 'scripts/run_initialized_policy.py', ROOT / 'src/policy_portfolio.py']
    sources = {**parent['source_hashes'], **{str(p.relative_to(PROJECT)): digest(p) for p in files}}
    manifest = dict(status='running', dataset=account['dataset'], source_hashes=sources,
                    execution_input_hashes=parent['input_hashes'], supplement=str(supplement.relative_to(PROJECT)),
                    supplement_manifest_sha256=digest(supplement / 'manifest.json'), unbiased_oos=False)
    save_json(run / 'manifest.json', manifest)
    (run / 'sources').mkdir()
    for p in files:
        shutil.copy2(p, run / 'sources' / p.name)
    try:
        G['sector_features'] = build_features(DataHub(), supplement, cfg, run)
        defaults = default_params(cfg)
        tasks = [('default', defaults, 'full'), ('no_leader', defaults, 'no_leader'), ('stock_only', defaults, 'stock_only')]
        keys = list(cfg['search'])
        tasks += [(f'grid_{i:03d}', dict(zip(keys, values)), 'full')
                  for i, values in enumerate(itertools.product(*[cfg['search'][k] for k in keys]))]
        save_json(run / 'tasks.json', tasks)
        print('BACKTEST_ACCOUNTS', len(tasks), flush=True)
        rows = []
        with mp.get_context('fork').Pool(cfg['workers']) as pool:
            for row in pool.imap_unordered(run_account, tasks):
                rows.append(row)
                print('ACCOUNT', row['id'], 'return', round(row['total'], 6), 'buys', row['buys'], flush=True)
        rows.sort(key=lambda r: r['id'])
        path = report(run, cfg, rows)
        assert all(digest(PROJECT / p) == h for p, h in sources.items())
        assert all(digest(supplement / p) == h for p, h in s['hashes'].items())
        manifest.update(status='complete', accounts=len(rows), report=str(path.relative_to(PROJECT)), report_sha256=digest(path))
        save_json(run / 'artifact_hashes.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file() and p.name != 'manifest.json'})
        save_json(run / 'manifest.json', manifest)
        print('COMPLETE', path, flush=True)
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc())
        save_json(run / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
