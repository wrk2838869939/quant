# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""固定数据上的基线风格替换及固定参数跨年诊断。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import multiprocessing as mp
import shutil
import sys
import traceback
import numpy as np
import pandas as pd
import scipy
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_policy_balance as B
from baseline_style_controls import style_ranks, make_control, account_exposure
from baseline_combination import baseline_scores, blend_scores
from rule_search import decision_dates
from run_classic_strategy_smoke import digest, save_json
from run_rule_strategy_search import write_result
G = B.G


def annual(result):
    previous, rows = G['portfolio']['capital'], []
    for year, part in result.daily.groupby(result.daily.index.year):
        rows.append(dict(year=int(year), total=float(part.equity.iloc[-1] / previous - 1), net_pnl=float(part.equity.iloc[-1] - previous)))
        previous = part.equity.iloc[-1]
    return rows


def freeze_signals(name):
    cfg, registry = G['study'], G['registry']
    params = [entry['annual_parameters'][cfg['anchor_year']]['params'] for entry in registry['baselines'].values()]
    raw = baseline_scores(G['wide'], G['eligible'], G['calendar'], params)
    if name == 'equal_weight':
        p = G['choices'][str(cfg['anchor_year'])]['equal_weight']['params']
        score = blend_scores(raw, p['weights'])
    else:
        i = list(registry['baselines']).index(name)
        p, score = params[i], raw[i]
    # 保留每年首个交易日的启动约定，避免把它混入参数冻结效应。
    dates = G['eligible'].loc[cfg['start']:cfg['end']].index
    selected = []
    for year in sorted(set(dates.year)):
        selected.extend(decision_dates(dates[dates.year == year], p['frequency'], G['calendar']))
    signals = score.loc[pd.DatetimeIndex(selected)]
    return signals, pd.Series(p['top_n'], index=signals.index)


def control_task(task):
    name, mode, seed = task
    cfg = G['study']
    source, numbers = G['signals'][name]
    dims = cfg['modes'][mode]
    signals, pairs = make_control(source, numbers, G['ranks'], dims, seed, cfg['matching'])
    folder = G['run'] / 'controls' / name / mode / str(seed)
    folder.mkdir(parents=True)
    result = B.R.original.execute(signals, numbers, cfg['start'], cfg['end'])
    audit = B.R.original.audit(result)
    write_result(folder, result, signals)
    pairs.to_parquet(folder / 'pairs.parquet', index=False)
    exposure = account_exposure(result, G['ranks'])
    exposure.to_parquet(folder / 'exposure.parquet')
    base = G['base'][name]
    gaps = {k: float((exposure[k] - G['exposure'][name][k]).abs().mean()) for k in dims}
    coverage = min([float(exposure[k + '_coverage'].mean()) for k in dims] or [1.])
    metrics = B.R.metric(result)
    ratio = metrics['turnover'] / base['metrics']['turnover'] if base['metrics']['turnover'] else np.inf
    support = bool(dims and pairs.passed.mean() >= cfg['matching']['minimum_pair_pass_rate']
                   and all(v <= cfg['matching']['maximum_actual_mean_abs_gap'] for v in gaps.values())
                   and coverage >= cfg['matching']['minimum_pair_pass_rate'])
    credible = support and 1 / cfg['matching']['maximum_turnover_ratio'] <= ratio <= cfg['matching']['maximum_turnover_ratio']
    row = dict(baseline=name, mode=mode, seed=seed, total=metrics['total'], mdd=metrics['mdd'],
               fees=metrics['total_fees'], turnover=metrics['turnover'], turnover_ratio=ratio,
               net_gap=base['metrics']['total'] - metrics['total'],
               same_path_prefee_gap=base['metrics']['total'] - metrics['total'] + (base['metrics']['total_fees'] - metrics['total_fees']) / G['portfolio']['capital'],
               pair_pass=float(pairs.passed.mean()), actual_gaps=gaps, style_support=support,
               credible_control=bool(credible), coverage=coverage, invested=metrics['mean_invested'],
               signal_sha256=digest(folder / 'signals.parquet'), annual=annual(result))
    save_json(folder / 'metrics.json', row)
    save_json(folder / 'audit.json', audit)
    if seed == cfg['seeds'][0]:
        cutoff = pd.Timestamp(cfg['prefix_date'])
        short, short_pairs = make_control(source.loc[:cutoff], numbers.loc[:cutoff], G['prefix_ranks'], dims, seed, cfg['matching'])
        pd.testing.assert_frame_equal(short, signals.loc[:cutoff], check_exact=True)
        pd.testing.assert_frame_equal(short_pairs, pairs[pairs.date <= cutoff], check_exact=True)
        replay = B.R.original.execute(short, numbers.loc[:cutoff], cfg['start'], cfg['prefix_date'])
        B.R.original.audit(replay)
        B.R.original.compare_prefix(replay, result, cutoff)
        save_json(folder / 'prefix_check.json', dict(raw_features_signals_accounts=True))
    return row


def report(rows):
    cfg, run = G['study'], G['run']
    frame = pd.DataFrame(rows)
    frame.drop(columns=['annual', 'actual_gaps']).to_csv(run / 'controls_summary.csv', index=False)
    grouped, yearly = [], []
    for (name, mode), part in frame.groupby(['baseline', 'mode'], sort=False):
        grouped.append(dict(baseline=name, mode=mode, baseline_return=G['base'][name]['metrics']['total'],
                            control_median=part.total.median(), control_min=part.total.min(), control_max=part.total.max(),
                            net_gap_median=part.net_gap.median(), prefee_gap_median=part.same_path_prefee_gap.median(),
                            baseline_beats_fraction=(part.net_gap > 0).mean(), pair_pass_mean=part.pair_pass.mean(),
                            support_count=int(part.style_support.sum()), credible_count=int(part.credible_control.sum()),
                            unique_signals=part.signal_sha256.nunique(), turnover_ratio_median=part.turnover_ratio.median(),
                            max_actual_gap_mean=max([np.mean([r[k] for r in part.actual_gaps]) for k in cfg['modes'][mode]] or [np.nan])))
        for reference in G['base'][name]['annual']:
            returns = [next(a['total'] for a in row['annual'] if a['year'] == reference['year']) for row in part.to_dict('records')]
            yearly.append(dict(baseline=name, mode=mode, year=reference['year'], baseline_return=reference['total'],
                               control_median=float(np.median(returns)), gap_median=float(reference['total'] - np.median(returns))))
    pd.DataFrame(grouped).to_csv(run / 'comparison.csv', index=False)
    pd.DataFrame(yearly).to_csv(run / 'annual_comparison.csv', index=False)
    frozen = pd.DataFrame(G['frozen'])
    frozen.drop(columns='annual').to_csv(run / 'fixed_parameters.csv', index=False)
    lines = ['# 基线风格对照与跨年稳定性', '',
             f'运行：`{run.relative_to(PROJECT)}`。冻结 93 股，初始 10 万，2021–2023 连续账户，费用与成交引擎不变。4 条基线 × 3 类对照 × 12 个预定种子，共 {len(rows)} 条控制账户。未按收益挑选种子。', '',
             '## 风格匹配对照', '',
             '| 基线 | 对照 | 基线收益 | 对照中位收益 | 净收益差 | 风格合格/12 | 风格及换手均合格/12 | 配对合格率 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in grouped:
        lines.append(f'| {r["baseline"]} | {r["mode"]} | {r["baseline_return"]:+.2%} | {r["control_median"]:+.2%} | {r["net_gap_median"]*100:+.2f}pp | {r["support_count"]} | {r["credible_count"]} | {r["pair_pass_mean"]:.1%} |')
    lines += ['', '合格门槛事先固定：配对 RMS≤0.15、每维差≤0.25，整体配对合格率≥80%，实际每日持仓风格绝对差均值每维≤0.10，数据覆盖≥80%；再要求账户换手比在 [0.5,2]。未达门槛的差异不能解释为已排除风格和交易频率差异后的选股贡献。random_priority 不尝试匹配，合格计数不适用。', '',
              'match_size_vol_value 匹配市值、20 日波动与正盈利收益率；match_all 再控制个股换手率。对照与原 top-N 当日不重合。候选来自原非空评分股票且全部风格可用；这并不等于完全控制原资格边界、行业、现金比例或所有风险因子。', '',
              '## 费用与交易路径', '', '| 基线 | 对照 | 净收益差 | 原路径费用加回后的利润差 | 账户换手比中位数 | 不同信号轨迹数 |', '|---|---|---:|---:|---:|---:|']
    for r in grouped:
        lines.append(f'| {r["baseline"]} | {r["mode"]} | {r["net_gap_median"]*100:+.2f}pp | {r["prefee_gap_median"]*100:+.2f}pp | {r["turnover_ratio_median"]:.2f} | {r["unique_signals"]} |')
    lines += ['', '此处加回费用仅是固定交易路径的账面分解，不是零费用重新分配资金的回测。费用包含滑点：原引擎按开盘价记成交 gross，并把 gross×slippage_rate 加入 fee。前份归因报告“滑点嵌入成交价”的文字应更正；归因数值已按完整 fee 扣除，不需再扣一次滑点。', '',
              '## 各年全风格对照', '', '| 基线 | 年份 | 基线收益 | 对照中位收益 | 差值 |', '|---|---:|---:|---:|---:|']
    for r in yearly:
        if r['mode'] == 'match_all':
            lines.append(f'| {r["baseline"]} | {r["year"]} | {r["baseline_return"]:+.2%} | {r["control_median"]:+.2%} | {r["gap_median"]*100:+.2f}pp |')
    lines += ['', '## 冻结 2021 参数后的跨年结果', '', '| 基线 | 原年度路径累计收益 | 固定参数累计收益 | 固定参数回撤 | 2021 | 2022 | 2023 |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in G['frozen']:
        yr = row['annual']
        lines.append(f'| {row["baseline"]} | {G["base"][row["baseline"]]["metrics"]["total"]:+.2%} | {row["total"]:+.2%} | {abs(row["mdd"]):.2%} | {yr[0]["total"]:+.2%} | {yr[1]["total"]:+.2%} | {yr[2]["total"]:+.2%} |')
    lines += ['', '固定参数仍按每年首交易日启动的原约定生成信号，第一年四张账户表与原路径完全一致。此项检查有助于观察规则不再年度重选后的变化；由于历史已观察、初始参数本身经过筛选，不构成新样本外认证。', '',
              '## 验证及边界', '',
              '- 四条原基线逐表复现；所有控制和固定参数账户通过独立账本审计。',
              '- 三类对照的首种子共 12 条通过原始数据截断、风格、信号与账户前缀检查。',
              '- 六项针对性测试通过；缺少相近替代股时保留失败标记，未把失败日期删除。',
              '- 本轮不将相互依赖的 12 个种子当成独立市场样本，不提供显著性结论。',
              '- 93 股与基线选择的既有偏差、行业与其他未控暴露仍在。风格匹配不合格时，应报告识别条件不足，不能宣称存在或不存在独立 alpha。', '',
              '下一步取决于匹配质量：先补足同风格比较的可识别性；对固定参数能够保持正收益的规则，再预先锁定跨股票分组和未来观察期，继续验证收益是否可迁移。']
    path = ROOT / 'docs/BASELINE_STYLE_CONTROLS_RESULTS_20260921.md'
    path.write_text('\n'.join(lines) + '\n')
    return path


def main():
    config = ROOT / 'configs/baseline_style_controls_v1.yaml'
    cfg = yaml.safe_load(config.read_text())
    run = ROOT / 'outputs/baseline_style_controls' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    print('RUN=' + str(run), flush=True)
    _, loader_parent, account = B.setup(PROJECT / cfg['loader_config'], run)
    assert cfg['max_drawdown'] == account['risk']['max_drawdown']
    source = PROJECT / cfg['source_run']
    parent = json.loads((source / 'manifest.json').read_text())
    assert parent['status'] == 'complete'
    hashes = json.loads((source / 'artifact_hashes.json').read_text())
    assert digest(source / 'all_locked_choices.json') == hashes['all_locked_choices.json']
    files = [Path(__file__), config, ROOT / 'src/baseline_style_controls.py', ROOT / 'tests/test_baseline_style_controls.py',
             ROOT / 'docs/BASELINE_STYLE_CONTROLS_PREREG_20260921.md', ROOT / 'scripts/run_policy_balance.py',
             ROOT / 'scripts/run_policy_convergence.py', PROJECT / cfg['loader_config'], ROOT / 'src/policy_balance.py',
             ROOT / 'src/policy_evolution.py', ROOT / 'src/policy_portfolio_cached.py']
    sources = {**loader_parent['source_hashes'], **{str(p.relative_to(PROJECT)): digest(p) for p in files}}
    manifest = dict(status='running', study=cfg['study'], dataset=account['dataset'], source_hashes=sources,
                    input_hashes=loader_parent['input_hashes'], parent_manifest_sha256=digest(source / 'manifest.json'),
                    scipy_version=scipy.__version__, unbiased_oos=False)
    save_json(run / 'manifest.json', manifest)
    (run / 'sources').mkdir()
    for path in sources:
        shutil.copy2(PROJECT / path, run / 'sources' / Path(path).name)
    try:
        G.update(study=cfg, signals={}, base={}, exposure={}, frozen=[],
                 registry=yaml.safe_load((ROOT / 'configs/selection_baselines_v1.yaml').read_text()),
                 choices=json.loads((source / 'all_locked_choices.json').read_text()))
        G['ranks'] = style_ranks(G['wide'], G['eligible'], cfg['style_window'])
        cut = cfg['prefix_date']
        G['prefix_ranks'] = style_ranks({k: v.loc[:cut] for k, v in G['wide'].items()}, G['eligible'].loc[:cut], cfg['style_window'])
        for k in G['ranks']:
            pd.testing.assert_frame_equal(G['ranks'][k].loc[:cut], G['prefix_ranks'][k], check_exact=True)
        for name in cfg['baselines']:
            old = source / 'outer' / name
            for filename in [*[f'{t}.parquet' for t in B.R.original.TABLES], 'signals.parquet', 'top_n_schedule.parquet']:
                assert digest(old / filename) == hashes[str((old / filename).relative_to(source))]
            sig = pd.read_parquet(old / 'signals.parquet')
            n = pd.read_parquet(old / 'top_n_schedule.parquet').iloc[:, 0]
            G['signals'][name] = sig, n
            result = B.R.original.execute(sig, n, cfg['start'], cfg['end'])
            for table in B.R.original.TABLES:
                pd.testing.assert_frame_equal(getattr(result, table), pd.read_parquet(old / f'{table}.parquet'), check_exact=True)
            B.R.original.audit(result)
            write_result(run / 'baseline' / name, result, sig)
            G['base'][name] = dict(metrics=B.R.metric(result), annual=annual(result))
            G['exposure'][name] = account_exposure(result, G['ranks'])
            G['exposure'][name].to_parquet(run / 'baseline' / name / 'exposure.parquet')
            fixed, ns = freeze_signals(name)
            frozen = B.R.original.execute(fixed, ns, cfg['start'], cfg['end'])
            B.R.original.audit(frozen)
            first_end = f'{pd.Timestamp(cfg["start"]).year}-12-31'
            first = B.R.original.execute(fixed.loc[:first_end], ns.loc[:first_end], cfg['start'], first_end)
            B.R.original.compare_prefix(first, result, pd.Timestamp(first_end))
            write_result(run / 'fixed_parameters' / name, frozen, fixed)
            G['frozen'].append(dict(baseline=name, total=frozen.summary['total'], mdd=frozen.summary['mdd'], annual=annual(frozen)))
        tasks = [(name, mode, seed) for name in cfg['baselines'] for mode in cfg['modes'] for seed in cfg['seeds']]
        rows = []
        with mp.get_context('fork').Pool(cfg['workers']) as pool:
            for row in pool.imap_unordered(control_task, tasks):
                rows.append(row)
                save_json(run / 'progress.json', rows)
                if len(rows) % len(cfg['seeds']) == 0:
                    print(f'CONTROLS {len(rows)}/{len(tasks)}', flush=True)
        rows.sort(key=lambda r: (cfg['baselines'].index(r['baseline']), list(cfg['modes']).index(r['mode']), r['seed']))
        path = report(rows)
        assert all(digest(PROJECT / p) == h for p, h in sources.items())
        manifest.update(status='complete', control_accounts=len(rows), prefix_checks=len(cfg['baselines'])*len(cfg['modes']),
                        baseline_replays=len(cfg['baselines']), report=str(path.relative_to(PROJECT)), report_sha256=digest(path))
        save_json(run / 'manifest.json', manifest)
        save_json(run / 'artifact_hashes.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
        print('COMPLETE ' + str(path), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        save_json(run / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    main()
