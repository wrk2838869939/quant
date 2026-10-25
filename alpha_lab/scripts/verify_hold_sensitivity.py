"""独立核验退出敏感性：原价重建证据、逐前缀因果、36账户及真实倒序。"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from verify_recent_low_turnover import (digest, require, close, verify_case,
    verify_copy_provenance, prepare_execution_evidence, verify_market_fills, performance_report)
from verify_entry_guard_research import csv, identical, verify_first_exit_requests
VARIANTS = ['baseline', 'fast', 'slow', 'persist5', 'persist10', 'persist15']


def check_hashes(run, manifest):
    for name, expected in manifest['source_hashes'].items():
        require(digest(ROOT/name) == expected, '研究源码漂移: '+name)
        require(digest(run/'sources'/name) == expected, '冻结源码快照漂移: '+name)
    for name, expected in manifest['input_hashes'].items():
        require(digest(run/name) == expected, '冻结输入漂移: '+name)
    require(digest(run/'PROTOCOL.md') == manifest['protocol_sha256'], '协议漂移')
    require(digest(Path(manifest['projection_source'])) == manifest['projection_sha256'], '原价投影源漂移')
    parent = Path(manifest['parent'])
    for name in ['execution_predictions.parquet', 'execution/monthly_hold_allowed.parquet',
                 'execution/low_turnover_signals.parquet', 'execution/low_turnover_point_forecasts.parquet']:
        require(digest(run/name) == digest(parent/name), '候选/预测/原hold不再等于parent: '+name)
    require(manifest['refit'] is False and manifest['fresh_holdout'] is False, '未冻结不重拟合/已见历史口径')
    spec = yaml.safe_load((run/'sources/configs/hold_sensitivity_research.yaml').read_text())
    require(spec == manifest['parameters'], 'manifest参数与冻结YAML不同')
    require(spec['persistence_window'] == 20 and spec['primary_variant'] == 'persist10'
            and list(spec['variants']) == VARIANTS, '预登记六政策/主假设不同')
    require(spec['trend'] == dict(medium_window=60, long_window=120, min_return=0., require_above_long_ma=True), '趋势周期/门槛改变')
    rules = {'baseline': {'monthly_confirmations': 2}, 'fast': {'monthly_confirmations': 1},
             'slow': {'monthly_confirmations': 3}, **{f'persist{k}': {'monthly_confirmations': 2,
             'persistent_weak_days': k} for k in [5, 10, 15]}}
    require(spec['variants'] == rules, '政策规则改变')
    return spec


def verify_matrices(run, manifest, spec):
    prices = pd.read_parquet(run/'daily_close.parquet')
    signals = pd.read_parquet(run/'execution/low_turnover_signals.parquet')
    parent_hold = pd.read_parquet(run/'execution/monthly_hold_allowed.parquet')
    reviews = signals.index
    universe = sorted(set().union(*(set(row.dropna().sort_values(ascending=False, kind='stable').head(20).index)
                                   for _, row in signals.iterrows())))
    require(universe == manifest['execution_universe'] == list(prices.columns), '持有代码不是原Top20并集')
    require(prices.index.is_monotonic_increasing and not prices.index.has_duplicates, '原价日历非法')
    require(prices.index.max() == pd.Timestamp(manifest['end']), '原价端点不一致')
    require(reviews.equals(parent_hold.index) and reviews.isin(prices.index).all(), '改变原月频review日历')
    raw = pd.read_parquet(manifest['projection_source'], columns=['date', 'ts_code', 'hfq_close'], filters=[('ts_code', 'in', universe)])
    require(not raw.duplicated(['date', 'ts_code']).any(), '原投影重复键')
    pd.testing.assert_frame_equal(prices, raw.pivot(index='date', columns='ts_code', values='hfq_close').sort_index(), check_exact=True)
    value = prices.to_numpy(float)
    require(not np.isinf(value).any(), '价格含无限值')
    clean = np.where(value > 0, value, np.nan)
    n, width = value.shape
    prior60, prior120 = np.full_like(clean, np.nan), np.full_like(clean, np.nan)
    prior60[60:] = clean[:-60]; prior120[120:] = clean[:-120]
    average = np.full_like(clean, np.nan)
    average[119:] = np.lib.stride_tricks.sliding_window_view(clean, 120, axis=0).mean(axis=-1)
    known = np.isfinite(clean) & np.isfinite(prior60) & np.isfinite(prior120) & np.isfinite(average)
    with np.errstate(invalid='ignore', divide='ignore'):
        trend = (clean/prior60 - 1 > 0) & (clean/prior120 - 1 > 0) & (clean > average)
    numpy_trend = trend.copy()
    recorded_trend = pd.read_parquet(run/'daily_trend.parquet').to_numpy(bool)
    boundaries = []
    differences = np.argwhere(trend != recorded_trend)
    if len(differences):
        # 严格比较在机器精度边界可能不同。显式披露并复现冻结pandas语义，不改原策略。
        native_mean = pd.DataFrame(clean).rolling(120, min_periods=120).mean().to_numpy()
        native_trend = (clean/prior60-1 > 0) & (clean/prior120-1 > 0) & (clean > native_mean)
        for i, j in differences:
            scale = max(1., abs(clean[i, j]))
            require(np.isfinite([clean[i,j], average[i,j], native_mean[i,j]]).all()
                    and abs(clean[i,j]-average[i,j]) <= np.finfo(float).eps*scale*8
                    and abs(clean[i,j]-native_mean[i,j]) <= np.finfo(float).eps*scale*8,
                    '非机器精度边界的趋势重建差异')
            require(native_trend[i,j] == recorded_trend[i,j], '冻结pandas语义也不能复现趋势')
            boundaries.append(dict(date=str(prices.index[i].date()), ts_code=prices.columns[j],
                close=float(clean[i,j]), numpy_mean=float(average[i,j]), pandas_mean=float(native_mean[i,j]),
                numpy_trend=bool(trend[i,j]), frozen_trend=bool(recorded_trend[i,j])))
            trend[i,j] = native_trend[i,j]
    pd.DataFrame(boundaries).to_csv(run/'matrix_numerical_boundaries.csv', index=False)
    trend_frame = pd.DataFrame(trend, index=prices.index, columns=prices.columns)
    known_frame = pd.DataFrame(known, index=prices.index, columns=prices.columns)
    pd.testing.assert_frame_equal(trend_frame, pd.read_parquet(run/'daily_trend.parquet'))
    pd.testing.assert_frame_equal(known_frame, pd.read_parquet(run/'daily_observable.parquet'))
    indices = prices.index.get_indexer(reviews); weak = known & ~trend
    count = np.full((len(reviews), width), np.nan); coverage = count.copy()
    streaks = np.zeros((len(reviews), width), dtype=int)
    for j, i in enumerate(indices):
        if i >= 19:
            count[j] = weak[i-19:i+1].sum(axis=0); coverage[j] = known[i-19:i+1].sum(axis=0)
        previous = streaks[j-1] if j else np.zeros(width, dtype=int)
        streaks[j] = np.where(weak[i], previous+1, 0)
    def frame(array):
        return pd.DataFrame(array, index=reviews, columns=prices.columns)
    evidence = dict(monthly_observable=known[indices], monthly_weak=weak[indices],
                    monthly_failure_streak=streaks, daily_weak_count=count, daily_known_count=coverage)
    for name, array in evidence.items():
        pd.testing.assert_frame_equal(frame(array), pd.read_parquet(run/(name+'.parquet')), check_exact=True)
    expected = {}
    for name, rule in spec['variants'].items():
        exit_now = streaks >= rule['monthly_confirmations']
        if 'persistent_weak_days' in rule:
            exit_now |= weak[indices] & (coverage == 20) & (count >= rule['persistent_weak_days'])
        expected[name] = frame(np.where(known[indices], np.logical_not(exit_now).astype(float), np.nan))
        pd.testing.assert_frame_equal(expected[name], pd.read_parquet(run/'hold_variants'/(name+'.parquet')), check_exact=True)
    pd.testing.assert_frame_equal(expected['baseline'], parent_hold.loc[:, universe], check_exact=True)
    # 单独反转全部数值边界，重建六政策；不把该审计写回任何策略输入。
    alt_weak = known & ~numpy_trend
    alt_count = np.full_like(count, np.nan)
    alt_streaks = np.zeros_like(streaks)
    for j, i in enumerate(indices):
        if i >= 19:
            alt_count[j] = alt_weak[i-19:i+1].sum(axis=0)
        previous = alt_streaks[j-1] if j else np.zeros(width, dtype=int)
        alt_streaks[j] = np.where(alt_weak[i], previous+1, 0)
    changed_counts = []
    for i,j in np.argwhere(np.isfinite(count) & (count != alt_count)):
        changed_counts.append(dict(date=str(reviews[i].date()), ts_code=prices.columns[j],
                                   frozen_weak_count=float(count[i,j]), flipped_weak_count=float(alt_count[i,j])))
    pd.DataFrame(changed_counts).to_csv(run/'matrix_numerical_weak_count_changes.csv', index=False)
    changed_holds = {}
    for name, rule in spec['variants'].items():
        alt_exit = alt_streaks >= rule['monthly_confirmations']
        if 'persistent_weak_days' in rule:
            alt_exit |= alt_weak[indices] & (coverage == 20) & (alt_count >= rule['persistent_weak_days'])
        alt_states = np.where(known[indices], (~alt_exit).astype(float), np.nan)
        original_states = expected[name].to_numpy()
        changed_holds[name] = int(((alt_states != original_states) &
                                   ~(np.isnan(alt_states) & np.isnan(original_states))).sum())
    numerical = dict(boundaries=boundaries, changed_monthly_counts=changed_counts,
                     changed_hold_cells=changed_holds,
                     account_decision_inputs_unchanged=not any(changed_holds.values()),
                     account_rerun_performed=False,
                     conclusion='六hold若逐格相同，则只翻转持有证据的所有账户决策输入完全不变；原入场信号未改')
    (run/'matrix_numerical_sensitivity.json').write_text(json.dumps(numerical, ensure_ascii=False, indent=2)+'\n')
    require(not any(changed_holds.values()), '机器精度边界影响了hold许可，须独立评估而非自动通过')
    from hold_sensitivity import build_hold_variants
    prefix_checks = 0
    for position, cutoff in enumerate(reviews):
        partial, _ = build_hold_variants(trend_frame.loc[:cutoff], known_frame.loc[:cutoff], reviews[:position+1], spec)
        for name in VARIANTS:
            pd.testing.assert_frame_equal(partial[name], expected[name].loc[:cutoff], check_exact=True)
            prefix_checks += 1
    from run_hold_sensitivity import effective_hold
    for name in VARIANTS:
        actual = effective_hold(parent_hold, expected[name])
        reference = parent_hold.copy(); reference.loc[:, universe] = expected[name]
        pd.testing.assert_frame_equal(actual, reference, check_exact=True)
        if name == 'baseline':
            pd.testing.assert_frame_equal(actual, parent_hold, check_exact=True)
    report = dict(status='complete', daily_rows=n, codes=width, reviews=len(reviews),
                  daily_trend_observable_cells=n*width*2, variant_cells=len(reviews)*width*len(VARIANTS),
                  prefix_checks=prefix_checks, baseline_parent_exact=True, effective_full_schema_preserved=True,
                  entry_inputs_parent_hash_exact=True, numerical_boundary_cells=len(boundaries),
                  numerical_sensitivity_sha256=digest(run/'matrix_numerical_sensitivity.json'),
                  projection_sha256=manifest['projection_sha256'],
                  input_hashes=manifest['input_hashes'], verifier_sha256=digest(Path(__file__)),
                  scope='numpy重建逐日趋势/可观测性、20日窗口、月度计数；原月度候选/进入预测不变')
    (run/'independent_hold_matrix_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'input_hashes'}, ensure_ascii=False), flush=True)
    return report


def verify_reverse(folder):
    evidence = json.loads((folder/'reverse_input_evidence.json').read_text())
    require(evidence['input_reversal_performed'] is True, '未实际反转输入')
    records = evidence['records']
    require(records and any(len(r['before']) > 1 for r in records), '缺真实多候选倒序调用')
    for record in records:
        require(record['actual'] == record['before'][::-1] and record['function_module'] == 'selection_first', '实际选择器未接收一次反序输入')
    return dict(case=folder.name, calls=len(records), sha256=digest(folder/'reverse_input_evidence.json'))


def compare_outputs(run):
    out = run/'execution'; manifest = json.loads((run/'manifest.json').read_text())
    reverse, baseline, evidence, deferred = [], [], [], []
    for variant in VARIANTS:
        left, right = out/('cold_'+variant), out/('reverse_'+variant)
        evidence.append(verify_reverse(right))
        names = {p.name for p in left.glob('*.csv')}
        require(names == {p.name for p in right.glob('*.csv')}, '真实倒序CSV文件集不同')
        for name in sorted(names):
            reverse.append(dict(variant=variant, **identical(left/name, right/name)))
    parent = Path(manifest['parent'])/'execution'
    mapping = dict(cold='cold_50000', continued='continued_50000', small='cold_20000', large='cold_100000', cost2='cost2_50000', reverse='reverse_50000')
    for kind, reference in mapping.items():
        folder = out/(kind+'_baseline')
        for file in sorted((parent/reference).glob('*.csv')):
            if not (folder/file.name).exists() and file.name in ['recent_daily.csv', 'recent_daily_asset_pnl.csv']:
                deferred.append(dict(case=folder.name, file=file.name)); continue
            require((folder/file.name).exists(), '基线缺原CSV: '+file.name)
            baseline.append(dict(case=folder.name, **identical(file, folder/file.name)))
    return reverse, baseline, evidence, deferred


def write_report(run, report):
    out = run/'execution'
    (out/'independent_hold_sensitivity_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    cases = report['cases']
    counts = dict(cases=len(cases), trades=sum(c['trades'] for c in cases), marks=sum(c['valid_close_marks'] for c in cases),
                  reverse_csv=len(report['reverse_comparisons']), baseline_csv=len(report['baseline_comparisons']))
    (run/'INDEPENDENT_VERIFICATION.md').write_text('# 退出确认敏感性独立核验\n\n状态：'+report['status']+'。\n\n'
        f"{counts['cases']}账户、{counts['trades']}笔真实成交、{counts['marks']}次原价收盘估值检查；{counts['reverse_csv']}项真实倒序CSV、{counts['baseline_csv']}项原基线CSV对照通过。\n\n"
        '持有矩阵由冻结原价用numpy重建，172原Top20并集代码之外保持原hold；所有原月度review前缀逐一核验，候选/预测/进入条件未改变。\n\n'
        '退出请求逐hold许可和订单生命周期检查；买入限定原月度、原Top20与原趋势恢复门，pending继续每日重试不等于每日选股。\n\n'
        '矩阵：independent_hold_matrix_verification.json；账户：execution/independent_hold_sensitivity_verification.json；成交：execution/independent_hold_sensitivity_fill_checks.csv。\n\n'
        '核验的是账本与因果实现，不认证政策有效；persist10为已登记主假设，不能择优用persist5/15替换。\n')
    print(json.dumps(dict(status=report['status'], **counts), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--matrix-only', action='store_true')
    args = parser.parse_args(); run = args.run.resolve(); out = run/'execution'
    manifest = json.loads((run/'manifest.json').read_text()); spec = check_hashes(run, manifest)
    matrix_path = run/'independent_hold_matrix_verification.json'
    if args.matrix_only or not matrix_path.exists():
        matrix = verify_matrices(run, manifest, spec)
    else:
        matrix = json.loads(matrix_path.read_text())
        require(matrix['status'] == 'complete' and matrix['input_hashes'] == manifest['input_hashes'], '矩阵核验输入变化')
        require(matrix['verifier_sha256'] == digest(Path(__file__)), '矩阵核验器变化')
    if args.matrix_only:
        return
    cases = manifest['cases']
    require(len(cases) == len({c['name'] for c in cases}) == 36, '预登记非36账户')
    require(set(p.parent.name for p in out.glob('*/summary.json')) == {c['name'] for c in cases}, '36账户未齐或额外账户')
    provenance = verify_copy_provenance(); data_manifest = json.loads((out/'data_manifest.json').read_text())
    for name, expected in data_manifest['hashes'].items():
        require(digest(out/name) == expected, '公司行动/行情执行证据变化: '+name)
    predictions = pd.read_parquet(run/'execution_predictions.parquet'); predictions['date'] = pd.to_datetime(predictions.date)
    require(digest(run/'execution_predictions.parquet') == data_manifest['prediction_hash'], '执行预测哈希变化')
    pred = predictions.set_index(['date', 'ts_code']); require(not pred.index.has_duplicates, '预测重复键')
    signals = pd.read_parquet(out/'low_turnover_signals.parquet')
    top20 = {date:set(row.dropna().sort_values(ascending=False, kind='stable').head(20).index) for date,row in signals.iterrows()}
    require(set(signals.index).issubset(set(predictions.date)), '整月预测缺失')
    market, limits = prepare_execution_evidence(pd.read_parquet(out/'market.parquet'), pd.read_parquet(out/'limits.parquet'))
    cfg = yaml.safe_load((out/'execution_config.yaml').read_text()); reports, fills = [], []
    for case in cases:
        folder = out/case['name']; hold = pd.read_parquet(run/'hold_variants'/(case['variant']+'.parquet'))
        saved = json.loads((folder/'portfolio_config.json').read_text()); portfolio = saved.get('portfolio', saved)
        expected = dict(cfg['portfolio']); expected.update(capital=case['capital'], top_n=5)
        for field in ['buy_cost', 'sell_cost', 'min_fee']:
            expected[field] *= case['cost_multiplier']
        require(portfolio == expected, '账户配置偏离预登记本金/费用')
        summary = json.loads((folder/'summary.json').read_text()); close(summary['capital'], case['capital'], '本金不符')
        require(summary['policy'] == case['policy'] == 'low_turnover', '账户策略改变')
        window = (pd.Timestamp(case['start']), pd.Timestamp(manifest['end']))
        account = verify_case(folder, predictions, {'verification_window':window}, {'portfolio':portfolio})
        marks = verify_market_fills(folder, portfolio, market, limits, window)
        fills.extend(dict(case=case['name'], **item) for item in marks.pop('fills'))
        exit_checks = verify_first_exit_requests(folder, hold)
        trades = csv(folder/'trades.csv'); buys = trades[trades.side.eq('buy')]
        for trade in buys.itertuples():
            decision = pd.Timestamp(trade.signal_date)
            require(decision in top20 and trade.ts_code in top20[decision], '新增/重入改变原monthly Top20')
            source = pred.loc[(decision, trade.ts_code)]
            require(bool(source.trend) and np.isfinite(signals.loc[decision, trade.ts_code]), '趋势未恢复/缺原分数仍买入')
        reports.append(dict(**account, **marks, first_exit_lifecycle_checks=exit_checks, original_monthly_entry_checks=len(buys),
                            performance=performance_report(folder, case['capital'], pd.Timestamp(manifest['start']))))
        print('verified '+case['name'], flush=True)
    reverse, baseline, evidence, deferred = compare_outputs(run)
    report = dict(status='complete' if not deferred else 'derivative_outputs_pending', cases=reports,
                  matrix_report_sha256=digest(matrix_path), reverse_comparisons=reverse, baseline_comparisons=baseline,
                  reverse_input_evidence=evidence, deferred=deferred, copied_account_state_machine=provenance,
                  manifest_sha256=digest(run/'manifest.json'), input_hashes=manifest['input_hashes'],
                  verifier_sha256=digest(Path(__file__)), inherited_verifier_sha256=digest(ROOT/'scripts/verify_recent_low_turnover.py'),
                  exit_lifecycle_verifier_sha256=digest(ROOT/'scripts/verify_entry_guard_research.py'))
    pd.DataFrame(fills).to_csv(out/'independent_hold_sensitivity_fill_checks.csv', index=False); write_report(run, report)


if __name__ == '__main__':
    main()
