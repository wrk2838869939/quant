"""独立复核新仓2×2：完整账本、真实倒序、原基线及历史相关/暂停证据。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from verify_recent_low_turnover import (digest, require, close, verify_case,
    verify_copy_provenance, prepare_execution_evidence, verify_market_fills, performance_report)


def csv(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def codes(value):
    return [] if pd.isna(value) or value == '' else str(value).split('|')


def identical(left, right):
    a, b = csv(left), csv(right)
    # 账户名称不是经济字段；除此以外保留全部列、行序和数值。
    for frame in [a, b]:
        if 'name' in frame:
            frame.drop(columns='name', inplace=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=False)
    return dict(file=left.name, rows=len(a), left_sha256=digest(left), right_sha256=digest(right))


def verify_reverse_evidence(folder):
    path = folder / 'reverse_input_evidence.json'
    require(path.exists(), f'缺少真实倒序调用证据: {folder.name}')
    evidence = json.loads(path.read_text())
    records = evidence['records'] if isinstance(evidence, dict) else evidence
    require(records and any(len(r['before']) > 1 for r in records), '倒序未覆盖多候选实际调用')
    for record in records:
        require(record['actual'] == record['before'][::-1], '真实函数未接收反向候选')
        require(record['function_module'] == 'guarded_selection', '反转未作用于实际selector模块')
    return dict(case=folder.name, calls=len(records), sha256=digest(path))



def verify_first_exit_requests(folder, hold):
    """独立跟踪退出确认生命周期，pending/ST不能伪装首次趋势失败。"""
    events, trades, holdings = [csv(folder / name) for name in ['events.csv', 'trades.csv', 'holdings.csv']]
    confirmed = set()
    checked = 0
    dates = sorted(set(events.date) | set(trades.date) | set(holdings.date))
    for date in dates:
        for event in events[events.date.eq(date)].itertuples():
            if event.kind != 'hold_exit_requested':
                continue
            if event.reason == 'hold_false':
                require(event.ts_code not in confirmed, 'pending被重复记作首次趋势退出')
                signal = pd.Timestamp(event.signal_date)
                require(signal in hold.index and event.ts_code in hold, '趋势退出缺原始hold单元')
                value = hold.loc[signal, event.ts_code]
                require(pd.notna(value) and float(value) == 0., '趋势退出不是原hold明确false')
                checked += 1
            confirmed.add(event.ts_code)
        for trade in trades[trades.date.eq(date)].itertuples():
            if trade.side == 'buy':
                confirmed.discard(trade.ts_code)
        # 清仓或退市后结束确认生命周期；下一次重新建仓可再次形成首次退出。
        confirmed.intersection_update(set(holdings.loc[holdings.date.eq(date), 'ts_code']))
    return checked


def verify_guard(folder, case, parameters, returns):
    pairs, reviews, candidates = [csv(folder / (name + '.csv'))
                                  for name in ['guard_pairs', 'guard_reviews', 'guard_candidates']]
    events = csv(folder / 'events.csv')
    events['signal_date'] = pd.to_datetime(events.signal_date)
    selection = events[events.kind.eq('selection_decision')]
    allocation = events[events.kind.eq('allocation_decision')]
    exits = events[events.kind.eq('hold_exit_requested') & events.reason.eq('hold_false')]
    require(not reviews.date.duplicated().any(), '重复review审计')
    require(set(pd.to_datetime(reviews.date)) == set(allocation.signal_date), 'review审计与实际分配日期不一致')
    pair_values = {}
    max_error = 0.
    for pair in pairs.itertuples():
        decision = pd.Timestamp(pair.date)
        require(decision in set(pd.to_datetime(reviews.date)), '相关记录不是实际review')
        require(pair.left in returns and pair.right in returns, '相关股票缺冻结收益')
        # 明确截断再取最近交易日，不以未来行补足缺失样本；相关系数使用numpy独立重建。
        block = returns.loc[returns.index <= decision, [pair.left, pair.right]].iloc[-parameters['correlation_window']:]
        mask = np.isfinite(block.to_numpy()).all(axis=1)
        sample = block.to_numpy()[mask]
        require(len(sample) == pair.observations, '相关有效样本数错误')
        last = block.index[mask][-1] if len(sample) else pd.NaT
        require((pd.isna(last) and pd.isna(pair.last_observation)) or last == pd.Timestamp(pair.last_observation),
                '相关最后观测日错误')
        require(pd.isna(last) or last <= decision, '相关使用未来数据')
        rho = np.nan
        if len(sample) >= parameters['minimum_observations'] and np.all(np.std(sample, axis=0) > 0):
            rho = float(np.corrcoef(sample.T)[0, 1])
        require((pd.isna(rho) and pd.isna(pair.correlation)) or
                (np.isfinite(rho) and np.isfinite(pair.correlation) and abs(rho-pair.correlation) < 1e-12),
                'numpy独立相关值不等于记录')
        if np.isfinite(rho):
            max_error = max(max_error, abs(rho-pair.correlation))
        key = (decision, *sorted([pair.left, pair.right]))
        require(key not in pair_values, '相关缓存出现重复键')
        pair_values[key] = rho
    paused_count, candidate_count = 0, 0
    for review in reviews.itertuples():
        decision = pd.Timestamp(review.date)
        alloc = allocation[allocation.signal_date.eq(decision)]
        old = alloc[alloc.old_shares.gt(0)]
        held = set(old.ts_code)
        fresh = exits[exits.signal_date.eq(decision)]
        require(not fresh.ts_code.duplicated().any(), '相同review重复退出请求')
        failed = set(fresh.ts_code) & held
        require(review.held_before == len(held) and review.newly_confirmed_exits == len(failed),
                '暂停计数偏离复核前真实持仓/首次hold_false退出')
        require(set(codes(review.exit_codes)) == failed, '暂停包含非首次趋势退出')
        trigger = len(failed) >= parameters['minimum_collective_exits'] and len(failed) >= len(held)*parameters['collective_exit_fraction']
        paused = trigger and case['variant'] in ['pause', 'combined']
        require(bool(review.collective_trigger) == trigger and bool(review.paused) == paused, '暂停触发/政策错误')
        paused_count += int(paused)
        retained = sorted(old.loc[old.target_shares.gt(0), 'ts_code'])
        selected = []
        rows = selection[selection.signal_date.eq(decision)]
        observed = candidates[pd.to_datetime(candidates.date).eq(decision)]
        require(len(rows) == len(observed), '候选审计与真实选择事件数不等')
        for choice, logged in zip(rows.itertuples(), observed.itertuples()):
            candidate_count += 1
            require(choice.ts_code == logged.ts_code, '候选审计顺序偏离实际选择')
            references = retained + selected
            require(codes(logged.references) == references, '相关参考不是留任旧仓+成功预约新仓')
            expected_reason, maximum = 'allowed', np.nan
            if paused:
                expected_reason = 'collective_exit_pause'
            elif case['variant'] in ['correlation', 'combined'] and references:
                values = []
                for other in references:
                    key = (decision, *sorted([choice.ts_code, other]))
                    require(key in pair_values, '候选判断缺独立相关证据')
                    values.append(pair_values[key])
                finite = np.isfinite(values)
                maximum = max(np.array(values)[finite]) if finite.any() else np.nan
                if not finite.all():
                    expected_reason = 'unknown_correlation'
                elif maximum > parameters['maximum_correlation']:
                    expected_reason = 'correlation_limit'
            require(logged.guard_reason == expected_reason, '候选门原因不符独立重建')
            require((pd.isna(maximum) and pd.isna(logged.maximum_correlation)) or
                    (np.isfinite(maximum) and abs(maximum-logged.maximum_correlation) < 1e-12),
                    '候选最大相关值错误')
            if choice.reason == 'selected':
                require(expected_reason == 'allowed', '被guard拒绝的股票仍被选中')
                require(choice.reserved_shares >= 100 and choice.reserved_shares % 100 == 0
                        and choice.reserved_net_profit > 0, '非法经济预约')
                selected.append(choice.ts_code)
            elif choice.reason in ['collective_exit_pause', 'unknown_correlation', 'correlation_limit']:
                require(choice.reason == expected_reason, '实际拒绝与guard不一致')
    return dict(reviews=len(reviews), paused_reviews=paused_count, pair_checks=len(pairs),
                candidate_checks=candidate_count, rho_max_error=max_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(); out = run / 'execution'
    manifest = json.loads((run / 'manifest.json').read_text())
    cases = manifest['cases']; parameters = manifest['parameters']
    require(len(cases) == 24 and len({c['name'] for c in cases}) == 24, '预登记非24个唯一账户')
    require(parameters == dict(correlation_window=120, minimum_observations=100, maximum_correlation=.7,
                              minimum_collective_exits=2, collective_exit_fraction=.5), '登记参数变化')
    require(set(p.parent.name for p in out.glob('*/summary.json')) == {c['name'] for c in cases}, '执行账户不齐或额外账户')
    source_provenance = verify_copy_provenance()
    for name, expected in manifest['source_hashes'].items():
        require(digest(ROOT / name) == expected, '登记源码漂移: '+name)
        require(digest(run / 'sources' / name) == expected, '冻结源码快照漂移: '+name)
    for name, expected in manifest['input_hashes'].items():
        require(digest(run / name) == expected, '登记输入漂移: '+name)
    data_manifest = json.loads((out / 'data_manifest.json').read_text())
    for name, expected in data_manifest['hashes'].items():
        require(digest(out / name) == expected, '执行数据漂移: '+name)
    require(digest(run / 'execution_predictions.parquet') == data_manifest['prediction_hash'], '执行预测漂移')
    parent_run = Path(manifest['parent'])
    for name in ['execution_predictions.parquet', 'execution/monthly_hold_allowed.parquet',
                 'execution/low_turnover_signals.parquet', 'execution/low_turnover_point_forecasts.parquet']:
        require(digest(run/name) == digest(parent_run/name), '冻结策略输入偏离修复parent: '+name)
    returns = pd.read_parquet(run / 'correlation_returns.parquet')
    require(returns.index.is_monotonic_increasing and not returns.index.has_duplicates, '相关收益日期未排序或重复')
    require(not returns.columns.has_duplicates, '相关收益代码重复')
    predictions = pd.read_parquet(run / 'execution_predictions.parquet')
    predictions['date'] = pd.to_datetime(predictions.date)
    require(not predictions.duplicated(['date', 'ts_code']).any(), '预测重复')
    hold = pd.read_parquet(out / 'monthly_hold_allowed.parquet')
    required_reviews = pd.DatetimeIndex(hold.index)
    required_reviews = required_reviews[(required_reviews >= min(pd.Timestamp(c['start']) for c in cases)) &
                                         (required_reviews <= pd.Timestamp(manifest['end']))]
    require(set(required_reviews).issubset(set(predictions.date)), '月度预测缺整月覆盖')
    market, limits = prepare_execution_evidence(pd.read_parquet(out / 'market.parquet'), pd.read_parquet(out / 'limits.parquet'))
    cfg = yaml.safe_load((out / 'execution_config.yaml').read_text())
    reports, fills = [], []
    for case in cases:
        folder = out / case['name']
        saved = json.loads((folder / 'portfolio_config.json').read_text())
        spec = saved.get('portfolio', saved)
        expected = dict(cfg['portfolio']); expected.update(capital=case['capital'], top_n=5)
        for key in ['buy_cost', 'sell_cost', 'min_fee']:
            expected[key] *= case['cost_multiplier']
        require(spec == expected, '账户配置偏离预登记: '+case['name'])
        summary = json.loads((folder / 'summary.json').read_text())
        close(summary['capital'], case['capital'], '账户本金不符')
        require(summary['policy'] == case['policy'], '账户policy不符')
        window = (pd.Timestamp(case['start']), pd.Timestamp(manifest['end']))
        account = verify_case(folder, predictions, {'verification_window': window}, {'portfolio': spec})
        market_checks = verify_market_fills(folder, spec, market, limits, window)
        fills.extend(dict(case=case['name'], **item) for item in market_checks.pop('fills'))
        guard = verify_guard(folder, case, parameters, returns)
        guard['first_exit_lifecycle_checks'] = verify_first_exit_requests(folder, hold)
        reports.append(dict(**account, **market_checks, guard=guard,
                            performance=performance_report(folder, case['capital'], pd.Timestamp(manifest['start']))))
        print('verified '+case['name'], flush=True)
    reverse_checks, reverse_evidence, baseline_checks, deferred = [], [], [], []
    for variant in ['baseline', 'correlation', 'pause', 'combined']:
        a, b = out / ('cold_'+variant), out / ('reverse_'+variant)
        reverse_evidence.append(verify_reverse_evidence(b))
        names_a, names_b = {p.name for p in a.glob('*.csv')}, {p.name for p in b.glob('*.csv')}
        require(names_a == names_b, '倒序CSV集合不一致: '+variant)
        for name in sorted(names_a):
            reverse_checks.append(dict(variant=variant, **identical(a/name, b/name)))
    parent = Path(manifest['parent']) / 'execution'
    mapping = dict(cold='cold_50000', continued='continued_50000', small='cold_20000',
                   large='cold_100000', cost2='cost2_50000', reverse='reverse_50000')
    for kind, reference in mapping.items():
        folder = out / (kind+'_baseline')
        for path in sorted((parent / reference).glob('*.csv')):
            if not (folder / path.name).exists() and path.name in ['recent_daily.csv', 'recent_daily_asset_pnl.csv']:
                deferred.append(dict(case=folder.name, file=path.name)); continue
            require((folder / path.name).exists(), '基线缺少原CSV: '+str(path))
            baseline_checks.append(dict(case=folder.name, **identical(path, folder/path.name)))
    report = dict(status='complete' if not deferred else 'derivative_outputs_pending', cases=reports,
                  reverse_comparisons=reverse_checks, reverse_input_evidence=reverse_evidence,
                  baseline_comparisons=baseline_checks, deferred=deferred,
                  source_provenance=source_provenance, manifest_sha256=digest(run/'manifest.json'),
                  input_hashes=manifest['input_hashes'], verifier_sha256=digest(Path(__file__)),
                  inherited_verifier_sha256=digest(ROOT/'scripts/verify_recent_low_turnover.py'),
                  scope='只核验冻结账户与研究门机制，不认证策略有效；未修改策略/下载/训练')
    (out/'independent_entry_guard_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    pd.DataFrame(fills).to_csv(out/'independent_entry_guard_fill_checks.csv', index=False)
    text = ('# 新仓2×2独立核验\n\n状态：'+report['status']+'。\n\n'
            f'逐账户账本与真实成交检查：{len(reports)}组、{len(fills)}笔；倒序全部CSV：{len(reverse_checks)}项；原基线CSV：{len(baseline_checks)}项。\n\n'
            '独立numpy重建最近120交易日/至少100有效观测相关值、全部候选参考与预约顺序；暂停计数只接受实际events中的首次hold_false，分母由分配前旧股数重建。\n\n'
            '完整结果见 execution/independent_entry_guard_verification.json；成交原价/官方限价证据见 execution/independent_entry_guard_fill_checks.csv。\n\n'
            '账户守恒与机制一致不代表参数有效或未来收益；历史窗口已见，政策为适应性研究。\n')
    (run/'INDEPENDENT_VERIFICATION.md').write_text(text)
    print(json.dumps(dict(status=report['status'], cases=len(reports), fills=len(fills),
                          reverse_csv=len(reverse_checks), baseline_csv=len(baseline_checks)), ensure_ascii=False))


if __name__ == '__main__':
    main()
