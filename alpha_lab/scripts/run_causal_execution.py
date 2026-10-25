"""冻结因果订单协议、隔离回放baseline/Hybrid，并保留全部失败记录。"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def case_matrix(config):
    cases = []
    for variant in config['variants']:
        for scenario, options in config['scenarios'].items():
            cases.append(dict(name=f'{scenario}_{variant}', variant=variant,
                              scenario=scenario, buffer=config['causal']['buy_limit_buffer'], **options))
        for buffer in config['buffer_sensitivity']:
            for scenario in ['cold', 'continued']:
                cases.append(dict(name=f'{scenario}_{variant}_buffer{int(buffer*100):02d}',
                    variant=variant, scenario=scenario, buffer=buffer,
                    **config['scenarios'][scenario]))
    return cases


def prepare(run, config_path):
    if run.exists():
        raise ValueError('新运行目录必须不存在，禁止覆盖证据')
    run.mkdir(parents=True)
    config = yaml.safe_load(config_path.read_text())
    parent = ROOT / config['parent']
    shutil.copy2(config_path, run / 'config.yaml')
    inputs = run / 'inputs'
    inputs.mkdir()
    for name in ['low_turnover_signals', 'low_turnover_point_forecasts', 'monthly_hold_allowed',
                 'basics', 'st', 'limits', 'normalized_actions', 'raw_actions']:
        shutil.copy2(parent / 'execution' / f'{name}.parquet', inputs / f'{name}.parquet')
    for variant in config['variants']:
        shutil.copy2(parent / 'hold_variants' / f'{variant}.parquet', inputs / f'hold_{variant}.parquet')
    market = pd.read_parquet(parent / 'execution/market.parquet')
    panel = ROOT / 'data/panel/daily_panel.parquet'
    raw = pd.read_parquet(panel, columns=['date', 'ts_code', 'open', 'close', 'high', 'low'],
        filters=[('date', '>=', market.date.min()), ('date', '<=', market.date.max()),
                 ('ts_code', 'in', sorted(market.ts_code.unique()))])
    merged = market.merge(raw, on=['date', 'ts_code'], how='left', suffixes=('', '_check'), validate='one_to_one')
    for name in ['open', 'close']:
        assert np.array_equal(merged[name], merged[name + '_check'], equal_nan=True)
    assert merged[['high', 'low']].notna().all().all()
    assert (merged.high >= merged[['open', 'close', 'low']].max(axis=1)).all()
    assert (merged.low <= merged[['open', 'close']].min(axis=1)).all()
    merged.drop(columns=['open_check', 'close_check']).to_parquet(inputs / 'market.parquet', index=False)
    # 公告字段单独回接到新的行动输入，取全部组成的最晚实施公告，未知即失败。
    actions = pd.read_parquet(inputs / 'normalized_actions.parquet')
    raw_actions = pd.read_parquet(inputs / 'raw_actions.parquet')
    implemented = raw_actions.loc[raw_actions.div_proc.eq('实施')].copy()
    for name in ['imp_ann_date', 'ex_date']:
        implemented[name] = pd.to_datetime(implemented[name].astype(str), errors='coerce', format='mixed')
    assert implemented.imp_ann_date.notna().all()
    known = implemented.groupby(['ts_code', 'ex_date']).imp_ann_date.max().rename('known_date')
    actions = actions.merge(known, on=['ts_code', 'ex_date'], how='left', validate='one_to_one')
    assert actions.known_date.notna().all() and actions.known_date.le(actions.record_date).all()
    actions.to_parquet(inputs / 'normalized_actions.parquet', index=False)
    names_path = ROOT / 'data/raw/namechange_all.parquet'
    names = pd.read_parquet(names_path)
    st = names.loc[names.name.str.contains('ST', case=False, na=False)].copy()
    for old, new in [('start_date', 'start'), ('end_date', 'end'), ('ann_date', 'known_date')]:
        st[new] = pd.to_datetime(st[old].astype(str), errors='coerce', format='mixed')
    assert st.known_date.notna().all() and st.known_date.le(st.start).all()
    old_st = pd.read_parquet(inputs / 'st.parquet')
    pd.testing.assert_frame_equal(st[['ts_code', 'start', 'end']].reset_index(drop=True), old_st.reset_index(drop=True))
    st[['ts_code', 'start', 'end', 'known_date']].to_parquet(inputs / 'st.parquet', index=False)
    for name in ['symmetric_phase_research.yaml', 'corporate_action_components.yaml']:
        shutil.copy2(parent / 'sources/configs' / name, inputs / name)
    shutil.copy2(parent / 'execution/execution_config.yaml', inputs / 'execution_config.yaml')
    manifest = dict(status='protocol_frozen_inputs_prepared', created_at_utc=datetime.now(timezone.utc).isoformat(),
        parent=str(parent), parent_manifest_sha256=digest(parent / 'manifest.json'),
        parent_completion_sha256=digest(parent / 'completion.json'), fresh_holdout=False,
        end=config['end'], cases=case_matrix(config), inputs_sha256={p.name:digest(p) for p in inputs.iterdir()},
        source_data_sha256={str(panel):digest(panel), str(names_path):digest(names_path)},
        input_extension=dict(high_low_rows=len(merged), original_open_close_exact=True,
                             action_known_dates=len(actions), st_known_dates=len(st)),
        source_hashes={}, strategy_admitted=False, execution_certified=False)
    save(run / 'manifest.json', manifest)
    (run / 'PROTOCOL.md').write_text('''# 因果订单修复与固定比较协议

本协议在新回放收益产生前登记。原研究均已反复观察，不称新样本外。

## 固定目标

原baseline与Hybrid的分数、常数、月度持有许可保持不变，首先隔离执行修复。历史身份修复另行报告，不将两层收益差异混为一谈。

## 可提交的日限价订单

月度信号日收盘后，以截至当时的原始收盘、现金和股数冻结订单。新买主限价=前收×1.02，按报价单位取整，股数和费用按限价足额预留。买单不预支下一日卖款，不因看到开盘价重新分配股数或换股票。卖单前晚确定股数及主板10%/已知ST5%的向下限价。官方当日界仅验证订单是否合法。

次日限价单若开盘可成交，按开盘撮合；否则只在日内最低/最高严格穿过限价时按限价撮合，触及不算。高低价仅用于冻结订单后的成交判断。日单未成于收盘结束，已确认卖出退出在随后收盘重提，买单最多在月度首轮与紧随的补位轮两次规划，之后等待下次月度复核。只有卖出后的收盘现金可支持次日新买。未卖出的持仓继续占名额。跨除权订单取消并记录，不用未知开盘重新调整股数。

日线模型仍假定符合条件订单可足量成交，缺成交队列及分钟证据，不能称实盘认证。补充高低价来自原面板，校验原open/close逐键不变。任何现金、股数、公司行动或订单时点检查失败先停止对应结果使用。

## 比较矩阵与停止条件

两策略×六主情景（2024空仓5万、2022连续5万、2万、10万、双费用、接收端倒序）共12组；买限价缓冲0%/5%仅跑两主时期两策略，共8组。主2%不根据结果更换。逐年、近期、完整窗口以及逐股损益与原账本差异一并输出，不只呈现赢家。

先完成独立时序对抗测试，再封存源码和回放；任何源码改变须新建运行目录。封存原研究不覆盖。缺失行情、未认证数据版本、ST/换码覆盖范围和日线撮合假设必须列入报告。复验完成不自动准入。
''')
    print(str(run), flush=True)


def freeze(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['source_hashes']:
        raise ValueError('源码已冻结')
    files = ['src/causal_portfolio.py', 'src/portfolio.py', 'src/selection_first.py',
             'src/retail_rebalance.py', 'scripts/run_causal_execution.py',
             'scripts/summarize_causal_execution.py',
             'scripts/audit_monthly_portfolio.py', 'scripts/run_patient_value_execution.py']
    for name in files:
        destination = run / 'sources' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    manifest['source_hashes'] = {name:digest(ROOT / name) for name in files}
    manifest['protocol_sha256'] = digest(run / 'PROTOCOL.md')
    manifest['config_sha256'] = digest(run / 'config.yaml')
    manifest['status'] = 'sources_frozen'
    save(run / 'manifest.json', manifest)


def verify_quotes(run):
    """新加high/low逐日追到raw缓存；通过仅表示同源加工一致。"""
    market = pd.read_parquet(run / 'inputs/market.parquet')
    hashes, errors = {}, []
    rows = 0
    for date, group in market.groupby('date'):
        path = ROOT / 'data/raw/by_date/daily' / f'{date:%Y%m%d}.parquet'
        raw = pd.read_parquet(path, columns=['ts_code', 'high', 'low']).set_index('ts_code')
        actual = group.set_index('ts_code')[['high', 'low']]
        if not np.array_equal(actual.to_numpy(), raw.reindex(actual.index).to_numpy(), equal_nan=True):
            errors.append(str(date))
        hashes[str(path)] = digest(path)
        rows += len(actual)
    save(run / 'high_low_source_verification.json', dict(rows=rows, days=len(hashes),
        all_exact=not errors, errors=errors, source_sha256=hashes,
        scope='raw daily cache consistency only; not independent exchange authenticity'))
    assert not errors, errors


@contextmanager
def reverse_candidates(enabled, records):
    import selection_first
    original = selection_first.plan_selection_first
    if enabled:
        def reversed_call(candidates, **kwargs):
            actual = candidates.iloc[::-1].copy()
            records.append(dict(before=candidates.ts_code.tolist(), actual=actual.ts_code.tolist()))
            return original(actual, **kwargs)
        selection_first.plan_selection_first = reversed_call
    try:
        yield
    finally:
        selection_first.plan_selection_first = original


def run_case(run, name):
    import causal_portfolio
    from audit_monthly_portfolio import reconcile
    from portfolio import daily_metrics
    from run_patient_value_execution import process_report
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['source_hashes'], '先完成独立测试并冻结源码'
    assert digest(run / 'PROTOCOL.md') == manifest['protocol_sha256'], '协议改变'
    assert digest(run / 'config.yaml') == manifest['config_sha256'], '配置改变'
    for rel, expected in manifest['source_hashes'].items():
        assert digest(ROOT / rel) == expected, '源码改变: ' + rel
    inputs = run / 'inputs'
    for rel, expected in manifest['inputs_sha256'].items():
        assert digest(inputs / rel) == expected, '输入改变: ' + rel
    case = next(x for x in manifest['cases'] if x['name'] == name)
    folder = run / 'execution' / name
    folder.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((run / 'config.yaml').read_text())
    cfg = deepcopy(yaml.safe_load((inputs / 'execution_config.yaml').read_text())['portfolio'])
    cfg.update(capital=case['capital'], top_n=5)
    for key in ['buy_cost', 'sell_cost', 'min_fee']:
        cfg[key] *= case['cost_multiplier']
    causal = {**config['causal'], 'buy_limit_buffer':case['buffer']}
    save(folder / 'portfolio_config.json', dict(portfolio=cfg, causal=causal))
    retail = yaml.safe_load((inputs / 'symmetric_phase_research.yaml').read_text())['retail']
    actions_cfg = yaml.safe_load((inputs / 'corporate_action_components.yaml').read_text())
    signals = pd.read_parquet(inputs / 'low_turnover_signals.parquet')
    forecasts = pd.read_parquet(inputs / 'low_turnover_point_forecasts.parquet')
    hold = pd.read_parquet(inputs / 'monthly_hold_allowed.parquet')
    subset = pd.read_parquet(inputs / f"hold_{case['variant']}.parquet")
    assert hold.index.equals(subset.index) and subset.columns.isin(hold.columns).all()
    hold.loc[:, subset.columns] = subset
    frames = [pd.read_parquet(inputs / f'{key}.parquet') for key in
              ['market', 'basics', 'st', 'limits', 'normalized_actions']]
    reversed_inputs = []
    try:
        with reverse_candidates(case['reverse_input'], reversed_inputs):
            result = causal_portfolio.run_backtest(signals, *frames, case['start'], manifest['end'],
                cfg, actions_cfg.get('rights_issues', []), retail_forecasts=forecasts,
                retail_config=retail, hold_allowed=hold, causal_config=causal)
        checks = reconcile(result, case['capital'])
        process, pnl, periods, exits, statistics = process_report(result, case['capital'])
        for key, frame in [('daily', result.daily), ('trades', result.trades),
            ('holdings', result.holdings), ('events', result.events), ('process_daily', process),
            ('asset_pnl', pnl), ('holding_periods', periods), ('exit_reasons', exits)]:
            frame.to_csv(folder / f'{key}.csv', index=key in ['daily', 'process_daily'])
        # 引擎可提供额外订单表，事件表仍始终封存。
        if hasattr(result, 'orders'):
            result.orders.to_csv(folder / 'orders.csv', index=False)
        summary = {**case, **result.summary, **checks, **statistics}
        save(folder / 'summary.json', summary)
        previous, rows = case['capital'], []
        for year, daily in result.daily.groupby(result.daily.index.year):
            metrics, _ = daily_metrics(daily.equity, previous, cfg['annual_days'])
            rows.append(dict(year=year, start_equity=previous, end_equity=daily.equity.iloc[-1], **metrics))
            previous = daily.equity.iloc[-1]
        pd.DataFrame(rows).to_csv(folder / 'yearly.csv', index=False)
        recent = result.daily.loc['2024-01-01':]
        earlier = result.daily.loc[:'2023-12-31']
        initial = earlier.equity.iloc[-1] if len(earlier) else case['capital']
        metrics, _ = daily_metrics(recent.equity, initial, cfg['annual_days'])
        save(folder / 'recent_summary.json', dict(name=name, initial_equity=initial,
            end_equity=recent.equity.iloc[-1], mean_exposure=float((recent.stock_value/recent.equity).mean()), **metrics))
        if case['reverse_input']:
            assert reversed_inputs, '未在真实接收端执行倒序'
            save(folder / 'reverse_input_evidence.json', reversed_inputs)
        save(folder / 'completion.json', dict(status='complete', strategy_admitted=False,
            artifacts={p.name:digest(p) for p in folder.iterdir() if p.is_file()}))
        print(json.dumps(dict(name=name, total=summary['total'], mdd=summary['mdd'],
                             sharpe=summary['sharpe'], trades=summary['n_trades']), ensure_ascii=False), flush=True)
    except Exception as exc:
        save(folder / 'failure.json', dict(type=type(exc).__name__, error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/causal_execution_research.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--freeze-sources', action='store_true')
    parser.add_argument('--verify-quotes', action='store_true')
    parser.add_argument('--case-name')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.run.resolve(), args.config)
    elif args.freeze_sources:
        freeze(args.run.resolve())
    elif args.verify_quotes:
        verify_quotes(args.run.resolve())
    elif args.case_name:
        run_case(args.run.resolve(), args.case_name)
    else:
        parser.error('必须指定准备、冻结或单案例运行')


if __name__ == '__main__':
    main()
