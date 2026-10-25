"""5万元为主的风险恢复/整手缓冲2×2实验，不重训或优化风险模型。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import difflib
import sys
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from research_profit_defense import digest, save
from risk_recovery_portfolio import run_backtest, daily_metrics
from audit_monthly_portfolio import reconcile
from run_patient_value_execution import process_report


def prepare():
    cfg = yaml.safe_load((ROOT / 'configs/risk_recovery_research.yaml').read_text())
    for key in ['parent', 'signal_parent']:
        folder = ROOT / cfg[key]
        old = json.loads((folder / 'completion.json').read_text())
        for name, expected in old['artifact_sha256'].items():
            assert digest(folder / name) == expected, name
    run = ROOT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    shutil.copy2(ROOT / 'configs/risk_recovery_research.yaml', run / 'config.yaml')
    sources = ['scripts/research_risk_recovery.py', 'src/risk_recovery_portfolio.py',
        'src/risk_budget_portfolio.py', 'src/portfolio.py', 'src/selection_first.py', 'src/retail_rebalance.py',
        'scripts/audit_monthly_portfolio.py', 'scripts/run_patient_value_execution.py',
        'scripts/run_risk_recovery_tests.py', 'tests/test_risk_recovery.py', 'tests/test_risk_budget.py',
        'tests/test_causal_portfolio_adversarial.py']
    for name in sources:
        target = run / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in ['recovery_unit.log', 'recovery_risk.log', 'recovery_causal.log']:
        shutil.copy2(Path('/tmp') / name, run / name)
        assert (run / name).read_text().rstrip().endswith('OK')
    (run / 'engine_changes.diff').write_text(''.join(difflib.unified_diff(
        (ROOT / 'src/risk_budget_portfolio.py').read_text().splitlines(True),
        (ROOT / 'src/risk_recovery_portfolio.py').read_text().splitlines(True),
        fromfile='frozen_risk_budget_engine', tofile='recovery_research_engine')))
    (run / 'PROTOCOL.md').write_text('''# 风险恢复与整手缓冲：固定2×2实验

5万元为主，2万元仅沿用上轮参考，不以其结果优化本轮。价格、股票池、持有退出规则、风险模型与全部年度风险折扣冻结复用，不调模型或阈值。
基线是上轮full_risk；四政策baseline/restore/grace/combined独立检验及交互。
恢复：仅实际风险上限卖出成交后、仍持有的股票有资格；全卖/退市结束资格。原复核日当日完整入场分数非空、持有许可为真、没有待卖、当时风险折扣严格高于上次真实风险减持或恢复成交时的折扣，才考虑恢复。模型年初更新也会改变风险估计，不解释为纯行情变化。
优先按原规则预约新仓，旧仓恢复只使用剩余收盘现金；不预支卖款，不挤占新仓已冻结预算。旧仓之间用当日原排名排序，不要求重入Top20。恢复至当前风险许可上限，数量按100股取整，并通过费用后预期净利门槛；不限制为过去卖出的原股数。成交保留原持仓及原锁定股，并将新增股锁到T+1。
恢复只在原复核日发DAY单，不在补位日或每日追买；未成交不更新恢复参考，不在次日追加重试。下一次恢复需要风险折扣进一步改善。未来涨跌不能决定恢复资格。
整手缓冲：仅原要求减仓股数大于0且不足100股，且当前市值未超过原25%上限时，暂缓该次减仓；达到一手或超过原25%仍执行。风险折扣上限因此成为有一手容忍带的复核约束，不能宣称完全无超限。明确趋势退出/ST退出不受缓冲影响。
所有复核仍最多5股，持有期间不因新排名强卖旧仓；次日限价、100股、现金、分红与费用沿用冻结执行。新增股与旧股的可卖数量分别保留，禁止覆盖旧仓对象。
主实验8日历×2起点×4政策=64账户；5万元双费用同月初四政策×两起点=8；10万元月初baseline/combined×两起点=4，共76。按全部日历及逐一配对报告，不选最好日期。2万元不新增参数试验。
验证37测试、16主基线四类账本逐格复现、3个实际历史前缀、每账户冻结现金/股数/逐股损益，逐笔恢复及缓冲条件独立复核；分析六月、成交日、现金和旧仓恢复后路径。旧模型和过去实验封存保持不变。
全为已观察历史，未自动实盘准入；不以个别恢复成功案例事后挑阈值。
''')
    cases = []
    for window, start in cfg['windows'].items():
        for schedule in cfg['schedules']:
            for policy in cfg['policies']:
                cases.append(dict(name=f'{window}_{schedule}_{policy}', window=window, start=start,
                    schedule=schedule, policy=policy, capital=cfg['capital'], cost_multiplier=1, stress='main'))
        for policy in cfg['policies']:
            cases.append(dict(name=f'{window}_cost2_{policy}', window=window, start=start,
                schedule='calendar_01', policy=policy, capital=cfg['capital'], cost_multiplier=cfg['stress_cost_multiplier'], stress='cost2'))
        for policy in ['baseline', 'combined']:
            cases.append(dict(name=f'{window}_large_{policy}', window=window, start=start,
                schedule='calendar_01', policy=policy, capital=cfg['reference_capital'], cost_multiplier=1, stress='large'))
    save(run / 'manifest.json', dict(cases=cases, config_sha256=digest(run / 'config.yaml'),
        protocol_sha256=digest(run / 'PROTOCOL.md'), source_hashes={name: digest(ROOT / name) for name in sources},
        parent_hashes={cfg[key]: digest(ROOT / cfg[key] / 'completion.json') for key in ['parent', 'signal_parent']},
        root_agents_sha256=digest(ROOT.parent / 'AGENTS.md')))
    print(run, flush=True)


def verify(run):
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    m = json.loads((run / 'manifest.json').read_text())
    assert digest(run / 'config.yaml') == m['config_sha256']
    assert digest(run / 'PROTOCOL.md') == m['protocol_sha256']
    for name, expected in m['source_hashes'].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in m['parent_hashes'].items():
        assert digest(ROOT / name / 'completion.json') == expected
    return cfg, m


def simulate(run, case, stop=None):
    cfg, _ = verify(run)
    parent = ROOT / cfg['signal_parent']
    inp = parent / 'inputs'
    end = cfg['end'] if stop is None else stop
    source = parent / 'schedules' / case['schedule']
    sig, mu, hold = [pd.read_parquet(source / f'{name}.parquet').loc[:end] for name in ['signals', 'forecasts', 'hold']]
    needed = sorted(set().union(*(set(row.dropna().sort_values(ascending=False, kind='stable').head(20).index) for _, row in sig.iterrows())))
    frames = [pd.read_parquet(inp / f'{name}.parquet') for name in ['market', 'basics', 'st', 'limits', 'normalized_actions']]
    local = [frame[frame.ts_code.isin(needed)].copy() for frame in frames]
    local[0] = local[0][local[0].date <= pd.Timestamp(end)]
    local[3] = local[3][local[3].date <= pd.Timestamp(end)]
    local[4] = local[4][local[4].known_date <= pd.Timestamp(end)]
    portfolio = yaml.safe_load((inp / 'execution_config.yaml').read_text())['portfolio']
    portfolio.update(capital=case['capital'], top_n=5)
    for key in ['buy_cost', 'sell_cost', 'min_fee']:
        portfolio[key] *= case['cost_multiplier']
    retail = yaml.safe_load((inp / 'symmetric_phase_research.yaml').read_text())['retail']
    ac = yaml.safe_load((inp / 'corporate_action_components.yaml').read_text())
    cc = yaml.safe_load((ROOT / 'outputs/causal_execution/20260913T130810_374654Z/config.yaml').read_text())['causal']
    discounts = pd.read_parquet(ROOT / cfg['parent'] / 'risk/discounts.parquet').reindex(index=sig.index, columns=sig.columns)
    return run_backtest(sig, *local, case['start'], end, portfolio, ac.get('rights_issues', []),
        retail_forecasts=mu, retail_config=retail, hold_allowed=hold, causal_config=cc,
        risk_multipliers=discounts, adjust_existing_risk=True, recovery_config=cfg['policies'][case['policy']])


def execute(run, window, family):
    cfg, m = verify(run)
    cases = [c for c in m['cases'] if c['window'] == window and
        ((family == 'stress' and c['stress'] != 'main') or (c['stress'] == 'main' and c['schedule'].startswith(family)))]
    for case in cases:
        folder = run / 'execution' / case['name']
        folder.mkdir(parents=True)
        result = simulate(run, case)
        checks = reconcile(result, case['capital'])
        process, pnl, periods, exits, stats = process_report(result, case['capital'])
        for name, frame in [('daily', result.daily), ('trades', result.trades), ('holdings', result.holdings),
                            ('events', result.events), ('process_daily', process), ('asset_pnl', pnl),
                            ('holding_periods', periods), ('exit_reasons', exits)]:
            frame.to_csv(folder / f'{name}.csv', index=name in ['daily', 'process_daily'])
        if case['policy'] == 'baseline':
            original = f"{window}_{case['schedule'] if case['stress'] == 'main' else case['stress']}_full_risk"
            old = ROOT / cfg['parent'] / 'execution' / original
            for name in ['daily', 'trades', 'holdings', 'events']:
                pd.testing.assert_frame_equal(pd.read_csv(folder / f'{name}.csv'), pd.read_csv(old / f'{name}.csv'), check_exact=True)
        summary = {**case, **result.summary, **stats, **checks}
        save(folder / 'summary.json', summary)
        previous, years = case['capital'], []
        for year, group in result.daily.groupby(result.daily.index.year):
            metrics, _ = daily_metrics(group.equity, previous, 244)
            years.append(dict(name=case['name'], year=year, **metrics))
            previous = group.equity.iloc[-1]
        pd.DataFrame(years).to_csv(folder / 'yearly.csv', index=False)
        save(folder / 'completion.json', dict(status='complete', hashes={p.name: digest(p) for p in folder.iterdir()}))
        print(json.dumps({key: summary[key] for key in ['name', 'total', 'mdd', 'sharpe', 'n_trades']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--window', choices=['continued', 'cold'])
    parser.add_argument('--family', choices=['calendar', 'rolling', 'stress'])
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        execute(args.run.resolve(), args.window, args.family)
