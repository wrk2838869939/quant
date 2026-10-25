"""冻结盈利来源与防守机制对照，使用同一因果订单及现金账本。"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from defense_diagnostics import DefenseOverlay, risk_evidence
from neutralize import attach_industry_pit
from audit_monthly_portfolio import reconcile
from run_patient_value_execution import process_report
from causal_portfolio import daily_metrics
import defense_research_portfolio as engine


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')


def prepare(config_path):
    config = yaml.safe_load(config_path.read_text())
    parent, signal_parent = [ROOT/config[key] for key in ['parent', 'signal_parent']]
    run = ROOT/config['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run/'config.yaml')
    sources = ['scripts/research_profit_defense.py', 'src/defense_diagnostics.py',
               'src/defense_research_portfolio.py', 'src/causal_portfolio.py', 'src/portfolio.py',
               'src/selection_first.py', 'src/retail_rebalance.py', 'src/patient_trend.py',
               'src/neutralize.py', 'scripts/audit_monthly_portfolio.py',
               'scripts/run_patient_value_execution.py', 'tests/test_defense_diagnostics.py']
    for name in sources:
        target = run/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    (run/'engine_changes.diff').write_text(''.join(difflib.unified_diff(
        (ROOT/'src/causal_portfolio.py').read_text().splitlines(True),
        (ROOT/'src/defense_research_portfolio.py').read_text().splitlines(True),
        fromfile='sealed_causal_engine', tofile='research_defense_engine')))
    protocol = '''# 盈利来源与防守能力：固定机制对照

历史截至2026-09-08，全部已观察，不是新样本外。协议在本轮对照回放前冻结。

## 盈利来源
逐日现金账本拆价格变动、税后模型分红与费用；行业使用每日之前已生效的申万分类。
可执行对照均5万元、最多5只、100股整手、相同费用、预定DAY限价，不主动排除高价股。
initial_buyhold只在起始月度复核按原信号规划一次，随后不再月度调仓/减持上限；法定事件仍按原引擎处理。
bank_buyhold在该日全合格主板银行中按流通市值降序取5只（代码打破并列），只初次规划，随后同上。
银行比较是事后提出的机制诊断，不代表当时已选择该行业，也不是据后续利润选择五个赢家。
参考行业收益使用昨日成分、昨日流通市值及复权价格；含分红的理论参考无整手和费用，不能冒充可执行账户或把残差称独立alpha。

## 防守2×2及完整退出
hybrid保持冻结月度规则。weekly_exit仅每周首个交易日收盘检查长期两项均弱且过去完整20日中至少10日弱，向次日提交退出。
pause_only保持原月度退出；周初收盘若当前持仓市值至少一半满足上述条件，暂停新买并冻结当时风险篮子及权重。
weekly_pause合并两者。weekly_liquidate在暂停触发时额外请求退出全部旧仓。所有已确认卖出仍重试，不能用反弹撤销。
暂停后仅月度复核可恢复：原冻结篮子超过一半权重满足原60/120日正收益且高于MA120，并且信息可知。
未知不作恢复证据；卖出造成的空篮子不算市场恢复；新买仍只在原月度/一次补位窗口，不增加日频换股。
半数阈值是预先登记的诊断参数，未优化；不根据本轮收益换阈值。

## 全部报告与核验
每种方案跑2022连续、2024现金启动，共14个账户；报告逐年及回撤全过程，包含失败方案。
原Hybrid四张账本与原封存逐值等价；新增状态机历史截断、未知和冻结篮子测试；每账户会计对账。
个股风险证据每日输出，回撤定位使用事后高低点只作描述，不进入决策。
原引擎与封存结果不改，不进行任何实盘操作。阈值有效性、首发数据版本及成交队列仍未认证。
'''
    (run/'PROTOCOL.md').write_text(protocol)
    # 固定银行参考只读取开始日的信息，不看未来收益或最终赢家名单。
    signals = pd.read_parquet(parent/'inputs/low_turnover_signals.parquet')
    forecasts = pd.read_parquet(parent/'inputs/low_turnover_point_forecasts.parquet')
    reviews = {name: signals.index[signals.index >= pd.Timestamp(start)][0]
               for name, start in config['windows'].items()}
    panel_path = ROOT/'data/panel/daily_panel.parquet'
    snapshot = pd.read_parquet(panel_path, columns=['date', 'ts_code', 'circ_mv_yuan'],
                              filters=[('date', 'in', list(reviews.values()))])
    industry_path = ROOT/'data/raw/sw_l1_member_hist.parquet'
    snapshot = attach_industry_pit(snapshot, pd.read_parquet(industry_path))
    rows = []
    for name, date in reviews.items():
        pool = snapshot[snapshot.date.eq(date)].copy()
        eligible = forecasts.loc[date].dropna().index
        pool = pool[pool.ts_code.isin(eligible) & pool.sw_l1.eq(config['bank_industry'])
            & pool.ts_code.str.match(r'^(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)$')
            & pool.circ_mv_yuan.gt(0)]
        pool = pool.sort_values(['circ_mv_yuan', 'ts_code'], ascending=[False, True])
        pool['window'] = name
        pool['selected'] = np.arange(len(pool)) < config['benchmark_size']
        assert pool.selected.sum() == config['benchmark_size']
        rows.append(pool)
    bank = pd.concat(rows, ignore_index=True)
    bank.to_csv(run/'bank_initial_universe.csv', index=False)
    codes = set(pd.read_parquet(parent/'inputs/market.parquet', columns=['ts_code']).ts_code)
    absent = set(bank.loc[bank.selected, 'ts_code'])-codes
    if absent:
        raise ValueError('银行对照必须扩展真实行情及行动，不许缩减候选: '+str(absent))
    data_paths = [parent/'completion.json', parent/'manifest.json', signal_parent/'daily_close.parquet',
                  industry_path, panel_path]
    data_paths += list((parent/'inputs').glob('*'))
    data_paths += [parent/'execution'/name/f'{file}.csv' for name in ['cold_hybrid', 'continued_hybrid']
                   for file in ['daily', 'trades', 'holdings', 'events']]
    manifest = dict(status='frozen', fresh_holdout=False, created=str(datetime.now(timezone.utc)),
        cases=[dict(name=f'{w}_{v}', window=w, variant=v, start=start)
               for w, start in config['windows'].items() for v in config['variants']],
        source_hashes={name:digest(ROOT/name) for name in sources},
        inputs_sha256={str(p):digest(p) for p in data_paths},
        config_sha256=digest(run/'config.yaml'), protocol_sha256=digest(run/'PROTOCOL.md'),
        bank_universe_sha256=digest(run/'bank_initial_universe.csv'))
    save(run/'manifest.json', manifest)
    print(str(run), flush=True)
    return run


def execute(run):
    config = yaml.safe_load((run/'config.yaml').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    for name, h in manifest['source_hashes'].items():
        assert digest(ROOT/name) == h, name
    for name, h in manifest['inputs_sha256'].items():
        assert digest(name) == h, name
    assert digest(run/'PROTOCOL.md') == manifest['protocol_sha256']
    assert digest(run/'config.yaml') == manifest['config_sha256']
    assert digest(run/'bank_initial_universe.csv') == manifest['bank_universe_sha256']
    parent, signal_parent = [ROOT/config[key] for key in ['parent', 'signal_parent']]
    inp = parent/'inputs'
    signals = pd.read_parquet(inp/'low_turnover_signals.parquet')
    forecasts = pd.read_parquet(inp/'low_turnover_point_forecasts.parquet')
    holds = pd.read_parquet(inp/'monthly_hold_allowed.parquet')
    hybrid = pd.read_parquet(inp/'hold_hybrid.parquet')
    holds.loc[:, hybrid.columns] = hybrid
    frames = [pd.read_parquet(inp/f'{key}.parquet') for key in
              ['market', 'basics', 'st', 'limits', 'normalized_actions']]
    base_cfg = yaml.safe_load((inp/'execution_config.yaml').read_text())['portfolio']
    base_cfg.update(capital=config['capital'], top_n=5)
    retail = yaml.safe_load((inp/'symmetric_phase_research.yaml').read_text())['retail']
    actions_cfg = yaml.safe_load((inp/'corporate_action_components.yaml').read_text())
    causal = yaml.safe_load((parent/'config.yaml').read_text())['causal']
    prices = pd.read_parquet(signal_parent/'daily_close.parquet')
    evidence = risk_evidence(prices, config['defense'])
    bank = pd.read_csv(run/'bank_initial_universe.csv', parse_dates=['date'])
    summaries, years, checks = [], [], []
    for case in manifest['cases']:
        folder = run/'execution'/case['name']
        if folder.exists():
            raise ValueError('禁止覆盖已有案例: '+str(folder))
        folder.mkdir(parents=True)
        sig, mu, hold = signals.copy(), forecasts.copy(), holds.copy()
        if case['variant'].endswith('buyhold'):
            date = sig.index[sig.index >= pd.Timestamp(case['start'])][0]
            sig, mu = sig.loc[[date]], mu.loc[[date]]
            hold = pd.DataFrame(1., index=sig.index, columns=sig.columns)
            if case['variant'] == 'bank_buyhold':
                chosen = bank[bank.window.eq(case['window']) & bank.selected].ts_code.tolist()
                sig.loc[:] = np.nan
                sig.loc[date, chosen] = np.arange(len(chosen), 0, -1.)
        overlay = (DefenseOverlay(evidence, config['defense'], case['variant'])
                   if case['variant'] in ['weekly_exit', 'pause_only', 'weekly_pause', 'weekly_liquidate'] else None)
        result = engine.run_backtest(sig, *frames, case['start'], config['end'], deepcopy(base_cfg),
            actions_cfg.get('rights_issues', []), retail_forecasts=mu, retail_config=retail,
            hold_allowed=hold, causal_config=causal, defense_overlay=overlay)
        account_check = reconcile(result, config['capital'])
        process, pnl, periods, exits, stats = process_report(result, config['capital'])
        for name, frame in [('daily',result.daily), ('holdings',result.holdings), ('trades',result.trades),
            ('events',result.events), ('process_daily',process), ('asset_pnl',pnl),
            ('holding_periods',periods), ('exit_reasons',exits)]:
            frame.to_csv(folder/f'{name}.csv', index=name in ['daily','process_daily'])
        if overlay:
            pd.DataFrame(overlay.records).to_csv(folder/'defense_events.csv', index=False)
        if case['variant'] == 'hybrid':
            for name in ['daily', 'holdings', 'trades', 'events']:
                old = pd.read_csv(parent/'execution'/case['name']/f'{name}.csv')
                new = pd.read_csv(folder/f'{name}.csv')
                pd.testing.assert_frame_equal(old, new, check_exact=True)
        # 所有实际订单必须先于成交；现金约束仍由原引擎逐单断言。
        assert (result.trades.decision_date < result.trades.date).all()
        summary = dict(**case, **result.summary, **stats, **account_check)
        save(folder/'summary.json', summary)
        summaries.append(summary)
        previous = config['capital']
        for year, group in result.daily.groupby(result.daily.index.year):
            metrics, _ = daily_metrics(group.equity, previous, base_cfg['annual_days'])
            years.append(dict(name=case['name'], year=year, **metrics, end_equity=group.equity.iloc[-1]))
            previous = group.equity.iloc[-1]
        checks.append(dict(name=case['name'], **account_check,
                           baseline_exact=case['variant']=='hybrid', decision_precedes_fill=True))
        print(json.dumps({k:summary[k] for k in ['name','total','mdd','mean_invested','n_trades']}, ensure_ascii=False), flush=True)
        save(folder/'completion.json', dict(status='complete', artifacts={p.name:digest(p) for p in folder.iterdir()}))
    pd.DataFrame(summaries).to_csv(run/'summary.csv', index=False)
    pd.DataFrame(years).to_csv(run/'yearly.csv', index=False)
    save(run/'account_checks.json', checks)
    save(run/'execution_completion.json', dict(status='complete', cases=len(summaries),
                                              strategy_admitted=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/profit_defense_research.yaml')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run = args.run or prepare(args.config)
    if not args.prepare_only:
        execute(run)


if __name__ == '__main__':
    main()
