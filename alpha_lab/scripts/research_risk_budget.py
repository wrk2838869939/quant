"""生命周期风险估计进入五股账户：统一降仓、仅入场和持仓风险折扣。"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse
import difflib
import json
import shutil
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from research_profit_defense import digest, save
from exit_evidence import forward_marks, nonoverlapping_events
from exit_history import annual_fit, predict_frame
from lifecycle_risk import lifecycle_reviews, risk_discount
from risk_budget_portfolio import run_backtest, daily_metrics
from audit_monthly_portfolio import reconcile
from run_patient_value_execution import process_report


def risk_features(price):
    clean = price.where(price > 0)
    return dict(volatility20=clean.pct_change(fill_method=None).rolling(20, min_periods=20).std(),
                drawdown120=clean / clean.rolling(120, min_periods=120).max() - 1)


def prepare():
    cfg = yaml.safe_load((ROOT / 'configs/risk_budget_research.yaml').read_text())
    run = ROOT / cfg['output_root'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    shutil.copy2(ROOT / 'configs/risk_budget_research.yaml', run / 'config.yaml')
    sources = ['scripts/research_risk_budget.py', 'src/risk_budget_portfolio.py', 'src/causal_portfolio.py',
        'src/lifecycle_risk.py', 'src/exit_history.py', 'src/exit_evidence.py', 'src/portfolio.py',
        'src/selection_first.py', 'src/retail_rebalance.py', 'scripts/audit_monthly_portfolio.py',
        'scripts/run_patient_value_execution.py', 'scripts/run_risk_budget_tests.py',
        'tests/test_risk_budget.py', 'tests/test_causal_portfolio_adversarial.py']
    for name in sources:
        target = run / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in ['risk_budget_unit.log', 'risk_budget_adversarial.log']:
        shutil.copy2(Path('/tmp') / name, run / name)
        assert (run / name).read_text().rstrip().endswith('OK')
    (run / 'engine_changes.diff').write_text(''.join(difflib.unified_diff(
        (ROOT / 'src/causal_portfolio.py').read_text().splitlines(True),
        (ROOT / 'src/risk_budget_portfolio.py').read_text().splitlines(True),
        fromfile='original_causal_engine', tofile='research_risk_budget_engine')))
    (run / 'PROTOCOL.md').write_text('''# 生命周期风险折扣：固定执行协议

上一轮只取退出时状态，不能直接推广至正常持仓。本轮用2015起当时Top20候选完整生命周期的月初状态（入场/持有/退出）重建训练池，同股顺序保留间隔严格超过60交易日的状态，不按未来收益采样。
特征固定20日波动率及距120日高点回撤，目标下一日收盘开始60交易日内MAE，缺失不填。每年初只用标签完全结束的样本拟合标准化Ridge alpha=10，至少30事件/15日期。原账户不进入训练。
参考风险取当年成熟训练MAE绝对值中位数；单股上限折扣=min(1,参考风险/max(预测浮亏绝对值,参考风险))。预测浮亏绝对值=max(-预测MAE,0)。这不是组合最大回撤保证。
统一折扣取该年训练样本的模型预测折扣均值，仅用历史信息；它是少持仓对照，不是用未来实现仓位匹配。
四政策：baseline原上限；uniform所有新旧仓统一折扣；entry_risk只对新买入按股票风险折扣；full_risk新旧仓都按风险折扣。旧仓只在原复核日按上限减仓，不额外增加交易日期，风险改善后不主动补足旧仓，沿用原无补仓逻辑。
实际持仓或候选未知特征时回退当年统一折扣，未知不视为零风险；缺风险矩阵直接报错。保留原25%单股上限，风险折扣只能缩小，不能杠杆或突破。候选排序与Top20池不变，100股现金约束造成不可买需如实记录，不另外筛高价股。
每年模型更新不引入额外交易日；补位沿用最近一次复核风险比例。原持有退出规则、5只上限、DAY限价、分红/费用、不得预支卖款均不变。
主实验8日历×2启动窗口×4政策=64账户，5万元。压力检验仅月初baseline/full_risk：2万、10万和5万元双费用，两启动窗口共12账户，总76账户。全部报告，不按结果挑日历/阈值/资金量。
检查16原基线账本逐格复现，6风险测试+25原撮合对抗测试，年度标签扰动与独立矩阵复算，真实历史截断，逐日现金/逐股损益/冻结订单对账。剖析减仓后现金是否补回、整手完全退出、交易次数、六月损失及收益来源。
退出风险和现金配置不是整体择时。本轮不假设组合个股风险独立，不将各股MAE简单相加当作真实组合回撤，也不自动加入强制全空仓规则。
全部历史已被项目观察；这是新的研究实验，不是新鲜样本外。没有自动实盘准入，不修改根AGENTS或生产。
''')
    cases = []
    for window, start in cfg['windows'].items():
        for schedule in cfg['schedules']:
            for policy in cfg['policies']:
                cases.append(dict(name=f'{window}_{schedule}_{policy}', window=window, start=start,
                    schedule=schedule, policy=policy, capital=cfg['capital'], cost_multiplier=1, stress='main'))
        for stress, capital, cost in [('small', cfg['stress_capitals'][0], 1),
                                      ('large', cfg['stress_capitals'][1], 1),
                                      ('cost2', cfg['capital'], cfg['stress_cost_multiplier'])]:
            for policy in ['baseline', 'full_risk']:
                cases.append(dict(name=f'{window}_{stress}_{policy}', window=window, start=start,
                    schedule='calendar_01', policy=policy, capital=capital, cost_multiplier=cost, stress=stress))
    save(run / 'manifest.json', dict(cases=cases, config_sha256=digest(run / 'config.yaml'),
        protocol_sha256=digest(run / 'PROTOCOL.md'), source_hashes={name: digest(ROOT / name) for name in sources},
        parent_receipts={cfg[key]: digest(ROOT / cfg[key] / 'completion.json') for key in ['parent', 'history_parent']},
        root_agents_sha256=digest(ROOT.parent / 'AGENTS.md')))
    print(run, flush=True)


def verify(run):
    m = json.loads((run / 'manifest.json').read_text())
    assert digest(run / 'config.yaml') == m['config_sha256']
    assert digest(run / 'PROTOCOL.md') == m['protocol_sha256']
    for name, expected in m['source_hashes'].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in m['parent_receipts'].items():
        assert digest(ROOT / name / 'completion.json') == expected, name
    return yaml.safe_load((run / 'config.yaml').read_text()), m


def build(run):
    cfg, m = verify(run)
    for path in m['parent_receipts']:
        parent = ROOT / path
        receipt = json.loads((parent / 'completion.json').read_text())
        for name, expected in receipt['artifact_sha256'].items():
            assert digest(parent / name) == expected, name
    out = run / 'risk'
    out.mkdir()
    hist = ROOT / cfg['history_parent'] / 'data'
    prices = pd.read_parquet(hist / 'close.parquet')
    sig, hold = [pd.read_parquet(hist / f'{name}.parquet') for name in ['signals', 'hold']]
    states = lifecycle_reviews(sig, hold, 20)
    features = risk_features(prices)
    states = nonoverlapping_events(states, prices.index, cfg['nonoverlap_days'])
    rows = []
    for r in states.to_dict('records'):
        r.update({key: value.at[r['date'], r['ts_code']] for key, value in features.items()})
        r['features_complete'] = bool(np.isfinite([r[key] for key in cfg['features']]).all())
        if r['nonoverlap_primary']:
            r.update({f'{key}_60': value for key, value in forward_marks(prices, r['date'], r['ts_code'], 60).items()})
        rows.append(r)
    all_states = pd.DataFrame(rows)
    all_states.to_parquet(out / 'lifecycle_states.parquet', index=False)
    frame = all_states[all_states.nonoverlap_primary].copy()
    frame.groupby([frame.date.dt.year, 'phase']).size().rename('n').to_csv(out / 'training_population.csv')
    inputs = ROOT / cfg['parent'] / 'inputs'
    actual_price = pd.read_parquet(inputs / 'daily_close.parquet')
    real_features = risk_features(actual_price)
    dates = sorted(set().union(*(set(pd.read_parquet(ROOT / cfg['parent'] / 'schedules' / schedule / 'signals.parquet').index)
                                  for schedule in cfg['schedules'])))
    template = pd.DataFrame(np.nan, index=pd.DatetimeIndex(dates, name='date'), columns=actual_price.columns)
    caps, predicted = template.copy(), template.copy()
    uniform, audits, predictions, checks = [], [], [], []
    for cutoff in pd.to_datetime(['2019-06-28', '2025-06-30']):
        short = lifecycle_reviews(sig.loc[:cutoff], hold.loc[:cutoff], 20)
        pd.testing.assert_frame_equal(short, states[states.date <= cutoff].drop(columns='nonoverlap_primary').reset_index(drop=True))
        for price, full in [(prices, features), (actual_price, real_features)]:
            small = risk_features(price.loc[:cutoff])
            for key in full:
                pd.testing.assert_frame_equal(small[key], full[key].loc[:cutoff], check_exact=True)
        checks.append(dict(cutoff=cutoff, lifecycle_exact=True, feature_checks=4))
    for year in cfg['prediction_years']:
        train, model, mean, _ = annual_fit(frame, year, 60, 'mae', cfg)
        assert model is not None
        reference = float((-train.mae_60).median())
        training_pred = predict_frame(train, model, mean, cfg['features'])
        constant = float(risk_discount(training_pred, reference).mean())
        uniform.append(dict(year=year, reference_risk=reference, uniform_discount=constant,
                            train_events=len(train), train_dates=train.date.nunique(), max_label_end=train.label_end_60.max()))
        train.to_parquet(out / f'train_{year}.parquet', index=False)
        flat = pd.DataFrame({key: value.loc[template.index[template.index.year == year]].stack(future_stack=True)
                             for key, value in real_features.items()}).reset_index()
        flat['features_complete'] = np.isfinite(flat[cfg['features']]).all(axis=1)
        values = predict_frame(flat, model, mean, cfg['features'])
        discounts = risk_discount(values, reference)
        discounts[~flat.features_complete.to_numpy()] = constant
        flat['prediction'] = values
        flat['discount'] = discounts
        for name, result in [('discount', caps), ('prediction', predicted)]:
            block = flat.pivot(index='date', columns='ts_code', values=name)
            result.loc[block.index, block.columns] = block
        altered = frame.copy()
        unknown = altered.label_end_60.ge(pd.Timestamp(f'{year}-01-01')) | altered.label_end_60.isna()
        altered.loc[unknown, 'mae_60'] = -1e6
        train2, model2, mean2, _ = annual_fit(altered, year, 60, 'mae', cfg)
        assert train.index.equals(train2.index) and mean == mean2
        np.testing.assert_array_equal(values, predict_frame(flat, model2, mean2, cfg['features']))
        # 独立标准化和矩阵解检查，不调用Ridge进行复算。
        x = train[cfg['features']].to_numpy()
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale == 0] = 1.
        z = (x-center)/scale
        zmean = z.mean(axis=0)
        z -= zmean
        coef = np.linalg.solve(z.T@z + cfg['model']['alpha']*np.eye(len(cfg['features'])), z.T@(train.mae_60-mean))
        good = flat.features_complete.to_numpy()
        manual = np.full(len(flat), mean)
        manual[good] = ((flat.loc[good, cfg['features']].to_numpy()-center)/scale-zmean)@coef+mean
        error = float(np.max(np.abs(manual-values)))
        assert error < 1e-10
        audits.append(dict(year=year, future_perturbation_exact=True, manual_max_error=error,
            reference_risk=reference, uniform_discount=constant, unknown_features=int((~good).sum()),
            intercept=float(model[-1].intercept_), coefficients=model[-1].coef_.tolist(),
            center=model[0].mean_.tolist(), scale=model[0].scale_.tolist()))
        predictions.append(flat)
    assert caps.notna().all().all()
    caps.to_parquet(out / 'discounts.parquet')
    predicted.to_parquet(out / 'predicted_mae.parquet')
    pd.concat(predictions).to_parquet(out / 'predictions.parquet', index=False)
    pd.DataFrame(uniform).to_csv(out / 'annual_reference.csv', index=False)
    save(out / 'verification.json', dict(prefix_checks=checks, annual_checks=audits))
    save(out / 'completion.json', dict(status='complete', lifecycle_states=len(states), primary_states=len(frame),
        artifacts={p.name: digest(p) for p in out.iterdir()}))
    print('Risk built:', len(states), 'lifecycle states;', len(frame), 'primary; annual reference:', uniform, flush=True)


def simulate(run, case, stop=None):
    cfg, _ = verify(run)
    parent = ROOT / cfg['parent']
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
    for name in ['buy_cost', 'sell_cost', 'min_fee']:
        portfolio[name] *= case['cost_multiplier']
    retail = yaml.safe_load((inp / 'symmetric_phase_research.yaml').read_text())['retail']
    ac = yaml.safe_load((inp / 'corporate_action_components.yaml').read_text())
    cc = yaml.safe_load((ROOT / 'outputs/causal_execution/20260913T130810_374654Z/config.yaml').read_text())['causal']
    scales = None
    if case['policy'] != 'baseline':
        scales = pd.read_parquet(run / 'risk/discounts.parquet').reindex(index=sig.index, columns=sig.columns)
        if case['policy'] == 'uniform':
            annual = pd.read_csv(run / 'risk/annual_reference.csv').set_index('year').uniform_discount
            for date in scales.index:
                scales.loc[date] = annual.loc[date.year]
    return run_backtest(sig, *local, case['start'], end, portfolio, ac.get('rights_issues', []),
        retail_forecasts=mu, retail_config=retail, hold_allowed=hold, causal_config=cc,
        risk_multipliers=scales, adjust_existing_risk=case['policy'] in ['uniform', 'full_risk'])


def execute(run, window, family):
    cfg, m = verify(run)
    receipt = json.loads((run / 'risk/completion.json').read_text())
    for name, expected in receipt['artifacts'].items():
        assert digest(run / 'risk' / name) == expected
    cases = [c for c in m['cases'] if c['window'] == window and
             ((family == 'stress' and c['stress'] != 'main') or
              (c['stress'] == 'main' and c['schedule'].startswith(family)))]
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
        if case['policy'] == 'baseline' and case['stress'] == 'main':
            old = ROOT / cfg['parent'] / 'execution' / f"{window}_{case['schedule']}_hybrid"
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
    parser.add_argument('--stage', required=True, choices=['prepare', 'build', 'execute'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--window', choices=['continued', 'cold'])
    parser.add_argument('--family', choices=['calendar', 'rolling', 'stress'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'build':
        build(args.run.resolve())
    else:
        execute(args.run.resolve(), args.window, args.family)
