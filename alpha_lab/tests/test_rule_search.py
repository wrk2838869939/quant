"""规则搜索的时序、全域联合采样和真实账本接口验证。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from rule_search import RuleSignals, decision_dates, sample_configs, local_configs, config_id, legal, select_trial
import test_portfolio as fixture_module
from portfolio import run_backtest


def synthetic():
    rng = np.random.default_rng(10)
    dates = pd.bdate_range('2017-01-02', periods=600)
    columns = ['A', 'B', 'C', 'D']
    close = pd.DataFrame(20 * np.exp(np.cumsum(rng.normal(0, .015, (600, 4)), axis=0)), index=dates, columns=columns)
    wide = {k: close.copy() for k in ['hfq_close', 'hfq_high', 'hfq_low', 'open', 'close', 'high', 'low', 'pre_close']}
    wide['pre_close'] = close.shift()
    for k in ['high', 'hfq_high']:
        wide[k] *= 1.02
    for k in ['low', 'hfq_low']:
        wide[k] *= .98
    for k in ['turnover_rate', 'pe_ttm', 'pb', 'circ_mv_yuan', 'amount_yuan', 'vol_share',
              'buy_lg_amount_yuan', 'sell_lg_amount_yuan', 'buy_elg_amount_yuan', 'sell_elg_amount_yuan']:
        wide[k] = pd.DataFrame(rng.uniform(1, 10, (600, 4)), index=dates, columns=columns)
    eligible = pd.DataFrame(True, index=dates[300:], columns=columns)
    eligible.iloc[5:15, 1] = False
    return wide, eligible, dates


def test_every_family_prefix_and_missing_eligibility():
    plan = yaml.safe_load((ROOT / 'configs/rule_strategy_search_v1.yaml').read_text())
    wide, eligible, calendar = synthetic()
    cut = eligible.index[170]
    full = RuleSignals(wide, eligible, calendar)
    short = RuleSignals({k: v.loc[:cut] for k, v in wide.items()}, eligible.loc[:cut], calendar)
    for family in list(plan['family_spaces']) + list(plan['classic_tunable_variants']):
        for p in sample_configs(plan, family, 3, 47):
            a, b = full.score(p), short.score(p)
            pd.testing.assert_frame_equal(a.loc[:cut], b)
            assert a.where(~eligible).isna().all().all()


def test_direction_known_return_and_turnover():
    wide, eligible, calendar = synthetic()
    model = RuleSignals(wide, eligible, calendar)
    p = dict(family='S05', variant='absolute_reversal', window=5)
    actual = model.score(p)
    expected = -(wide['hfq_close'] / wide['hfq_close'].shift(5) - 1).reindex_like(eligible).where(eligible)
    pd.testing.assert_frame_equal(actual, expected)
    wide['turnover_rate'].loc[:, 'A'] = 1
    wide['turnover_rate'].loc[:, 'B'] = 9
    score = RuleSignals(wide, eligible, calendar).score(dict(family='S01', window=20))
    assert score.iloc[-1].A > score.iloc[-1].B


def test_partial_week_or_month_is_not_new_rebalance():
    calendar = pd.bdate_range('2023-01-02', '2023-03-31')
    for freq in ['weekly', 'monthly']:
        full = decision_dates(calendar, freq, calendar)
        cut = calendar[:13]
        short = decision_dates(cut, freq, calendar)
        assert short.equals(full[full <= cut[-1]])


def test_sampler_covers_noncoarse_values_and_joint_categories():
    plan = yaml.safe_load((ROOT / 'configs/rule_strategy_search_v1.yaml').read_text())
    rows = sample_configs(plan, 'S06', 128, 19)
    assert rows == sample_configs(plan, 'S06', 128, 19)
    assert len({config_id(x) for x in rows}) == 128
    assert all(legal(x) for x in rows)
    assert len({x['frequency'] for x in rows}) == 3
    assert len({x['top_n'] for x in rows}) == 3
    assert any(x['lookback'] not in plan['family_spaces']['S06']['lookback'] for x in rows)
    local = local_configs(plan, 'S06', rows[:4], 32, 20, [config_id(x) for x in rows])
    assert len(local) == 32 and all(legal(x) for x in local)
    assert not {config_id(x) for x in local} & {config_id(x) for x in rows}
    assert sample_configs(plan, 'S06', 0, 19) == []


def test_no_feasible_is_not_relaxed_or_outer_ranked():
    row = dict(id='x', status='ok', params=dict(family='S01', window=20, frequency='monthly', top_n=5),
               metrics=dict(feasible=False, median_ann=10., turnover=1.))
    assert select_trial([row]) is None


def test_fixed_schedule_exactly_matches_original_account():
    f = fixture_module.PortfolioTest()
    f.setUp()
    f.signals = pd.DataFrame([[1., 2.], [3., 1.]], index=f.dates[[0, 10]], columns=f.codes)
    old = f.run_case()
    new = run_backtest(f.signals, f.market, f.basics, f.st, f.limits, f.actions,
                       f.dates[0], f.dates[-1], f.cfg, top_n_schedule=pd.Series(1, index=f.signals.index))
    for key in ['daily', 'trades', 'events', 'holdings']:
        pd.testing.assert_frame_equal(getattr(old, key), getattr(new, key))


def test_changed_schedule_carries_cash_and_positions():
    f = fixture_module.PortfolioTest()
    f.setUp()
    signals = pd.DataFrame([[2., 1.], [2., 1.]], index=f.dates[[0, 10]], columns=f.codes)
    result = run_backtest(signals, f.market, f.basics, f.st, f.limits, f.actions,
                          f.dates[0], f.dates[-1], f.cfg, top_n_schedule=pd.Series([2, 1], index=signals.index))
    assert result.daily.loc[f.dates[1], 'n_hold'] == 2
    assert result.daily.loc[f.dates[11], 'n_hold'] == 1
    assert result.daily.equity.eq(10000).all()
    assert result.daily.iloc[-1].cash == 5000
    assert result.holdings.iloc[-1].shares == 500
    with pytest.raises(ValueError):
        run_backtest(signals, f.market, f.basics, f.st, f.limits, f.actions,
                     f.dates[0], f.dates[-1], f.cfg, top_n_schedule=pd.Series([2, 1.5], index=signals.index))
