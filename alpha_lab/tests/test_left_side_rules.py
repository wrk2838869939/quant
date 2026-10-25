"""验证时序、历史归属冲突、量价顺序与不可撤销退出。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from left_side_rules import membership_matrix, capped_weights, stock_shape, sector_states, LeftSidePolicy

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/left_side_rules_v1.yaml').read_text())


def test_membership_intervals_and_conflicts():
    dates = pd.date_range('2020-01-01', periods=5)
    catalog = pd.DataFrame(dict(index_code=['a', 'b'], src=['SW2014', 'SW2014']))
    members = pd.DataFrame([
        ['a', 'x', '20200101', '20200103'],
        ['b', 'x', '20200103', None],
        ['a', 'y', '20200101', None],
        ['b', 'y', '20200102', '20200104'],
    ], columns=['index_code', 'con_code', 'in_date', 'out_date'])
    result, _ = membership_matrix(members, catalog, dates, ['x', 'y'], CFG['membership'])
    assert result[:, 0].tolist() == [0, 0, 1, 1, 1]
    assert result[:, 1].tolist() == [0, -2, -2, 0, 0]
    prefix, _ = membership_matrix(members, catalog, dates[:3], ['x', 'y'], CFG['membership'])
    np.testing.assert_array_equal(prefix, result[:3])


def test_capped_weights():
    w = capped_weights(np.array([10000., 2., 3., 4., 5., 6.]), .2)
    assert abs(w.sum() - 1) < 1e-12
    assert w.max() <= .2 + 1e-12


def test_shape_requires_correct_sequence_and_accepts_new_low():
    dates = pd.bdate_range('2020-01-01', periods=30)
    close = np.r_[np.full(25, 100.), [98., 95., 92., 91.8, 91.7]]
    turn = np.r_[np.full(25, 1.), [2., 2., 2., .7, .5]]
    wide = {'hfq_close': pd.DataFrame({'x': close}, index=dates),
            'turnover_rate': pd.DataFrame({'x': turn}, index=dates)}
    params = {k: CFG['stock'][k] for k in ['shape_window', 'turnover_expansion', 'turnover_contraction']}
    good = stock_shape(wide, CFG['stock'], params)
    assert good['shape'].iloc[-1, 0]
    reversed_wide = {k: v.copy() for k, v in wide.items()}
    reversed_wide['turnover_rate'].iloc[-5:, 0] = [.5, .7, 2., 2., 2.]
    assert not stock_shape(reversed_wide, CFG['stock'], params)['shape'].iloc[-1, 0]
    prefix = stock_shape({k: v.iloc[:-1] for k, v in wide.items()}, CFG['stock'], params)
    for key in prefix:
        pd.testing.assert_frame_equal(prefix[key], good[key].iloc[:-1], check_exact=True)


def test_sector_anchor_does_not_change_after_future_shock():
    price = np.r_[np.tile([100., 101.], 12), np.linspace(105, 140, 20), np.linspace(138, 110, 10), [109., 108.8, 108.7]]
    dates = pd.bdate_range('2020-01-01', periods=len(price))
    frame = pd.DataFrame(dict(date=dates, sector='a', sector_id=0, close=price))
    full = sector_states(frame, CFG['sector'])
    short = sector_states(frame.iloc[:-3], CFG['sector'])
    pd.testing.assert_frame_equal(short, full.iloc[:-3], check_exact=True)
    assert full.rallied.any()
    assert (full.loc[full.anchor_i.ge(0), 'anchor_date'] < full.loc[full.anchor_i.ge(0), 'date']).all()


def test_exit_is_not_revoked_and_trailing_never_lowers():
    dates = pd.bdate_range('2020-01-01', periods=10)
    close = pd.DataFrame({'x': [100.] * 6 + [110., 111., 90., 115.]}, index=dates)
    wide = {'hfq_close': close, 'close': close, 'hfq_low': close - 1}
    scores = pd.DataFrame({'x': [np.nan] * 10}, index=dates)
    policy = LeftSidePolicy(scores, wide, CFG)
    stops = []
    for i in range(5, 10):
        trades = [dict(side='buy', ts_code='x', signal_date=dates[4], price=100.)] if i == 5 else []
        a = policy(dict(date=dates[i], positions={'x': {}}, trades=trades, pending_sell={}))
        stops.append(policy.state['x']['stop'])
        if i >= 8:
            assert 'x' in a.exit
    assert np.diff(stops).min() >= 0
