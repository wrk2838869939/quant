"""续训必须保留原随机流、优化更新及账本语义。"""
from pathlib import Path
import sys
import copy
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from policy_evolution import candidates, update, plateau, restore_rng
from policy_portfolio_cached import run_backtest as cached, PreparedMarket
from policy_portfolio import run_backtest as original
from initialized_policy import PolicyAction
from test_portfolio import PortfolioTest


def test_random_stream_continues_after_checkpoint():
    cfg=dict(population=12,scales=[.15,.6,1.8])
    rng=np.random.default_rng(99)
    z=np.zeros(100)
    for _ in range(8): candidates(z,z,np.ones(100),.4,rng,cfg)
    other=restore_rng(rng.bit_generator.state)
    a=candidates(z,z,np.ones(100),.4,rng,cfg)
    b=candidates(z,z,np.ones(100),.4,other,cfg)
    for x,y in zip(a,b): np.testing.assert_array_equal(x[0],y[0])


def row(value,feasible=True):
    return dict(metrics=dict(total=value,feasible=feasible,mdd=-.1,worst_block_mdd=-.1,turnover=1.))


def test_plateau_requires_three_complete_flat_blocks_and_feasibility():
    cfg=dict(block_generations=8,required_flat_blocks=3,min_log_wealth_improvement=.005)
    archive=[row(.1)]*65
    assert plateau(archive,64,cfg)['plateau']
    assert not plateau(archive,16,cfg)['plateau']
    modified=copy.deepcopy(archive);modified[64]=row(.2)
    assert not plateau(modified,64,cfg)['plateau']
    assert not plateau([row(.1,False)]*65,64,cfg)['plateau']


@pytest.mark.parametrize('case',['normal','dividend','bonus','deferred','partial','cash','empty'])
def test_cached_rollout_exact(case):
    f=PortfolioTest();f.setUp()
    if case=='dividend': f.action(cash=1.)
    if case=='bonus': f.action(cash=.2,bonus=.2)
    if case=='deferred': f.limits.loc[f.limits.date==f.dates[3],'down_limit']=10.
    def callback(s):
        if case=='empty': return None
        if s['date']==f.dates[0]:
            return PolicyAction(f.signals.iloc[0],1,.6 if case=='cash' else 1.)
        if s['date']==f.dates[2] and case in ['deferred','partial']:
            return PolicyAction(f.signals.iloc[0],1,reduce={'000001.SZ':.4})
    args=[f.signals.iloc[:0].copy(),f.market,f.basics,f.st,f.limits,f.actions,f.dates[0],f.dates[-1],f.cfg]
    a=original(*args,policy=callback)
    prep=PreparedMarket(f.market,f.basics,f.st,f.limits,f.dates[0],f.dates[-1])
    for _ in range(2):
        b=cached(*args,policy=callback,prepared=prep)
        for table in ['daily','trades','events','holdings']:
            pd.testing.assert_frame_equal(getattr(a,table),getattr(b,table),check_exact=True)
        assert a.summary==b.summary or case=='empty'
