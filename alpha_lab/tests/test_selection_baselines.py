"""基线年度锁定、候选资格约束与未来信息隔离的契约测试。"""
from copy import deepcopy
from pathlib import Path
import sys
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_rule_search import synthetic
from rule_search import config_id
from selection_baselines import build_signals,validate_entry,validate_candidate
from selection_baseline_plugins import identity


def example():
    wide,eligible,calendar=synthetic()
    entry=dict(source_family='S01',annual_parameters={})
    for year in sorted(set(eligible.index.year)):
        p=dict(family='S01',variant='low_mean_turnover',window=5+(year%2)*4,frequency='weekly',top_n=3)
        entry['annual_parameters'][year]=dict(params=p,selection_id=config_id(p),inner_end=f'{year-1}-12-31')
    return entry,wide,eligible,calendar


def test_no_implicit_future_parameters_or_changed_lock():
    entry,_,eligible,_=example();years=sorted(set(eligible.index.year))
    validate_entry(entry,years)
    with pytest.raises(ValueError,match='年度锁定'):validate_entry(entry,years+[years[-1]+1])
    altered=deepcopy(entry);altered['annual_parameters'][years[0]]['params']['window']+=1
    with pytest.raises(ValueError,match='锁定ID'):validate_entry(altered,years)
    altered=deepcopy(entry);altered['annual_parameters'][years[0]]['inner_end']=f'{years[0]}-01-01'
    with pytest.raises(ValueError,match='选择窗口'):validate_entry(altered,years)


def test_identity_and_truncated_prefix_match():
    entry,wide,eligible,calendar=example();start,end=eligible.index[[0,-1]]
    a,n=build_signals(entry,wide,eligible,calendar,start,end)
    b,m=build_signals(entry,wide,eligible,calendar,start,end,identity,{})
    pd.testing.assert_frame_equal(a,b);pd.testing.assert_series_equal(n,m)
    cut=eligible.index[171]
    c,k=build_signals(entry,{x:v.loc[:cut] for x,v in wide.items()},eligible.loc[:cut],calendar,start,cut,identity,{})
    pd.testing.assert_frame_equal(c,a.loc[:cut]);pd.testing.assert_series_equal(k,n.loc[:cut])


def test_future_centering_is_detected_by_prefix_check():
    entry,wide,eligible,calendar=example();start,end=eligible.index[[0,-1]];cut=eligible.index[91]
    def future_mean(scores,**kwargs):return scores-scores.mean(axis=0)
    full,_=build_signals(entry,wide,eligible,calendar,start,end,future_mean,{})
    prefix,_=build_signals(entry,{k:v.loc[:cut] for k,v in wide.items()},eligible.loc[:cut],calendar,start,cut,future_mean,{})
    with pytest.raises(AssertionError):pd.testing.assert_frame_equal(prefix,full.loc[:cut])


def test_ineligible_scores_and_overwriting_baseline_rejected():
    entry,wide,eligible,calendar=example();start,end=eligible.index[[0,-1]]
    def fill(scores,**kwargs):return scores.fillna(1.)
    with pytest.raises(ValueError,match='不合格'):build_signals(entry,wide,eligible,calendar,start,end,fill,{})
    config=dict(candidate_id='B01',parent_baseline='B01',plugin_module='selection_baseline_plugins',plugin_function='identity')
    with pytest.raises(ValueError,match='覆盖'):validate_candidate(config,{'baselines':{'B01':entry}})
