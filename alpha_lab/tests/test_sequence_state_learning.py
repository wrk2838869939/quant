"""序列窗口、在线状态过滤、成熟标签和双模型重放的独立核验。"""
from copy import deepcopy
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pytest,yaml
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_rule_search import synthetic
from learning_search import build_features,build_labels,prepare_arrays
from learning_extensions import attach_context
from sequence_state_learning import sequence_allowed,sequence_rows,filtered_probabilities,predict_walk_forward,infer_bundle
PLAN=yaml.safe_load((ROOT/'configs/sequence_state_learning_v1.yaml').read_text())


def test_window_alignment_and_missing_history():
    x=np.arange(8*2*3,dtype=np.float32).reshape(8,2,3)
    rows=sequence_rows({'x':x},np.array([3,6]),np.array([0,1]),3)
    np.testing.assert_array_equal(rows[0],x[1:4,0].T)
    np.testing.assert_array_equal(rows[1],x[4:7,1].T)
    allowed=np.ones((8,2),bool);allowed[3,0]=False
    actual=sequence_allowed(allowed,3)
    assert not actual[:2].any()
    assert not actual[3:6,0].any() and actual[6:,0].all() and actual[2:,1].all()


def test_filter_matches_bayes_and_does_not_smooth():
    from hmmlearn.hmm import GaussianHMM
    model=GaussianHMM(2,covariance_type='diag');model.n_features=1
    model.startprob_=np.array([.6,.4]);model.transmat_=np.array([[.9,.1],[.2,.8]])
    model.means_=np.array([[-1.],[1.]]);model.covars_=np.array([[1.],[1.]])
    x=np.array([[-.5],[.2],[1.3]])
    result=filtered_probabilities(model,x,1e-12)
    state=model.startprob_.copy()
    for at,value in enumerate(x[:,0]):
        if at:state=state@model.transmat_
        state=state*norm.pdf(value,[-1.,1.],1.)
        state/=state.sum()
        np.testing.assert_allclose(result[at],state,atol=1e-14)
    np.testing.assert_array_equal(filtered_probabilities(model,np.vstack([x,[[999.]]]),1e-12)[:3],result)


@pytest.fixture(scope='module')
def prepared():
    wide,eligible,calendar=synthetic()
    cfg=dict(PLAN['training'],min_dates=30,min_rows=80,min_features=3,min_ic_stocks=3,expert_min_rows=30,batch_size=128,hmm_iterations=12)
    f=build_features(wide,eligible,calendar,PLAN['features'])
    data=prepare_arrays(f,eligible,build_labels(wide,eligible.index,[5,20]),cfg)
    attach_context(data,wide,eligible)
    return data,cfg


@pytest.mark.parametrize('family',['cnn','hmm'])
def test_future_poison_and_saved_pair_replay(prepared,family,tmp_path):
    data,cfg=prepared;variant=next(iter(PLAN['families'][family]))
    axes={**PLAN['common_axes'],**PLAN['families'][family][variant]}
    p=dict(family=family,variant=variant,**{k:v[0] for k,v in axes.items()})
    cut,start=data['dates'][160],data['dates'][100]
    a,b,fits=predict_walk_forward(data,p,cfg,17,start,cut,tmp_path)
    altered=deepcopy(data)
    for key in ['x','ranks','context','stock_returns']:altered[key][161:]=999
    for label in altered['labels'].values():
        unknown=pd.isna(label['end']) | (label['end']>=np.datetime64(cut))
        for key in ['returns','risk','ic']:label[key][unknown]=999
    aa,bb,_=predict_walk_forward(altered,p,cfg,17,start,cut)
    pd.testing.assert_frame_equal(a,aa);pd.testing.assert_frame_equal(b,bb)
    assert a.notna().any().any() and b.notna().any().any()
    for record in fits:
        if record['status']!='ok':continue
        assert pd.Timestamp(record['max_label_end'])<pd.Timestamp(record['fit_date'])
        bundle=joblib.load(tmp_path/record['model_file'])
        date=pd.Timestamp(record['fit_date']);at=data['dates'].get_loc(date)
        x,y=infer_bundle(bundle,data,at)
        np.testing.assert_array_equal(x,a.loc[date]);np.testing.assert_array_equal(y,b.loc[date])
