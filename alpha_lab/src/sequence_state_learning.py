"""CNN序列与HMM在线状态学习；每次拟合附带同参数点模型/无转移状态对照。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from learning_search import training_mask


def sequence_allowed(allowed, lookback):
    """要求整个回看窗都有足够可得特征，主模型与点模型使用相同样本。"""
    result=allowed.copy()
    result[:lookback-1]=False
    for lag in range(1,lookback):
        result[lag:] &= allowed[:-lag]
    return result


def sequence_rows(data, times, stocks, lookback):
    offsets=np.arange(lookback-1,-1,-1)
    return np.ascontiguousarray(data['x'][times[:,None]-offsets,stocks[:,None]].transpose(0,2,1))


def filtered_probabilities(model, observations, floor):
    """逐日前向过滤，不调用包含未来观测的后向平滑或Viterbi全序列解码。"""
    variance=np.maximum(np.diagonal(model.covars_,axis1=1,axis2=2),floor)
    log_emission=-.5*(np.log(2*np.pi*variance).sum(axis=1)[None,:]
                     +((observations[:,None,:]-model.means_[None,:,:])**2/variance[None,:,:]).sum(axis=2))
    transition=np.log(np.maximum(model.transmat_,floor))
    posterior=np.empty((len(observations),len(model.startprob_)))
    state=np.log(np.maximum(model.startprob_,floor))
    for at,emission in enumerate(log_emission):
        if at:
            state=logsumexp(state[:,None]+transition,axis=0)
        state=state+emission
        state-=logsumexp(state)
        posterior[at]=np.exp(state)
    return posterior


def torch_network(features,p,paired=False):
    import torch.nn as nn
    if paired:
        return nn.Sequential(nn.Linear(features,p['width']),nn.ReLU(),nn.Linear(p['width'],1))
    return nn.Sequential(nn.Conv1d(features,p['width'],p['kernel'],padding=p['kernel']//2),nn.ReLU(),
                         nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(p['width'],1))


def fit_torch(x,y,weights,p,cfg,seed,paired):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=cfg['mkldnn']
    torch.manual_seed(seed)
    model=torch_network(x.shape[1],p,paired)
    inputs=torch.from_numpy(np.ascontiguousarray(x[:,:,-1] if paired else x))
    target=torch.as_tensor(y*cfg['target_scale'],dtype=torch.float32)
    weight=torch.as_tensor(weights,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    rng=np.random.default_rng(seed)
    losses=[]
    for _ in range(p['epochs']):
        total=0.;denominator=0.
        order=rng.permutation(len(x))
        for start in range(0,len(order),cfg['batch_size']):
            batch=order[start:start+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            prediction=model(inputs[batch]).flatten()
            loss=((prediction-target[batch])**2*weight[batch]).mean()
            assert torch.isfinite(loss),'神经网络训练出现非有限损失'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            optimizer.step()
            total+=loss.item()*len(batch);denominator+=len(batch)
        losses.append(total/denominator)
    model.eval()
    return model,losses


def fit_bundle(data,p,cfg,seed,at):
    dates=data['dates'];date=dates[at]
    mask=training_mask(data,p,date,cfg)
    if p['family']=='cnn':
        mask &= sequence_allowed(data['allowed'],p['lookback'])
    used=np.flatnonzero(mask.any(axis=1))
    record=dict(fit_date=str(date.date()),rows=int(mask.sum()),dates=len(used),params=p)
    if len(used)<cfg['min_dates'] or mask.sum()<cfg['min_rows']:
        record['status']='insufficient_training';return None,record
    latest=pd.Timestamp(data['labels'][p['horizon']]['end'][used].max());assert latest<date
    record.update(status='ok',first_signal_date=str(dates[used[0]].date()),last_signal_date=str(dates[used[-1]].date()),
                  max_label_end=str(latest.date()),training_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    bundle=dict(params=p,target_scale=cfg['target_scale'],mkldnn=cfg['mkldnn'])
    weights=np.broadcast_to(1/np.maximum(mask.sum(axis=1),1)[:,None],mask.shape)[mask]
    weights*=len(weights)/weights.sum()
    y=data['labels'][p['horizon']]['returns'][mask]
    if p['family']=='cnn':
        times,stocks=np.nonzero(mask)
        x=sequence_rows(data,times,stocks,p['lookback'])
        bundle['model'],record['training_losses']=fit_torch(x,y,weights,p,cfg,seed,False)
        bundle['paired'],record['paired_training_losses']=fit_torch(x,y,weights,p,cfg,seed,True)
    elif p['family']=='hmm':
        start=max(0,at-p['train_window'])
        scaler=StandardScaler().fit(data['context'][start:at])
        z=scaler.transform(data['context'][start:at])
        prior=np.ones((p['states'],p['states']))+(p['persistence']-1)*np.eye(p['states'])
        hmm=GaussianHMM(n_components=p['states'],covariance_type='diag',n_iter=cfg['hmm_iterations'],
                        tol=cfg['hmm_tolerance'],min_covar=cfg['hmm_covariance_floor'],
                        transmat_prior=prior,random_state=seed).fit(z)
        assert np.isfinite(hmm.means_).all() and np.isfinite(hmm.covars_).all()
        assert np.allclose(hmm.transmat_.sum(axis=1),1)
        prob=filtered_probabilities(hmm,z,cfg['probability_floor'])
        kmeans=KMeans(n_clusters=p['states'],n_init=cfg['kmeans_n_init'],random_state=seed).fit(z)
        groups=kmeans.predict(z)
        x=data['x'][mask];row_dates=np.nonzero(mask)[0]-start
        pooled=Ridge(alpha=p['alpha']).fit(x,y,sample_weight=weights)
        experts={};paired_experts={}
        for group in range(p['states']):
            mass=prob[row_dates,group]
            if mass.sum()>=cfg['expert_min_rows']:
                experts[group]=Ridge(alpha=p['alpha']).fit(x,y,sample_weight=weights*mass)
            selected=groups[row_dates]==group
            if selected.sum()>=cfg['expert_min_rows']:
                paired_experts[group]=Ridge(alpha=p['alpha']).fit(x[selected],y[selected],sample_weight=weights[selected])
        bundle.update(hmm=hmm,kmeans=kmeans,scaler=scaler,filter_start=start,floor=cfg['probability_floor'],
                      model=pooled,experts=experts,paired_experts=paired_experts)
        history=list(hmm.monitor_.history)
        record.update(em_iterations=hmm.monitor_.iter,em_likelihood_last=history[-1],
                      em_delta=history[-1]-history[-2] if len(history)>1 else None,
                      em_budget_exhausted=hmm.monitor_.iter>=cfg['hmm_iterations'],
                      observations=len(z),last_unsupervised_date=str(dates[at-1].date()),
                      transition_matrix=hmm.transmat_.tolist(),expert_count=len(experts),paired_expert_count=len(paired_experts))
    else:raise ValueError(p['family'])
    return bundle,record


def infer_bundle(bundle,data,at,posterior=None):
    p=bundle['params']; valid=data['allowed'][at].copy()
    prediction=np.full(len(valid),np.nan);paired=prediction.copy()
    if p['family']=='cnn':
        if at<p['lookback']-1:return prediction,paired
        valid &= data['allowed'][at-p['lookback']+1:at+1].all(axis=0)
    if not valid.any():return prediction,paired
    if p['family']=='cnn':
        import torch
        torch.set_num_threads(1);torch.backends.mkldnn.enabled=bundle['mkldnn']
        stocks=np.flatnonzero(valid)
        x=sequence_rows(data,np.full(len(stocks),at),stocks,p['lookback'])
        with torch.no_grad():
            prediction[valid]=bundle['model'](torch.from_numpy(x)).flatten().numpy()/bundle['target_scale']
            paired[valid]=bundle['paired'](torch.from_numpy(np.ascontiguousarray(x[:,:,-1]))).flatten().numpy()/bundle['target_scale']
    else:
        if posterior is None:
            z=bundle['scaler'].transform(data['context'][bundle['filter_start']:at+1])
            posterior=filtered_probabilities(bundle['hmm'],z,bundle['floor'])[-1]
        x=data['x'][at,valid];pool=bundle['model'].predict(x)
        local=np.zeros_like(pool)
        for group,mass in enumerate(posterior):
            local+=mass*bundle['experts'].get(group,bundle['model']).predict(x)
        current=bundle['scaler'].transform(data['context'][at:at+1])
        group=int(bundle['kmeans'].predict(current)[0])
        control=bundle['paired_experts'].get(group,bundle['model']).predict(x)
        prediction[valid]=(1-p['expert_weight'])*pool+p['expert_weight']*local
        paired[valid]=(1-p['expert_weight'])*pool+p['expert_weight']*control
    assert np.isfinite(prediction[valid]).all() and np.isfinite(paired[valid]).all()
    return prediction,paired


def score_predictions(pred,paired,params,remove_risk=False):
    return pred


def predict_walk_forward(data,params,cfg,seed,start,end,folder=None):
    dates=data['dates'];evaluation=np.flatnonzero((dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end)))
    prediction=np.full(data['allowed'].shape,np.nan);paired=prediction.copy();fits=[]
    blocks=(dates.year*12+dates.month-1)//params['retrain_months'];previous=None;bundle=None;posterior=None
    for at in evaluation:
        if previous!=blocks[at]:
            previous=blocks[at]
            bundle,record=fit_bundle(data,params,cfg,seed,at)
            if bundle is not None:
                if folder:
                    path=Path(folder);path.mkdir(parents=True,exist_ok=True)
                    artifact=path/f'model_{dates[at]:%Y%m%d}.joblib'
                    joblib.dump(bundle,artifact)
                    record.update(model_file=artifact.name,model_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
                if params['family']=='hmm':
                    # 前向递推的各日前缀彼此独立于后缀；仅缓存计算结果，不写入模型文件。
                    z=bundle['scaler'].transform(data['context'][bundle['filter_start']:evaluation[-1]+1])
                    posterior=filtered_probabilities(bundle['hmm'],z,bundle['floor'])
            fits.append(record)
        if bundle is not None:
            prob=posterior[at-bundle['filter_start']] if params['family']=='hmm' else None
            prediction[at],paired[at]=infer_bundle(bundle,data,at,prob)
    return (pd.DataFrame(prediction[evaluation],index=dates[evaluation],columns=data['codes']),
            pd.DataFrame(paired[evaluation],index=dates[evaluation],columns=data['codes']),fits)
