"""按已知趋势阶段分治；局部模型输出统一的收益量纲，不分配组别名额。"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from value_forecast import fit_value_model


def phase_features(frame, features):
    """从两个相邻60日区间定义阶段；不含任何未来路径标签。"""
    out=frame.copy()
    if not np.isfinite(out[['mom60','mom120']].to_numpy()).all() or (out.mom60<=-1).any():
        raise ValueError('阶段需要有效的已知60/120日收益')
    previous=(1+out.mom120)/(1+out.mom60)-1
    out['phase']=np.select([previous.le(0),out.mom60.gt(previous)],
                          ['recovery','accelerating'],default='decelerating')
    # 修复期作参考；交互项共同正则化，保留全局共享主效应。
    extra=[]
    for phase in ['accelerating','decelerating']:
        key='phase_'+phase
        out[key]=out.phase.eq(phase).astype(float);extra.append(key)
        for feature in features:
            col=key+'__'+feature;out[col]=out[key]*out[feature];extra.append(col)
    return out,list(features)+extra


@dataclass
class DividedModels:
    global_model: object
    shared_model: object
    local_models: dict
    constant_model: object
    features: tuple
    metadata: dict

    def predict(self, frame):
        x,_=phase_features(frame,self.features)
        preds={'global_return':self.global_model.predict(x),
               'shared_experts':self.shared_model.predict(x),
               'constant':self.constant_model.predict(x)}
        local=[]
        for phase,g in x.groupby('phase',sort=True):
            model=self.local_models.get(phase,self.global_model)
            local.append(model.predict(g))
        preds['phase_experts']=pd.concat(local).reindex(x.index)
        result=x[['date','ts_code','phase','trend']].copy()
        for name,pred in preds.items():
            for col in pred:result[name+'__'+col]=pred[col]
        return result


def fit_divided(frame,cfg,asof):
    """全局、完全分开和共享收缩三种固定模型；稀疏组回退全局。"""
    x,shared=phase_features(frame,cfg['features'])
    opts=dict(cfg['forecast']);end=pd.Timestamp(asof)
    eligible=x.label_end_60.le(end)&x.entry_missing_60.eq(False)&x.ret_60.notna()&x.mae_60.notna()
    global_model=fit_value_model(x,cfg['features'],asof=end,**opts)
    shared_model=fit_value_model(x,shared,asof=end,**opts)
    constant=fit_value_model(x,[],asof=end,**opts)
    locals={};metadata={}
    for phase in cfg['phases']:
        group=x[x.phase.eq(phase)];valid=group.loc[eligible.reindex(group.index)]
        counts=valid.groupby('date').size()
        enough=len(counts)>=opts['min_dates']
        if enough:locals[phase]=fit_value_model(group,cfg['features'],asof=end,**opts)
        metadata[phase]=dict(rows=len(valid),dates=len(counts),minimum_names_per_date=int(counts.min()) if len(counts) else 0,
                             fallback=not enough,model=locals[phase].metadata if enough else None)
    return DividedModels(global_model,shared_model,locals,constant,tuple(cfg['features']),
        dict(asof=str(end),local=metadata,global_model=global_model.metadata,
             shared_model=shared_model.metadata,constant=constant.metadata))


def policy_predictions(frame,models,cfg):
    """规则与估值各司其职：规则不是收益，所有策略经过同一新仓收益/风险门槛。"""
    preds=models.predict(frame)
    for policy in cfg['policies']:
        model='constant' if policy in ['low_turnover','continuous_trend'] else policy
        mu=preds[model+'__expected_return'];adverse=preds[model+'__expected_adverse']
        if policy=='low_turnover':score=-frame.x_turnover
        elif policy=='continuous_trend':score=sum(frame[key]*weight for key,weight in cfg['continuous_score'].items())
        else:score=mu
        gate=frame.trend&mu.gt(cfg['entry']['min_expected_return'])&adverse.le(cfg['entry']['max_expected_adverse'])
        preds[policy+'__score']=score.where(gate)
        preds[policy+'__expected_return']=mu
    return preds
