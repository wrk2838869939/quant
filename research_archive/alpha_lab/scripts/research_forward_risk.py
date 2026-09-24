# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
"""先固定历史顺序检验风险预测和现金效用，再决定是否值得交易回放。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from forward_risk_evidence import FAMILIES,market_features,forward_outcomes,matured_weekly_rows


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str))


def evaluate(rows,prob,alarm,labels,prior):
    y=labels.loc[rows,'loss_event'].to_numpy()
    p=prob.reindex(rows).to_numpy();a=alarm.reindex(rows).to_numpy(dtype=bool)
    benefit=labels.loc[rows,'cash_benefit_proxy'].to_numpy()
    terminal=labels.loc[rows,'terminal'].to_numpy()
    rng=np.random.default_rng(1701);boot=[]
    n=len(rows)
    for _ in range(1000):
        starts=rng.integers(0,n,size=int(np.ceil(n/4)))
        ix=np.concatenate([(np.arange(4)+start)%n for start in starts])[:n]
        if a[ix].any():boot.append(float(benefit[ix][a[ix]].mean()))
    base=float(y.mean());precision=float(y[a].mean()) if a.any() else np.nan
    return dict(n=n,alarms=int(a.sum()),alarm_fraction=float(a.mean()),base_loss_rate=base,
        alarm_loss_rate=precision,precision_lift=precision/base if base else np.nan,
        recall=float(y[a].sum()/y.sum()) if y.sum() else np.nan,
        auc=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
        brier=float(brier_score_loss(y,p)),prior_brier=float(brier_score_loss(y,np.repeat(prior,n))),
        alarm_terminal_return=float(terminal[a].mean()) if a.any() else np.nan,
        mean_cash_benefit=float(benefit[a].mean()) if a.any() else np.nan,
        cash_benefit_p05=float(np.quantile(boot,.05)) if boot else np.nan,
        mean_worst_return=float(labels.loc[rows[a],'worst'].mean()) if a.any() else np.nan)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path)
    args=p.parse_args();run=args.run or ROOT/'outputs/forward_risk'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True,exist_ok=False)
    panel=ROOT/'data/panel/daily_panel.parquet';idx_path=ROOT/'data/raw/index_daily.parquet'
    repair=ROOT/'outputs/forward_risk/repair_20211213'
    parent=ROOT/'outputs/cash_robustness/20260914T102042_513326Z'
    for name,value in json.loads((parent/'completion.json').read_text())['artifact_sha256'].items():
        assert digest(parent/name)==value,name
    splits=dict(train=('2014-01-01','2018-12-31'),validation=('2019-01-01','2021-12-31'),observed_late=('2022-01-01','2026-09-08'))
    protocol=dict(parent=str(parent),parent_completion_sha256=digest(parent/'completion.json'),
        input_sha256={str(panel):digest(panel),str(idx_path):digest(idx_path),
            str(repair/'projection.parquet'):digest(repair/'projection.parquet'),
            str(repair/'provenance.json'):digest(repair/'provenance.json')},
        source_sha256={name:digest(ROOT/name) for name in ['src/forward_risk_evidence.py','scripts/research_forward_risk.py']},
        root_agents_sha256=digest(ROOT.parent/'AGENTS.md'),splits=splits,families=FAMILIES,
        target='CSI300未来20个交易日收盘相对次日开盘最低收益<=-5%；固定20日持有收益以第21日开盘计。',
        selection='仅2019—2021选择；2022+已观察历史不能翻盘验证输家。',
        models='LogisticRegression C=1、L2、无类别重加权；标准化只拟合TRAIN。joint_utility附加Ridge alpha=10预测固定期限收益。',
        alarms='TRAIN拟合概率80分位阈值冻结；joint_utility另要求预测收益<-40bp。',
        validation_gate=dict(min_alarms=12,min_auc=.55,min_precision_lift=1.25,max_alarm_fraction=.35,
            brier_better_than_train_prior=True,mean_cash_benefit_positive=True,block_bootstrap_p05_positive=True),
        uncertainty='周首样本，标签重叠；4周循环块bootstrap1000次，只作不确定性诊断，不称多重检验校正显著性。',
        comparison_cost_roundtrip=.004,independent_oos=False,live_orders=False)
    save(run/'protocol.json',protocol);print(run,flush=True)
    raw=pd.read_parquet(panel,columns=['date','ts_code','hfq_close','amount_yuan','close','adj_factor','amount'])
    patch=pd.read_parquet(repair/'projection.parquet')
    assert not raw.date.eq(pd.Timestamp('2021-12-13')).any(),'修复仅适用明确整日缺口'
    raw=pd.concat([raw,patch],ignore_index=True)
    assert not raw.duplicated(['date','ts_code']).any()
    price_mask=raw[['hfq_close','close','adj_factor']].notna().all(axis=1)
    amount_mask=raw[['amount_yuan','amount']].notna().all(axis=1)
    assert np.allclose(raw.loc[price_mask,'hfq_close'],raw.loc[price_mask,'close']*raw.loc[price_mask,'adj_factor'],rtol=1e-7,atol=1e-7)
    assert np.allclose(raw.loc[amount_mask,'amount_yuan'],raw.loc[amount_mask,'amount']*1000,rtol=1e-7,atol=1e-5)
    assert raw.loc[price_mask,'hfq_close'].gt(0).all()
    idx=pd.read_parquet(idx_path);idx=idx.loc[idx.ts_code.eq('000300.SH')].copy()
    idx['date']=pd.to_datetime(idx.trade_date);idx=idx.set_index('date').sort_index()
    assert not idx.index.has_duplicates
    missing_sessions=idx.index.difference(pd.DatetimeIndex(raw.date.unique()))
    assert len(missing_sessions)==0,f'全市场整日缺失，禁止继续：{missing_sessions}'
    close=raw.pivot(index='date',columns='ts_code',values='hfq_close').sort_index().reindex(idx.index)
    amount=raw.pivot(index='date',columns='ts_code',values='amount_yuan').sort_index().reindex(idx.index)
    save(run/'data_audit.json',dict(rows=len(raw),codes=raw.ts_code.nunique(),dates=len(idx),
        source_start=str(raw.date.min()),source_end=str(raw.date.max()),
        price_unit_rows=int(price_mask.sum()),amount_unit_rows=int(amount_mask.sum()),
        current_constituents_used=False,repaired_rows=len(patch),missing_market_sessions=0,
        eligibility='至少120个历史报价、最近60日价格完整；当日可知筛选，含退市股票历史。'))
    del raw
    features=market_features(close,amount,idx.close)
    features.to_parquet(run/'features.parquet');idx.to_parquet(run/'index.parquet')
    print('features',features.shape,'coverage',features.price_coverage.describe().to_dict(),flush=True)
    # 全市场真实前缀按相同原始输入重算，不仅检查合成样本。
    prefixes=[]
    for cutoff in [pd.Timestamp('2018-12-28'),pd.Timestamp('2021-12-31')]:
        actual=market_features(close.loc[:cutoff],amount.loc[:cutoff],idx.close.loc[:cutoff])
        pd.testing.assert_frame_equal(actual,features.loc[:cutoff],check_exact=True)
        prefixes.append(str(cutoff.date()))
        print('feature prefix',cutoff.date(),flush=True)
    del close,amount
    labels=forward_outcomes(idx.open,idx.close)
    labels.to_parquet(run/'labels.parquet')
    samples={name:matured_weekly_rows(features,labels,*bounds) for name,bounds in splits.items()}
    save(run/'sample_audit.json',{name:dict(n=len(rows),first=str(rows.min()),last=str(rows.max()),
        latest_label_end=str(labels.loc[rows,'label_end'].max())) for name,rows in samples.items()})
    train=samples['train'];y=labels.loc[train,'loss_event'];prior=float(y.mean())
    predictions=pd.DataFrame(index=features.index);thresholds={};coefficients=[]
    models={};all_known=features[FAMILIES['joint']].notna().all(axis=1)
    for name,columns in FAMILIES.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=1))
        model.fit(features.loc[train,columns],y)
        pred=pd.Series(np.nan,index=features.index)
        pred.loc[all_known]=model.predict_proba(features.loc[all_known,columns])[:,1]
        threshold=float(pred.loc[train].quantile(.8));alarm=pred.ge(threshold)&pred.notna()
        predictions[name]=pred;predictions[name+'_alarm']=alarm;thresholds[name]=threshold;models[name]=model
        lr=model[-1];scaler=model[0]
        coefficients.extend(dict(model=name,feature=c,mean=scaler.mean_[j],scale=scaler.scale_[j],coefficient=lr.coef_[0,j],intercept=lr.intercept_[0]) for j,c in enumerate(columns))
    utility=make_pipeline(StandardScaler(),Ridge(alpha=10))
    utility.fit(features.loc[train,FAMILIES['joint']],labels.loc[train,'terminal'])
    predicted_ret=pd.Series(np.nan,index=features.index)
    predicted_ret.loc[all_known]=utility.predict(features.loc[all_known,FAMILIES['joint']])
    predictions['predicted_terminal']=predicted_ret
    predictions['joint_utility']=predictions.joint
    predictions['joint_utility_alarm']=predictions.joint_alarm&predicted_ret.lt(-.004)
    thresholds['joint_utility']=thresholds['joint']
    for j,c in enumerate(FAMILIES['joint']):
        coefficients.append(dict(model='utility',feature=c,mean=utility[0].mean_[j],scale=utility[0].scale_[j],coefficient=utility[-1].coef_[j],intercept=utility[-1].intercept_))
    predictions.to_parquet(run/'predictions.parquet');pd.DataFrame(coefficients).to_csv(run/'coefficients.csv',index=False)
    save(run/'thresholds.json',dict(train_prior=prior,thresholds=thresholds))
    # TRAIN结束后的标签和特征不参与拟合；只复制已确定行重新拟合检查参数。
    for name,model in models.items():
        cols=FAMILIES[name]
        shortened=features.loc[:splits['train'][1]]
        fresh=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=1))
        fresh.fit(shortened.loc[train,cols],labels.loc[train,'loss_event'])
        np.testing.assert_array_equal(model[-1].coef_,fresh[-1].coef_)
    metrics=[];gates=[]
    for name in list(FAMILIES)+['joint_utility']:
        for split,rows in samples.items():
            m=evaluate(rows,predictions[name],predictions[name+'_alarm'],labels,prior)
            metrics.append(dict(model=name,split=split,**m))
            if split=='validation':
                checks=dict(sample=m['alarms']>=12,auc=m['auc']>=.55,precision=m['precision_lift']>=1.25,
                    exposure=m['alarm_fraction']<=.35,calibration=m['brier']<m['prior_brier'],
                    economic_mean=m['mean_cash_benefit']>0,economic_uncertainty=m['cash_benefit_p05']>0)
                gates.append(dict(model=name,checks=checks,passed=all(checks.values())))
    pd.DataFrame(metrics).to_csv(run/'metrics.csv',index=False)
    save(run/'acceptance.json',dict(validation_gates=gates,eligible=[x['model'] for x in gates if x['passed']],
        independent_oos=False,late_results_cannot_override_validation=True))
    save(run/'verification.json',dict(full_market_feature_prefixes=prefixes,train_model_parameters_prefix_exact=True,
        matured_labels_checked=True,source_units_checked=True,root_agents_unchanged=digest(ROOT.parent/'AGENTS.md')==protocol['root_agents_sha256']))
    print(pd.DataFrame(metrics).loc[lambda x:x.split.ne('train'),['model','split','n','alarms','auc','precision_lift','brier','prior_brier','mean_cash_benefit','cash_benefit_p05']].to_string(index=False),flush=True)
    print(json.dumps(gates,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
