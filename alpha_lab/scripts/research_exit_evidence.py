"""冻结实际退出/恢复机会事件，按成熟标签逐年训练，先验证可预测性再讨论交易。"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,json,shutil
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from research_profit_defense import digest,save
from exit_evidence import state_features,forward_marks,nonoverlapping_events

def prepare():
    source=ROOT/'configs/exit_evidence_research.yaml';cfg=yaml.safe_load(source.read_text());parent=ROOT/cfg['parent']
    old=json.loads((parent/'completion.json').read_text())
    for name,h in old['artifact_sha256'].items():assert digest(parent/name)==h,name
    run=ROOT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    shutil.copy2(source,run/'config.yaml');shutil.copy2('/tmp/exit_evidence_tests.log',run/'tests.log')
    assert (run/'tests.log').read_text().rstrip().endswith('OK')
    sources=['src/exit_evidence.py','scripts/research_exit_evidence.py','tests/test_exit_evidence.py']
    for name in sources:
        dest=run/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    save(run/'manifest.json',dict(config_sha256=digest(source),parent_completion_sha256=digest(parent/'completion.json'),sources={name:digest(ROOT/name) for name in sources}))
    (run/'PROTOCOL.md').write_text('''# 退出和恢复事件的证据验证协议

不改组合、不调阈值。主事件只取原16个Hybrid账户，禁止用后来收益选事件或把80变体当独立样本。
退出用实际持仓的趋势退出请求决策日；恢复机会只在真实完整退出后接下来两次原复核中，取第一次满足完整原入场资格的日期。恢复不要求Top20，不筛掉无名额事件，但须明确它是机会层而非可买组合。
先合并同股票同日重复账户事件，再按股票时间顺序保留相距严格超过60交易日的事件，形成主样本；所有被排除事件和恢复未出现者仍存档。该去重只看日期，不能按后续收益选样本。
9个特征只用决策当日及之前价格/已知趋势；未知不填为正常。标签从下一交易日收盘到20/60交易日后收盘，记录最差路径、缺失与右截断。不把复权行情代理称为整数股、股息现金或可成交组合收益。
2024/2025/2026年初各训练一次：训练事件日期在年初之前，标签结束日期严格在年初之前，且输入完整。标准化只fit训练集，Ridge alpha=10，不搜索。最低30事件/15日期；不足时回退训练均值。缺失特征的测试事件也回退均值，不从评价中删掉。
分别预测退出后和恢复后的20/60日收益。基准为每次训练池均值；高低预测组以当次训练预测的中位数划分，不能用测试期分位数决定阈值。
每个事件族独立过门：两个期限均至少50个可评价主事件；相对历史均值的时序R2>0；总体预测秩相关>0，至少两个各有10样本的年份秩相关>0；预测低组实际均值<0，高组>0。任何一项失败则不把该模型接入组合，不追加阈值搜索。
样本不足也不能证明信号不存在。所有2024+均已被项目看过，只是按历史时间顺序评价，不叫全新样本外。即使过门也仅允许另行执行层验证，不自动上线。
验证：原封存哈希、6单元测试、真实特征截断、严格标签成熟时间、未来标签扰动不改既有拟合、恢复有真实完整退出与当时资格；最后保留完整预测和不过门理由。
''')
    print(run,flush=True)

def verify(run):
    cfg=yaml.safe_load((run/'config.yaml').read_text());m=json.loads((run/'manifest.json').read_text())
    assert digest(run/'config.yaml')==m['config_sha256']
    for name,h in m['sources'].items():assert digest(ROOT/name)==h,name
    assert digest(ROOT/cfg['parent']/'completion.json')==m['parent_completion_sha256']
    return cfg

def build(run):
    cfg=verify(run);parent=ROOT/cfg['parent'];inp=parent/'inputs'
    p,t,o=[pd.read_parquet(inp/f'{n}.parquet') for n in ['daily_close','daily_trend','daily_observable']]
    features=state_features(p,t,o);prefix=[]
    for stop in pd.to_datetime(['2023-06-30','2025-06-30']):
        short=state_features(p.loc[:stop],t.loc[:stop],o.loc[:stop])
        for key in features:pd.testing.assert_frame_equal(short[key],features[key].loc[:stop],check_exact=True)
        prefix.append(dict(stop=stop,features=len(features)))
    raw=[];rec=[];status=[]
    for folder in sorted((parent/'execution').glob('*_hybrid')):
        e=pd.read_csv(folder/'events.csv',parse_dates=['date']);trades=pd.read_csv(folder/'trades.csv',parse_dates=['date','signal_date']);holds=pd.read_csv(folder/'holdings.csv',parse_dates=['date']);daily=pd.read_csv(folder/'daily.csv',index_col='date',parse_dates=True)
        schedule=folder.name.split('_',1)[1].removesuffix('_hybrid');sig=pd.read_parquet(parent/'schedules'/schedule/'signals.parquet')
        requests=e[e.kind.eq('hold_exit_requested')&e.reason.eq('hold_false')]
        for row in requests.itertuples():
            raw.append(dict(date=row.date,ts_code=row.ts_code,case=folder.name,schedule=schedule))
            sales=trades[trades.ts_code.eq(row.ts_code)&trades.side.eq('sell')&trades.signal_date.eq(row.date)]
            item=dict(case=folder.name,ts_code=row.ts_code,exit_decision=row.date)
            if sales.empty:status.append(dict(**item,status='no_completed_sale'));continue
            completed=sales.date.max()
            if len(holds[holds.date.eq(completed)&holds.ts_code.eq(row.ts_code)]):status.append(dict(**item,status='partial_sale_only'));continue
            reviews=sig.index[sig.index>completed][:cfg['recovery_review_limit']];found=False
            for date in reviews:
                if pd.isna(sig.at[date,row.ts_code]):continue
                ranks=sig.loc[date].dropna().sort_values(ascending=False,kind='stable')
                rank=int(ranks.index.get_loc(row.ts_code)+1)
                rec.append(dict(date=date,ts_code=row.ts_code,case=folder.name,schedule=schedule,exit_decision=row.date,sale_completed=completed,rank=rank,slots_at_review=max(0,5-int(daily.at[date,'n_hold']))))
                status.append(dict(**item,status='requalified',recovery_date=date,sale_completed=completed));found=True;break
            if not found:status.append(dict(**item,status='not_requalified' if len(reviews)==cfg['recovery_review_limit'] else 'recovery_window_right_censored',sale_completed=completed))
    out=run/'data';out.mkdir(exist_ok=False);pd.DataFrame(status).to_csv(out/'recovery_search_status.csv',index=False)
    summaries=[]
    for cohort,records in [('exit',raw),('recovery',rec)]:
        allrows=pd.DataFrame(records);allrows.to_csv(out/f'{cohort}_account_events.csv',index=False)
        counts=allrows.groupby(['date','ts_code']).size().rename('account_multiplicity')
        unique=allrows.drop_duplicates(['date','ts_code']).merge(counts,on=['date','ts_code'],validate='one_to_one')
        # slots等跨账户不同，不作为预测特征；保留全部原始记录便于检查可买性。
        unique=nonoverlapping_events(unique,p.index,cfg['nonoverlap_days'])
        rows=[]
        for item in unique.to_dict('records'):
            date,code=item['date'],item['ts_code']
            for name in cfg['features']:item[name]=features[name].at[date,code]
            item['features_complete']=bool(np.isfinite([item[n] for n in cfg['features']]).all())
            item['state']='long_broken' if item['ret120']<=0 and item['ma120_gap']<=0 else ('long_intact' if item['ret120']>0 and item['ma120_gap']>0 else 'mixed')
            for horizon in cfg['horizons']:
                for key,value in forward_marks(p,date,code,horizon).items():item[f'{key}_{horizon}']=value
            rows.append(item)
        frame=pd.DataFrame(rows);frame.to_parquet(out/f'{cohort}_events.parquet',index=False);frame.to_csv(out/f'{cohort}_events.csv',index=False)
        primary=frame[frame.nonoverlap_primary]
        summaries.append(dict(cohort=cohort,account_events=len(allrows),unique_events=len(frame),primary_events=len(primary),stocks=primary.ts_code.nunique(),missing_features=int((~primary.features_complete).sum())))
    pd.DataFrame(summaries).to_csv(out/'coverage.csv',index=False)
    save(run/'data_completion.json',dict(status='complete',prefix_checks=prefix,hashes={f.name:digest(f) for f in out.iterdir()}))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

def fit_one(frame,year,horizon,cfg):
    cutoff=pd.Timestamp(f'{year}-01-01');endkey=f'label_end_{horizon}';target=f'ret_{horizon}'
    train=frame[frame.date.lt(cutoff)&frame[endkey].lt(cutoff)&frame[target].notna()&frame.features_complete]
    test=frame[frame.date.dt.year.eq(year)].copy()
    assert train[endkey].lt(cutoff).all()
    mean=float(train[target].mean()) if len(train) else 0.
    enough=len(train)>=cfg['model']['min_train_events'] and train.date.nunique()>=cfg['model']['min_train_dates']
    pred=np.full(len(test),mean);model=None;threshold=mean
    if enough:
        model=make_pipeline(StandardScaler(),Ridge(alpha=cfg['model']['alpha']))
        model.fit(train[cfg['features']],train[target]);valid=test.features_complete.to_numpy()
        if valid.any():pred[valid]=model.predict(test.loc[valid,cfg['features']])
        threshold=float(np.median(model.predict(train[cfg['features']])))
    return train,test,pred,mean,threshold,model

def fit(run):
    cfg=verify(run);out=run/'models';out.mkdir(exist_ok=False);predictions=[];audits=[];coefficients=[]
    for cohort in cfg['cohorts']:
        frame=pd.read_parquet(run/'data'/f'{cohort}_events.parquet');frame=frame[frame.nonoverlap_primary].reset_index(drop=True)
        for horizon in cfg['horizons']:
            for year in cfg['prediction_years']:
                train,test,pred,mean,threshold,model=fit_one(frame,year,horizon,cfg)
                # 改写该年初尚不可知的所有标签；训练、标准化、系数和既有预测不能变。
                altered=frame.copy();future=altered[f'label_end_{horizon}'].ge(pd.Timestamp(f'{year}-01-01'))|altered[f'label_end_{horizon}'].isna();altered.loc[future,f'ret_{horizon}']=1e6
                t2,te2,p2,mean2,th2,model2=fit_one(altered,year,horizon,cfg)
                np.testing.assert_array_equal(pred,p2);assert mean==mean2 and threshold==th2
                if model is not None:
                    np.testing.assert_array_equal(model[-1].coef_,model2[-1].coef_)
                    for feature,value in zip(cfg['features'],model[-1].coef_):coefficients.append(dict(cohort=cohort,horizon=horizon,year=year,feature=feature,standardized_coefficient=value))
                split=f'{cohort}_h{horizon}_{year}'
                train[['date','ts_code',f'label_end_{horizon}',f'ret_{horizon}']].to_csv(out/f'{split}_train.csv',index=False)
                audits.append(dict(cohort=cohort,horizon=horizon,year=year,train_events=len(train),train_dates=train.date.nunique(),max_label_end=train[f'label_end_{horizon}'].max(),model_fitted=model is not None,test_events=len(test),future_label_perturbation_exact=True))
                for (_,row),value in zip(test.iterrows(),pred):
                    predictions.append(dict(cohort=cohort,horizon=horizon,year=year,date=row.date,ts_code=row.ts_code,prediction=value,baseline=mean,train_threshold=threshold,high_group=value>threshold,actual=row[f'ret_{horizon}'],mae=row[f'mae_{horizon}'],mfe=row[f'mfe_{horizon}'],label_end=row[f'label_end_{horizon}'],model_fitted=model is not None,feature_fallback=not row.features_complete))
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False);pd.DataFrame(audits).to_csv(out/'training_audit.csv',index=False);pd.DataFrame(coefficients).to_csv(out/'coefficients.csv',index=False)
    metrics=[];yearly=[];gate={}
    def stats(g):
        g=g[g.actual.notna()];den=((g.actual-g.baseline)**2).sum();r2=1-((g.actual-g.prediction)**2).sum()/den if den>0 else np.nan
        ic=float(spearmanr(g.prediction,g.actual).statistic) if len(g)>2 and g.prediction.nunique()>1 else np.nan
        return dict(n=len(g),r2=r2,ic=ic,rmse=float(np.sqrt(np.mean((g.actual-g.prediction)**2))),baseline_rmse=float(np.sqrt(np.mean((g.actual-g.baseline)**2))),mean_actual=g.actual.mean(),high_mean=g[g.high_group].actual.mean(),low_mean=g[~g.high_group].actual.mean(),high_n=int(g.high_group.sum()),low_n=int((~g.high_group).sum()),high_mae_mean=g[g.high_group].mae.mean(),low_mae_mean=g[~g.high_group].mae.mean())
    predictions=pd.DataFrame(predictions)
    for cohort in cfg['cohorts']:
        passed=[];reasons=[]
        for horizon in cfg['horizons']:
            group=predictions[predictions.cohort.eq(cohort)&predictions.horizon.eq(horizon)]
            total=stats(group);metrics.append(dict(cohort=cohort,horizon=horizon,**total));annual=[]
            for year,g in group.groupby('year'):
                z=stats(g);annual.append(z);yearly.append(dict(cohort=cohort,horizon=horizon,year=year,**z))
            checks=dict(enough_events=total['n']>=cfg['gate']['min_scored_events'],positive_r2=total['r2']>0,positive_ic=total['ic']>0,positive_years=sum(x['n']>=cfg['gate']['min_year_events'] and x['ic']>0 for x in annual)>=cfg['gate']['min_positive_ic_years'],low_negative_high_positive=total['low_mean']<0<total['high_mean'])
            checks={key:bool(value) for key,value in checks.items()}
            passed.append(all(checks.values()));reasons.append(dict(horizon=horizon,checks=checks))
        gate[cohort]=dict(passed=all(passed),horizons=reasons)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(yearly).to_csv(out/'yearly_metrics.csv',index=False);save(out/'gate.json',gate)
    save(out/'completion.json',dict(status='complete',hashes={f.name:digest(f) for f in out.iterdir()}))
    print(pd.DataFrame(metrics).round(4).to_string(index=False),flush=True);print(json.dumps(gate),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','build','fit'],required=True);p.add_argument('--run',type=Path);args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='build':build(args.run)
    else:fit(args.run)
