# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""已锁定学习策略的同信号成本重放、同交易日规则对照和预测失效归因。"""
from pathlib import Path
import sys,json,hashlib,shutil,os
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml,joblib
ROOT=Path(__file__).resolve().parents[1]; PROJECT=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from portfolio import run_backtest
from classic_smoke import verify_ledger
from run_rule_strategy_search import write_result
from learning_search import training_mask
from learning_extensions import infer_bundle,score_predictions
from run_classic_strategy_smoke import digest,save_json


def load_data(run,plan):
    eligible=pd.read_parquet(run/'inputs/eligible.parquet')
    frame=pd.read_parquet(run/'inputs/learning_features.parquet')
    raw=np.stack([frame[k].unstack().reindex(index=eligible.index,columns=eligible.columns).to_numpy() for k in plan['features']],axis=2)
    missing=~np.isfinite(raw)
    data=dict(dates=eligible.index,codes=eligible.columns,names=list(plan['features']),ranks=np.nan_to_num(raw).astype(np.float32),
              x=np.concatenate([np.nan_to_num(raw),missing.astype(float)],axis=2).astype(np.float32),
              allowed=eligible.to_numpy() & ((~missing).sum(axis=2)>=plan['training']['min_features']))
    ends=pd.read_parquet(run/'inputs/label_end_dates.parquet')
    data['labels']={h:dict(returns=pd.read_parquet(run/'inputs'/f'label_returns_{h}.parquet').to_numpy(),
                           risk=pd.read_parquet(run/'inputs'/f'label_risk_{h}.parquet').to_numpy(),end=ends[str(h)].to_numpy()) for h in plan['common_axes']['horizon']}
    if (run/'inputs/market_context.parquet').exists():
        data['context']=pd.read_parquet(run/'inputs/market_context.parquet').fillna(0).to_numpy()
    return data,frame


def rank_ic(pred,target,minimum=10):
    a,b=pd.DataFrame(pred),pd.DataFrame(target)
    valid=a.notna() & b.notna()
    a=a.where(valid).rank(axis=1);b=b.where(valid).rank(axis=1)
    a=a.sub(a.mean(axis=1),axis=0);b=b.sub(b.mean(axis=1),axis=0)
    denom=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    return ((a*b).sum(axis=1)/denom.where(denom>0)).where(valid.sum(axis=1)>=minimum)


def diagnostic_model_predictions(saved,data,p,indices,extension):
    out=np.full((len(indices),len(data['codes'])),np.nan)
    for j,at in enumerate(indices):
        if extension:
            out[j],_=infer_bundle(saved,data,at)
        else:
            valid=data['allowed'][at]
            if saved['weights'] is not None:
                out[j,valid]=data['ranks'][at,valid]@saved['weights']
            elif valid.any():
                out[j,valid]=saved['model'].predict(data['x'][at,valid])
                if p['family']=='dual':
                    risk=np.maximum(0.,saved['risk_model'].predict(data['x'][at,valid]))
                    out[j,valid]-=p['risk_aversion']*risk
                    out[j,np.flatnonzero(valid)[risk>p['risk_cap']]]=np.nan
    return out


def main():
    plan_path=ROOT/'configs/learning_diagnosis_v1.yaml'; plan=yaml.safe_load(plan_path.read_text())
    cfg_account=yaml.safe_load((PROJECT/plan['account_config']).read_text())
    assert os.environ.get('DATAHUB_CONTENT_ID')==cfg_account['dataset']['content_id']
    run=ROOT/'outputs/learning_diagnosis'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    print('RUN='+str(run),flush=True)
    source_paths=[Path(__file__),plan_path,ROOT/'src/portfolio.py',ROOT/'src/classic_smoke.py',ROOT/'src/learning_search.py',ROOT/'src/learning_extensions.py']
    hashes={str(p.relative_to(PROJECT)):digest(p) for p in source_paths}
    (run/'sources').mkdir()
    for p in source_paths: shutil.copy2(p,run/'sources'/p.name)
    manifest=dict(status='running',source_hashes=hashes,started_at=datetime.now(timezone.utc).isoformat(),input_runs=plan['sources'],parameter_selection=False)
    save_json(run/'manifest.json',manifest)
    rows=[];quality=[];stability=[];audits=[]
    try:
        for source in plan['sources']:
            old=PROJECT/source; extension='learning_extensions' in source
            old_manifest=json.loads((old/'manifest.json').read_text());assert old_manifest['status']=='complete'
            for key in ['alpha_lab/src/portfolio.py','alpha_lab/src/classic_smoke.py']:
                assert digest(PROJECT/key)==old_manifest['source_hashes'][key]
            assert all(digest(old/'inputs'/p)==h for p,h in json.loads((old/'input_hashes.json').read_text()).items())
            old_plan=yaml.safe_load((old/'sources'/('learning_extensions_v1.yaml' if extension else 'learning_strategy_search_v1.yaml')).read_text())
            data,features=load_data(old,old_plan)
            host=pd.DataFrame(data['ranks'].mean(axis=2),index=data['dates'],columns=data['codes']).where(data['allowed'])
            controls={'equal_factor':host,**{k:features[k].unstack().reindex_like(host).where(data['allowed']) for k in plan['rule_controls'] if k!='equal_factor'}}
            frames={k:pd.read_parquet(old/'inputs'/f'{k}.parquet') for k in ['market','basics','st','limits','actions']}
            smoke=yaml.safe_load((old/'sources/classic_strategy_smoke.yaml').read_text())
            base=yaml.safe_load((old/'sources/monthly_portfolio_audit.yaml').read_text())['portfolio']
            base.update(capital=cfg_account['account']['capital'],top_n=5,band_mult=1.,**{k:smoke[k] for k in ['buy_cost','sell_cost','min_fee','cash_dividend_haircut','annual_days']})
            rules=yaml.safe_load((old/'sources/corporate_action_components.yaml').read_text())
            choices=json.loads((old/'all_locked_choices.json').read_text())
            for family in old_plan['families']:
                folder=old/'outer'/family
                signals=pd.read_parquet(folder/'signals.parquet');schedule=pd.read_parquet(folder/'top_n_schedule.parquet').iloc[:,0]
                variants={'learned':signals}
                for name,score in controls.items():
                    matched=score.reindex(signals.index).copy()
                    # 保留预先锁定的无可行年度；模型自身过滤导致的空仓不复制到规则对照。
                    for c in choices:
                        if c['family']==family and c['selected'] is None:
                            matched.loc[matched.index.year==c['outer_year']]=np.nan
                    variants[name]=matched
                for name,signal in variants.items():
                    for costs in ['net','zero_cost']:
                        config=dict(base);slip=smoke['slippage_rate']
                        if costs=='zero_cost':
                            config.update(buy_cost=0.,sell_cost=0.,min_fee=0.);slip=0.
                        result=run_backtest(signal,frames['market'],frames['basics'],frames['st'],frames['limits'],frames['actions'],plan['start'],plan['end'],config,rules.get('rights_issues',[]),slippage_rate=slip,top_n_schedule=schedule)
                        dest=run/'accounts'/family/name/costs
                        write_result(dest,result,signal)
                        check=verify_ledger(result,config['capital'],frames['market'],frames['limits'],frames['actions'],config,slip,smoke['audit']['money_atol'])
                        save_json(dest/'audit.json',check);audits.append(check)
                        if name=='learned' and costs=='net':
                            for table in ['daily','trades','events','holdings']:
                                pd.testing.assert_frame_equal(getattr(result,table),pd.read_parquet(folder/f'{table}.parquet'))
                        rows.append(dict(family=family,signal=name,cost=costs,**result.summary))
                for c in [x for x in choices if x['family']==family]:
                    year=c['outer_year'];sel=c['selected']
                    trials=[json.loads(line) for line in (old/'search'/str(year)/family/'trials.jsonl').read_text().splitlines()]
                    success=[r for r in trials if r['status']=='ok'];feasible=[r for r in success if r['metrics']['feasible']]
                    item=dict(family=family,year=year,attempts=len(trials),feasible=len(feasible),selected=sel is not None)
                    if sel:
                        item.update(selected_inner_median_ann=sel['metrics']['median_ann'],selected_inner_mdd=sel['metrics']['mdd'],
                                    median_candidate_ann=float(np.median([r['metrics']['median_ann'] for r in success])),
                                    neighbor_count=sel.get('neighbor_count',0),parameters=json.dumps(sel['params'],sort_keys=True))
                    stability.append(item)
                    if not sel or family in ['trade_filter','market_gate']:
                        continue
                    p=sel['params'];paths=list((old/'predictions'/f'outer_{year}'/family).glob('*/fits.json'));assert len(paths)==1
                    fit_path=paths[0];fits=[f for f in json.loads(fit_path.read_text()) if f['status']=='ok'];fit=fits[0]
                    model_path=fit_path.parent/'models'/fit['model_file'];assert digest(model_path)==fit['model_sha256']
                    saved=joblib.load(model_path);date=pd.Timestamp(fit['fit_date'])
                    mask=training_mask(data,p,date,old_plan['training'])
                    if family=='multi_horizon':
                        for h in old_plan['training']['joint_horizons']:
                            mask &= training_mask(data,dict(p,horizon=h),date,old_plan['training'])
                    assert hashlib.sha256(mask.tobytes()).hexdigest()==fit['training_mask_sha256']
                    indices=np.flatnonzero(mask.any(axis=1))
                    ins=diagnostic_model_predictions(saved,data,p,indices,extension)
                    target=np.where(mask[indices],data['labels'][p['horizon']]['returns'][indices],np.nan)
                    train_ic=rank_ic(ins,target)
                    # 比较同一个已保存模型的下一重训段；不拿不同模型拼出的预测充当该模型未来表现。
                    next_date=pd.Timestamp(fits[1]['fit_date']) if len(fits)>1 else pd.Timestamp(f'{year+1}-01-01')
                    prediction=pd.read_parquet(fit_path.parent/'return_predictions.parquet')
                    if family=='dual':
                        from learning_search import score_predictions as first_score
                        prediction=first_score(prediction,pd.read_parquet(fit_path.parent/'risk_predictions.parquet'),p)
                    oos=prediction.loc[(prediction.index>=date)&(prediction.index<next_date)]
                    observed=data['dates'].get_indexer(oos.index)
                    y=data['labels'][p['horizon']]['returns'][observed].copy()
                    ready=data['labels'][p['horizon']]['end'][observed] < np.datetime64(next_date)
                    y[~ready]=np.nan
                    future_ic=rank_ic(oos.to_numpy(),y)
                    quality.append(dict(family=family,year=year,fit_date=str(date.date()),next_fit_date=str(next_date.date()),
                                        training_daily_ic=float(train_ic.mean()),future_daily_ic=float(future_ic.mean()),training_ic_days=int(train_ic.notna().sum()),future_ic_days=int(future_ic.notna().sum()),
                                        descriptive_only=True))
                print('DIAGNOSED '+family,flush=True)
        accounts=pd.DataFrame(rows);accounts.to_csv(run/'account_comparisons.csv',index=False)
        pd.DataFrame(quality).to_csv(run/'same_model_train_future_ic.csv',index=False)
        pd.DataFrame(stability).to_csv(run/'selection_stability.csv',index=False)
        result=[]
        for family,g in accounts.groupby('family',sort=False):
            fetch=lambda signal,cost:g[(g.signal==signal)&(g.cost==cost)].iloc[0]
            net,gross=fetch('learned','net'),fetch('learned','zero_cost')
            item=dict(family=family,net_return=net.total,zero_cost_return=gross.total,cost_path_difference=gross.total-net.total,net_mdd=net.mdd,turnover_fees=net.total_fees)
            for name in plan['rule_controls']:
                item[name+'_net_return']=fetch(name,'net').total
                item[name+'_zero_cost_return']=fetch(name,'zero_cost').total
            result.append(item)
        summary=pd.DataFrame(result);summary.to_csv(run/'diagnosis_summary.csv',index=False)
        assert all(digest(PROJECT/p)==h for p,h in hashes.items())
        manifest.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),accounts=len(audits),archived_account_replays=11,
                        cash_max_error=max(x['cash_max_error'] for x in audits),trades_checked=sum(x['trades_checked'] for x in audits))
        print(summary.to_string(index=False),flush=True)
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:save_json(run/'manifest.json',manifest)


if __name__=='__main__':main()
