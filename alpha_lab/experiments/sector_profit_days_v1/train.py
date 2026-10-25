"""按原年度训练框架替换盈亏天数标签；输出独立实验模型。"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import copy
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path('/root/autodl-tmp')
HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE/'config.yaml').read_text())
OUT = ROOT/CFG['output']; BASE = ROOT/CFG['source_run']; DATA = ROOT/CFG['dataset']
sys.path.insert(0, str(HERE))
from model import FuzzyNetwork, predict_network
from sector_asof_replay import training_plan, fit_asof, predict_asof

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_json(p, value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str))

def main():
    torch.set_num_threads(1)
    manifest=json.loads((BASE/'manifest.json').read_text())
    for name,h in manifest['artifacts'].items():assert digest(BASE/name)==h,name
    sources=OUT/'sources';sources.mkdir(exist_ok=True)
    for name in ['train.py','model.py','sector_asof_replay.py','config.yaml','PREREG.md']:
        shutil.copy2(HERE/name,sources/name)
    original=pd.read_parquet(BASE/'research_inputs_WITH_LABELS.parquet')
    oldpred=pd.read_parquet(BASE/'predictions_BEFORE_EVALUATION.parquet')
    oldfits={int(f['year']):f for f in json.loads((BASE/'fits.json').read_text()) if f['status']=='complete'}
    panel=pd.read_parquet(DATA/'daily.parquet',columns=['date','sector_code','index_valid','index_close','segment_id'])
    calendar=pd.DatetimeIndex(sorted(panel.date.unique()))
    groups={c:g.set_index('date') for c,g in panel.groupby('sector_code')}
    rows=[]
    for r in original.itertuples():
        start=calendar.get_loc(r.date);dates=calendar[start:start+CFG['label']['horizon']+1]
        path=groups[r.sector_code].reindex(dates)
        mature=len(dates)==CFG['label']['horizon']+1
        complete=mature and path.index_valid.eq(True).all() and path.segment_id.eq(path.segment_id.iloc[0]).all() and path.index_close.gt(0).all()
        target=terminal=np.nan;positive=negative=flat=np.nan
        if complete:
            ret=path.index_close.to_numpy()[1:]/path.index_close.iloc[0]-1
            positive=int((ret>0).sum());negative=int((ret<0).sum());flat=int((ret==0).sum())
            target=(positive-negative)/CFG['label']['horizon'];terminal=ret[-1]
            assert positive+negative+flat==CFG['label']['horizon']
        assert bool(complete)==bool(r.complete),r.event_id
        end=dates[-1] if mature else pd.NaT
        assert (pd.isna(end) and pd.isna(r.end_date)) or end==r.end_date,r.event_id
        if complete:np.testing.assert_allclose(terminal,r.target,atol=1e-12,rtol=1e-10)
        rows.append(dict(event_id=r.event_id,target=target,terminal_return=terminal,positive_days=positive,negative_days=negative,flat_days=flat))
    data=original.drop(columns='target').merge(pd.DataFrame(rows),on='event_id',validate='one_to_one')
    data.to_parquet(OUT/'research_inputs_WITH_NEW_LABELS.parquet',index=False)
    save_json(OUT/'label_audit.json',dict(events=len(data),complete=int(data.complete.sum()),eligibility_unchanged=True,old_terminal_labels_matched=True,horizon=CFG['label']['horizon']))
    model_cfg=yaml.safe_load((BASE/'sources/config.yaml').read_text())
    features=model_cfg['features'];audits=[];allpred=[];fits=[];allsearch=[]
    # 先检查所有年度训练集合，再产生实验预测。
    for year in CFG['test_years']:
        start=pd.Timestamp(year=year,month=1,day=1)
        clean=training_plan(data,start,calendar,CFG['model'])
        for ids,frame in zip(['train_ids','fit_ids','valid_ids'],clean[:3]):
            assert frame.event_id.tolist()==oldfits[year][ids],(year,ids)
        mutated=data.copy();future=mutated.date.ge(start)
        mutated.loc[future,'target']=12345.0
        mutated.loc[future,features]=-98765.0
        changed=training_plan(mutated,start,calendar,CFG['model'])
        for a,b in zip(clean[:3],changed[:3]):pd.testing.assert_frame_equal(a,b)
        assert clean[3:]==changed[3:]
        audits.append(dict(year=year,training_ids_match_original=True,future_label_feature_perturbation_pass=True))
    save_json(OUT/'timing_plan_audit.json',audits)
    for variant in CFG['variants']:
        name=variant['name'];vcfg=copy.deepcopy(model_cfg);vcfg['target_scale']=variant['target_scale']
        folder=OUT/name;folder.mkdir(exist_ok=True);(folder/'models').mkdir(exist_ok=True)
        (folder/'model_config.yaml').write_text(yaml.safe_dump(vcfg,allow_unicode=True,sort_keys=False))
        for year in CFG['test_years']:
            start=pd.Timestamp(year=year,month=1,day=1)
            fp=folder/f'predictions_{year}_BEFORE_EVALUATION.parquet'
            ap=folder/f'fit_{year}.json'
            if fp.exists() and ap.exists():
                allpred.append(pd.read_parquet(fp));fits.append(json.loads(ap.read_text()))
                allsearch.extend(json.loads((folder/f'search_{year}.json').read_text()))
                print('RESUME',name,year,flush=True);continue
            # 拟合入口不给当年记录；预测入口只给特征白名单。
            historical=data[data.date.lt(start)].copy()
            fitted=fit_asof(historical,start,calendar,CFG['model'],vcfg)
            feature_frame=data[data.date.dt.year.eq(year)][['event_id','date','sector_code']+features].copy()
            scores,selected=predict_asof(fitted,feature_frame,vcfg)
            result=feature_frame[['event_id','date','sector_code']].copy()
            result['variant']=name;result['model_year']=year;result['score']=scores
            result['selected']=selected;result['threshold']=fitted['audit']['threshold']
            result.to_parquet(fp,index=False)
            audit=dict(variant=name,year=year,status='complete',target_scale=variant['target_scale'],**fitted['audit'])
            assert pd.Timestamp(audit['train_last_label'])<pd.Timestamp(audit['cutoff'])<start
            reloaded=[]
            for seed,net in zip(CFG['model']['ensemble_seeds'],fitted['networks']):
                saved=dict(state_dict=net.state_dict(),feature_names=features,median=fitted['scaler'].median,
                           mean=fitted['scaler'].mean,scale=fitted['scaler'].scale,audit=audit,seed=seed)
                weight_path=folder/'models'/f'{year}_{seed}.pt';torch.save(saved,weight_path)
                loaded=torch.load(weight_path,map_location='cpu',weights_only=False)
                restored=FuzzyNetwork.__new__(FuzzyNetwork);torch.nn.Module.__init__(restored)
                for key,value in loaded['state_dict'].items():restored.register_parameter(key,torch.nn.Parameter(torch.empty_like(value)))
                restored.width_floor=vcfg['fnn']['width_floor'];restored.load_state_dict(loaded['state_dict']);restored.eval()
                reloaded.append(predict_network(restored,fitted['scaler'].transform(feature_frame[features])))
            np.testing.assert_array_equal(np.mean(reloaded,axis=0)*vcfg['target_scale'],scores)
            audit['saved_model_reload_equal']=True
            if year==CFG['prefix_refit_year'] and variant['role']=='primary_label_only_change':
                # 让入口接收全量含未来标签，再与已截断拟合的结果作真实重训比较。
                prefix=fit_asof(data,start,calendar,CFG['model'],vcfg)
                prefix_scores,_=predict_asof(prefix,feature_frame,vcfg)
                np.testing.assert_array_equal(prefix_scores,scores)
                assert prefix['audit']==fitted['audit']
                audit['full_vs_prefix_actual_refit_equal']=True
            save_json(ap,audit)
            search=[dict(variant=name,year=year,**r) for r in fitted['search']]
            save_json(folder/f'search_{year}.json',search)
            allsearch.extend(search);allpred.append(result);fits.append(audit)
            print('TRAINED',name,year,'train',len(audit['train_ids']),'rules',audit['rules'],'epochs',audit['epochs'],
                  'threshold',round(audit['threshold'],6),'pass',int(selected.sum()),'/',len(selected),flush=True)
    predictions=pd.concat(allpred,ignore_index=True)
    for name,g in predictions.groupby('variant'):
        assert set(g.event_id)==set(oldpred.event_id),name
    predictions.to_parquet(OUT/'predictions_BEFORE_EVALUATION.parquet',index=False)
    save_json(OUT/'fits.json',fits);pd.DataFrame(allsearch).to_csv(OUT/'inner_search.csv',index=False)
    checks=json.loads((OUT/'original_hashes_before.json').read_text())
    for name,h in checks.items():assert digest(ROOT/name)==h,name
    save_json(OUT/'models_complete.json',dict(status='complete',variants=len(CFG['variants']),predictions=len(predictions),original_hashes_unchanged=True,training_sets_match_original=True))
    print('MODELS_COMPLETE',OUT,flush=True)

if __name__=='__main__':main()
