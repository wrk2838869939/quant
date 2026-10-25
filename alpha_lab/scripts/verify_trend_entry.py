#!/usr/bin/env python3
"""独立复核趋势路径、模型训练边界，并追加明确标注的简单风格对照。"""
import argparse,hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from patient_trend import review_dates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);args=ap.parse_args();out=args.prepared.resolve()
    cfg=yaml.safe_load((out/'config.yaml').read_text());manifest=json.loads((out/'manifest.json').read_text());assert manifest['status']=='complete'
    for name,digest in manifest['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    addendum=dict(purpose='Check whether learned path score adds anything beyond low volatility or low turnover',
        controls=['low_volatility','low_turnover'],post_hoc_diagnostic=True,no_parameter_selection=True)
    (out/'control_addendum.json').write_text(json.dumps(addendum,indent=2))
    panel=pd.read_parquet(out/'entry_paths.parquet');h=cfg['primary_horizon']
    for horizon in cfg['horizons']:
        assert panel[f'label_end_{horizon}'].dropna().max()<pd.Timestamp('2022-01-01')
    train=panel[panel[cfg['fit_frequency']]&panel.trend&panel[f'label_end_{h}'].le(pd.Timestamp(cfg['fit_end']))]
    assert train[f'label_end_{h}'].max()<=pd.Timestamp(cfg['fit_end'])
    weights=1/train.groupby('date').ts_code.transform('size');xcols=['x_'+c for c in cfg['features']]
    coefs=pd.read_csv(out/'model_coefficients.csv')
    for target in ['sustained','roundtrip']:
        model=LogisticRegression(**cfg['model']).fit(train[xcols],train[f'{target}_{h}'].astype(int),sample_weight=weights)
        expect=coefs[coefs.target.eq(target)].set_index('feature').coefficient
        np.testing.assert_allclose(np.r_[model.intercept_,model.coef_[0]],expect.loc[['intercept']+xcols],atol=1e-12)
        sample=panel.iloc[::137]
        np.testing.assert_allclose(model.predict_proba(sample[xcols])[:,1],sample['p_'+target],atol=1e-12)
    # 50个固定随机观测，直接从原始开发行情独立重建端点和最大回撤。
    check=panel[panel[f'label_end_{h}'].notna()].sample(50,random_state=42)
    raw=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',columns=['date','ts_code','hfq_open','hfq_close'],
        filters=[('date','>=',pd.Timestamp(cfg['history'][0])),('date','<=',pd.Timestamp(cfg['history'][1])),('ts_code','in',check.ts_code.unique().tolist())])
    # 完整市场日历由原数据文件的日期列恢复，不用抽样股票的停牌日期构造日历。
    calendar=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',columns=['date'],filters=[('date','>=',pd.Timestamp(cfg['history'][0])),('date','<=',pd.Timestamp(cfg['history'][1]))])
    dates=pd.DatetimeIndex(sorted(calendar.date.unique()));basic=pd.read_parquet(ROOT/'data/raw/stock_basic_all.parquet').set_index('ts_code')
    max_error=0.
    for row in check.itertuples():
        i=dates.get_loc(row.date);hist=raw[raw.ts_code.eq(row.ts_code)].set_index('date').reindex(dates)
        entry=hist.hfq_open.iloc[i+1];terminal=hist.hfq_open.iloc[i+h+1]
        sequence=np.r_[entry,hist.hfq_close.iloc[i+1:i+h+1],terminal]
        if not np.isfinite(entry) or entry<=0:path=np.ones(h+2)
        else:
            path=pd.Series(sequence).where(lambda x:x>0).ffill().to_numpy()/entry
            dd=pd.to_datetime(basic.loc[row.ts_code,'delist_date'],errors='coerce')
            if pd.notna(dd):
                path_dates=pd.DatetimeIndex([dates[i+1]]+list(dates[i+1:i+h+1])+[dates[i+h+1]])
                path[path_dates>=dd]=0.
                path[0]=1.
        ret=path[-1]-1;mdd=(path/np.maximum.accumulate(path)-1).min()
        max_error=max(max_error,abs(ret-getattr(row,f'ret_{h}')),abs(mdd-getattr(row,f'mdd_{h}')))
    assert max_error<1e-10
    evaluation=panel[panel.date.between(*map(pd.Timestamp,cfg['evaluation']))&panel.trend]
    rows=[];tails=[];ranking=[]
    for frequency in cfg['review_frequencies']:
        for date,g in evaluation[evaluation[frequency]].groupby('date'):
            if g[f'label_end_{h}'].notna().any():
                for target in ['sustained','roundtrip']:
                    truth=g[f'{target}_{h}'].astype(int)
                    ranking.append(dict(frequency=frequency,date=date,target=target,
                        auc=roc_auc_score(truth,g['p_'+target]) if truth.nunique()==2 else np.nan))
            for policy,field in [('low_volatility','volatility'),('low_turnover','turnover')]:
                chosen=g.sort_values([field,'ts_code']).head(cfg['max_positions'])
                for horizon in cfg['horizons']:
                    if chosen[f'label_end_{horizon}'].isna().all():continue
                    rows.append(dict(frequency=frequency,date=date,year=date.year,policy=policy,horizon=horizon,
                        ret=chosen[f'ret_{horizon}'].mean(),mae=chosen[f'mae_{horizon}'].mean(),
                        sustained=chosen[f'sustained_{horizon}'].mean(),roundtrip=chosen[f'roundtrip_{horizon}'].mean()))
            for policy in cfg['policies']:
                chosen=g[g.path_model.gt(cfg['path_model_min_score'])] if policy=='path_model' else g
                chosen=chosen.sort_values([policy,'ts_code'],ascending=[False,True]).head(cfg['max_positions'])
                if chosen[f'label_end_{h}'].isna().all():continue
                for row in chosen.itertuples():
                    tails.append(dict(frequency=frequency,date=date,policy=policy,ts_code=row.ts_code,
                        ret=getattr(row,f'ret_{h}'),mae=getattr(row,f'mae_{h}'),mfe=getattr(row,f'mfe_{h}'),
                        state=getattr(row,f'state_{h}'),missing_marks=getattr(row,f'missing_marks_{h}'),
                        entry_missing=getattr(row,f'entry_missing_{h}'),exit_marked=getattr(row,f'exit_marked_{h}'),delisted=getattr(row,f'delisted_{h}'),
                        volatility=row.volatility,turnover=row.turnover,log_size=row.log_size))
    pd.DataFrame(ranking).to_csv(out/'probability_rank_diagnostics.csv',index=False)
    controls=pd.DataFrame(rows);controls.to_csv(out/'simple_control_paths.csv',index=False)
    controls.groupby(['frequency','horizon','policy'])[['ret','mae','sustained','roundtrip']].mean().to_csv(out/'simple_control_summary.csv')
    selected=pd.DataFrame(tails);selected.to_csv(out/'selected_stock_paths.csv',index=False)
    selected.groupby(['frequency','policy']).agg(n=('ret','size'),median_return=('ret','median'),mean_return=('ret','mean'),
        worst_return=('ret','min'),best_return=('ret','max'),loss_fraction=('ret',lambda x:x.le(0).mean()),
        median_daily_vol=('volatility','median'),median_turnover=('turnover','median'),
        missing_mark_fraction=('missing_marks',lambda x:x.gt(0).mean()),entry_missing=('entry_missing','sum'),
        exit_marked=('exit_marked','sum'),delisted=('delisted','sum')).to_csv(out/'selected_distribution.csv')
    selected.groupby(['frequency','policy','state']).size().to_csv(out/'selected_states.csv')
    # 每项差异按观察日配对，仍是重叠价格路径诊断，不能作实盘收益。
    original=pd.read_csv(out/'policy_paths.csv',parse_dates=['date']);paired=original.merge(controls,on=['frequency','date','year','horizon'],suffixes=('_strategy','_control'))
    paired['return_difference']=paired.ret_strategy-paired.ret_control
    paired.groupby(['frequency','horizon','policy_strategy','policy_control']).return_difference.agg(['mean','median','count']).to_csv(out/'style_paired_differences.csv')
    members=pd.read_csv(out/'patient_memberships.csv',parse_dates=['date']);lifecycle=[]
    for (frequency,policy),g in members.groupby(['frequency','policy']):
        calendar=sorted(panel.loc[panel[frequency]&panel.date.between(*map(pd.Timestamp,cfg['evaluation'])),'date'].unique())
        previous=set();new=exits=empty=0;lengths=[];starts={}
        for j,date in enumerate(calendar):
            current=set(g.loc[g.date.eq(date),'ts_code']);assert len(current)<=cfg['max_positions']
            for code in previous-current:lengths.append(j-starts.pop(code))
            for code in current-previous:starts[code]=j
            new+=len(current-previous);exits+=len(previous-current);empty+=not current;previous=current
        lifecycle.append(dict(frequency=frequency,policy=policy,reviews=len(calendar),new_signals=new,exit_signals=exits,
            empty_reviews=empty,closed_episodes=len(lengths),median_closed_review_periods=float(np.median(lengths)) if lengths else np.nan,open_episodes=len(starts)))
    pd.DataFrame(lifecycle).to_csv(out/'signal_lifecycle_summary.csv',index=False)
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for name in manifest['source_hashes']:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    metadata=[ROOT/'data/raw/stock_basic_all.parquet',ROOT/'data/raw/namechange_all.parquet']
    result=dict(status='complete',source_hashes_verified=True,coefficient_reproduction=True,prediction_reproduction=True,
        independently_rebuilt_paths=50,max_path_error=max_error,max_positions=cfg['max_positions'],
        last_training_label=str(train[f'label_end_{h}'].max()),all_test_labels_before_2022=True,
        metadata_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata},
        verifier_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),controls=addendum)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
