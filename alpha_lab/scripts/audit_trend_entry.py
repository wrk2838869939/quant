#!/usr/bin/env python3
"""固定趋势入场对照：成熟标签拟合、历史顺序验证、逐股路径与信号生命周期。"""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from patient_trend import review_dates,trend_state
from trend_entry import entry_features,marked_paths,patient_memberships
from ledger import ExperimentLedger


def main():
    cp=ROOT/'configs/trend_entry_audit.yaml';cfg=yaml.safe_load(cp.read_text())
    out=ROOT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    (out/'config.yaml').write_bytes(cp.read_bytes());sources=[cp,Path(__file__),ROOT/'src/trend_entry.py',ROOT/'src/patient_trend.py']
    manifest=dict(status='registered',fresh_holdout=False,test_market_data_used=False,valid_2022_plus_used=False,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out,flush=True)
    lo,hi=map(pd.Timestamp,cfg['history']);assert hi<pd.Timestamp('2022-01-01')
    raw=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',columns=['date','ts_code','hfq_open','hfq_close','total_mv_yuan','turnover_rate'],filters=[('date','>=',lo),('date','<=',hi)])
    manifest['input_hash']=hashlib.sha256(pd.util.hash_pandas_object(raw,index=False).values.tobytes()).hexdigest()
    wide={c:raw.pivot(index='date',columns='ts_code',values=c).sort_index() for c in ['hfq_open','hfq_close','total_mv_yuan','turnover_rate']}
    close=wide['hfq_close'];dates=close.index;codes=close.columns;opening=wide['hfq_open'].to_numpy(float);prices=close.to_numpy(float)
    features=entry_features(close,wide['total_mv_yuan'],wide['turnover_rate']);score,trend=trend_state(close,cfg['trend'])
    for end in ['2017-06-30','2020-06-30']:
        short=entry_features(close.loc[:end],wide['total_mv_yuan'].loc[:end],wide['turnover_rate'].loc[:end])
        for col in cfg['features']:pd.testing.assert_frame_equal(short[col],features[col].loc[:end])
    basic=pd.read_parquet(ROOT/'data/raw/stock_basic_all.parquet').set_index('ts_code')
    listed=pd.to_datetime(basic.list_date).reindex(codes);delisted=pd.to_datetime(basic.delist_date).reindex(codes)
    names=pd.read_parquet(ROOT/'data/raw/namechange_all.parquet');names=names[names.name.str.contains('ST',case=False,na=False)].copy()
    names['start']=pd.to_datetime(names.start_date);names['end']=pd.to_datetime(names.end_date)
    calendars={f:set(review_dates(dates,f)) for f in cfg['review_frequencies']};decisions=sorted(set.union(*calendars.values()))
    blocks=[]
    for date in decisions:
        i=dates.get_loc(date)
        if i<cfg['trend']['long_window']:continue
        f=pd.DataFrame({key:value.loc[date] for key,value in features.items()})
        # 只依赖入场前的可用性；不因未来缺口删除股票。
        eligible=(date-listed).dt.days.ge(cfg['min_listing_days'])&(delisted.isna()|delisted.gt(date))&np.isfinite(f).all(axis=1)
        st=names.loc[names.start.le(date)&(names.end.isna()|names.end.ge(date)),'ts_code'];eligible.loc[eligible.index.isin(st)]=False
        ix=np.flatnonzero(eligible.to_numpy())
        if len(ix)<cfg['min_cross_section']:continue
        frame=f.iloc[ix].copy();ranks=frame.rank(pct=True)-.5
        for key in cfg['features']:frame['x_'+key]=ranks[key]
        frame['date']=date;frame['trend']=trend.loc[date].iloc[ix].to_numpy(bool)
        frame['momentum']=score.loc[date].iloc[ix].to_numpy()
        frame['continuous']=(ranks.continuity-ranks.jump_share-ranks.volatility-ranks.extension)/4
        for freq,calendar in calendars.items():frame[freq]=date in calendar
        for h in cfg['horizons']:
            if i+h+1>=len(dates):continue
            d=delisted.iloc[ix].fillna(pd.Timestamp.max).to_numpy()
            # 第一处日历日期>=摘牌日即减记；超过本路径则不会触发。
            offsets=np.searchsorted(dates.to_numpy(),d,side='left')-i
            labels=marked_paths(opening[i+1,ix],prices[i+1:i+h+1,ix],opening[i+h+1,ix],offsets,cfg['path_labels'])
            for key in labels:frame[f'{key}_{h}']=labels[key].to_numpy()
            frame[f'label_end_{h}']=dates[i+h+1]
        blocks.append(frame.reset_index(names='ts_code'))
    panel=pd.concat(blocks,ignore_index=True);print('已知特征与逐股路径生成',panel.shape,flush=True)
    h=cfg['primary_horizon'];cut=pd.Timestamp(cfg['fit_end'])
    train=panel[panel[cfg['fit_frequency']]&panel.trend&(panel[f'label_end_{h}']<=cut)].copy()
    assert train.date.nunique()>=cfg['min_fit_dates']
    weights=1/train.groupby('date').ts_code.transform('size')
    xcols=['x_'+c for c in cfg['features']];models={};coeff=[]
    for target in ['sustained','roundtrip']:
        model=LogisticRegression(**cfg['model']).fit(train[xcols],train[f'{target}_{h}'].astype(int),sample_weight=weights)
        panel['p_'+target]=model.predict_proba(panel[xcols])[:,1];models[target]=model
        coeff.extend(dict(target=target,feature=name,coefficient=value) for name,value in zip(['intercept']+xcols,np.r_[model.intercept_,model.coef_[0]]))
    panel['path_model']=panel.p_sustained-panel.p_roundtrip
    panel.to_parquet(out/'entry_paths.parquet',index=False);pd.DataFrame(coeff).to_csv(out/'model_coefficients.csv',index=False)
    test=panel[panel.date.between(*map(pd.Timestamp,cfg['evaluation']))].copy();comparison=[];calibration=[];memberships=[]
    metrics=['ret','mae','mfe','mdd','giveback','late_return','sustained','roundtrip','entry_missing','missing_marks','exit_marked','delisted']
    for freq in cfg['review_frequencies']:
        sampled=test[test[freq]]
        for policy in cfg['policies']:
            members=patient_memberships(sampled,policy,cfg['max_positions'],cfg['exit_confirmations'],cfg['path_model_min_score'] if policy=='path_model' else -np.inf);members['frequency']=freq;memberships.append(members)
        for date,g in sampled.groupby('date'):
            eligible=g[g.trend]
            if eligible.empty:continue
            groups={'trend_pool':eligible}
            for policy in cfg['policies']:
                pool=eligible[eligible.path_model.gt(cfg['path_model_min_score'])] if policy=='path_model' else eligible
                groups[policy]=pool.sort_values([policy,'ts_code'],ascending=[False,True]).head(cfg['max_positions'])
            for horizon in cfg['horizons']:
                if f'label_end_{horizon}' not in g or g[f'label_end_{horizon}'].isna().all():continue
                for policy,chosen in groups.items():
                    row=dict(frequency=freq,date=date,year=date.year,horizon=horizon,policy=policy,n=len(chosen))
                    row.update({key:pd.to_numeric(chosen[f'{key}_{horizon}']).mean() for key in metrics})
                    if chosen.empty:row.update(ret=0.,entry_missing=0.)
                    comparison.append(row)
            if eligible[f'label_end_{h}'].notna().any():
                for target in ['sustained','roundtrip']:
                    pred=eligible['p_'+target];truth=eligible[f'{target}_{h}'].astype(float)
                    prior=np.average(train[f'{target}_{h}'].astype(float),weights=weights)
                    calibration.append(dict(frequency=freq,date=date,year=date.year,target=target,n=len(eligible),
                        predicted=pred.mean(),observed=truth.mean(),brier=((pred-truth)**2).mean(),prior_brier=((prior-truth)**2).mean()))
    comparisons=pd.DataFrame(comparison);comparisons.to_csv(out/'policy_paths.csv',index=False)
    comparisons.groupby(['frequency','horizon','policy'])[metrics].mean().to_csv(out/'policy_summary.csv')
    comparisons.groupby(['frequency','year','horizon','policy'])[metrics].mean().to_csv(out/'policy_yearly.csv')
    calibration=pd.DataFrame(calibration);calibration.to_csv(out/'calibration_dates.csv',index=False)
    calibration.groupby(['frequency','year','target'])[['predicted','observed','brier','prior_brier']].mean().to_csv(out/'calibration_yearly.csv')
    pd.concat(memberships,ignore_index=True).to_csv(out/'patient_memberships.csv',index=False)
    # 仅做描述分层；未来路径类别不得回填入场条件。
    past=train.copy();past.groupby(f'state_{h}')[cfg['features']].median().to_csv(out/'fit_state_feature_medians.csv')
    diagnostic=[]
    for date,g in test[test.monthly & test.trend & test[f'label_end_{h}'].notna()].groupby('date'):
        g=g.copy();g['momentum_bin']=pd.qcut(g.momentum.rank(method='first'),5,labels=False,duplicates='drop')
        for col in ['continuity','jump_share','extension','volatility']:
            for band,cell in g.groupby('momentum_bin'):
                if len(cell)<10:continue
                ranked=cell.sort_values(col);k=max(1,len(cell)//5)
                for level,sub in [('low',ranked.head(k)),('high',ranked.tail(k))]:
                    diagnostic.append(dict(date=date,feature=col,momentum_bin=band,level=level,n=len(sub),
                        ret=sub[f'ret_{h}'].mean(),sustained=sub[f'sustained_{h}'].mean(),roundtrip=sub[f'roundtrip_{h}'].mean()))
    diagnostic=pd.DataFrame(diagnostic);diagnostic.to_csv(out/'matched_momentum_diagnostics.csv',index=False)
    diagnostic.groupby(['feature','level'])[['ret','sustained','roundtrip']].mean().to_csv(out/'matched_summary.csv')
    manifest.update(status='complete',fit_dates=int(train.date.nunique()),fit_rows=len(train),last_fit_label=str(train[f'label_end_{h}'].max()),
        prefix_checks=18,selection_note='Fixed three policies, no tuning on 2019-2021; these years were previously inspected',
        path_convention='Adjusted price marking, missing entry cash, missing marks carried, delisting zero; no costs or execution; membership is only a signal simulation')
    ExperimentLedger().log('trend_entry_quality','60d_path_probabilities','FIT2014-2018 development-forward2019-2021',np.nan,np.nan,np.nan,np.nan,
        params=cfg,verdict='DIAGNOSTIC_NOT_ADMITTED',experiment_id=str(out.relative_to(ROOT)),source_hashes=manifest['source_hashes'],
        note='Repeated development data; retained all policies; no probability score used as expected return')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(comparisons.groupby(['frequency','horizon','policy'])[['ret','mae','sustained','roundtrip']].mean().to_string(),flush=True)


if __name__=='__main__':main()
