#!/usr/bin/env python3
"""只在TRAIN检验趋势持有的价格路径；不是可交易回测或新样本外成绩。"""
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from patient_trend import review_dates,trend_state
from ledger import ExperimentLedger


def old_holding_paths(out,cfg):
    source=ROOT/cfg['reference_execution'];rows=[];ages=[]
    for rule in ['none','extreme_original','extreme_recovery']:
        name=f'base_50000_{rule}'
        states=pd.read_parquet(source/f'{name}_asset_states.parquet').sort_values(['ts_code','date'])
        daily=pd.read_csv(source/f'{name}_daily.csv',index_col='date',parse_dates=True)
        new=states.previous_value.eq(0)&states.value.gt(0)
        states['episode']=new.groupby(states.ts_code).cumsum()
        loc=pd.Series(np.arange(len(daily)),index=daily.index)
        states['day']=states.date.map(loc)
        states['age']=states.day-states.groupby(['ts_code','episode']).day.transform('min')+1
        for (code,episode),g in states.groupby(['ts_code','episode']):
            rows.append(dict(rule=rule,ts_code=code,episode=episode,start=g.date.min(),end=g.date.max(),
                holding_trading_days=int(g.age.max()),closed=bool(g.iloc[-1].value==0),pnl=g.pnl.sum()))
        states['age_bucket']=pd.cut(states.age,[0,5,20,60,120,np.inf],labels=['1_5','6_20','21_60','61_120','121_plus'])
        for bucket,g in states.groupby('age_bucket',observed=True):
            ages.append(dict(rule=rule,age_bucket=bucket,pnl=g.pnl.sum(),capital_days=g.previous_value.sum()))
    pd.DataFrame(rows).to_csv(out/'existing_continuous_holdings.csv',index=False)
    pd.DataFrame(ages).to_csv(out/'existing_age_pnl.csv',index=False)


def main():
    path=ROOT/'configs/patient_trend_research.yaml';cfg=yaml.safe_load(path.read_text())
    start,end=map(pd.Timestamp,cfg['train'])
    if end>=pd.Timestamp('2022-01-01'):raise ValueError('本轮趋势假设只允许TRAIN行情')
    out=ROOT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    (out/'config.yaml').write_bytes(path.read_bytes())
    sources=[path,Path(__file__),ROOT/'src/patient_trend.py',ROOT/'src/portfolio.py']
    manifest=dict(status='registered',test_market_data_used=False,new_valid_market_data_used=False,
        returns='unhedged adjusted-price path proxy; no execution, fees or portfolio CAGR',
        no_independence_claim='overlapping horizons; dates/stocks not independent trials',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out,flush=True)
    old_holding_paths(out,cfg)
    frame=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',columns=['date','ts_code','hfq_close','hfq_open'],
        filters=[('date','>=',start),('date','<=',end)])
    close=frame.pivot(index='date',columns='ts_code',values='hfq_close').sort_index()
    opening=frame.pivot(index='date',columns='ts_code',values='hfq_open').reindex_like(close)
    score,trend=trend_state(close,cfg['trend'])
    for cutoff in [pd.Timestamp('2017-06-30'),pd.Timestamp('2020-06-30')]:
        short,_=trend_state(close.loc[:cutoff],cfg['trend']);pd.testing.assert_frame_equal(short,score.loc[:cutoff])
    basic=pd.read_parquet(ROOT/'data/raw/stock_basic_all.parquet').set_index('ts_code')
    listed=pd.to_datetime(basic.list_date).reindex(close.columns)
    delisted=pd.to_datetime(basic.delist_date).reindex(close.columns)
    names=pd.read_parquet(ROOT/'data/raw/namechange_all.parquet');names=names[names.name.str.contains('ST',case=False,na=False)].copy()
    names['start']=pd.to_datetime(names.start_date);names['end']=pd.to_datetime(names.end_date)
    dates=close.index;position={d:i for i,d in enumerate(dates)};arr=close.to_numpy(float);op=opening.to_numpy(float)
    rows=[]
    for freq in cfg['review_frequencies']:
        for date in review_dates(dates,freq):
            i=position[date]
            if i<cfg['trend']['long_window']:continue
            eligible=(date-listed).dt.days.ge(cfg['min_listing_calendar_days'])&(delisted.isna()|delisted.gt(date))&close.loc[date].notna()
            st=names.loc[names.start.le(date)&(names.end.isna()|names.end.ge(date)),'ts_code'];eligible.loc[eligible.index.isin(st)]=False
            if eligible.sum()<cfg['min_cross_section']:continue
            ranked=score.loc[date].where(eligible).dropna().sort_values(ascending=False,kind='stable')
            groups={'eligible_market':np.flatnonzero(eligible.to_numpy()),
                'trend_eligible':np.flatnonzero((eligible&trend.loc[date]).to_numpy()),
                'trend_top5':close.columns.get_indexer(ranked.head(cfg['retail']['max_positions']).index)}
            for h in cfg['label_horizons']:
                if i+h+1>=len(dates):continue
                entry=op[i+1];finish=op[i+h+1]
                observed=arr[i+1:i+h+1]
                # 缺失退出、退市、途中行情缺口分别记录；不把不完整路径的好成绩当证据。
                for group,ix in groups.items():
                    if not len(ix):continue
                    e=entry[ix];f=finish[ix];raw=observed[:,ix]
                    terminated=delisted.iloc[ix].notna().to_numpy()&(delisted.iloc[ix].fillna(pd.Timestamp.max).to_numpy()<=dates[i+h+1].to_datetime64())
                    valid_entry=np.isfinite(e)&(e>0);valid_exit=np.isfinite(f)&(f>0)
                    complete=valid_entry&valid_exit&np.isfinite(raw).all(axis=0)&~terminated
                    # 存续条件下的完整路径与缺失覆盖率并列，另列退市按零回收的端点收益。
                    terminal=np.where(valid_entry&valid_exit,f/e-1,np.nan)
                    terminal[valid_entry&terminated]=-1.
                    wealth=raw[:,complete]/e[complete]
                    path=np.vstack([np.ones(int(complete.sum())),wealth,f[complete]/e[complete]])
                    dd=(path/np.maximum.accumulate(path,axis=0)-1).min(axis=0)
                    peak=path.max(axis=0)-1;trough=path.min(axis=0)-1
                    final=path[-1]-1
                    late=(path[-1]-path[min(5,len(path)-1)])
                    rows.append(dict(frequency=freq,date=date,label_end=dates[i+h+1],horizon=h,group=group,
                        n_candidates=len(ix),n_complete=int(complete.sum()),n_missing_entry=int((~valid_entry).sum()),
                        n_missing_exit=int((~valid_exit).sum()),n_terminated=int(terminated.sum()),
                        endpoint_return=np.nanmean(terminal) if np.isfinite(terminal).any() else np.nan,
                        complete_return=final.mean() if len(final) else np.nan,
                        positive_fraction=(final>0).mean() if len(final) else np.nan,
                        late_pnl_per_initial=late.mean() if len(late) else np.nan,
                        max_drawdown=dd.mean() if len(dd) else np.nan,mae=trough.mean() if len(trough) else np.nan,
                        mfe=peak.mean() if len(peak) else np.nan,giveback=(peak-final).mean() if len(peak) else np.nan))
        print(freq,'完成',flush=True)
    result=pd.DataFrame(rows);result.to_csv(out/'train_paths.csv',index=False)
    valuecols=['endpoint_return','complete_return','positive_fraction','late_pnl_per_initial','max_drawdown','mae','mfe','giveback']
    summary=result.groupby(['frequency','horizon','group'])[valuecols].mean()
    summary.to_csv(out/'train_path_summary.csv')
    result.assign(year=pd.to_datetime(result.date).dt.year).groupby(['frequency','horizon','group','year'])[valuecols].mean().to_csv(out/'train_path_yearly.csv')
    coverage=result.groupby(['frequency','horizon','group'])[['n_candidates','n_complete','n_missing_entry','n_missing_exit','n_terminated']].sum()
    coverage.to_csv(out/'path_coverage.csv')
    manifest.update(status='complete',prefix_checks=2,last_label=str(result.label_end.max()),
        data_rows=len(frame),paths_rows=len(result),input_slice_hash=hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest())
    ledger=ExperimentLedger();ledger.log('patient_trend_horizon_diagnostic','20_60_120d_price_paths','TRAIN2014-2021',np.nan,np.nan,np.nan,np.nan,
        params=cfg,verdict='DIAGNOSTIC_NOT_TRADING_RESULT',experiment_id=str(out.relative_to(ROOT)),source_hashes=manifest['sources'],
        note='All groups/horizons retained, overlapping dates; complete path selection coverage reported; original VALID ledger durations only',
        selection_note='User preference max5 weekly/monthly long trend; no champion selected')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(summary.to_string(),flush=True)


if __name__=='__main__':main()
