"""独立矩阵解复算预测、成熟标签和原始恢复资格，避免只检验同一拟合函数。"""
from pathlib import Path
import sys,json,platform
import pandas as pd
import numpy as np
import sklearn,scipy
from research_exit_evidence import ROOT,verify,save,digest
run=Path(sys.argv[1]).resolve();cfg=verify(run);parent=ROOT/cfg['parent'];price=pd.read_parquet(parent/'inputs/daily_close.parquet');pred=pd.read_csv(run/'models/predictions.csv',parse_dates=['date']);records=[]
for cohort in cfg['cohorts']:
    data=pd.read_parquet(run/'data'/f'{cohort}_events.parquet')
    for row in data.itertuples():
        i=price.index.get_loc(row.date)
        for h in cfg['horizons']:
            start=i+1;end=start+h;actual=getattr(row,f'ret_{h}')
            if end>=len(price):assert np.isnan(actual);continue
            values=price.iloc[start:end+1][row.ts_code].to_numpy()
            if not np.isfinite(values).all() or (values<=0).any():assert np.isnan(actual);continue
            assert abs(actual-(values[-1]/values[0]-1))<1e-12
            assert getattr(row,f'label_end_{h}')==price.index[end]
    data=data[data.nonoverlap_primary]
    for _,g in data.groupby('ts_code'):
        indexes=price.index.get_indexer(g.sort_values('date').date)
        assert (np.diff(indexes)>cfg['nonoverlap_days']).all()
    for h in cfg['horizons']:
        for year in cfg['prediction_years']:
            cutoff=pd.Timestamp(f'{year}-01-01');train=data[data.date.lt(cutoff)&data[f'label_end_{h}'].lt(cutoff)&data[f'ret_{h}'].notna()&data.features_complete]
            old=pd.read_csv(run/'models'/f'{cohort}_h{h}_{year}_train.csv',parse_dates=['date',f'label_end_{h}'])
            assert set(zip(train.date,train.ts_code))==set(zip(old.date,old.ts_code));assert old[f'label_end_{h}'].lt(cutoff).all()
            p=pred[pred.cohort.eq(cohort)&pred.horizon.eq(h)&pred.year.eq(year)]
            x=data.set_index(['date','ts_code']).loc[p.set_index(['date','ts_code']).index,cfg['features']].to_numpy()
            y=train[f'ret_{h}'].to_numpy();mean=float(y.mean()) if len(y) else 0.
            predicted=np.full(len(x),mean);fit=len(train)>=cfg['model']['min_train_events'] and train.date.nunique()>=cfg['model']['min_train_dates']
            if fit:
                xt=train[cfg['features']].to_numpy();center=xt.mean(axis=0);scale=xt.std(axis=0);scale[scale==0]=1
                z=(xt-center)/scale;zcenter=z.mean(axis=0);z-=zcenter
                coef=np.linalg.solve(z.T@z+cfg['model']['alpha']*np.eye(z.shape[1]),z.T@(y-mean))
                good=np.isfinite(x).all(axis=1);predicted[good]=((x[good]-center)/scale-zcenter)@coef+mean
            error=float(np.max(np.abs(predicted-p.prediction))) if len(p) else 0.;assert error<1e-10
            records.append(dict(cohort=cohort,horizon=h,year=year,manual_ridge_max_error=error,fitted=bool(fit),train_labels_mature=True))
# 每个恢复机会均发生在真实完成卖出之后，且当日原信号非空。
raw=pd.read_csv(run/'data/recovery_account_events.csv',parse_dates=['date','sale_completed','exit_decision'])
for case,g in raw.groupby('case'):
    folder=parent/'execution'/case;schedule=case.split('_',1)[1].removesuffix('_hybrid');sig=pd.read_parquet(parent/'schedules'/schedule/'signals.parquet')
    trades=pd.read_csv(folder/'trades.csv',parse_dates=['date','signal_date']);holds=pd.read_csv(folder/'holdings.csv',parse_dates=['date'])
    for r in g.itertuples():
        assert r.date>r.sale_completed and pd.notna(sig.at[r.date,r.ts_code])
        assert len(trades[trades.date.eq(r.sale_completed)&trades.ts_code.eq(r.ts_code)&trades.side.eq('sell')&trades.signal_date.eq(r.exit_decision)])
        assert holds[holds.date.eq(r.sale_completed)&holds.ts_code.eq(r.ts_code)].empty
save(run/'independent_verification.json',dict(status='passed',models=records,recovery_events_verified=len(raw),labels_recomputed=True,nonoverlap_checked=True,source_sha256=digest(__file__)))
save(run/'environment.json',dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__))
print('Independent prediction/label/time verification passed for',len(records),'annual model slots;',len(raw),'recovery events')
