#!/usr/bin/env python3
"""固定完整检验矩阵，并只用TRAIN估计排名分组收益与滚动误差。"""
from datetime import datetime,timezone
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_gp_mining import build_wide
from synthesis_and_test import build_all_factors
from labels import purged_dates
from risk_states import fixed_risk_plan


def recovery_plan(features,cfg):
    spec=yaml.safe_load((ROOT/cfg['risk_config']).read_text())
    original=fixed_risk_plan(features,'extreme_downside',spec)
    recovery=cfg['risk_recovery']
    safe=(~original.raw_risk)&original.valid & (features.short_return>recovery['short_return_min']) & (features.trend_return>recovery['trend_return_min']) & (features.above_ma_ratio>recovery['breadth_min'])
    allow=0;streak=0;rows=[]
    for date in features.index:
        if original.loc[date,'raw_risk'] or not original.loc[date,'valid']:
            allow=0;streak=0
        elif allow==0:
            streak=streak+1 if safe.loc[date] else 0
            if streak>=recovery['confirmation_days']:allow=1
        rows.append(allow)
    return pd.Series(rows,index=features.index,name='allowed')


def main():
    cp=ROOT/'configs/retail_validation.yaml';cfg=yaml.safe_load(cp.read_text())
    if pd.Timestamp(cfg['train'][1])>=pd.Timestamp('2022-01-01') or pd.Timestamp(cfg['valid'][1])>=pd.Timestamp('2024-01-01'):raise ValueError('时间边界错误')
    out=ROOT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.yaml').write_text(cp.read_text())
    manifest=dict(status='preregistered',test_market_data_used=False,fresh_holdout=False,
        scope=['TRAIN风险持续与恢复','TRAIN收益估计滚动校准','整手旧仓联合优化','资金矩阵','成本压力','执行延迟','风险上限压力','前视和对账'],
        config_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(f'完整检验登记 {out}',flush=True)
    risk_features=pd.read_parquet(ROOT/cfg['risk_features'])
    rc=yaml.safe_load((ROOT/cfg['risk_config']).read_text())
    old=fixed_risk_plan(risk_features,'extreme_downside',rc).allowed
    new=recovery_plan(risk_features,cfg)
    pd.DataFrame(dict(extreme_original=old,extreme_recovery=new)).to_csv(out/'risk_plans.csv')
    # 仅用于风险持续性诊断的未来市场代理收益，不冒充可交易指数回测。
    forward=(1+risk_features.market_ret).rolling(5).apply(np.prod,raw=True).shift(-5)-1
    risk_rows=[]
    for year in cfg['walk_forward_years']:
        for name,plan in [('extreme_original',old),('extreme_recovery',new)]:
            for kind,condition in [('exit',plan.diff().eq(-1)),('reentry',plan.diff().eq(1))]:
                dates=plan.index[condition & plan.index.year.eq(year)] if hasattr(plan.index.year,'eq') else plan.index[condition & (plan.index.year==year)]
                for date in dates:
                    # 标签不得跨年末；没有成熟标签时保留缺失。
                    idx=risk_features.index.get_loc(date)
                    end=risk_features.index[idx+5] if idx+5<len(risk_features) else pd.NaT
                    value=forward.loc[date] if pd.notna(end) and end.year==year else np.nan
                    risk_rows.append(dict(rule=name,year=year,event=kind,date=date,forward5=value,label_end=end))
    pd.DataFrame(risk_rows).to_csv(out/'train_risk_events.csv',index=False)
    print('TRAIN风险事件冻结，开始同口径收益校准',flush=True)
    cols=dict(ret1='pct_chg',vol='vol_share',amt='amount_yuan',to='turnover_rate',high='high',low='low',close='close',hclose='hfq_close',hopen='hfq_open',buy_lg='buy_lg_amount_yuan',sell_lg='sell_lg_amount_yuan',pe='pe_ttm',pb='pb')
    w=build_wide(cols,end_date=cfg['train'][1]);w['ret1']/=100
    w['vwap_dev']=w['amt']/w['vol']/w['close']-1
    w['lg_share']=(w['buy_lg']+w['sell_lg'])/(w['amt']+1e-6)
    factors=build_all_factors(w,cfg['horizon'])
    factors['ep']=1/w['pe'].where(w['pe']>0);factors['bp']=1/w['pb'].where(w['pb']>0)
    dates=purged_dates(w['close'].loc[slice(*cfg['train'])].index,cfg['horizon']+1,cfg['embargo'])[::cfg['horizon']]
    common=np.logical_and.reduce([np.isfinite(f.loc[dates].to_numpy()) for f in factors.values()])
    scores=factors['lit:amihud'].loc[dates].where(common)
    basics=pd.read_parquet(ROOT/'data/raw/stock_basic_all.parquet').set_index('ts_code')
    listing=pd.to_datetime(basics.list_date,errors='coerce').reindex(scores.columns)
    delisting=pd.to_datetime(basics.delist_date,errors='coerce').reindex(scores.columns)
    names=pd.read_parquet(ROOT/'data/raw/namechange_all.parquet')
    names=names[names.name.str.contains('ST',case=False,na=False)].copy()
    names['start']=pd.to_datetime(names.start_date);names['end']=pd.to_datetime(names.end_date)
    h=cfg['horizon'];entry=w['hopen'].shift(-1).loc[dates];exit=w['hopen'].shift(-h-1).loc[dates]
    last=w['hclose'].ffill().shift(-h).loc[dates]
    labels=exit.fillna(last).div(entry)-1
    labels=labels.where(entry.notna(),0.)
    rows=[];coverage=[]
    for date in dates:
        end=w['close'].index[w['close'].index.get_loc(date)+h+1]
        eligible=(date-listing).dt.days.ge(60)&(delisting.isna()|(date<delisting))
        st=set(names.loc[names.start.le(date)&(names.end.isna()|names.end.ge(date)),'ts_code'])
        eligible.loc[eligible.index.isin(st)]=False
        rank=scores.loc[date].where(eligible).rank(ascending=False,method='first')
        bucket=pd.cut(rank,cfg['rank_edges'],labels=False)
        y=labels.loc[date].copy();y.loc[delisting.notna()&delisting.le(end)&entry.loc[date].notna()]=-1.
        for b in sorted(bucket.dropna().unique()):
            codes=bucket.index[bucket==b];values=y[codes]
            rows.append(dict(date=date,label_end=end,year=date.year,bucket=int(b),mean_return=values.mean(),n=values.count(),n_missing=values.isna().sum(),min_return=values.min(),max_return=values.max()))
        coverage.append(dict(date=date,n_scores=rank.count(),entry_missing=int(entry.loc[date,rank.notna()].isna().sum()),exit_marked=int(exit.loc[date,rank.notna()].isna().sum())))
    obs=pd.DataFrame(rows);obs.to_csv(out/'train_rank_returns.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'training_coverage.csv',index=False)
    validation=[]
    for year in cfg['walk_forward_years']:
        train=obs[obs.label_end<pd.Timestamp(f'{year}-01-01')]
        test=obs[obs.year.eq(year)]
        means=train.groupby('bucket').mean_return.mean()
        for bucket,g in test.groupby('bucket'):
            validation.append(dict(year=year,bucket=bucket,predicted_return=means.get(bucket,np.nan),realized_return=g.mean_return.mean(),n_dates=len(g)))
    pd.DataFrame(validation).to_csv(out/'return_calibration_walk_forward.csv',index=False)
    model=obs.groupby('bucket').agg(expected_return=('mean_return','mean'),std_between_dates=('mean_return','std'),n_dates=('mean_return','size'))
    if (model.n_dates<cfg['min_training_dates']).any() or not np.isfinite(model.expected_return).all():raise ValueError('收益校准数据不足')
    model.to_csv(out/'frozen_return_model.csv')
    manifest.update(status='prepared',training_last_label=str(obs.label_end.max()),
        label_convention='next-open adjusted-price proxy; missing entry cash, missing exit last mark, delisting zero; actual execution separately audited',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/risk_states.py',ROOT/'scripts/run_gp_mining.py',ROOT/'scripts/synthesis_and_test.py']},
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.csv')})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(model.to_string(),flush=True);print(out,flush=True)


if __name__=='__main__':main()
