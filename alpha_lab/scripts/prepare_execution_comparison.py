"""重建真实历史14:30特征，与EOD特征分别封存；未来成交行情不参与选股。"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from execution_market_feed import file_hash
from trend_entry import entry_features
from patient_trend import trend_state
from conditional_hold import build_conditional_holds
from selection_stability import buffered_holds
from rolling_review import decision_dates
from portfolio_cash_state import basket_features
from monthly_risk_live import risk_history,read_index_close


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str))


def load_minutes(folder):
    frames=[]
    for receipt in sorted(folder.glob('*.json')):
        item=json.loads(receipt.read_text());path=receipt.with_suffix('.parquet')
        if item['status']!='downloaded' or file_hash(path)!=item['sha256']:raise ValueError('分钟下载缺失或原件改变')
        x=pd.read_parquet(path)
        if len(x):frames.append(x)
    x=pd.concat(frames,ignore_index=True)
    x['timestamp']=pd.to_datetime(x.trade_time);x['date']=x.timestamp.dt.normalize();x['time']=x.timestamp.dt.strftime('%H:%M:%S')
    if x.duplicated(['ts_code','timestamp']).any():raise ValueError('分钟主键重复')
    if not np.isfinite(x[['close','vol']]).all().all() or x.close.le(0).any() or x.vol.lt(0).any():raise ValueError('分钟价格或量非法')
    return x.sort_values(['ts_code','timestamp'])


def build_matrices(raw,basics,st,index,calendar,cfg,model,out,start,end):
    out.mkdir(parents=True,exist_ok=False)
    wide={k:raw.pivot(index='date',columns='ts_code',values=k).sort_index() for k in ['hfq_close','total_mv_yuan','turnover_rate']}
    close=wide['hfq_close'];f=entry_features(close,wide['total_mv_yuan'],wide['turnover_rate'])
    _,trend=trend_state(close,cfg['trend'])
    observable=close.gt(0)&close.shift(60).gt(0)&close.shift(120).gt(0)&close.where(close>0).rolling(120,min_periods=120).mean().notna()
    b=basics.set_index('ts_code');listing=b.list_date.reindex(close.columns);delisting=b.delist_date.reindex(close.columns)
    valid=pd.DataFrame(np.logical_and.reduce([np.isfinite(v.to_numpy()) for v in f.values()]),index=close.index,columns=close.columns)
    clean=close.where(close>0);vol=clean.pct_change(fill_method=None).rolling(20,min_periods=20).std();dd=clean/clean.rolling(120,min_periods=120).max()-1
    needed=set();recovery=basket_features(close)
    audits=[]
    for offset in [1,8,15,22]:
        local=json.loads(json.dumps(cfg));local['schedule']['parameter']=offset
        reviews=decision_dates(close.index[close.index>='2022-01-01'],'calendar',offset,20,'2022-01-01')
        holds,_=build_conditional_holds(close,trend,observable,reviews,local['hold'])
        score=pd.DataFrame(np.nan,index=reviews,columns=close.columns)
        for day in reviews:
            ok=valid.loc[day]&(day-listing).dt.days.ge(60)&(delisting.isna()|delisting.gt(day))
            active=st.loc[st.known_date.le(day)&st.start.le(day)&(st.end.isna()|st.end.ge(day)),'ts_code'];ok.loc[ok.index.isin(active)]=False
            ranks=f['turnover'].loc[day,ok].rank(pct=True)-.5
            codes=ranks.index[ranks.index.str.match(r'^(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)$')]
            score.loc[day,codes]=(-ranks.loc[codes]).where(trend.loc[day,codes])
        hold,_,exits=buffered_holds(score,holds['hybrid'].loc[reviews],10,2,5)
        signals=score.copy()*np.nan
        for day,row in score.iterrows():
            top=row.dropna().sort_values(ascending=False,kind='stable').head(20)
            signals.loc[day,top.index]=top.mask(exits.loc[day,top.index])
        active_reviews=reviews[(reviews>=start)&(reviews<=end)]
        codes=sorted(set().union(*(set(signals.loc[d].dropna().index) for d in active_reviews)))
        needed.update(codes)
        x=np.stack([vol.loc[reviews,codes].to_numpy(),dd.loc[reviews,codes].to_numpy()],axis=-1)
        pred=((x-np.array(model['center']))/np.array(model['scale']))@np.array(model['coefficients'])+model['intercept']
        risk=pd.DataFrame(np.minimum(1.,model['reference_risk']/np.maximum(-pred,model['reference_risk'])),index=reviews,columns=codes)
        if (risk.loc[active_reviews].isna() & signals.loc[active_reviews,codes].notna()).any().any():raise ValueError('当期候选股票风险覆盖缺失')
        folder=out/f'calendar_{offset:02d}';folder.mkdir()
        for name,frame in [('signals',signals[codes]),('hold',hold[codes]),('risk',risk),
            ('forecasts',pd.DataFrame(model['expected_return'],index=reviews,columns=codes))]:frame.to_parquet(folder/(name+'.parquet'))
        rh=risk_history(index,str(pd.Timestamp(end).date()),calendar,local);save(folder/'market_risk.json',rh)
        audits.append(dict(offset=offset,review_dates=[str(d.date()) for d in active_reviews],candidates=len(codes)))
    for key,frame in recovery.items():frame.loc[start:end,sorted(needed)].to_parquet(out/('recovery_'+key+'.parquet'))
    # 截断重算代表性股票的全部9类特征，不能让7月之后的分钟输入改变6月已知值。
    cut=pd.Timestamp('2026-06-30');selected=sorted(needed)
    prefix=entry_features(close.loc[:cut,selected],wide['total_mv_yuan'].loc[:cut,selected],wide['turnover_rate'].loc[:cut,selected])
    for key,value in prefix.items():pd.testing.assert_frame_equal(value,f[key].loc[:cut,selected],check_exact=True)
    close.loc[start:end,selected].to_parquet(out/'decision_adjusted_close.parquet')
    save(out/'verification.json',dict(full_A_columns=len(close.columns),schedules=audits,nine_feature_prefixes_exact=True,codes=selected))
    return needed


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    protocol=json.loads((run/'minutes/protocol.json').read_text());start=pd.Timestamp(protocol['start']);end=pd.Timestamp(protocol['end'])
    done=json.loads((run/'minutes/download_summary.json').read_text())
    if done['failed']:raise ValueError('下载仍有失败项')
    out=run/'prepared';out.mkdir(exist_ok=False)
    cfg=yaml.safe_load((ROOT/'configs/monthly_risk_close_day15.yaml').read_text())
    old=ROOT/'outputs/monthly_manual_day15/observation_final'
    raw=pd.read_parquet(old/'feature_panel.parquet');raw=raw[raw.date<=end]
    market=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',columns=['date','ts_code','open','high','low','close','pre_close','adj_factor','hfq_close','vol_share','total_mv_yuan','float_share'],
        filters=[('date','>=',start-pd.Timedelta(days=200)),('date','<=',end)]).sort_values(['ts_code','date'])
    base=market.set_index(['date','ts_code']);daily=market[market.date>=start].copy();expected=daily.set_index(['date','ts_code'])
    minute=load_minutes(run/'minutes/stocks30');index_minute=load_minutes(run/'minutes/index30')
    aggregate=minute.groupby(['date','ts_code']).agg(close=('close','last'),volume=('vol','sum'))
    missing=expected.index.difference(aggregate.index);extra=aggregate.index.difference(expected.index)
    save(out/'coverage.json',dict(expected_pairs=len(expected),minute_pairs=len(aggregate),missing=list(missing),extra=list(extra)))
    if len(missing) or len(extra):raise ValueError('全A分钟与日线覆盖不一致')
    aggregate=aggregate.reindex(expected.index)
    price_error=(aggregate.close-expected.close).abs();volume_error=(aggregate.volume-expected.vol_share).abs()
    bad=(price_error>.011)|(volume_error>np.maximum(100.,expected.vol_share*1e-5))
    if bad.any():
        aggregate.join(expected[['close','vol_share']],rsuffix='_daily').loc[bad].to_csv(out/'bad_daily_agreement.csv')
        raise ValueError('分钟/日线校验不一致')
    at=minute[minute.time.eq('14:30:00')].set_index(['date','ts_code'])
    cum=minute[minute.time.le('14:30:00')].groupby(['date','ts_code']).vol.sum()
    if len(expected.index.difference(at.index)):raise ValueError('缺完整14:30截面')
    market['previous_mv']=market.groupby('ts_code').total_mv_yuan.shift(1)
    market['previous_close']=market.groupby('ts_code').close.shift(1)
    market['previous_hfq']=market.groupby('ts_code').hfq_close.shift(1)
    market['previous_float']=market.groupby('ts_code').float_share.shift(1)*10000
    reference=market[market.date>=start].set_index(['date','ts_code'])
    proxy=pd.DataFrame(index=expected.index)
    proxy['hfq_close']=reference.previous_hfq*at.close/reference.pre_close
    proxy['total_mv_yuan']=reference.previous_mv*at.close/reference.previous_close
    proxy['turnover_rate']=cum/reference.previous_float*100
    projected=pd.concat([raw[raw.date<start],proxy.reset_index()[raw.columns]],ignore_index=True)
    projected.to_parquet(out/'feature_proxy.parquet',index=False)
    marks=at[['close']].join(cum.rename('vol_share')).reset_index();marks.to_parquet(out/'decision_market.parquet',index=False)
    daily.to_parquet(out/'market.parquet',index=False)
    idx=read_index_close(ROOT/'outputs/monthly_manual_current/index_verified/index_close.parquet');idx=idx.loc[:end]
    actual_index=index_minute.groupby('date').close.last()
    if not actual_index.index.equals(pd.DatetimeIndex(protocol['sessions'])):raise ValueError('指数分钟历史覆盖不完整')
    error=(actual_index-idx.loc[actual_index.index]).abs()
    if error.max()>.011:raise ValueError('指数分钟/日线收盘不一致')
    pi=index_minute[index_minute.time.eq('14:30:00')].set_index('date').close
    if not pi.index.sort_values().equals(actual_index.index):raise ValueError('指数缺14:30报价')
    proxy_index=idx.copy();proxy_index.loc[pi.index]=pi
    idx.rename('close').to_frame().to_parquet(out/'index_eod.parquet')
    proxy_index.rename('close').to_frame().to_parquet(out/'index_proxy.parquet')
    calendar=dict(source='sealed EOD historical sessions',sessions=[str(d.date()) for d in idx.index])
    basics=pd.read_parquet(old/'basics.parquet');st=pd.read_parquet(ROOT/'outputs/selection_stability/20260914T070228_976007Z/inputs/st.parquet')
    model=json.loads((ROOT/'outputs/shadow_execution/20260914T054517_770961Z/model_2026.json').read_text())
    needed=set()
    for name,features,index in [('eod',raw,idx),('proxy1430',projected,proxy_index)]:
        print('building',name,flush=True);needed|=build_matrices(features,basics,st,index,calendar,cfg,model,out/name,start,end)
    basics.to_parquet(out/'basics.parquet',index=False);st.to_parquet(out/'st.parquet',index=False)
    save(out/'verification.json',dict(start=start,end=end,codes=sorted(needed),rows=len(daily),
        stock_close_max_error=float(price_error.max()),stock_volume_max_relative_error=float((volume_error/expected.vol_share).max()),
        index_close_max_error=float(error.max()),input_sha256={str(p):file_hash(p) for p in
            [old/'feature_panel.parquet',ROOT/'configs/monthly_risk_close_day15.yaml',run/'minutes/protocol.json']},
        intraday_vintage='historical bars downloaded now; not proof of original live feed availability',live_orders=False))
    print('prepared',len(needed),'execution candidates',flush=True)


if __name__=='__main__':main()
