"""逐日会计归因、滞后行业参考和真实持仓风险过程；不拟合交易参数。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from defense_diagnostics import risk_evidence
from neutralize import attach_industry_pit
from patient_trend import review_dates
from research_profit_defense import digest, save


def prepare_reference(run):
    """行业参考是全行业理论收益，不用回测最终赢家定义成分。"""
    config = yaml.safe_load((run/'config.yaml').read_text())
    parent = ROOT/config['parent']
    out = run/'reference'
    out.mkdir(exist_ok=False)
    panel_path = ROOT/'data/panel/daily_panel.parquet'
    industry_path = ROOT/'data/raw/sw_l1_member_hist.parquet'
    # 给起始日保留足够前值；所有权重和行业在收益发生前一日确定。
    start = pd.Timestamp(min(config['windows'].values())) - pd.Timedelta(days=40)
    raw = pd.read_parquet(panel_path, columns=['date','ts_code','hfq_close','circ_mv_yuan'],
                         filters=[('date','>=',start),('date','<=',pd.Timestamp(config['end']))])
    raw = raw[raw.ts_code.str.match(r'^(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)$')].copy()
    # 研究窗口已在两只主板换码生效之后，不重复计入历史别名。
    aliases = ['000022.SZ','000043.SZ']
    alias_rows = int(raw.ts_code.isin(aliases).sum())
    raw = raw[~raw.ts_code.isin(aliases)]
    assert not raw.duplicated(['date','ts_code']).any()
    dates = pd.DatetimeIndex(sorted(raw.date.unique()))
    codes = sorted(raw.ts_code.unique())
    close = raw.pivot(index='date',columns='ts_code',values='hfq_close').reindex(index=dates,columns=codes)
    mv = raw.pivot(index='date',columns='ts_code',values='circ_mv_yuan').reindex_like(close)
    # 停牌日按前值标记零收益；不把缺行情股票从该日参考篮子删除。
    prices = close.where(close>0).ffill()
    ret = prices.pct_change(fill_method=None)
    grid = pd.MultiIndex.from_product([dates,codes],names=['date','ts_code']).to_frame(index=False)
    industry = attach_industry_pit(grid,pd.read_parquet(industry_path)).pivot(
        index='date',columns='ts_code',values='sw_l1').reindex_like(close)
    basics = pd.read_parquet(parent/'inputs/basics.parquet').set_index('ts_code')
    listing = pd.to_datetime(basics.list_date).reindex(codes)
    delisting = pd.to_datetime(basics.delist_date).reindex(codes)
    eligible = pd.DataFrame(((dates.to_numpy()[:,None]-listing.to_numpy()[None,:])/np.timedelta64(1,'D'))>=60,
                            index=dates,columns=codes)
    eligible &= pd.DataFrame((dates.to_numpy()[:,None]<delisting.to_numpy()[None,:]) |
                             pd.isna(delisting.to_numpy())[None,:],index=dates,columns=codes)
    st = pd.read_parquet(parent/'inputs/st.parquet')
    for row in st.itertuples():
        if row.ts_code in eligible.columns:
            mask = (dates>=row.start) & (pd.isna(row.end) | (dates<=row.end))
            eligible.loc[mask,row.ts_code] = False
    previous_mv = mv.ffill().shift(1)
    previous_industry = industry.shift(1)
    valid = eligible.shift(1,fill_value=False)&previous_mv.gt(0)&ret.notna()&previous_industry.notna()
    long = pd.DataFrame(dict(date=np.repeat(dates,len(codes)),ts_code=np.tile(codes,len(dates)),
        industry=previous_industry.to_numpy().ravel(),ret=ret.to_numpy().ravel(),
        lag_mv=previous_mv.to_numpy().ravel(),eligible=valid.to_numpy().ravel()))
    selected = long[long.eligible].copy()
    selected['weighted_ret'] = selected.ret*selected.lag_mv
    reference = selected.groupby(['date','industry']).agg(sum_weight=('lag_mv','sum'),
        sum_weighted_ret=('weighted_ret','sum'),sum_ret=('ret','sum'),n=('ret','size'))
    reference['vw_ret'] = reference.sum_weighted_ret/reference.sum_weight
    reference['ew_ret'] = reference.sum_ret/reference.n
    reference.to_parquet(out/'industry_returns.parquet')
    needed = set(pd.read_parquet(parent/'inputs/market.parquet',columns=['ts_code']).ts_code)
    long[long.ts_code.isin(needed)].to_parquet(out/'stock_reference.parquet',index=False)
    market = selected.groupby('date').agg(sum_weight=('lag_mv','sum'),sum_return=('weighted_ret','sum'))
    market['ret'] = market.sum_return/market.sum_weight
    market.to_csv(out/'mainboard_reference.csv')
    names = pd.read_parquet(ROOT/'data/raw/sw_l1_classify.parquet')
    names.to_csv(out/'industry_names.csv',index=False)
    save(out/'manifest.json',dict(status='complete',source_sha256={str(panel_path):digest(panel_path),
        str(industry_path):digest(industry_path),str(Path(__file__)):digest(__file__)},
        alias_rows_removed=alias_rows,rows=len(selected),lagged_industry_and_weights=True,
        missing_stock_return_policy='carry previous mark while suspended; delisting liquidation not simulated',
        caveat='gross adjusted-price theoretical reference; no transaction costs/lots; vendor vintage unverified',
        artifacts={p.name:digest(p) for p in out.iterdir()}))
    print('Reference ready:',len(selected),flush=True)


def ledger(folder, capital):
    daily = pd.read_csv(folder/'daily.csv',index_col='date',parse_dates=True)
    h = pd.read_csv(folder/'holdings.csv',parse_dates=['date'])
    t = pd.read_csv(folder/'trades.csv',parse_dates=['date','decision_date'])
    e = pd.read_csv(folder/'events.csv',parse_dates=['date'])
    values = h.pivot(index='date',columns='ts_code',values='value').reindex(daily.index).fillna(0.)
    t['flow'] = np.where(t.side.eq('sell'),t.gross,-t.gross)
    def pivot(frame, col):
        if len(frame)==0:
            return values*0
        return frame.pivot_table(index='date',columns='ts_code',values=col,aggfunc='sum').reindex_like(values).fillna(0.)
    price_pnl = values.diff().fillna(values)+pivot(t,'flow')
    fees = pivot(t,'fee')
    dividends = pivot(e[e.kind.eq('dividend_receivable')],'amount')
    if 'cash_recovery' in e:
        price_pnl += pivot(e[e.kind.eq('delist_writeoff')],'cash_recovery')
    pnl = price_pnl+dividends-fees
    expected = daily.equity.diff().fillna(daily.equity.iloc[0]-capital)
    np.testing.assert_allclose(pnl.sum(axis=1),expected,atol=1e-6)
    return daily,h,t,e,values,price_pnl,dividends,fees,pnl


def episodes(equity, capital, threshold):
    """事后划分完整回撤段，仅用于描述，不输入任何策略。"""
    peak,peak_date = capital,equity.index[0]
    active = None
    rows=[]
    for date,value in equity.items():
        if value>=peak:
            if active is not None:
                active.update(recovery=date,unrecovered=False)
                if active['mdd']<=threshold:
                    rows.append(active)
            peak,peak_date,active=value,date,None
        else:
            dd=value/peak-1
            if active is None:
                active=dict(peak=peak_date,peak_equity=peak,trough=date,trough_equity=value,mdd=dd)
            if dd<active['mdd']:
                active.update(trough=date,trough_equity=value,mdd=dd)
    if active is not None and active['mdd']<=threshold:
        active.update(recovery=pd.NaT,unrecovered=True)
        rows.append(active)
    return rows


def report(run):
    config=yaml.safe_load((run/'config.yaml').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    out=run/'analysis';out.mkdir(exist_ok=False)
    stock=pd.read_parquet(run/'reference/stock_reference.parquet').set_index(['date','ts_code'])
    industry=pd.read_parquet(run/'reference/industry_returns.parquet')
    names=pd.read_csv(run/'reference/industry_names.csv').set_index('index_code').industry_name
    parent=ROOT/config['signal_parent']
    evidence=risk_evidence(pd.read_parquet(parent/'daily_close.parquet'),config['defense'])
    hold=pd.read_parquet(ROOT/config['parent']/'inputs/hold_hybrid.parquet')
    streak=pd.read_parquet(parent/'monthly_failure_streak.parquet')
    regimes=pd.read_parquet(parent/'weak_regime.parquet')
    rows,industry_rows,drawdowns,summary,check_rows,stock_year_rows=[],[],[],[],[],[]
    bspec=config['bootstrap']
    rng=np.random.default_rng(bspec['seed'])
    for case in manifest['cases']:
        folder=run/'execution'/case['name']
        d,h,t,e,v,price,div,fee,pnl=ledger(folder,config['capital'])
        component=pd.DataFrame(dict(price=price.sum(axis=1),dividends=div.sum(axis=1),
                                   fees=fee.sum(axis=1),net=pnl.sum(axis=1)))
        component.to_csv(out/f"{case['name']}_daily_components.csv")
        for year,group in component.groupby(component.index.year):
            rows.append(dict(name=case['name'],year=year,**group.sum().to_dict()))
        for code in pnl:
            for year,series in pnl[code].groupby(pnl.index.year):
                stock_year_rows.append(dict(name=case['name'],year=year,ts_code=code,pnl_yuan=series.sum()))
        for ep in episodes(d.equity,config['capital'],config['episode_threshold']):
            peak_i,trough_i=d.index.get_loc(ep['peak']),d.index.get_loc(ep['trough'])
            span=d.iloc[peak_i+1:trough_i+1]
            ep.update(name=case['name'],days_to_trough=trough_i-peak_i,
                duration_calendar=((ep['recovery'] if pd.notna(ep['recovery']) else d.index[-1])-ep['peak']).days,
                mean_exposure=(span.stock_value/span.equity).mean())
            drawdowns.append(ep)
        # 逐日前值持仓权重；行业匹配参考排除自身避免自我解释。
        lagv=v.shift(1).fillna(0.)
        theoretical=pd.Series(0.,index=d.index)
        theoretical_ew=theoretical.copy()
        known_value=theoretical.copy()
        covered_net=theoretical.copy()
        uncovered_rows=[]
        case_ind=[]
        for code in v:
            idx=pd.MultiIndex.from_product([d.index,[code]],names=['date','ts_code'])
            ref=stock.reindex(idx).reset_index().set_index('date')
            sector_key=pd.MultiIndex.from_arrays([d.index,ref.industry],names=['date','industry'])
            sector=industry.reindex(sector_key).reset_index(drop=True).set_axis(d.index)
            own_w=ref.lag_mv.where(ref.eligible.fillna(False),0.).fillna(0.)
            own_r=ref.ret.fillna(0.)
            bw=sector.sum_weight-own_w
            br=(sector.sum_weighted_ret-own_w*own_r)/bw.where(bw>0)
            bn=sector.n-ref.eligible.fillna(False).astype(int)
            er=(sector.sum_ret-own_r*ref.eligible.fillna(False).astype(int))/bn.where(bn>0)
            known_value+=lagv[code].where(br.notna(),0.)
            covered_net+=pnl[code].where(br.notna(),0.)
            absent=br.isna() & ((lagv[code]>0) | pnl[code].ne(0))
            if absent.any():
                uncovered_rows.append(dict(ts_code=code,days=int(absent.sum()),
                    first=d.index[absent][0],last=d.index[absent][-1],
                    pnl_yuan=pnl[code].where(absent,0.).sum(),
                    exposure_yuan_days=lagv[code].where(absent,0.).sum()))
            theoretical+=lagv[code]*br.fillna(0.)
            theoretical_ew+=lagv[code]*er.fillna(0.)
            for year in sorted(d.index.year.unique()):
                frame=pd.DataFrame(dict(industry=ref.industry.fillna('__unknown__'),
                    pnl=pnl[code],price=price[code],dividends=div[code],fees=fee[code],
                    prior_value=lagv[code]))
                frame=frame.loc[frame.index.year==year]
                for sector_name,part in frame.groupby('industry'):
                    case_ind.append(dict(name=case['name'],year=year,industry=sector_name,
                        industry_name=names.get(sector_name,'未知'),**part.drop(columns='industry').sum().to_dict()))
        coverage=known_value.sum()/lagv.sum().sum() if lagv.sum().sum()>0 else 1.
        # 行业原表缺失不得静默用当前行业回填，也不能把未知行业收益算成alpha。
        pd.DataFrame(uncovered_rows,columns=['ts_code','days','first','last','pnl_yuan','exposure_yuan_days']).to_csv(
            out/f"{case['name']}_missing_industry.csv",index=False)
        initial=d.equity.shift(1).fillna(config['capital'])
        reference=pd.DataFrame(dict(account_ret=d.ret,covered_account_ret=covered_net/initial,
            sector_vw_ret=theoretical/initial,
            sector_ew_ret=theoretical_ew/initial,exposure=lagv.sum(axis=1)/initial,
            sector_proxy_pnl=theoretical,residual_pnl=covered_net-theoretical,
            unclassified_pnl=component.net-covered_net))
        reference.to_csv(out/f"{case['name']}_sector_reference.csv")
        delta=(reference.covered_account_ret-reference.sector_vw_ret).to_numpy()
        n=len(delta);block=bspec['block_days']
        starts=rng.integers(0,n,size=(bspec['samples'],int(np.ceil(n/block))))
        ix=((starts[:,:,None]+np.arange(block))%n).reshape(len(starts),-1)[:,:n]
        lo,hi=np.quantile(delta[ix].mean(axis=1)*244,bspec['quantiles'])
        summary.append(dict(name=case['name'],net_profit=component.net.sum(),price_pnl=component.price.sum(),
            dividends=component.dividends.sum(),fees=component.fees.sum(),sector_proxy_pnl=theoretical.sum(),
            residual_pnl=(covered_net-theoretical).sum(),unclassified_pnl=(component.net-covered_net).sum(),
            mean_annual_residual=delta.mean()*244,
            descriptive_block_low=lo,descriptive_block_high=hi,industry_coverage=coverage,
            exposure=(d.stock_value/d.equity).mean(),cash_days=int(d.n_hold.eq(0).sum()),
            stock_return_reference='covered portion only; gross adjusted; not executable or pure alpha'))
        industry_rows+=case_ind
        check_rows.append(dict(name=case['name'],pnl_error=float(abs(pnl.sum().sum()-(d.equity.iloc[-1]-config['capital']))),
                               source_account_verified=True))
        if case['variant']!='hybrid':
            continue
        # 所有实际持仓日逐股状态，月度复核与周度潜在警报并列，未知独立记录。
        state=h.copy()
        state['weight']=state.value/state.date.map(d.equity)
        keys=pd.MultiIndex.from_frame(state[['date','ts_code']])
        for key,matrix in evidence.items():
            state[key]=matrix.stack(future_stack=True).reindex(keys).to_numpy()
        state['monthly_review']=state.date.isin(hold.index)
        state['weekly_review']=state.date.isin(review_dates(evidence['alarm'].index,'weekly'))
        for key,matrix in [('hybrid_allowed',hold),('failure_streak',streak),('monthly_regime',regimes)]:
            state[key]=matrix.stack(future_stack=True).reindex(keys).to_numpy()
        state.to_csv(out/f"{case['name']}_daily_held_states.csv",index=False)
        state['alarm_value']=state.value*state.alarm.fillna(False)
        state['weak_value']=state.value*state.weak.fillna(False)
        state['broken_value']=state.value*state.broken.fillna(False)
        risk=state.groupby('date')[['value','alarm_value','weak_value','broken_value']].sum().reindex(d.index,fill_value=0.)
        for col in ['alarm','weak','broken']:
            risk[f'{col}_fraction']=risk[f'{col}_value']/risk.value.where(risk.value>0)
        risk['equity']=d.equity
        risk['exposure']=d.stock_value/d.equity
        risk['drawdown']=d.equity/d.equity.cummax().clip(lower=config['capital'])-1
        risk['monthly_review']=risk.index.isin(hold.index)
        risk['weekly_review']=risk.index.isin(review_dates(evidence['alarm'].index,'weekly'))
        risk.to_csv(out/f"{case['name']}_risk_timeline.csv")
        pnl.to_csv(out/f"{case['name']}_daily_asset_pnl.csv")
    pd.DataFrame(rows).to_csv(out/'annual_components.csv',index=False)
    pd.DataFrame(stock_year_rows).to_csv(out/'stock_year_pnl.csv',index=False)
    ind=pd.DataFrame(industry_rows)
    ind.groupby(['name','year','industry','industry_name'],dropna=False).sum().reset_index().to_csv(out/'industry_pnl.csv',index=False)
    pd.DataFrame(drawdowns).to_csv(out/'drawdown_episodes.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'attribution_summary.csv',index=False)
    save(out/'checks.json',check_rows)
    save(out/'completion.json',dict(status='complete',analysis_source_sha256=digest(__file__),
        artifacts={p.name:digest(p) for p in out.iterdir()}))
    print(pd.DataFrame(summary)[['name','net_profit','sector_proxy_pnl','residual_pnl','cash_days']].to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--prepare-reference',action='store_true')
    args=p.parse_args()
    if args.prepare_reference:
        prepare_reference(args.run)
    else:
        report(args.run)


if __name__=='__main__':
    main()
