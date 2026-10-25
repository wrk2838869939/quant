"""冻结错开月度与20交易日滚动复核对照；重新生成当日全A排名，不平移旧信号。"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from rolling_review import decision_dates,schedule_specs
from trend_entry import entry_features
from patient_trend import trend_state
from conditional_hold import build_conditional_holds
from research_profit_defense import digest,save
from audit_monthly_portfolio import reconcile
from run_patient_value_execution import process_report
from portfolio import normalize_actions
from causal_portfolio import run_backtest,daily_metrics


def prepare(config_path):
    config=yaml.safe_load(config_path.read_text())
    run=ROOT/config['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True,exist_ok=False);(run/'inputs').mkdir();(run/'schedules').mkdir()
    shutil.copy2(config_path,run/'config.yaml')
    parent=ROOT/config['parent'];inp=parent/'inputs'
    sources=['scripts/research_rolling_review.py','src/rolling_review.py','src/trend_entry.py',
        'src/patient_trend.py','src/conditional_hold.py','src/hold_sensitivity.py','src/causal_portfolio.py',
        'src/portfolio.py','src/selection_first.py','src/retail_rebalance.py','scripts/audit_monthly_portfolio.py',
        'scripts/run_patient_value_execution.py','configs/conditional_hold_research.yaml','tests/test_rolling_review.py']
    for name in sources:
        target=run/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    protocol='''# 月内起点与滚动复核：固定协议

旧历史已观察，不是新样本外。价格、换手率滚动窗口不变；只检验决策日历及其持有确认。
每月自然日1/8/15/22之后当月首个交易日四组，20交易日复核偏移0/5/10/15四组。
滚动锚点固定2022年第一个交易日，2024现金启动不重置日历；首次等待保持现金，所有账户按同一完整期间报告。
每种日历均重算当日全A合格截面排名、趋势与Hybrid持有许可，不能平移月初旧信号。
60/120日指标本已逐日滚动，新的20日是决策间隔。两/三次失败仍指复核次数，不把每日重叠窗口当作月度确认。
保持2021常数、5万元、最多5只、原整数手分配及费率、DAY预定订单。旧仓不按排名替换。
每种日历跑2022连续、2024现金启动，另跑相同首次买入后一直持有的对照，共32账户。
初始持有对照保留分红现金，不再修正单股上限；它分离起始组合影响，但差值仍含再配置、费用与再投资。
报告所有起点、年化/回撤分布、逐年、买卖日期、持仓重合、六月过程、初始现金等待和初始组合，不选择冠军。
新增候选须补真实行情、官方限价及分红，不准因为数据缺失缩减交易池；关键时序、行情或公司行动不明则停对应账户。
跨日历平均净值仅是描述，不能作为把5万元切成多个小账户的可执行方案。
月初参照需先逐格复现旧信号/持有许可，再复现两个封存账户；日历和价格历史截断测试通过后运行。
原封存、原引擎、生产和AGENTS.md均不修改；结果不足以自动实盘准入。
'''
    (run/'PROTOCOL.md').write_text(protocol)
    cases=[dict(name=f"{window}_{spec['name']}_{policy}",window=window,start=start,schedule=spec['name'],policy=policy)
           for window,start in config['windows'].items() for spec in schedule_specs(config)
           for policy in config['holding_policies']]
    inputs=[ROOT/config['feature_projection'],ROOT/'data/panel/daily_panel.parquet',
            inp/'basics.parquet',inp/'st.parquet',inp/'low_turnover_signals.parquet',
            inp/'low_turnover_point_forecasts.parquet',inp/'hold_hybrid.parquet',
            inp/'symmetric_phase_research.yaml',inp/'execution_config.yaml',inp/'corporate_action_components.yaml',
            parent/'completion.json']
    manifest=dict(status='protocol_frozen',cases=cases,fresh_holdout=False,
        source_hashes={name:digest(ROOT/name) for name in sources},
        input_hashes={str(p):digest(p) for p in inputs},config_sha256=digest(run/'config.yaml'),
        protocol_sha256=digest(run/'PROTOCOL.md'))
    save(run/'manifest.json',manifest)
    print(run,flush=True)
    return run


def verify_manifest(run):
    m=json.loads((run/'manifest.json').read_text())
    for name,h in m['source_hashes'].items():assert digest(ROOT/name)==h,name
    assert digest(run/'config.yaml')==m['config_sha256']
    assert digest(run/'PROTOCOL.md')==m['protocol_sha256']
    return m


def build_signals(run):
    m=verify_manifest(run)
    cfg=yaml.safe_load((run/'config.yaml').read_text())
    parent=ROOT/cfg['parent'];inp=parent/'inputs'
    for name,h in m['input_hashes'].items():assert digest(name)==h,name
    raw=pd.read_parquet(ROOT/cfg['feature_projection'])
    cols=['hfq_close','total_mv_yuan','turnover_rate']
    wide={c:raw.pivot(index='date',columns='ts_code',values=c).sort_index() for c in cols}
    close=wide['hfq_close'];spec=cfg['trend']
    assert raw.date.max()==pd.Timestamp(cfg['end'])
    del raw
    features=entry_features(close,wide['total_mv_yuan'],wide['turnover_rate'],
        medium=spec['medium_window'],long=spec['long_window'])
    _,trend=trend_state(close,spec)
    observable=(close.gt(0)&close.shift(spec['medium_window']).gt(0)&close.shift(spec['long_window']).gt(0)
        &close.where(close>0).rolling(spec['long_window'],min_periods=spec['long_window']).mean().notna())
    calendar=close.index[(close.index>=pd.Timestamp(cfg['start']))&(close.index<=pd.Timestamp(cfg['end']))]
    schedules={s['name']:decision_dates(calendar,s['kind'],s['parameter'],cfg['rolling_interval'],cfg['start'])
               for s in schedule_specs(cfg)}
    reviews=sorted(set().union(*map(set,schedules.values())))
    basics=pd.read_parquet(inp/'basics.parquet').set_index('ts_code')
    listing=pd.to_datetime(basics.list_date).reindex(close.columns)
    delisting=pd.to_datetime(basics.delist_date).reindex(close.columns)
    st=pd.read_parquet(inp/'st.parquet')
    old_mu=pd.read_parquet(inp/'low_turnover_point_forecasts.parquet')
    constants=np.unique(old_mu.to_numpy()[np.isfinite(old_mu.to_numpy())]);assert len(constants)==1
    constant=float(constants[0]);assert constant>cfg['entry']['min_expected_return']
    assert cfg['constant_expected_adverse']<=cfg['entry']['max_expected_adverse']
    signals=pd.DataFrame(np.nan,index=pd.DatetimeIndex(reviews,name='date'),columns=close.columns)
    forecasts=signals.copy();coverage=[]
    main=close.columns.str.match(r'^(?:(?:000|001|002|003)\d{3}\.SZ|(?:600|601|603|605)\d{3}\.SH)$')
    for date in reviews:
        frame=pd.DataFrame({k:v.loc[date] for k,v in features.items()})
        eligible=(date-listing).dt.days.ge(cfg['min_listing_days'])&(delisting.isna()|delisting.gt(date))
        eligible &= np.isfinite(frame).all(axis=1)
        active=st.loc[st.start.le(date)&(st.end.isna()|st.end.ge(date)),'ts_code']
        eligible.loc[eligible.index.isin(active)]=False
        ranks=frame.loc[eligible].rank(pct=True)-.5
        chosen=ranks.index[ranks.index.isin(close.columns[main])]
        scores=(-ranks.loc[chosen,'turnover']).where(trend.loc[date,chosen])
        signals.loc[date,chosen]=scores;forecasts.loc[date,chosen]=constant
        coverage.append(dict(date=date,all_eligible=int(eligible.sum()),main_eligible=len(chosen),admitted=int(scores.notna().sum())))
    # 新日期全截面不是从原172只候选里再选。
    pd.DataFrame(coverage).to_csv(run/'signal_coverage.csv',index=False)
    old_signal=pd.read_parquet(inp/'low_turnover_signals.parquet')
    pd.testing.assert_frame_equal(signals.loc[old_signal.index],old_signal,check_exact=True)
    pd.testing.assert_frame_equal(forecasts.loc[old_mu.index],old_mu,check_exact=True)
    union=set()
    parameters=yaml.safe_load((ROOT/'configs/conditional_hold_research.yaml').read_text())
    parameters['variants']={'hybrid':parameters['variants']['hybrid']}
    calendar_rows=[]
    for name,dates in schedules.items():
        folder=run/'schedules'/name;folder.mkdir(exist_ok=False)
        sig,mu=signals.loc[dates],forecasts.loc[dates]
        variants,_=build_conditional_holds(close,trend,observable,dates,parameters)
        hold=variants['hybrid']
        if name=='calendar_01':
            original=pd.read_parquet(inp/'hold_hybrid.parquet')
            pd.testing.assert_frame_equal(hold[original.columns],original,check_exact=True)
        for key,frame in [('signals',sig),('forecasts',mu),('hold',hold)]:
            frame.to_parquet(folder/f'{key}.parquet')
        for date,row in sig.iterrows():
            ranked=row.dropna().sort_values(ascending=False,kind='stable').head(20)
            union.update(ranked.index)
            calendar_rows.append(dict(schedule=name,date=date,top20='|'.join(ranked.index)))
    pd.DataFrame(calendar_rows).to_csv(run/'decision_calendar.csv',index=False)
    codes=sorted(union)
    pd.DataFrame(dict(ts_code=codes)).to_csv(run/'execution_codes.csv',index=False)
    close[codes].to_parquet(run/'inputs/daily_close.parquet')
    trend[codes].to_parquet(run/'inputs/daily_trend.parquet')
    observable[codes].to_parquet(run/'inputs/daily_observable.parquet')
    # 截断后同函数重算全部9项特征；每个前缀的值须逐格相同。
    prefix_checks=[]
    for stop in pd.to_datetime(['2023-06-30','2025-06-30']):
        recalculated=entry_features(close.loc[:stop],wide['total_mv_yuan'].loc[:stop],
            wide['turnover_rate'].loc[:stop],medium=spec['medium_window'],long=spec['long_window'])
        for key in features:
            pd.testing.assert_frame_equal(recalculated[key],features[key].loc[:stop],check_exact=True)
        prefix_checks.append(dict(stop=stop,features_exact=len(features)))
    save(run/'signal_completion.json',dict(status='complete',n_codes=len(codes),reviews=len(reviews),
        old_monthly_signal_exact=True,old_monthly_forecasts_exact=True,old_hybrid_hold_exact=True,
        constant=constant,prefix_checks=prefix_checks,
        artifacts={str(p.relative_to(run)):digest(p) for p in (run/'schedules').rglob('*.parquet')}))
    print('Signals frozen:',len(reviews),'review dates;',len(codes),'execution codes',flush=True)


def prepare_execution(run):
    verify_manifest(run)
    cfg=yaml.safe_load((run/'config.yaml').read_text());parent=ROOT/cfg['parent'];inp=parent/'inputs'
    codes=pd.read_csv(run/'execution_codes.csv').ts_code.tolist()
    ec=yaml.safe_load((inp/'execution_config.yaml').read_text())
    ec['valid']=[cfg['start'],cfg['end']]
    ec['data_workers']=2
    # 已有按代码缓存优先，缺失才下载；远端错误只报告类型，不输出凭据。
    from prepare_monthly_audit import fetch_execution_data
    cache=fetch_execution_data(codes,tuple(map(pd.Timestamp,ec['valid'])),ec)
    market=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',
        columns=['date','ts_code','open','high','low','close','pre_close','adj_factor','vol_share'],
        filters=[('date','>=',pd.Timestamp(cfg['start'])),('date','<=',pd.Timestamp(cfg['end'])),('ts_code','in',codes)])
    limits=pd.concat([pd.read_parquet(cache/'stk_limit'/f'{c}.parquet') for c in codes],ignore_index=True)
    limits['date']=pd.to_datetime(limits.trade_date)
    assert not limits.duplicated(['date','ts_code']).any()
    cover=market.merge(limits[['date','ts_code','up_limit','down_limit']],on=['date','ts_code'],how='left',validate='one_to_one')
    missing=cover[cover[['up_limit','down_limit']].isna().any(axis=1)]
    missing.to_csv(run/'missing_limits.csv',index=False)
    assert missing.empty,'官方限价缺失，不能缩减股票池'
    actions=pd.concat([pd.read_parquet(cache/'dividend'/f'{c}.parquet') for c in codes],ignore_index=True)
    action_cfg=yaml.safe_load((inp/'corporate_action_components.yaml').read_text())
    normalized=normalize_actions(actions,action_cfg['components'],action_cfg.get('reference_adjustments',[]),action_cfg.get('source_corrections',[]))
    imp=actions[actions.div_proc.eq('实施')].copy()
    for col in ['imp_ann_date','ex_date']:imp[col]=pd.to_datetime(imp[col])
    assert imp.imp_ann_date.notna().all()
    known=imp.groupby(['ts_code','ex_date']).imp_ann_date.max().rename('known_date')
    normalized=normalized.merge(known,on=['ts_code','ex_date'],how='left',validate='one_to_one')
    assert normalized.known_date.notna().all() and normalized.known_date.le(normalized.record_date).all()
    for name,frame in [('market',market),('limits',limits),('raw_actions',actions),('normalized_actions',normalized)]:
        target=run/'inputs'/f'{name}.parquet';assert not target.exists();frame.to_parquet(target,index=False)
    for name in ['basics.parquet','st.parquet','execution_config.yaml','symmetric_phase_research.yaml','corporate_action_components.yaml']:
        shutil.copy2(inp/name,run/'inputs'/name)
    old_market=pd.read_parquet(inp/'market.parquet')
    pd.testing.assert_frame_equal(market.set_index(['date','ts_code'])[old_market.columns.difference(['date','ts_code'])].sort_index().loc[
        old_market.set_index(['date','ts_code']).sort_index().index],
        old_market.set_index(['date','ts_code'])[old_market.columns.difference(['date','ts_code'])].sort_index(),check_exact=True)
    save(run/'execution_input_completion.json',dict(status='complete',codes=len(codes),rows=len(market),
        old_market_exact=True,missing_limits=len(missing),actions=len(normalized),cache=str(cache),
        hashes={str(p.relative_to(run)):digest(p) for p in (run/'inputs').iterdir()}))
    print('Execution inputs ready:',len(codes),'codes',flush=True)


def run_window(run,window):
    m=verify_manifest(run);cfg=yaml.safe_load((run/'config.yaml').read_text())
    for name,h in json.loads((run/'signal_completion.json').read_text())['artifacts'].items():assert digest(run/name)==h,name
    for name,h in json.loads((run/'execution_input_completion.json').read_text())['hashes'].items():assert digest(run/name)==h,name
    inp=run/'inputs';parent=ROOT/cfg['parent']
    frames=[pd.read_parquet(inp/f'{key}.parquet') for key in ['market','basics','st','limits','normalized_actions']]
    portfolio=yaml.safe_load((inp/'execution_config.yaml').read_text())['portfolio']
    portfolio.update(capital=cfg['capital'],top_n=5)
    retail=yaml.safe_load((inp/'symmetric_phase_research.yaml').read_text())['retail']
    action_cfg=yaml.safe_load((inp/'corporate_action_components.yaml').read_text())
    causal=yaml.safe_load((parent/'config.yaml').read_text())['causal']
    for case in [c for c in m['cases'] if c['window']==window]:
        folder=run/'execution'/case['name'];folder.mkdir(parents=True,exist_ok=False)
        source=run/'schedules'/case['schedule']
        sig,mu,hold=[pd.read_parquet(source/f'{name}.parquet') for name in ['signals','forecasts','hold']]
        if case['policy']=='initial_buyhold':
            date=sig.index[sig.index>=pd.Timestamp(case['start'])][0]
            sig,mu=sig.loc[[date]],mu.loc[[date]];hold=pd.DataFrame(1.,index=sig.index,columns=sig.columns)
        # 每个案例只载入该日历Top20并集，不能用盈利/实际成交决定数据集合。
        needed=sorted(set().union(*(set(row.dropna().sort_values(ascending=False,kind='stable').head(20).index) for _,row in sig.iterrows())))
        local=[frame[frame.ts_code.isin(needed)].copy() for frame in frames]
        result=run_backtest(sig,*local,case['start'],cfg['end'],deepcopy(portfolio),action_cfg.get('rights_issues',[]),
            retail_forecasts=mu,retail_config=retail,hold_allowed=hold,causal_config=causal)
        checks=reconcile(result,cfg['capital'])
        process,pnl,periods,exits,stats=process_report(result,cfg['capital'])
        for name,frame in [('daily',result.daily),('trades',result.trades),('holdings',result.holdings),('events',result.events),
            ('process_daily',process),('asset_pnl',pnl),('holding_periods',periods),('exit_reasons',exits)]:
            frame.to_csv(folder/f'{name}.csv',index=name in ['daily','process_daily'])
        if case['schedule']=='calendar_01' and case['policy']=='hybrid':
            for name in ['daily','trades','holdings','events']:
                pd.testing.assert_frame_equal(pd.read_csv(folder/f'{name}.csv'),
                    pd.read_csv(parent/'execution'/f'{window}_hybrid'/f'{name}.csv'),check_exact=True)
        summary=dict(**case,**result.summary,**stats,**checks)
        save(folder/'summary.json',summary)
        previous=cfg['capital'];years=[]
        for year,group in result.daily.groupby(result.daily.index.year):
            metrics,_=daily_metrics(group.equity,previous,portfolio['annual_days'])
            years.append(dict(name=case['name'],year=year,**metrics,end_equity=group.equity.iloc[-1]))
            previous=group.equity.iloc[-1]
        pd.DataFrame(years).to_csv(folder/'yearly.csv',index=False)
        save(folder/'completion.json',dict(status='complete',hashes={p.name:digest(p) for p in folder.iterdir()}))
        print(json.dumps({k:summary[k] for k in ['name','total','mdd','sharpe','n_trades']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/rolling_review_research.yaml')
    p.add_argument('--run',type=Path)
    p.add_argument('--stage',choices=['prepare','signals','data','execute'],required=True)
    p.add_argument('--window',choices=['cold','continued'])
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.config)
    elif args.stage=='signals':build_signals(args.run)
    elif args.stage=='data':prepare_execution(args.run)
    else:
        assert args.window
        run_window(args.run,args.window)


if __name__=='__main__':main()
