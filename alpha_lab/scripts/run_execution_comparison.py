"""冻结策略下对照盘中信号收盘执行与EOD信号开市固定股数执行。"""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import execution_comparison_engine as engine
from predicted_risk_budget_state import PredictedRiskBudget
from execution_market_feed import file_hash
from audit_monthly_portfolio import reconcile
from portfolio import normalize_actions
from fetch_execution_comparison_minutes import fetch
from prepare_execution_comparison import save


def inputs(run):
    prepared=run/'prepared';info=json.loads((prepared/'verification.json').read_text());codes=info['codes']
    base=ROOT/'outputs/selection_stability/20260914T070228_976007Z/inputs'
    cache=ROOT/'data/raw/execution_audit/20220101_20260908'
    missing=[]
    for code in codes:
        for api in ['stk_limit','dividend']:
            if not (cache/api/(code+'.parquet')).exists():missing.append((code,api))
    # 补新候选原件到本轮，不改变封存缓存。
    extra=run/'execution_metadata';extra.mkdir(exist_ok=True)
    for code,api in missing:
        params=dict(ts_code=code)
        if api=='stk_limit':params.update(start_date='20220101',end_date='20260908')
        task=dict(folder=str(extra),name=code+'_'+api,api=api,params=params)
        # 通用分钟下载器不用于不同字段的元数据。
        from contextlib import redirect_stdout,redirect_stderr
        with open('/dev/null','w') as sink,redirect_stdout(sink),redirect_stderr(sink):
            from ts_client import get_pro
            frame=get_pro().query(api,**params)
        if frame is None or len(frame)>=5800:raise ValueError('新候选元数据缺失/截断')
        frame.to_parquet(extra/(task['name']+'.parquet'),index=False)
        save(extra/(task['name']+'.json'),dict(api=api,params=params,rows=len(frame),
            sha256=file_hash(extra/(task['name']+'.parquet')),received_at=pd.Timestamp.now(tz='UTC').isoformat()))
    def read(api,code):
        path=cache/api/(code+'.parquet')
        return pd.read_parquet(path if path.exists() else extra/(code+'_'+api+'.parquet'))
    limits=pd.concat([read('stk_limit',c) for c in codes],ignore_index=True);limits['date']=pd.to_datetime(limits.trade_date)
    raw_actions=pd.concat([read('dividend',c) for c in codes],ignore_index=True)
    ac=yaml.safe_load((base/'corporate_action_components.yaml').read_text())
    # 仅窗口相关权益进入本轮归一化，早于起点已结清方案无需重新合并。
    dates=pd.to_datetime(raw_actions.ex_date,errors='coerce')
    relevant=raw_actions.loc[dates.ge(pd.Timestamp(info['start'])-pd.Timedelta(days=30))].copy()
    corrections=[r for r in ac.get('source_corrections',[]) if pd.Timestamp(str(r['match']['ex_date']))>=pd.Timestamp(info['start'])-pd.Timedelta(days=30)]
    actions=normalize_actions(relevant,ac['components'],ac.get('reference_adjustments',[]),corrections)
    imp=relevant[relevant.div_proc.eq('实施')].copy()
    for key in ['imp_ann_date','ex_date']:imp[key]=pd.to_datetime(imp[key])
    known=imp.groupby(['ts_code','ex_date']).imp_ann_date.max().rename('known_date')
    actions=actions.merge(known,on=['ts_code','ex_date'],how='left',validate='one_to_one')
    if actions.known_date.isna().any():raise ValueError('公司行动缺实施公告时间')
    limits.to_parquet(prepared/'limits.parquet',index=False);actions.to_parquet(prepared/'actions.parquet',index=False)
    cfg=yaml.safe_load((base/'execution_config.yaml').read_text())['portfolio'];cfg.update(capital=80000.,top_n=5)
    retail=yaml.safe_load((base/'symmetric_phase_research.yaml').read_text())['retail']
    return info,cfg,retail,ac.get('rights_issues',[])


def simulate(run,case,shared,stop=None):
    info,cfg,retail,rights=shared;folder=run/'prepared'/case['signal']
    matrices={k:pd.read_parquet(folder/f"calendar_{case['offset']:02d}"/(k+'.parquet')) for k in ['signals','hold','risk','forecasts']}
    codes=list(matrices['signals'].columns);prepared=run/'prepared'
    market=pd.read_parquet(prepared/'market.parquet');market=market[market.ts_code.isin(codes)]
    basics=pd.read_parquet(prepared/'basics.parquet');st=pd.read_parquet(prepared/'st.parquet')
    limits=pd.read_parquet(prepared/'limits.parquet');actions=pd.read_parquet(prepared/'actions.parquet')
    actions=actions[actions.ts_code.isin(codes)]
    recovery={k:pd.read_parquet(folder/('recovery_'+k+'.parquet')) for k in ['known','weak','ret20']}
    history=pd.DataFrame(json.loads((folder/f"calendar_{case['offset']:02d}/market_risk.json").read_text()))
    history['date']=pd.to_datetime(history.date);prob=history.set_index('date').probability
    overlay=PredictedRiskBudget(recovery,prob,.30245275630569923,.16239408013966727,.5)
    overlay.weekly=set(matrices['signals'].index)
    for date in sorted(d for d in overlay.weekly if d<pd.Timestamp(info['start'])):
        overlay.on_close(date,{},80000.)
    quotes=pd.read_parquet(prepared/(case['quotes']+'_quotes.parquet'))
    decision=pd.read_parquet(prepared/'decision_market.parquet') if case['signal']=='proxy1430' else None
    end=pd.Timestamp(stop or info['end'])
    result=engine.run_backtest(matrices['signals'].loc[:end],market,basics,st,limits,actions,
        info['start'],end,deepcopy(cfg),rights,retail_forecasts=matrices['forecasts'].loc[:end],retail_config=retail,
        hold_allowed=matrices['hold'].loc[:end],risk_multipliers=matrices['risk'].loc[:end],adjust_existing_risk=True,
        cash_overlay=overlay,restore_enabled=True,slippage_rate=case['cost_bps']/10000.,
        execution_mode=case['mode'],decision_market=decision,execution_quotes=quotes,
        execution_volume_fraction=case['execution_volume_fraction'])
    return result,overlay


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    shared=inputs(run);cases=[]
    for offset in [1,8,15,22]:
        for cost in [0,30]:
            for label,signal,mode in [('close_auction','proxy1430','close_auction'),('open_first','eod','open_fixed'),
                                     ('open_1m','eod','open_fixed'),('open_5m','eod','open_fixed')]:
                for fraction in ([1.,.01] if offset==15 else [1.]):
                    cases.append(dict(name=f'd{offset:02d}_{label}_cost{cost}_cap{fraction:g}',offset=offset,cost_bps=cost,
                        signal=signal,mode=mode,quotes=label,execution_volume_fraction=fraction))
    save(run/'comparison_protocol.json',dict(cases=cases,capital=80000,max_positions=5,
        start=shared[0]['start'],end=shared[0]['end'],optimization=False,
        opening_rule='prior_EOD_fixed_quantity; no discretionary timing; previous cash pool only; no automatic quantity resize',
        close_rule='1430_current_price_and_cumulative_volume; frozen_limit; same_day_close_price_proxy',
        volume_assumption='primary only requires order <= execution minute volume; 1% participation conservative sensitivity; neither proves queue fill',
        historical_vintage_certified=False,live_orders=False))
    rows=[]
    for case in cases:
        folder=run/'accounts'/case['name'];folder.mkdir(parents=True,exist_ok=False)
        result,overlay=simulate(run,case,shared)
        check=reconcile(result,80000.)
        assert result.daily.cash.min()>=-1e-6 and result.daily.n_hold.max()<=5
        assert np.max(np.abs(result.daily.equity-result.daily.cash-result.daily.stock_value-result.daily.receivables))<1e-6
        if case['mode']=='close_auction':assert result.trades.decision_date.eq(result.trades.date).all()
        else:assert result.trades.decision_date.lt(result.trades.date).all()
        assert result.trades.loc[result.trades.side.eq('buy'),'shares'].mod(100).eq(0).all()
        plans=result.events[result.events.kind.eq('order_planned')]
        for date,g in plans[plans.side.eq('buy')].groupby('date'):
            assert g.reserved_cash.sum()<=g.cash_at_decision.iloc[0]+1e-6
        for name in ['daily','trades','holdings','events']:getattr(result,name).to_parquet(folder/(name+'.parquet'),index=name=='daily')
        pd.DataFrame(overlay.records).to_csv(folder/'risk_states.csv',index=False)
        row=dict(case,**result.summary,**check,planned_orders=len(plans),
            missed_orders=int(result.events.kind.isin(['order_cancelled','order_deferred']).sum()),
            execution_days=int(result.trades.date.nunique()),status='passed' if result.summary['execution_data_sufficient'] else 'data_warning')
        save(folder/'summary.json',row);rows.append(row)
        print(len(rows),'/',len(cases),case['name'],'return',round(result.summary['total']*100,3),flush=True)
    pd.DataFrame(rows).to_csv(run/'summary.csv',index=False)
    # 两种时序各做真实账户前缀回放；末日订单可不同，已成交与净值必须相同。
    checks=[]
    for name in ['d15_close_auction_cost0_cap1','d15_open_1m_cost0_cap1']:
        case=next(c for c in cases if c['name']==name);stop='2026-07-31'
        short,_=simulate(run,case,shared,stop)
        full=pd.read_parquet(run/'accounts'/name/'daily.parquet').loc[:stop]
        pd.testing.assert_frame_equal(short.daily,full,check_exact=True)
        full_trades=pd.read_parquet(run/'accounts'/name/'trades.parquet');full_trades=full_trades[full_trades.date<=stop].reset_index(drop=True)
        pd.testing.assert_frame_equal(short.trades.reset_index(drop=True),full_trades,check_exact=True)
        checks.append(dict(case=name,cutoff=stop,daily_and_trades_exact=True))
    save(run/'prefix_verification.json',checks)
    save(run/'execution_complete.json',dict(accounts=len(rows),all_prefixes_exact=True,live_orders=False))


if __name__=='__main__':main()
