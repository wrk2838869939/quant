"""8万元首次入场与后续日历对照；固定协议、同窗口、逐账户状态对账。"""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from startup_timing import startup_matrices
from risk_budget_portfolio import run_backtest,daily_metrics
from audit_monthly_portfolio import reconcile


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path)
    args=p.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/startup_timing_research.yaml').read_text())
    parent=ROOT/cfg['parent'];rp=ROOT/cfg['risk_parent'];inp=parent/'inputs'
    source_names=['src/startup_timing.py','src/risk_budget_portfolio.py','src/selection_first.py',
        'src/retail_rebalance.py','scripts/research_startup_timing.py','scripts/audit_monthly_portfolio.py',
        'tests/test_startup_timing.py','configs/startup_timing_research.yaml']
    schedules={}
    for offset in [1]+cfg['offset_days']:
        schedules[offset]={name:pd.read_parquet(parent/f'schedules/calendar_{offset:02d}/{name}.parquet') for name in ['signals','forecasts','hold']}
    discounts=pd.read_parquet(rp/'risk/discounts.parquet')
    frames={name:pd.read_parquet(inp/(name+'.parquet')) for name in ['market','basics','st','limits','normalized_actions']}
    dates=pd.DatetimeIndex(sorted(frames['market'].date.unique()))
    portfolio=yaml.safe_load((inp/'execution_config.yaml').read_text())['portfolio']
    portfolio.update(capital=cfg['capital'],top_n=cfg['max_positions'])
    retail=yaml.safe_load((inp/'symmetric_phase_research.yaml').read_text())['retail']
    rights=yaml.safe_load((inp/'corporate_action_components.yaml').read_text()).get('rights_issues',[])
    cc=yaml.safe_load((ROOT/'outputs/causal_execution/20260913T130810_374654Z/config.yaml').read_text())['causal']
    def simulate(matrices,scales,start,end,capital=None):
        needed=sorted(set().union(*(set(row.dropna().nlargest(20).index) for _,row in matrices['signals'].iterrows())))
        local=[frames[k][frames[k].ts_code.isin(needed)].copy() for k in ['market','basics','st','limits','normalized_actions']]
        local[0]=local[0][local[0].date.between(start,end)];local[3]=local[3][local[3].date.le(end)]
        local[4]=local[4][local[4].known_date.le(end)]
        pc=dict(portfolio)
        if capital:pc['capital']=capital
        return run_backtest(matrices['signals'],*local,start,end,pc,rights,
            retail_forecasts=matrices['forecasts'],retail_config=retail,hold_allowed=matrices['hold'],
            causal_config=cc,risk_multipliers=scales,adjust_existing_risk=True)
    if args.run:
        out=args.run;manifest=json.loads((out/'manifest.json').read_text())
        for name,h in manifest['source_sha256'].items():assert digest(ROOT/name)==h,name
        for name,h in manifest['input_sha256'].items():assert digest(name)==h,name
    else:
        out=ROOT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
        print(str(out),flush=True)
        checks={}
        for pr in [parent,rp]:
            receipt=json.loads((pr/'completion.json').read_text())
            for name,h in receipt['artifact_sha256'].items():assert digest(pr/name)==h,name
            checks[str(pr.relative_to(ROOT))]=len(receipt['artifact_sha256'])
        cases=[];omitted=[]
        for year in cfg['years']:
            for month in cfg['months']:
                period=pd.Period(f'{year}-{month:02d}',freq='M')
                boot={d:schedules[d]['signals'].index[schedules[d]['signals'].index.to_period('M')==period][0] for d in cfg['offset_days']}
                start=min(boot.values());ix=dates.get_loc(start)
                if ix+cfg['horizon_sessions']>len(dates):
                    omitted.append(str(period));continue
                end=dates[ix+cfg['horizon_sessions']-1]
                cohort=str(period)
                cases.append(dict(name=cohort+'_wait',cohort=cohort,offset=cfg['offset_days'][0],policy='wait',start=str(start.date()),end=str(end.date()),boot_date=str(start.date())))
                for day in cfg['offset_days']:
                    for policy in cfg['policies']:
                        cases.append(dict(name=f'{cohort}_{day:02d}_{policy}',cohort=cohort,offset=day,policy=policy,start=str(start.date()),end=str(end.date()),boot_date=str(boot[day].date())))
        sources={name:digest(ROOT/name) for name in source_names}
        files=list(inp.glob('*.yaml'))+[inp/(n+'.parquet') for n in frames]+[rp/'risk/discounts.parquet']
        files += list((parent/'schedules').glob('calendar_*/*.parquet'))
        manifest=dict(config=cfg,cases=cases,omitted_incomplete_horizon=omitted,source_sha256=sources,
            input_sha256={str(p):digest(p) for p in files},prior_verified=checks,root_agents_sha256=digest(ROOT.parent/'AGENTS.md'))
        save(out/'manifest.json',manifest)
        (out/'PROTOCOL.md').write_text("""# 首次入场日期敏感性固定协议

8万元，最多5股，100股整手，full_risk原引擎/年度模型/费用/分红/撮合完全不变。
2022至2025每季度起始月，8/15/22日或其后首个交易日为首次复核日。
每个季度窗口统一从8日起的首个交易日持有现金开始计收益，共244个交易日，包含等待期。
不足244日的2025年10月整组按预先条件排除，不观察结果再排除。
wait：等下个月月初复核；stale_once：月中使用本月月初旧名单和风险折扣，仅价格用入场日前晚真值；
fresh_once：月中当日完整截面首次评估，之后回原月初；shifted_calendar：首单同fresh，但后续改成对应月内日期。
首次插入行当天尚为空仓；之后每个原月初的hold值逐格不变，不额外累计/重置持有失败确认。
模型参数不重训，信号及风险来自已封存404复核日，source_date逐组保存且不晚于decision_date。
所有组逐现金/股数/净资产与委托对账；先精确复现2个5万元父账户，再执行150个8万元新账户。
分析初始组合重合、第一正常复核、逐股损益、偏移范围和误差；重叠窗口不是独立样本。
历史已观察，结果不算新鲜样本外，不选最好日期，不改生产或真实账户，不给当前买入指令。
""")
        for name in source_names:
            target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    # 每次恢复先核验父账户；不因资金不同而换模型。
    if not (out/'parent_parity.json').exists():
        parity=[]
        for label,start in [('continued','2022-01-01'),('cold','2024-01-01')]:
            m=schedules[1];scales=discounts.reindex(index=m['signals'].index,columns=m['signals'].columns)
            result=simulate(m,scales,pd.Timestamp(start),pd.Timestamp(cfg['end']),50000)
            reference=rp/f'execution/{label}_calendar_01_full_risk'
            for name,frame in [('daily',result.daily),('trades',result.trades),('holdings',result.holdings),('events',result.events)]:
                temporary=out/('parity_'+label+'_'+name+'.csv');frame.to_csv(temporary,index=name=='daily')
                pd.testing.assert_frame_equal(pd.read_csv(temporary),pd.read_csv(reference/(name+'.csv')),check_exact=True)
            parity.append(dict(window=label,exact=True,trades=len(result.trades)))
        save(out/'parent_parity.json',parity);print('parent parity passed',flush=True)
    results=[]
    for index,case in enumerate(manifest['cases']):
        folder=out/'accounts'/case['name']
        if (folder/'summary.json').exists():
            results.append(json.loads((folder/'summary.json').read_text()));continue
        folder.mkdir(parents=True,exist_ok=True)
        m,scales,lineage=startup_matrices(schedules[1],schedules[case['offset']],discounts,
            case['boot_date'],case['start'],case['end'],case['policy'])
        save(folder/'lineage.json',lineage)
        try:
            result=simulate(m,scales,pd.Timestamp(case['start']),pd.Timestamp(case['end']))
            checks=reconcile(result,cfg['capital'])
            assert result.summary['execution_data_sufficient']
            assert len(result.daily)==cfg['horizon_sessions']
            assert result.daily.n_hold.max()<=cfg['max_positions']
            assert result.trades.loc[result.trades.side.eq('buy'),'shares'].mod(100).eq(0).all()
            assert (result.trades.decision_date<result.trades.date).all()
            assert np.max(np.abs(result.daily.equity-result.daily.cash-result.daily.stock_value-result.daily.receivables))<1e-6
            orders=result.events.loc[result.events.kind.eq('order_planned')]
            for _,group in orders.loc[orders.side.eq('buy')].groupby('decision_date'):
                assert group.reserved_cash.sum()<=group.cash_at_decision.iloc[0]+1e-6
            for name,frame in [('daily',result.daily),('trades',result.trades),('holdings',result.holdings),('events',result.events)]:
                frame.to_parquet(folder/(name+'.parquet'),index=name=='daily')
            summary=dict(case,**result.summary,**checks,status='passed')
            save(folder/'summary.json',summary);results.append(summary)
            print(index+1,case['name'],round(result.summary['total']*100,2),round(result.summary['mdd']*100,2),flush=True)
        except Exception as exc:
            save(folder/'failure.json',dict(case,error_type=type(exc).__name__,message=str(exc)))
            raise
    pd.DataFrame(results).to_csv(out/'summary.csv',index=False)
    save(out/'execution_complete.json',dict(accounts=len(results),all_reconciled=True,
        total_trades=sum(r['n_trades'] for r in results),live_orders=False))


if __name__=='__main__':main()
