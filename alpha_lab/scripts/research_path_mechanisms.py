"""冻结并执行日期敏感性机制消融；每个改动独立与同日历基线比较。"""
import argparse,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from research_profit_defense import save,digest
from duration_hold import duration_holds
from path_mechanism_portfolio import run_backtest,daily_metrics
from audit_monthly_portfolio import reconcile
from run_patient_value_execution import process_report

def prepare(config):
    cfg=yaml.safe_load(config.read_text());parent=ROOT/cfg['parent']
    old=json.loads((parent/'completion.json').read_text())
    for n,h in old['artifact_sha256'].items():assert digest(parent/n)==h,n
    run=ROOT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    shutil.copy2(config,run/'config.yaml')
    source_names=['src/path_mechanism_portfolio.py','src/duration_hold.py','src/causal_portfolio.py','src/portfolio.py','src/selection_first.py','src/retail_rebalance.py','scripts/research_path_mechanisms.py','scripts/run_patient_value_execution.py','scripts/audit_monthly_portfolio.py','tests/test_path_mechanisms.py','tests/test_causal_portfolio_adversarial.py']
    for name in source_names:
        target=run/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    for name,source in [('unit_tests.log','/tmp/path_mechanism_tests_v2.log'),('adversarial_tests.log','/tmp/path_engine_adversarial_v2.log'),('initial_test_attempt.log','/tmp/path_mechanism_tests.log')]:shutil.copy2(source,run/name)
    for name in ['unit_tests.log','adversarial_tests.log']:assert (run/name).read_text().rstrip().endswith('OK')
    (run/'PROTOCOL.md').write_text('''# 路径机制独立消融协议

既有历史全部已观察。本轮8日历×2启动窗口×5政策＝80账户，不叠加改动、不按最终收益挑日历或调阈值。
基线：原Hybrid与原因果撮合。新分支关闭开关时16基线的四类原账本必须逐格相同。
持续证据：以过去20完整交易日弱势计数替代复核次数，当前仍弱才退；长期两项皆弱需10/20，其他需15/20。缺失不支持确认；只在原复核日执行。此规则仍可能在新华文轩8月3日退出，不为该个例定制。
等待：只有真实完成趋势全退出才保留一个名额到下次复核；未成交退出、风险减仓、ST均不创建记忆。等待期间名额不能被补掉，但不逐股隔离现金。
恢复：在相同等待之上，下次复核若原股票仍满足完整当日入场资格，可在预约优先级上超过普通Top20。收益预测、股数分配、现金及5股约束不变；不强卖其他旧仓。记忆仅持续这一复核周期，然后过期。允许的恢复证券必定曾从本日历Top20选出，因此原执行并集覆盖闭包，不补全市场报价或筛掉缺失候选。
整手缓冲：只有上限要求卖出不足100股且市值不超过账户35%时，暂缓原25%上限触发的整手减仓。超过一手或35%仍按原25%目标减仓。35%是复核阈值，非每日硬上限；增加集中风险须明确报告。原入场25%上限不改。
5万元、最多5只、前夜DAY订单、不能预支卖款、原费率/分红税假设保持。观察每组总收益/回撤、日历离散度、真实空仓、次数和状态事件，以及首次买入持有对照。初始单必须与同日历基线相同。
先通过新机制行为测试和原25项撮合对抗集，再运行。两种新路径做实际历史截断回放，订单/现金/股数及逐股损益独立对账。
只做机制有效性判断，未测试的组合/其他资金量不推断；不自动晋升研究冠军或上线。
''')
    cases=[dict(name=f'{w}_{schedule}_{policy}',window=w,start=start,schedule=schedule,policy=policy) for w,start in cfg['windows'].items() for schedule in cfg['schedules'] for policy in cfg['policies']]
    save(run/'manifest.json',dict(cases=cases,config_sha256=digest(run/'config.yaml'),parent_completion_sha256=digest(parent/'completion.json'),source_sha256={n:digest(ROOT/n) for n in source_names}))
    inp=parent/'inputs';p,t,o=[pd.read_parquet(inp/f'{name}.parquet') for name in ['daily_close','daily_trend','daily_observable']]
    (run/'holds').mkdir();checks=[]
    for schedule in cfg['schedules']:
        original=pd.read_parquet(parent/'schedules'/schedule/'hold.parquet')
        result=duration_holds(p,t,o,original.index,cfg['duration'])
        expanded=pd.DataFrame(np.nan,index=original.index,columns=original.columns);expanded.loc[:,result.columns]=result
        expanded.to_parquet(run/'holds'/f'{schedule}.parquet')
        for cut in pd.to_datetime(['2023-06-30','2025-06-30']):
            dates=original.index[original.index<=cut]
            short=duration_holds(p.loc[:cut],t.loc[:cut],o.loc[:cut],dates,cfg['duration'])
            pd.testing.assert_frame_equal(short,result.loc[dates],check_exact=True)
            checks.append(dict(schedule=schedule,cutoff=cut,exact=True))
    save(run/'hold_verification.json',dict(status='passed',checks=checks,hashes={f.name:digest(f) for f in (run/'holds').iterdir()}))
    print(run,flush=True)

def verify(run):
    cfg=yaml.safe_load((run/'config.yaml').read_text());m=json.loads((run/'manifest.json').read_text())
    assert digest(run/'config.yaml')==m['config_sha256']
    for n,h in m['source_sha256'].items():assert digest(ROOT/n)==h,n
    assert digest(ROOT/cfg['parent']/'completion.json')==m['parent_completion_sha256']
    for n,h in json.loads((run/'hold_verification.json').read_text())['hashes'].items():assert digest(run/'holds'/n)==h,n
    return cfg,m

def execute(run,window,family):
    cfg,m=verify(run);parent=ROOT/cfg['parent'];inp=parent/'inputs'
    frames=[pd.read_parquet(inp/f'{n}.parquet') for n in ['market','basics','st','limits','normalized_actions']]
    portfolio=yaml.safe_load((inp/'execution_config.yaml').read_text())['portfolio'];portfolio.update(capital=cfg['capital'],top_n=5)
    retail=yaml.safe_load((inp/'symmetric_phase_research.yaml').read_text())['retail']
    ac=yaml.safe_load((inp/'corporate_action_components.yaml').read_text());cc=yaml.safe_load((ROOT/'outputs/causal_execution/20260913T130810_374654Z/config.yaml').read_text())['causal']
    for case in [c for c in m['cases'] if c['window']==window and c['schedule'].startswith(family)]:
        folder=run/'execution'/case['name'];folder.mkdir(parents=True,exist_ok=False)
        source=parent/'schedules'/case['schedule'];sig,mu,hold=[pd.read_parquet(source/f'{n}.parquet') for n in ['signals','forecasts','hold']]
        policy=cfg['policies'][case['policy']]
        if policy['hold']=='duration':hold=pd.read_parquet(run/'holds'/f"{case['schedule']}.parquet")
        needed=sorted(set().union(*(set(row.dropna().sort_values(ascending=False,kind='stable').head(20).index) for _,row in sig.iterrows())))
        local=[f[f.ts_code.isin(needed)].copy() for f in frames]
        result=run_backtest(sig,*local,case['start'],cfg['end'],deepcopy(portfolio),ac.get('rights_issues',[]),retail_forecasts=mu,retail_config=retail,hold_allowed=hold,causal_config=cc,path_config={k:v for k,v in policy.items() if k!='hold'})
        checks=reconcile(result,cfg['capital']);process,pnl,periods,exits,stats=process_report(result,cfg['capital'])
        for name,frame in [('daily',result.daily),('trades',result.trades),('holdings',result.holdings),('events',result.events),('process_daily',process),('asset_pnl',pnl),('holding_periods',periods),('exit_reasons',exits)]:frame.to_csv(folder/f'{name}.csv',index=name in ['daily','process_daily'])
        if case['policy']=='baseline':
            for n in ['daily','trades','holdings','events']:
                pd.testing.assert_frame_equal(pd.read_csv(folder/f'{n}.csv'),pd.read_csv(parent/'execution'/f"{window}_{case['schedule']}_hybrid"/f'{n}.csv'),check_exact=True)
        summary=dict(**case,**result.summary,**stats,**checks);save(folder/'summary.json',summary)
        previous=cfg['capital'];rows=[]
        for year,group in result.daily.groupby(result.daily.index.year):
            metrics,_=daily_metrics(group.equity,previous,portfolio['annual_days']);rows.append(dict(name=case['name'],year=year,**metrics));previous=group.equity.iloc[-1]
        pd.DataFrame(rows).to_csv(folder/'yearly.csv',index=False)
        save(folder/'completion.json',dict(status='complete',hashes={f.name:digest(f) for f in folder.iterdir()}))
        print(json.dumps({k:summary[k] for k in ['name','total','mdd','sharpe','n_trades','n_cash_days']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--run',type=Path);p.add_argument('--window',choices=['continued','cold']);p.add_argument('--family',choices=['calendar','rolling']);args=p.parse_args()
    if args.prepare:prepare(ROOT/'configs/path_mechanism_research.yaml')
    else:execute(args.run,args.window,args.family)
