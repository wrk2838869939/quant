#!/usr/bin/env python3
"""分阶段价值研究的固定20组执行：先选名单，后定股数，倒序输入独立复跑。"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from portfolio import run_backtest, AccountingError, daily_metrics
from audit_monthly_portfolio import load_inputs, reconcile
from run_patient_value_execution import process_report, json_write, digest


def execution_cases(cfg):
    """先登记15基线和5个5万元倒序输入案例，顺序检验不是额外策略。"""
    cases = [dict(name=f'base_{capital:g}_{policy}', capital=capital, policy=policy,
                  reverse_input=False) for capital in cfg['capitals'] for policy in cfg['policies']]
    cases += [dict(name=f'reverse_{cfg["reference_capital"]:g}_{policy}',
        capital=cfg['reference_capital'], policy=policy, reverse_input=True) for policy in cfg['policies']]
    return cases


def make_matrices(predictions, hold, cfg):
    """只读冻结的分数和点收益，不重新筛门槛，不用rank冒充收益。"""
    lo, hi = map(pd.Timestamp, cfg['execution'])
    if (hi >= pd.Timestamp('2024-01-01') or not predictions.date.between(lo, hi).all()
            or predictions.duplicated(['date', 'ts_code']).any()
            or hold.index.has_duplicates or hold.columns.has_duplicates):
        raise ValueError('输入日期、代码或TEST边界非法')
    if not pd.DatetimeIndex(sorted(predictions.date.unique())).equals(hold.index):
        raise ValueError('冻结预测与月度持有日历不一致')
    matrices = {}
    for policy in cfg['policies']:
        signals = predictions.pivot(index='date', columns='ts_code', values=f'{policy}__score').reindex(
            index=hold.index, columns=hold.columns)
        forecasts = predictions.pivot(index='date', columns='ts_code', values=f'{policy}__expected_return').reindex(
            index=hold.index, columns=hold.columns)
        if np.isinf(signals.to_numpy()).any() or np.isinf(forecasts.to_numpy()).any():
            raise ValueError('冻结分数/预测含无限值')
        if (signals.notna() & forecasts.isna()).any().any():
            raise ValueError('准入候选必须有收益点估计')
        matrices[policy] = (signals, forecasts)
    return matrices


@contextmanager
def reversed_candidate_input(enabled):
    """只反转adapter收到的行；不改优先级、分数或返回顺序，退出时恢复函数。"""
    import selection_first
    original = selection_first.plan_selection_first
    if enabled:
        def reversed_plan(candidates, **kwargs):
            return original(candidates.iloc[::-1].copy(), **kwargs)
        selection_first.plan_selection_first = reversed_plan
    try:
        yield
    finally:
        selection_first.plan_selection_first = original


def entry_phase_attribution(result, predictions, signals):
    """按实际入场时已知阶段固定归因；分红权益和期末持仓不重复记利润。"""
    lookup = predictions.set_index(['date', 'ts_code'])
    rank_maps = {date: {code: rank for rank, code in enumerate(
        row.dropna().sort_values(ascending=False, kind='stable').index, 1)} for date, row in signals.iterrows()}
    active, episodes, buys = {}, [], []
    events = result.events
    for date in result.daily.index:
        day_events = events[events.date.eq(date)] if len(events) else pd.DataFrame()
        for event in day_events.to_dict('records'):
            code, kind = event['ts_code'], event['kind']
            if kind in ['dividend_receivable', 'bonus_shares', 'delist_writeoff']:
                if code not in active:
                    raise AccountingError(f'{date} {code}: 阶段归因缺少对应持仓段')
                episode = active[code]
                if kind == 'dividend_receivable':
                    episode['pnl_yuan'] += event['amount']
                    episode['dividends_yuan'] += event['amount']
                elif kind == 'bonus_shares':
                    episode['shares'] += event['shares']
                else:
                    episode['pnl_yuan'] += event['cash_recovery']
                    episode.update(shares=0, close_date=date)
                    del active[code]
            elif kind == 'hold_retained' and event.get('reason') == 'unknown' and code in active:
                active[code]['unknown_hold_reviews'] += 1
        trades = result.trades[result.trades.date.eq(date)] if len(result.trades) else pd.DataFrame()
        for trade in trades.to_dict('records'):
            code = trade['ts_code']
            if trade['side'] == 'buy':
                if code in active:
                    raise AccountingError('耐心持仓不应出现旧仓补仓，阶段归因拒绝隐式混合入场')
                known = lookup.loc[(trade['signal_date'], code)]
                rank = rank_maps[trade['signal_date']][code]
                episode = dict(episode_id=len(episodes)+1, ts_code=code, entry_date=date,
                    signal_date=trade['signal_date'], entry_phase=known.phase,
                    entry_priority_rank=rank, entry_score=signals.loc[trade['signal_date'], code],
                    entry_gross=trade['gross'], shares=trade['shares'], fees_yuan=trade['fee'],
                    dividends_yuan=0., pnl_yuan=-trade['gross']-trade['fee'],
                    terminal_value=0., close_date=pd.NaT, unknown_hold_reviews=0)
                active[code] = episode
                episodes.append(episode)
                selected = day_events[day_events.kind.eq('selection_decision') & day_events.ts_code.eq(code)]
                if len(selected) != 1 or selected.iloc[0].reason != 'selected':
                    raise AccountingError('真实买单缺少唯一的名单预约记录')
                decision = selected.iloc[0]
                if trade['shares'] < decision.reserved_shares or decision.reserved_net_profit <= 0:
                    raise AccountingError('真实入场未满足正净利股数预约')
                buys.append(dict(date=date, signal_date=trade['signal_date'], ts_code=code,
                    entry_phase=known.phase, priority_rank=rank, priority_score=decision.priority,
                    expected_return=decision.expected_return, reserved_shares=decision.reserved_shares,
                    reserved_net_profit=decision.reserved_net_profit, shares=trade['shares'],
                    price=trade['price'], gross=trade['gross'], fee=trade['fee']))
            else:
                if code not in active:
                    raise AccountingError('卖单缺少对应阶段持仓段')
                episode = active[code]
                episode['pnl_yuan'] += trade['gross']-trade['fee']
                episode['fees_yuan'] += trade['fee']
                episode['shares'] -= trade['shares']
                if episode['shares'] == 0:
                    episode['close_date'] = date
                    del active[code]
        actual = result.holdings[result.holdings.date.eq(date)] if len(result.holdings) else pd.DataFrame()
        if sum(e['shares'] for e in active.values()) != (actual.shares.sum() if len(actual) else 0):
            raise AccountingError('阶段归因的真实股数对账失败')
    last = result.holdings[result.holdings.date.eq(result.daily.index[-1])] if len(result.holdings) else pd.DataFrame()
    for row in last.itertuples():
        active[row.ts_code]['pnl_yuan'] += row.value
        active[row.ts_code]['terminal_value'] = row.value
    frame = pd.DataFrame(episodes, columns=['episode_id','ts_code','entry_date','signal_date','entry_phase',
        'entry_priority_rank','entry_score','entry_gross','shares','fees_yuan','dividends_yuan',
        'pnl_yuan','terminal_value','close_date','unknown_hold_reviews'])
    total = frame.pnl_yuan.sum()
    expected = result.daily.equity.iloc[-1]-result.summary['capital']
    error = float(abs(total-expected))
    if error > 1e-6:
        raise AccountingError(f'阶段利润不能还原账户净利润: {error}')
    phases = frame.groupby('entry_phase').agg(episodes=('episode_id','size'), stocks=('ts_code','nunique'),
        pnl_yuan=('pnl_yuan','sum'), entry_gross=('entry_gross','sum'), fees_yuan=('fees_yuan','sum'),
        dividends_yuan=('dividends_yuan','sum'), terminal_value=('terminal_value','sum'),
        unknown_hold_reviews=('unknown_hold_reviews','sum')).reset_index()
    buy_frame = pd.DataFrame(buys, columns=['date','signal_date','ts_code','entry_phase','priority_rank',
        'priority_score','expected_return','reserved_shares','reserved_net_profit','shares','price','gross','fee'])
    return frame, phases, buy_frame, error


def run_case(run, cfg, case):
    """每个独立进程只运行一组；倒序wrapper不会污染其它案例。"""
    out = run/'execution'
    p = yaml.safe_load((out/'execution_config.yaml').read_text())
    market, basics, st, limits, actions = load_inputs(out, p)
    hold = pd.read_parquet(out/'monthly_hold_allowed.parquet')
    signals = pd.read_parquet(out/f'{case["policy"]}_signals.parquet')
    forecasts = pd.read_parquet(out/f'{case["policy"]}_point_forecasts.parquet')
    predictions = pd.read_parquet(run/'execution_predictions.parquet')
    portfolio = deepcopy(p['portfolio'])
    portfolio.update(capital=case['capital'], top_n=cfg['retail']['max_positions'])
    actions_cfg = yaml.safe_load((ROOT/'configs/corporate_action_components.yaml').read_text())
    with reversed_candidate_input(case['reverse_input']):
        result = run_backtest(signals, market, basics, st, limits, actions, *cfg['execution'],
            portfolio, actions_cfg.get('rights_issues', []), retail_forecasts=forecasts,
            retail_config=cfg['retail'], hold_allowed=hold)
    check = reconcile(result, case['capital'])
    process, pnl, periods, exits, metrics = process_report(result, case['capital'])
    episodes, phases, buys, phase_error = entry_phase_attribution(result, predictions, signals)
    trace = result.events[result.events.kind.eq('selection_decision')].copy() if len(result.events) else pd.DataFrame()
    if len(trace):
        trace = trace.merge(predictions[['date','ts_code','phase']].rename(columns={'date':'signal_date'}),
                            on=['signal_date','ts_code'], how='left', validate='many_to_one')
    stale = result.holdings[result.holdings.stale_days.gt(0)] if len(result.holdings) else pd.DataFrame()
    unknown = result.events[result.events.kind.eq('hold_retained') & result.events.reason.eq('unknown')] if len(result.events) else pd.DataFrame()
    if result.daily.n_hold.max() > cfg['retail']['max_positions'] or (len(buys) and buys.priority_rank.max()>cfg['retail']['candidate_pool']):
        raise AccountingError('持仓数或真实入场优先名次超界')
    case_dir = out/case['name']
    for name, frame in [('daily',result.daily),('trades',result.trades),('holdings',result.holdings),
        ('events',result.events),('process_daily',process),('asset_pnl',pnl),('holding_periods',periods),
        ('exit_reasons',exits),('entry_phase_episodes',episodes),('entry_phase_attribution',phases),
        ('actual_entry_priorities',buys),('selection_trace',trace),('stale_holding_marks',stale),('unknown_hold_reviews',unknown)]:
        if frame.empty and not len(frame.columns):
            frame = frame.reindex(columns={'trades':['date','ts_code','side','shares','gross','fee'],
                'holdings':['date','ts_code','shares','value'], 'events':['date','ts_code','kind']}.get(name,['date','ts_code']))
        frame.to_csv(case_dir/f'{name}.csv',index=name in ['daily','process_daily'])
    report = {**case, **result.summary, **check, **metrics,
        'phase_pnl_max_error':phase_error,'actual_buys':len(buys),
        'mean_actual_entry_rank':float(buys.priority_rank.mean()) if len(buys) else np.nan,
        'max_actual_entry_rank':int(buys.priority_rank.max()) if len(buys) else 0,
        'stale_holding_rows':len(stale),'unknown_hold_reviews':len(unknown)}
    json_write(case_dir/'summary.json',report)
    years=[];previous=case['capital']
    for year,daily in result.daily.groupby(result.daily.index.year):
        values,_=daily_metrics(daily.equity,previous,portfolio['annual_days'])
        years.append(dict(name=case['name'],year=year,**values));previous=daily.equity.iloc[-1]
    pd.DataFrame(years).to_csv(case_dir/'yearly.csv',index=False)
    print(json.dumps({k:report[k] for k in ['name','total','mdd','n_trades','mean_holdings','phase_pnl_max_error']},ensure_ascii=False),flush=True)


def compare_reversed_cases(out, cfg):
    """真实复跑产物逐值比较，不能用排序输入前后的静态名单冒充验证。"""
    rows=[]
    for policy in cfg['policies']:
        base=out/f'base_{cfg["reference_capital"]:g}_{policy}'
        reverse=out/f'reverse_{cfg["reference_capital"]:g}_{policy}'
        record=dict(policy=policy, identical=True)
        for name in ['daily','trades','holdings','events','selection_trace','actual_entry_priorities','entry_phase_episodes']:
            a=pd.read_csv(base/f'{name}.csv');b=pd.read_csv(reverse/f'{name}.csv')
            try:
                pd.testing.assert_frame_equal(a,b,check_exact=True)
                record[f'{name}_identical']=True
            except AssertionError:
                record[f'{name}_identical']=False;record['identical']=False
        rows.append(record)
    frame=pd.DataFrame(rows);frame.to_csv(out/'candidate_order_invariance.csv',index=False)
    return frame


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,default=ROOT/'configs/divide_selection.yaml')
    ap.add_argument('--run',type=Path,default=None)
    ap.add_argument('--case-name',default=None)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args();run=args.run or Path(Path('/tmp/divide_selection_run_path').read_text().strip())
    cfg=yaml.safe_load(args.config.read_text());cases=execution_cases(cfg);out=run/'execution';out.mkdir(exist_ok=True)
    if args.case_name:
        case=next(c for c in cases if c['name']==args.case_name)
        (out/case['name']).mkdir(exist_ok=True)
        run_case(run,cfg,case)
        return
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('status')=='complete':
        raise ValueError('已经完成的固定执行矩阵不能覆盖')
    parent=ROOT/cfg['execution_parent']
    if pd.Timestamp(cfg['execution'][1])>=pd.Timestamp('2024-01-01') or not cfg['retail']['selection_first']:
        raise ValueError('禁止TEST或改变预登记的名单优先执行')
    sources=[Path(__file__),args.config,ROOT/'src/portfolio.py',ROOT/'src/retail_rebalance.py',
        ROOT/'src/selection_first.py',ROOT/'scripts/audit_monthly_portfolio.py',
        ROOT/'scripts/run_patient_value_execution.py',ROOT/'configs/corporate_action_components.yaml']
    manifest=dict(status='registered',fresh_holdout=False,test_market_data_used=False,cases=cases,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources},
        prediction_hash=digest(run/'execution_predictions.parquet'),
        hold_hash=digest(parent/'monthly_hold_allowed.parquet'),
        order_test='separate_process; reverse_adapter_frame_only; priorities_unchanged; finally_restore',
        stage_attribution='known_entry_phase_fixed_per_real_position_episode',
        limitation='unknown_trend_may_persist_about_half_year_after_missing_marks; retained_not_confirmed_trend')
    json_write(manifest_path,manifest)
    for path in sources:
        dest=out/'sources'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    hold=pd.read_parquet(parent/'monthly_hold_allowed.parquet');hold.to_parquet(out/'monthly_hold_allowed.parquet')
    predictions=pd.read_parquet(run/'execution_predictions.parquet')
    matrices=make_matrices(predictions,hold,cfg)
    p=yaml.safe_load((parent/'execution_config.yaml').read_text());p['valid']=cfg['execution']
    p['portfolio']['top_n']=cfg['retail']['max_positions']
    (out/'execution_config.yaml').write_text(yaml.safe_dump(p,allow_unicode=True,sort_keys=False))
    codes=set();coverage=[]
    for policy,(signals,forecasts) in matrices.items():
        signals.to_parquet(out/f'{policy}_signals.parquet');forecasts.to_parquet(out/f'{policy}_point_forecasts.parquet')
        for date,row in signals.iterrows():
            ranked=row.dropna().sort_values(ascending=False,kind='stable')
            codes.update(ranked.head(cfg['retail']['candidate_pool']).index)
            coverage.append(dict(policy=policy,date=date,admitted=len(ranked),candidate_pool=min(len(ranked),cfg['retail']['candidate_pool'])))
    manifest['calendar_only_anchor']=not bool(codes)
    if not codes:codes.add('000001.SZ')
    pd.DataFrame({'ts_code':sorted(codes)}).to_csv(out/'execution_codes.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'entry_coverage.csv',index=False)
    cache=Path((parent/'execution_cache.txt').read_text().strip())
    missing=[code for code in sorted(codes) if any(not(cache/api/f'{code}.parquet').exists() for api in ['dividend','stk_limit'])]
    pd.DataFrame({'ts_code':missing}).to_csv(out/'missing_cache_codes.csv',index=False)
    print(f'20个固定案例已登记；历史入场候选并集{len(codes)}只，待补缓存{len(missing)}只',flush=True)
    if missing:
        from prepare_monthly_audit import fetch_execution_data
        cache=fetch_execution_data(missing,tuple(map(pd.Timestamp,cfg['execution'])),p)
    (out/'execution_cache.txt').write_text(str(cache))
    manifest.update(status='prepared',execution_codes=len(codes),downloaded_missing_codes=len(missing))
    json_write(manifest_path,manifest)
    if args.prepare_only:return
    summaries=[];failures=[];yearly=[]
    for i,case in enumerate(cases,1):
        case_dir=out/case['name'];case_dir.mkdir(exist_ok=True)
        print(f'[{i}/{len(cases)}] {case["name"]}',flush=True)
        with (case_dir/'run.log').open('w') as log:
            completed=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
                '--run',str(run),'--case-name',case['name']],stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            # 仅输出本地账本异常尾行；远端下载异常在准备阶段由安全接口脱敏。
            last=(case_dir/'run.log').read_text().splitlines()[-1]
            failures.append(dict(name=case['name'],error=last,returncode=completed.returncode))
            print(last,flush=True)
        else:
            result=json.loads((case_dir/'summary.json').read_text());summaries.append(result)
            yearly.append(pd.read_csv(case_dir/'yearly.csv'))
            print(json.dumps({k:result[k] for k in ['name','total','mdd','n_trades','mean_actual_entry_rank']},ensure_ascii=False),flush=True)
        pd.DataFrame(summaries).to_csv(out/'execution_summary.csv',index=False)
        pd.DataFrame(failures,columns=['name','error','returncode']).to_csv(out/'execution_failures.csv',index=False)
        manifest.update(status='running',completed_cases=len(summaries),failed_cases=len(failures));json_write(manifest_path,manifest)
    if yearly:pd.concat(yearly,ignore_index=True).to_csv(out/'execution_yearly.csv',index=False)
    invariance=compare_reversed_cases(out,cfg) if not failures else None
    manifest.update(status='failed' if failures or not invariance.identical.all() else 'complete',
        order_invariance_pass=bool(invariance.identical.all()) if invariance is not None else False)
    json_write(manifest_path,manifest)
    if manifest['status']!='complete':raise AccountingError('真实执行或候选排列不变性检验未通过')


if __name__=='__main__':main()
