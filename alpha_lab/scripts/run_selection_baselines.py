#!/usr/bin/env python3
"""重建并锁定三条规则研究基线，可同条件评估一个候选分数插件。"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib,json,os,shutil,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from datahub import DataHub
from selection_baselines import build_signals,validate_candidate
from portfolio import run_backtest
from classic_smoke import verify_ledger
from rule_search import inner_metrics
from run_rule_strategy_search import write_result
from run_classic_strategy_smoke import digest,save_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registry',type=Path,default=ROOT/'configs/selection_baselines_v1.yaml')
    ap.add_argument('--candidate',type=Path)
    args=ap.parse_args();registry=yaml.safe_load(args.registry.read_text());base=PROJECT/registry['source_run']
    account=yaml.safe_load((PROJECT/registry['account_config']).read_text())
    assert os.environ.get('DATAHUB_CONTENT_ID')==account['dataset']['content_id'],'必须通过冻结启动器执行'
    assert account['risk']['max_drawdown']==registry['acceptance']['max_drawdown']
    assert digest(base/'manifest.json')==registry['source_manifest_sha256']
    assert digest(base/'all_locked_choices.json')==registry['source_choices_sha256']
    assert all(digest(PROJECT/p)==h for p,h in registry['protected_source_hashes'].items()),'基线代码或账户协议已改变；应单独版本化'
    source_choices=json.loads((base/'all_locked_choices.json').read_text())
    for entry in registry['baselines'].values():
        for year,item in entry['annual_parameters'].items():
            locked=next(c for c in source_choices if c['family']==entry['source_family'] and c['outer_year']==year)
            assert item['params']==locked['selected']['params'] and item['inner_end']==locked['inner_end']
    hashes=json.loads((base/'inputs/frozen_manifest.json').read_text())
    hashes['rule_history.parquet']=json.loads((base/'inputs/rule_history_manifest.json').read_text())['sha256']
    assert all(digest(base/'inputs'/p)==h for p,h in hashes.items())
    candidate=transform=module=None
    if args.candidate:
        candidate=yaml.safe_load(args.candidate.read_text());validate_candidate(candidate,registry)
        module=importlib.import_module(candidate['plugin_module'])
        module_path=Path(module.__file__).resolve()
        if not module_path.is_relative_to(ROOT/'src'):
            raise ValueError('策略插件须位于alpha_lab/src目录')
        transform=getattr(module,candidate['plugin_function'])
    run=ROOT/'outputs/selection_baselines'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');run.mkdir(parents=True)
    print('RUN='+str(run),flush=True)
    sources=[Path(__file__),ROOT/'src/selection_baselines.py',ROOT/'src/rule_search.py',ROOT/'src/portfolio.py',ROOT/'src/classic_smoke.py',
             ROOT/'scripts/run_rule_strategy_search.py',ROOT/'scripts/run_classic_strategy_smoke.py',ROOT/'scripts/audit_monthly_portfolio.py',args.registry,PROJECT/registry['account_config']]
    if candidate:sources.extend([args.candidate,module_path])
    source_hashes={str(p.resolve().relative_to(PROJECT)):digest(p) for p in sources}
    (run/'sources').mkdir()
    for p in sources:shutil.copy2(p,run/'sources'/p.name)
    save_json(run/'input_hashes.json',hashes)
    manifest=dict(status='running',registry_id=registry['registry_id'],registry_sha256=digest(args.registry),source_hashes=source_hashes,
                  source_run=registry['source_run'],dataset=account['dataset'],started_at=datetime.now(timezone.utc).isoformat(),
                  parameter_search=False,candidate=candidate)
    save_json(run/'manifest.json',manifest)
    try:
        frames={k:pd.read_parquet(base/'inputs'/f'{k}.parquet') for k in ['market','basics','st','limits','actions']}
        eligible=pd.read_parquet(base/'inputs/eligible.parquet');h=pd.read_parquet(base/'inputs/rule_history.parquet')
        calendar=DataHub().calendar.cn_days()
        grid=calendar[(calendar>=pd.Timestamp('2016-01-01'))&(calendar<=eligible.index.max())]
        wide={k:h.pivot(index='date',columns='ts_code',values=k).reindex(index=grid,columns=eligible.columns) for k in h if k not in ['date','ts_code']}
        smoke=yaml.safe_load((base/'sources/classic_strategy_smoke.yaml').read_text())
        cfg=yaml.safe_load((base/'sources/monthly_portfolio_audit.yaml').read_text())['portfolio']
        cfg.update(capital=account['account']['capital'],top_n=5,band_mult=1.,**{k:smoke[k] for k in ['buy_cost','sell_cost','min_fee','cash_dividend_haircut','annual_days']})
        rules=yaml.safe_load((base/'sources/corporate_action_components.yaml').read_text())
        start,end=registry['evaluation']['start'],registry['evaluation']['end']
        def execute(signals,schedule,end_date=end):
            return run_backtest(signals,frames['market'],frames['basics'],frames['st'],frames['limits'],frames['actions'],start,end_date,cfg,rules.get('rights_issues',[]),slippage_rate=smoke['slippage_rate'],top_n_schedule=schedule)
        jobs=[(bid,entry,None,{}) for bid,entry in registry['baselines'].items()]
        if candidate:jobs.append((candidate['candidate_id'],registry['baselines'][candidate['parent_baseline']],transform,candidate.get('params',{})))
        rows=[];yearly=[];checks=[];results={}
        for bid,entry,plugin,parameters in jobs:
            signals,schedule=build_signals(entry,wide,eligible,calendar,start,end,plugin,parameters)
            result=execute(signals,schedule);out=run/'accounts'/bid
            write_result(out,result,signals);schedule.to_frame().to_parquet(out/'top_n_schedule.parquet')
            audit=verify_ledger(result,cfg['capital'],frames['market'],frames['limits'],frames['actions'],cfg,smoke['slippage_rate'],smoke['audit']['money_atol'])
            save_json(out/'audit.json',audit)
            assert result.daily.ret.abs().max()<smoke['audit']['extreme_daily_return']
            if plugin is None:
                reference=base/'outer'/entry['source_family']
                assert all(digest(reference/p)==h for p,h in entry['reference_artifacts'].items())
                pd.testing.assert_frame_equal(signals,pd.read_parquet(reference/'signals.parquet'))
                pd.testing.assert_frame_equal(schedule.to_frame(),pd.read_parquet(reference/'top_n_schedule.parquet'))
                for table in ['daily','trades','events','holdings']:
                    pd.testing.assert_frame_equal(getattr(result,table),pd.read_parquet(reference/f'{table}.parquet'))
            for cutoff in registry['prefix_checks']:
                cut=pd.Timestamp(cutoff)
                short,short_schedule=build_signals(entry,{k:v.loc[:cut] for k,v in wide.items()},eligible.loc[:cut],calendar,start,cut,plugin,parameters)
                pd.testing.assert_frame_equal(short,signals.loc[:cut])
                pd.testing.assert_series_equal(short_schedule,schedule.loc[:cut])
                replay=execute(short,short_schedule,cut)
                for table in ['daily','trades','events','holdings']:
                    a,b=getattr(replay,table),getattr(result,table)
                    b=b.loc[a.index] if table=='daily' else (b[b.date<=cut].reset_index(drop=True) if len(b) else b.copy())
                    columns=sorted(set(a.columns)|set(b.columns))
                    pd.testing.assert_frame_equal(a.reindex(columns=columns).reset_index(drop=True),b.reindex(columns=columns).reset_index(drop=True),check_dtype=False)
                checks.append(dict(strategy=bid,cutoff=cutoff,signals=True,account_tables=True))
            metrics=inner_metrics(result,cfg['annual_days'],cfg['capital'])
            yearly.extend(dict(strategy=bid,**value) for value in metrics['yearly'])
            accepted=bool(result.summary['total']>0 and result.summary['mdd']>=-account['risk']['max_drawdown'])
            rows.append(dict(strategy=bid,name=entry['name'] if plugin is None else bid,kind='baseline' if plugin is None else 'candidate',
                             parent=bid if plugin is None else candidate['parent_baseline'],historical_acceptance=accepted,**result.summary))
            results[bid]=result
            print(f"VERIFIED {bid} total={result.summary['total']:.6f} mdd={result.summary['mdd']:.6f}",flush=True)
        summary=pd.DataFrame(rows)
        for i,row in summary.iterrows():
            parent=results[row.parent].summary
            summary.loc[i,'delta_total_pp']=(row.total-parent['total'])*100
            summary.loc[i,'delta_mdd_pp']=(row.mdd-parent['mdd'])*100
            summary.loc[i,'delta_fees_yuan']=row.total_fees-parent['total_fees']
        summary.to_csv(run/'summary.csv',index=False);pd.DataFrame(yearly).to_csv(run/'yearly.csv',index=False)
        save_json(run/'prefix_checks.json',checks)
        if candidate and candidate['plugin_module']=='selection_baseline_plugins' and candidate['plugin_function']=='identity':
            for table in ['daily','trades','events','holdings']:
                pd.testing.assert_frame_equal(getattr(results[candidate['candidate_id']],table),getattr(results[candidate['parent_baseline']],table))
            save_json(run/'identity_check.json',dict(passed=True,tables=4))
        assert all(digest(PROJECT/p)==h for p,h in source_hashes.items()),'运行中代码或配置发生变化'
        assert all(digest(base/'inputs'/p)==h for p,h in hashes.items())
        manifest.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),baselines=len(registry['baselines']),
                        accounts=len(rows),prefix_checks=len(checks),reference_replays=3,source_inputs_unchanged=True)
        lines=['# 固定规则基线重放','',f"登记版本：{registry['registry_id']}；运行：{run.relative_to(PROJECT)}。不重新选参，年度资金连续，含原费用与滑点。",'',
               '| 基线/候选 | 累计净收益 | 年化 | 最大回撤 | 交易笔数 | 历史验收 |','|---|---:|---:|---:|---:|---|']
        for row in summary.itertuples():lines.append(f'| {row.strategy} | {row.total:.2%} | {row.ann:.2%} | {abs(row.mdd):.2%} | {row.n_trades} | {"通过" if row.historical_acceptance else "未通过"} |')
        lines+=['','三条基线的信号、持股数日程及四张账户过程表与原归档一致；所有账户通过独立台账与三个截止点的信号/账户前缀核验。',
                '候选接口固定所属基线的频率、持股数、资金及执行，仅改变评分/过滤；delta_mdd_pp为正表示回撤变浅。候选不会自动覆盖基线。',
                '年度参数来自此前内层选参；选择这三条基线又使用了已观察外层结果，所以只能作为开发对照，不是新的独立样本外证明。93股有全期质量隔离偏差。2023之后没有登记参数，禁止默认延用。',
                '该版未组合三条策略，也未作为实盘参数部署。后续研发约定见docs/SELECTION_BASELINES_V1.md。','']
        (run/'REPORT.md').write_text('\n'.join(lines))
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:save_json(run/'manifest.json',manifest)


if __name__=='__main__':main()
