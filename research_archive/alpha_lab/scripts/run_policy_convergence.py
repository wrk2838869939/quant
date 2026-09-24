# 历史实验脚本归档：运行依赖原研究环境及未发布输入。
#!/usr/bin/env python3
"""从精确旧检查点延长全参数演化；训练平台判据独立于验证/外层。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_initialized_policy as R
from initialized_policy import NeuralPolicy, AnnualPolicy
from policy_portfolio_cached import PreparedMarket, run_backtest as cached_backtest
from policy_evolution import candidates, update, plateau, restore_rng
from datahub import DataHub
from run_classic_strategy_smoke import digest, save_json
G=R.G


def resolve(row):
    return Path(row['artifact_root'])/row['folder']


def old_row(path):
    row=json.loads(path.read_text())
    row['artifact_root']=str(G['parent_policy'])
    return row


def setup(config_path, run):
    extension=yaml.safe_load(config_path.read_text())
    parent=PROJECT/extension['parent_run']
    pm=json.loads((parent/'manifest.json').read_text())
    assert pm['status']=='complete'
    assert all(digest(PROJECT/p)==h for p,h in pm['source_hashes'].items())
    plan=yaml.safe_load((PROJECT/extension['parent_config']).read_text())
    registry=yaml.safe_load((PROJECT/plan['registry']).read_text())
    account=yaml.safe_load((PROJECT/registry['account_config']).read_text())
    assert os.environ.get('DATAHUB_CONTENT_ID')==account['dataset']['content_id']
    base=PROJECT/registry['source_run']
    assert all(digest(base/'inputs'/p)==h for p,h in pm['input_hashes'].items())
    frames={k:pd.read_parquet(base/'inputs'/f'{k}.parquet') for k in ['market','basics','st','limits','actions']}
    eligible=pd.read_parquet(base/'inputs/eligible.parquet')
    history=pd.read_parquet(base/'inputs/rule_history.parquet')
    calendar=DataHub().calendar.cn_days()
    grid=calendar[(calendar>=pd.Timestamp('2016-01-01'))&(calendar<=eligible.index.max())]
    wide={k:history.pivot(index='date',columns='ts_code',values=k).reindex(index=grid,columns=eligible.columns)
          for k in history if k not in ['date','ts_code']}
    smoke=yaml.safe_load((base/'sources/classic_strategy_smoke.yaml').read_text())
    portfolio=yaml.safe_load((base/'sources/monthly_portfolio_audit.yaml').read_text())['portfolio']
    portfolio.update(capital=account['account']['capital'],top_n=5,band_mult=1.,
                     **{k:smoke[k] for k in ['buy_cost','sell_cost','min_fee','cash_dividend_haircut','annual_days']})
    rules=yaml.safe_load((base/'sources/corporate_action_components.yaml').read_text())
    R.original.G.update(frames=frames,portfolio=portfolio,smoke=smoke,rules=rules,plan=plan)
    G.update(plan=plan,extension=extension,registry=registry,run=run,parent_policy=parent,
             parent=PROJECT/plan['parent_run'],portfolio=portfolio,frames=frames,smoke=smoke,rules=rules,
             eligible=eligible,wide=wide,calendar=calendar,prepared={},
             specs=yaml.safe_load((PROJECT/plan['features']).read_text())['features'],
             parent_choices=json.loads((PROJECT/plan['parent_run']/'all_locked_choices.json').read_text()))
    G['training']={y:R.get_package(y,f'{y-2}-12-31') for y in plan['outer_years']}
    G['validation']={y:R.window_package(R.get_package(y,f'{y-1}-12-31'),f'{y-1}-01-01',f'{y-1}-12-31') for y in plan['outer_years']}
    for y in plan['outer_years']:
        get_prepared(plan['train_start'],f'{y-2}-12-31')
        get_prepared(f'{y-1}-01-01',f'{y-1}-12-31')
    return pm,account


def get_prepared(start,end):
    key=(str(pd.Timestamp(start).date()),str(pd.Timestamp(end).date()))
    if key not in G['prepared']:
        f=G['frames']
        G['prepared'][key]=PreparedMarket(f['market'],f['basics'],f['st'],f['limits'],*key)
    return G['prepared'][key]


def execute_cached(policy,start,end):
    f=G['frames']
    signals=pd.DataFrame(index=pd.DatetimeIndex([],name='date'),columns=G['eligible'].columns,dtype=float)
    return cached_backtest(signals,f['market'],f['basics'],f['st'],f['limits'],f['actions'],start,end,G['portfolio'],
                           G['rules'].get('rights_issues',[]),policy=policy,slippage_rate=G['smoke']['slippage_rate'],
                           prepared=get_prepared(start,end))


def task_folder(task):
    year,host,initialization=task
    return G['run']/'search'/str(year)/host/initialization


def state_path(task,generation):
    return task_folder(task)/f'state_g{generation:03d}.json'


def save_state(task,generation,center,champion_theta,champion,multiplier,rng,archive):
    folder=task_folder(task)
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/f'center_g{generation:03d}.npy',center)
    np.save(folder/f'champion_g{generation:03d}.npy',champion_theta)
    state=dict(generation=generation,center=f'center_g{generation:03d}.npy',theta=f'champion_g{generation:03d}.npy',
               champion=champion,multiplier=multiplier,rng_state=rng.bit_generator.state,archive=archive)
    save_json(state_path(task,generation),state)
    return state


def reconstruct(task):
    """不运行账户，重建原8代全部候选和中心；随后另作真实账户等价验证。"""
    year,host,initialization=task
    old=G['parent_policy']/'search'/str(year)/host/initialization
    cfg=G['plan']['search'];seed=G['plan']['seed']+year*10+G['plan']['hosts'].index(host)
    initial=old_row(old/'train/g00_initial/trial.json')
    theta0=np.load(resolve(initial)/'theta.npy')
    compiled=NeuralPolicy.initialize(G['training'][year],G['plan']['hidden'],G['plan']['hosts'].index(host),seed,initialization=='random')
    np.testing.assert_array_equal(theta0,compiled)
    rng=np.random.default_rng(seed)
    scales=NeuralPolicy.scales(G['training'][year]['features'].shape[2],G['plan']['hidden'])
    center,theta,champion,multiplier=theta0.copy(),theta0.copy(),initial,1.
    old_updates=json.loads((old/'updates.json').read_text());archive=[initial]
    checked=0
    for gen in range(1,G['extension']['resume_generation']+1):
        evaluated=[]
        for j,(candidate,sigma,pair,direction) in enumerate(candidates(center,theta,scales,multiplier,rng,cfg)):
            row=old_row(old/f'train/g{gen:02d}_c{j:02d}/trial.json')
            np.testing.assert_array_equal(candidate,np.load(resolve(row)/'theta.npy'))
            assert (sigma,pair,direction)==(row['sigma'],row['pair'],row['direction'])
            evaluated.append((candidate,row));checked+=1
        center,theta,champion,multiplier,success,_=update(evaluated,theta,champion,multiplier,cfg)
        np.testing.assert_array_equal(center,np.load(old/f'center_g{gen:02d}.npy'))
        assert old_updates[gen-1]['champion']==champion['folder']
        assert old_updates[gen-1]['scale_multiplier']==multiplier
        assert old_updates[gen-1]['success_rate']==success
        archive.append(champion)
    state=save_state(task,8,center,theta,champion,multiplier,rng,archive)
    return dict(task=task,exact_candidate_vectors=checked,exact_centers=8),state


def train_job(task_and_target):
    task,target=task_and_target;task=tuple(task)
    year,host,initialization=task
    try:
        existing=sorted(task_folder(task).glob('state_g*.json'))
        path=existing[-1];state=json.loads(path.read_text())
        start=state['generation'];assert start<target
        center=np.load(path.parent/state['center']);theta=np.load(path.parent/state['theta'])
        champion=state['champion'];multiplier=state['multiplier'];archive=state['archive']
        rng=restore_rng(state['rng_state']);cfg=G['plan']['search']
        scales=NeuralPolicy.scales(G['training'][year]['features'].shape[2],G['plan']['hidden'])
        theta0=np.load(resolve(archive[0])/'theta.npy')
        records=[]
        for gen in range(start+1,target+1):
            evaluated=[]
            for j,(candidate,sigma,pair,direction) in enumerate(candidates(center,theta,scales,multiplier,rng,cfg)):
                folder=task_folder(task)/f'train/g{gen:03d}_c{j:02d}'
                row=R.evaluate(candidate,G['training'][year],G['plan']['train_start'],f'{year-2}-12-31',folder,
                               dict(outer_year=year,host=host,initialization=initialization,generation=gen,candidate=j,
                                    sigma=sigma,pair=pair,direction=direction,artifact_root=str(G['run']),
                                    distance_from_initial=float(np.linalg.norm((candidate-theta0)/scales))))
                evaluated.append((candidate,row))
            center,theta,champion,multiplier,success,elite=update(evaluated,theta,champion,multiplier,cfg)
            archive.append(champion)
            records.append(dict(generation=gen,success_rate=success,scale_multiplier=multiplier,
                                champion=champion,elite=[r['folder'] for r in elite],
                                center_distance=float(np.linalg.norm((center-theta0)/scales))))
            save_state(task,gen,center,theta,champion,multiplier,rng,archive)
            save_json(task_folder(task)/f'updates_{start+1:03d}_{target:03d}.json',records)
            if gen%8==0:
                print(f'GEN {year} {host} {initialization} {gen}/{target} return={champion["metrics"]["total"]:.5f} scale={multiplier:.5f}',flush=True)
        assessment=plateau(archive,target,G['extension']['convergence'])
        save_json(task_folder(task)/f'convergence_g{target:03d}.json',assessment)
        return dict(status='ok',task=task,generation=target,assessment=assessment)
    except Exception as exc:
        return dict(status='error',task=task,error=repr(exc),traceback=traceback.format_exc())


def validation_job(task_and_target):
    task,target=task_and_target;task=tuple(task);year,host,initialization=task
    try:
        state=json.loads(state_path(task,target).read_text());archive=state['archive']
        cache={};rows=[]
        for gen,row in enumerate(archive):
            identity=row['parameter_sha256']
            if identity not in cache:
                # 原8代的验证结果照样重算核验，保持过程可独立重现。
                theta=np.load(resolve(row)/'theta.npy')
                folder=task_folder(task)/'validation'/identity
                measured=R.evaluate(theta,G['validation'][year],f'{year-1}-01-01',f'{year-1}-12-31',folder,
                                    dict(outer_year=year,host=host,initialization=initialization,generation=gen,
                                         source=row,artifact_root=str(G['run'])))
                cache[identity]=measured
            rows.append(dict(cache[identity],archive_generation=gen))
        choices={}
        budgets=sorted(set(G['extension']['checkpoints']+[target]))
        for budget in budgets:
            eligible=[r for r in rows[:budget+1] if r['metrics']['feasible']]
            choice=max(eligible,key=R.key) if eligible else None
            choices[budget]=dict(selected=choice,training_champion_validation=rows[budget],
                                 nominal_validation_evaluations=budget+1,
                                 unique_validation_candidates=len({r['parameter_sha256'] for r in rows[:budget+1]}))
        result=dict(status='ok',task=task,final_generation=target,choices=choices,
                    unique_validation_evaluations=len(cache),locked_at=datetime.now(timezone.utc).isoformat())
        save_json(task_folder(task)/'locked_choices.json',result)
        return result
    except Exception as exc:
        return dict(status='error',task=task,error=repr(exc),traceback=traceback.format_exc())


def pool_run(function,tasks,label):
    results=[]
    with mp.get_context('fork').Pool(G['extension']['workers']) as pool:
        for result in pool.imap_unordered(function,tasks):
            results.append(result)
            save_json(G['run']/f'{label}_results.json',results)
            if result['status']!='ok':
                pool.terminate()
                raise RuntimeError(result['traceback'])
            print(label+' FINISHED '+str(result['task']),flush=True)
    return results


def verify_accounts(states):
    rows=[]
    for task,state in states.items():
        year,host,initialization=task
        package=G['training'][year];theta=np.load(resolve(state['champion'])/'theta.npy')
        policy=NeuralPolicy(package,theta,G['plan']['hidden'],G['portfolio']['capital'])
        begin=time.monotonic()
        result=execute_cached(policy,G['plan']['train_start'],f'{year-2}-12-31')
        seconds=time.monotonic()-begin
        for table in R.original.TABLES:
            pd.testing.assert_frame_equal(getattr(result,table),pd.read_parquet(resolve(state['champion'])/f'{table}.parquet'),check_exact=True)
        begin=time.monotonic();R.original.audit(result);audit_seconds=time.monotonic()-begin
        rows.append(dict(task=task,exact_four_tables=True,rollout_seconds=seconds,audit_seconds=audit_seconds))
    save_json(G['run']/'cached_engine_parity.json',rows)
    print('CACHED ENGINE PARITY 24/24; mean rollout seconds='+str(np.mean([r['rollout_seconds'] for r in rows])),flush=True)


def evaluate_outer(locked,final_targets):
    plan=G['plan'];summaries=[];checks=[]
    packages={y:R.get_package(y) for y in plan['outer_years']}
    prefix_packages={}
    for cutoff in plan['prefix_dates']:
        for year in plan['outer_years']:
            if year<=pd.Timestamp(cutoff).year:
                end=min(pd.Timestamp(cutoff),pd.Timestamp(f'{year}-12-31'))
                prefix_packages[(cutoff,year)]=R.window_package(R.get_package(year,str(end.date())),f'{year}-01-01',str(end.date()))
    selectors=['validation_selected','training_champion']
    for checkpoint in [*G['extension']['checkpoints'],'final']:
        for host in plan['hosts']:
            for initialization in plan['initializations']:
                for selector in selectors:
                    # 所有检查点记录冠军验证值；完整冠军外层仅8、64、最终三档。
                    if selector=='training_champion' and checkpoint not in [8,64,'final']:
                        continue
                    models={};thetas={};budgets={}
                    for year in plan['outer_years']:
                        task=(year,host,initialization)
                        budget=final_targets[task] if checkpoint=='final' else checkpoint
                        choice=locked[task]['choices'][budget]
                        row=choice['selected'] if selector=='validation_selected' else choice['training_champion_validation']
                        p=R.window_package(packages[year],f'{year}-01-01',f'{year}-12-31')
                        if row:
                            theta=np.load(resolve(row)/'theta.npy')
                        else:
                            shapes=NeuralPolicy.shapes(p['features'].shape[2],plan['hidden'])
                            theta=np.zeros(sum(int(np.prod(shape)) for _,shape in shapes))
                            theta[sum(int(np.prod(shape)) for _,shape in shapes[:-1])+plan['hidden']*3]=1.
                        thetas[year]=theta;budgets[year]=budget
                        models[year]=NeuralPolicy(p,theta,plan['hidden'],G['portfolio']['capital'])
                    result=execute_cached(AnnualPolicy(models),plan['outer_start'],plan['outer_end'])
                    folder=G['run']/'outer'/str(checkpoint)/host/initialization/selector
                    R.save_account(folder,result,list(models.values()),True)
                    for year,theta in thetas.items(): np.save(folder/f'theta_{year}.npy',theta)
                    if checkpoint==8 and selector=='validation_selected':
                        reference=G['parent_policy']/'outer'/host/initialization
                        for table in R.original.TABLES:
                            pd.testing.assert_frame_equal(getattr(result,table),pd.read_parquet(reference/f'{table}.parquet'),check_exact=True)
                    if checkpoint in [64,'final']:
                        for cutoff in plan['prefix_dates']:
                            short_models={}
                            for year in plan['outer_years']:
                                if year>pd.Timestamp(cutoff).year: continue
                                p=prefix_packages[(cutoff,year)];full=models[year].package;n=len(p['dates'])
                                np.testing.assert_array_equal(p['features'],full['features'][:n])
                                np.testing.assert_array_equal(p['schedules'],full['schedules'][:n])
                                short_models[year]=NeuralPolicy(p,thetas[year],plan['hidden'],G['portfolio']['capital'])
                            short=execute_cached(AnnualPolicy(short_models),plan['outer_start'],cutoff)
                            R.original.compare_prefix(short,result,pd.Timestamp(cutoff));R.original.audit(short)
                            checks.append(dict(checkpoint=checkpoint,host=host,initialization=initialization,selector=selector,cutoff=cutoff,passed=True))
                    summaries.append(dict(checkpoint=checkpoint,host=host,initialization=initialization,selector=selector,
                                          budgets=json.dumps(budgets),**result.summary,
                                          accepted=bool(result.summary['total']>0 and result.summary['mdd']>=-.30)))
                    pd.DataFrame(summaries).to_csv(G['run']/'outer_summary.csv',index=False)
                    save_json(G['run']/'prefix_checks.json',checks)
        print('OUTER CHECKPOINT '+str(checkpoint)+' complete',flush=True)
    return summaries,checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/policy_convergence_v1.yaml')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    run=args.resume.resolve() if args.resume else ROOT/'outputs/policy_convergence'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True,exist_ok=True);print('RUN='+str(run),flush=True)
    pm,account=setup(args.config,run)
    files=[Path(__file__),args.config,ROOT/'src/policy_evolution.py',ROOT/'src/policy_portfolio_cached.py',
           ROOT/'tests/test_policy_evolution.py',ROOT/'docs/POLICY_CONVERGENCE_PREREG_20260920.md']
    sources={**pm['source_hashes'],**{str(p.relative_to(PROJECT)):digest(p) for p in files}}
    if args.resume:
        manifest=json.loads((run/'manifest.json').read_text())
        assert manifest['source_hashes']==sources
        manifest.update(status='running',resumed_at=datetime.now(timezone.utc).isoformat())
        save_json(run/'manifest.json',manifest)
    else:
        (run/'sources').mkdir()
        for p in sources: shutil.copy2(PROJECT/p,run/'sources'/Path(p).name)
        manifest=dict(status='running',study=G['extension']['study'],source_hashes=sources,dataset=account['dataset'],
                      parent_run=str(G['parent_policy'].relative_to(PROJECT)),parent_manifest_sha256=digest(G['parent_policy']/'manifest.json'),
                      input_hashes=pm['input_hashes'],started_at=datetime.now(timezone.utc).isoformat(),unbiased_oos=False)
        save_json(run/'manifest.json',manifest)
    try:
        tasks=[(y,h,i) for y in G['plan']['outer_years'] for h in G['plan']['hosts'] for i in G['plan']['initializations']]
        if not args.resume:
            reconstructions=[];states={}
            for task in tasks:
                check,state=reconstruct(task);reconstructions.append(check);states[task]=state
            save_json(run/'resume_equivalence.json',reconstructions)
            verify_accounts(states)
        R.execute=execute_cached
        if args.verify_only:
            manifest.update(status='verification_complete');save_json(run/'manifest.json',manifest);return
        pending=[(t,64) for t in tasks if not state_path(t,64).exists()]
        if pending: pool_run(train_job,pending,'stage64')
        final_targets={};extension_decisions=[]
        for year in G['plan']['outer_years']:
            for host in G['plan']['hosts']:
                pair=[(year,host,i) for i in G['plan']['initializations']]
                assessments={t[2]:plateau(json.loads(state_path(t,64).read_text())['archive'],64,G['extension']['convergence']) for t in pair}
                extend=any(not a['plateau'] for a in assessments.values())
                target=256 if extend else 64
                for task in pair: final_targets[task]=target
                extension_decisions.append(dict(year=year,host=host,assessments=assessments,target=target))
        save_json(run/'extension_decisions.json',extension_decisions)
        print('EXTEND TO 256 TASKS='+str(sum(v==256 for v in final_targets.values())),flush=True)
        pending=[(t,b) for t,b in final_targets.items() if not state_path(t,b).exists()]
        if pending: pool_run(train_job,pending,'stage256')
        locked={}
        pending=[]
        for task,budget in final_targets.items():
            p=task_folder(task)/'locked_choices.json'
            if p.exists():
                row=json.loads(p.read_text());row['choices']={int(k):v for k,v in row['choices'].items()};locked[task]=row
            else: pending.append((task,budget))
        if pending:
            for row in pool_run(validation_job,pending,'validation'): locked[tuple(row['task'])]=row
        save_json(run/'all_locked_choices.json',list(locked.values()));lock_hash=digest(run/'all_locked_choices.json')
        summaries,checks=evaluate_outer(locked,final_targets)
        assert digest(run/'all_locked_choices.json')==lock_hash
        assert all(digest(PROJECT/p)==h for p,h in sources.items())
        manifest.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat(),
                        total_training_evaluations=sum(1+12*b for b in final_targets.values()),
                        new_training_evaluations=sum(12*(b-8) for b in final_targets.values()),
                        validation_unique_evaluations=sum(r['unique_validation_evaluations'] for r in locked.values()),
                        tasks_extended=sum(b==256 for b in final_targets.values()),outer_accounts=len(summaries),
                        prefix_checks=len(checks),locked_choices_sha256=lock_hash)
        save_json(run/'manifest.json',manifest)
        save_json(run/'artifact_hashes.json',{str(p.relative_to(run)):digest(p) for p in run.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc),traceback=traceback.format_exc());save_json(run/'manifest.json',manifest);raise


if __name__=='__main__': main()
