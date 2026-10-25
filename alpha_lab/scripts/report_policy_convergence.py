#!/usr/bin/env python3
"""从续训归档生成训练收敛、验证与连续账户的分离报告。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from policy_evolution import fitness_key


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str))


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def concentration(folder):
    """按全部账户交易日统计单只股票权重，现金日计零。"""
    daily=pd.read_parquet(folder/'daily.parquet')
    holdings=pd.read_parquet(folder/'holdings.parquet')
    if holdings.empty:
        return dict(max_stock_weight=0.,median_daily_max_weight=0.,days_over_80pct=0.,top_mean_weight_code=None,top_mean_stock_weight=0.)
    holdings=holdings.copy()
    holdings['weight']=holdings['value']/holdings['date'].map(daily['equity'])
    assert holdings['weight'].notna().all()
    maximum=holdings.groupby('date')['weight'].max().reindex(daily.index,fill_value=0.)
    means=holdings.groupby('ts_code')['weight'].sum()/len(daily)
    return dict(max_stock_weight=float(maximum.max()),median_daily_max_weight=float(maximum.median()),
                days_over_80pct=float((maximum>.8).mean()),top_mean_weight_code=str(means.idxmax()),
                top_mean_stock_weight=float(means.max()))


def read_states(run):
    result=[]
    for folder in sorted((run/'search').glob('*/*/*')):
        paths=sorted(folder.glob('state_g*.json'))
        if not paths: continue
        state=json.loads(paths[-1].read_text())
        result.append((folder,state))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--progress',action='store_true')
    args=parser.parse_args();run=args.run.resolve()
    manifest=json.loads((run/'manifest.json').read_text())
    states=read_states(run)
    if args.progress:
        print('completed_generation_evaluations',sum(1+12*s['generation'] for _,s in states),
              'finished256',sum(s['generation']==256 for _,s in states))
        for year in [2021,2022,2023]:
            group=[s for f,s in states if int(f.parts[-3])==year]
            print(year,'generations',[s['generation'] for s in group],
                  'median_log_wealth_gain_vs_g8',round(float(np.median([np.log1p(s['champion']['metrics']['total'])-np.log1p(s['archive'][8]['metrics']['total']) for s in group])),4))
        return
    assert manifest['status']=='complete'
    summary=pd.read_csv(run/'outer_summary.csv',dtype={'checkpoint':str})
    choices=json.loads((run/'all_locked_choices.json').read_text())
    assert digest(run/'all_locked_choices.json')==manifest['locked_choices_sha256']
    extensions=json.loads((run/'extension_decisions.json').read_text())
    output=run/'diagnostics';output.mkdir(exist_ok=True)
    curves=[];convergence=[];validation=[];exposures=[]
    max_drawdown=yaml.safe_load((run/'sources/baseline_initialized_policy_v1.yaml').read_text())['search']['max_drawdown']
    parameter_changes=equal_objective_changes=0
    for folder,state in states:
        year,host,initialization=int(folder.parts[-3]),folder.parts[-2],folder.parts[-1]
        for before,after in zip(state['archive'][8:-1],state['archive'][9:]):
            if before['parameter_sha256']!=after['parameter_sha256']:
                parameter_changes+=1
                equal_objective_changes+=int(fitness_key(before,max_drawdown)==fitness_key(after,max_drawdown))
        updates={}
        for p in folder.glob('updates_*.json'):
            for row in json.loads(p.read_text()): updates[row['generation']]=row
        validation_cache={}
        for gen,row in enumerate(state['archive']):
            identity=row['parameter_sha256']
            if identity not in validation_cache:
                validation_cache[identity]=json.loads((folder/'validation'/identity/'trial.json').read_text())['metrics']
            vm=validation_cache[identity]
            m=row['metrics'];u=updates.get(gen,{})
            curves.append(dict(year=year,host=host,initialization=initialization,generation=gen,
                               total=m['total'],ann=m['ann'],mdd=m['mdd'],feasible=m['feasible'],
                               validation_total=vm['total'],validation_ann=vm['ann'],validation_mdd=vm['mdd'],validation_feasible=vm['feasible'],
                               log_wealth=float(np.log1p(m['total'])),
                               scale_multiplier=u.get('scale_multiplier',np.nan),
                               center_distance=u.get('center_distance',np.nan),
                               champion_sha256=row['parameter_sha256']))
        assessment=json.loads((folder/f'convergence_g{state["generation"]:03d}.json').read_text())
        initial=state['archive'][8]['metrics'];last=state['champion']['metrics']
        champion_folder=Path(state['champion']['artifact_root'])/state['champion']['folder']
        exposures.append(dict(kind='final_training_champion',year=year,host=host,initialization=initialization,
                              total=last['total'],**concentration(champion_folder)))
        convergence.append(dict(year=year,host=host,initialization=initialization,final_generation=state['generation'],
                                empirical_plateau=assessment['plateau'],last_three_windows=assessment['windows'],
                                final_scale_multiplier=state['multiplier'],
                                at_original_scale_floor=bool(np.isclose(state['multiplier'],.05)),
                                generation8_total=initial['total'],final_total=last['total'],
                                wealth_gain_vs_g8=float((1+last['total'])/(1+initial['total'])-1)))
    for row in choices:
        year,host,initialization=row['task']
        for budget,item in row['choices'].items():
            selected=item['selected'];champ=item['training_champion_validation']
            validation.append(dict(year=year,host=host,initialization=initialization,budget=int(budget),
                                   selected_generation=selected['archive_generation'] if selected else None,
                                   selected_return=selected['metrics']['total'] if selected else 0.,
                                   selected_mdd=selected['metrics']['mdd'] if selected else 0.,
                                   champion_return=champ['metrics']['total'],champion_mdd=champ['metrics']['mdd'],
                                   champion_feasible=champ['metrics']['feasible'],
                                   unique_validation_candidates=item['unique_validation_candidates']))
    df=pd.DataFrame(curves);cd=pd.DataFrame(convergence);vd=pd.DataFrame(validation)
    for row in summary.itertuples():
        folder=run/'outer'/row.checkpoint/row.host/row.initialization/row.selector
        exposures.append(dict(kind='outer',checkpoint=row.checkpoint,selector=row.selector,
                              host=row.host,initialization=row.initialization,total=row.total,**concentration(folder)))
    ed=pd.DataFrame(exposures);ed.to_csv(output/'concentration.csv',index=False)
    gap_rows=[]
    for (year,host,initialization),group in df.groupby(['year','host','initialization']):
        first=group[group.generation==8].iloc[0]
        for budget in sorted(set([16,32,64,int(group.generation.max())])):
            last=group[group.generation==budget].iloc[0]
            gap_rows.append(dict(year=int(year),host=host,initialization=initialization,budget=budget,
                                 train_ann_change=float(last.ann-first.ann),
                                 validation_ann_change=float(last.validation_ann-first.validation_ann),
                                 train_up_validation_down=bool(last.ann>first.ann+1e-12 and last.validation_ann<first.validation_ann-1e-12)))
    gaps=pd.DataFrame(gap_rows)
    gaps.to_csv(output/'training_validation_gap.csv',index=False)
    df.to_csv(output/'training_curves.csv',index=False)
    cd.to_csv(output/'convergence_assessment.csv',index=False)
    vd.to_csv(output/'validation_checkpoints.csv',index=False)
    reports=[]
    for budget in ['8','16','32','64','final']:
        for initialization in ['baseline','random']:
            rows=summary[(summary.checkpoint==budget)&(summary.initialization==initialization)&(summary.selector=='validation_selected')]
            reports.append(dict(checkpoint=budget,initialization=initialization,mean_total=float(rows.total.mean()),
                                median_total=float(rows.total.median()),accepted=int(rows.accepted.sum())))
    stats=dict(total_training_evaluations=manifest['total_training_evaluations'],new_training_evaluations=manifest['new_training_evaluations'],
               unique_validation_evaluations=manifest['validation_unique_evaluations'],tasks_extended=manifest['tasks_extended'],
               final_empirical_plateau=int(cd.empirical_plateau.sum()),final_still_active=int((~cd.empirical_plateau).sum()),
               final_step_at_floor=int(cd.at_original_scale_floor.sum()),
               median_training_wealth_gain_vs_g8=float(cd.wealth_gain_vs_g8.median()),
               extended_champion_parameter_changes=parameter_changes,
               equal_objective_parameter_changes=equal_objective_changes,
               all_24_training_returns_nondecreasing=bool((cd.final_total>=cd.generation8_total).all()),
               summary_by_budget=reports,outer_accounts=manifest['outer_accounts'],prefix_checks=manifest['prefix_checks'],
               train_up_validation_down_at64=int(gaps[gaps.budget==64].train_up_validation_down.sum()),
               train_up_validation_down_at_final=int(gaps.sort_values('budget').groupby(['year','host','initialization']).tail(1).train_up_validation_down.sum()))
    save_json(output/'summary.json',stats)
    parent=PROJECT/manifest['parent_run']
    baseline={r['host']:r for r in json.loads((parent/'initialization_checks.json').read_text())}
    primary=summary[(summary.initialization=='baseline')&(summary.selector=='validation_selected')]
    lookup=lambda host,b:primary[(primary.host==host)&(primary.checkpoint==b)].iloc[0]
    low8,lowfinal=lookup('B01_low_turnover','8'),lookup('B01_low_turnover','final')
    value8,valuefinal=lookup('B03_valuation','8'),lookup('B03_valuation','final')
    passes8=int(primary[primary.checkpoint=='8'].accepted.sum())
    passesfinal=int(primary[primary.checkpoint=='final'].accepted.sum())
    passing_hosts=','.join(primary[(primary.checkpoint=='final')&primary.accepted].host)
    low_flat=all(lookup('B01_low_turnover',b).total==lowfinal.total for b in ['32','64'])
    example=df[(df.year==2023)&(df.host=='B03_valuation')&(df.initialization=='random')].sort_values('generation')
    example8,examplefinal=example[example.generation==8].iloc[0],example.iloc[-1]
    extreme_checks=json.loads((output/'high_return_engine_parity_final.json').read_text())
    assert len(extreme_checks)==3 and all(x['four_tables_original_cached_archive_exact'] for x in extreme_checks)
    low_choices=vd[(vd.host=='B01_low_turnover')&(vd.initialization=='baseline')].sort_values('budget').groupby('year').tail(1).set_index('year')
    low_choice_text='、'.join(f'{year}年选择第{int(low_choices.loc[year,"selected_generation"])}代' for year in [2021,2022,2023])
    lines=['# 完整策略学习延长训练结果','',f'运行：`{run.relative_to(PROJECT)}`。只延长原搜索轨迹；网络、目标、数据、种子、每代候选数和演化公式均未改变。','',
           f'累计训练评价 {manifest["total_training_evaluations"]:,} 次，其中新增 {manifest["new_training_evaluations"]:,} 次；精确恢复旧8代全部2,304个候选参数和192个中心。实际不同参数的验证评价 {manifest["validation_unique_evaluations"]:,} 次。','',
           f'24任务全部完成64代，其中 {manifest["tasks_extended"]} 个任务按仅使用训练记录的配对规则延长到256代。最后有 {stats["final_empirical_plateau"]} 个满足登记的平台判据，{stats["final_still_active"]} 个仍未满足。{stats["final_step_at_floor"]} 个搜索步长触及原下限，平台不能解释为高维全局收敛。','',
           '## 连续2021—2023账户（10万元，扣费后）','',
           '|基线初始化起点|原基线收益|8代收益 / 回撤|16代收益 / 回撤|32代收益 / 回撤|64代收益 / 回撤|最终预算收益 / 回撤|','|---|---:|---:|---:|---:|---:|---:|']
    for host,base in baseline.items():
        cells=[]
        for b in ['8','16','32','64','final']:
            row=summary[(summary.host==host)&(summary.initialization=='baseline')&(summary.selector=='validation_selected')&(summary.checkpoint==b)].iloc[0]
            cells.append(f'{row.total:.2%} / {abs(row.mdd):.2%}')
        lines.append('|'+host+'|'+f'{base["total"]:.2%}'+'|'+'|'.join(cells)+'|')
    lines+=['','“最终预算”按年度、宿主的训练平台判据分别采用64或256代，同配对随机组预算相同；全部年度选择先锁定，外层三年账户连续。','',
            f'最终通过正收益且回撤不超过30%历史验收的路径为 {passing_hosts}（{passesfinal}/4，8代为{passes8}/4）。低换手路径的三年累计收益从8代的{low8.total:+.2%}变为{lowfinal.total:+.2%}。'+('第32代预算已取得这一结果，继续到64/256代没有增加该账户收益。' if low_flat else '')+f'估值路径则从8代的{value8.total:+.2%}变为{valuefinal.total:+.2%}。不能从中挑选外层表现最好的预算当作未经选择的新结果。','',
            '## 最终各年选择的模型代数','',
            f'第0代表示回到原基线；这些代数来自验证选择，先于连续账户回放锁定。低换手路径{low_choice_text}，不能解释成三个年份都学到了持续增强。','',
            '|起点|2021|2022|2023|','|---|---:|---:|---:|']
    for host in baseline:
        cells=[]
        for year in [2021,2022,2023]:
            row=vd[(vd.host==host)&(vd.year==year)&(vd.initialization=='baseline')].sort_values('budget').iloc[-1]
            cells.append('现金' if pd.isna(row.selected_generation) else str(int(row.selected_generation)))
        lines.append('|'+host+'|'+'|'.join(cells)+'|')
    lines+=['','## 同预算随机初始化','', '|起点标识|8代收益 / 回撤|64代收益 / 回撤|最终收益 / 回撤|','|---|---:|---:|---:|']
    for host in baseline:
        cells=[]
        for b in ['8','64','final']:
            row=summary[(summary.host==host)&(summary.initialization=='random')&(summary.selector=='validation_selected')&(summary.checkpoint==b)].iloc[0]
            cells.append(f'{row.total:.2%} / {abs(row.mdd):.2%}')
        lines.append('|'+host+'|'+'|'.join(cells)+'|')
    lines+=['','## 不经验证选优的训练冠军对照','',
            '下面直接使用各预算的训练冠军。它回答训练目标改善是否随之带来跨期改善；与上表的差别是没有在历代冠军中再按验证收益选一次。两种选择方式都在观察外层结果前锁定。','',
            '|初始化起点|初始化方式|8代收益 / 回撤|64代收益 / 回撤|最终收益 / 回撤|',
            '|---|---|---:|---:|---:|']
    for host in baseline:
        for mode in ['baseline','random']:
            cells=[]
            for b in ['8','64','final']:
                row=summary[(summary.host==host)&(summary.initialization==mode)&(summary.selector=='training_champion')&(summary.checkpoint==b)].iloc[0]
                cells.append(f'{row.total:.2%} / {abs(row.mdd):.2%}')
            lines.append('|'+host+'|'+mode+'|'+'|'.join(cells)+'|')
    lines+=['','## 各训练任务的最终验证差异','',
            '与同一任务第8代相比，年化收益的变化以百分点计；每一列均在相同年份、相同口径内作差。验证列对应训练冠军自身，不是历代验证收益最大值。','',
            '|外层年份|起点|初始化|最终代数|训练年化变化|验证年化变化|',
            '|---|---|---|---:|---:|---:|']
    for row in gaps.sort_values('budget').groupby(['year','host','initialization']).tail(1).sort_values(['year','host','initialization']).itertuples():
        lines.append(f'|{row.year}|{row.host}|{row.initialization}|{row.budget}|{row.train_ann_change*100:+.2f}|{row.validation_ann_change*100:+.2f}|')
    lines+=['','## 每个训练窗口的最高收益冠军持仓集中度','',
            '允许持有1—5只股票不等于每只等权；现行执行约束没有另加单股权重上限。以下只作归因，不据此更改本轮搜索。权重以当日总资产为分母，现金日计零。','',
            '|外层年份|起点 / 初始化|训练累计收益|单股最大权重|每日最大单股权重中位数|单股超过80%仓位的天数占比|',
            '|---|---|---:|---:|---:|---:|']
    et=ed[ed.kind=='final_training_champion']
    for row in et.loc[et.groupby('year').total.idxmax()].itertuples():
        lines.append(f'|{int(row.year)}|{row.host} / {row.initialization}|{row.total:.2%}|{row.max_stock_weight:.2%}|{row.median_daily_max_weight:.2%}|{row.days_over_80pct:.2%}|')
    lines+=['','## 如何解释收敛与跨期表现','',
            f'相对第8代，最终训练期财富增益中位数为 {stats["median_training_wealth_gain_vs_g8"]:.2%}。训练目标和验证/外层表现必须分开看；更高训练回报不能直接当作未来收益提高。','',
            f'训练冠军在64代有 {stats["train_up_validation_down_at64"]}/24 个出现“训练年化收益高于8代、验证年年化收益却低于8代”；最终预算为 {stats["train_up_validation_down_at_final"]}/24 个。这是重复优化同段历史后过拟合风险的描述性检查，非独立统计检验。', '',
            f'延长阶段共有 {parameter_changes} 次冠军参数变化，其中 {equal_objective_changes} 次训练适应度完全持平。原演化排序允许持平的新候选替换旧冠军，因此账户收益平台不等于参数收敛；参数漂移仍可能改变验证期行为。', '',
            f'高收益反例：2023年估值随机起点的最终训练冠军，在2019—2021年训练累计{examplefinal.total:+.2%}，但2022年验证收益{examplefinal.validation_total:+.2%}、最大回撤{abs(examplefinal.validation_mdd):.2%}；其8代冠军在同一验证年为{example8.validation_total:+.2%}。原引擎、缓存引擎和归档四表零容差一致，仅排除了缓存实现差异，不能将这条历史高收益解释为泛化能力。','',
            '平台判据：最后三个连续8代区间的冠军均可行，且各区间log(期末财富)增益均<0.005；在第64代任一配对成员未达平台，则双方训练到256代。没有用验证或外层成绩决定训练停止。','',
            '验证规则保持“初始点＋每代历史冠军”不变，预算增加会扩大候选集合。已有可行候选时，优选验证收益不降是候选集嵌套的机械性质，不能据此证明泛化提高。报告另外记录各预算训练冠军自身的验证结果，并回放8代/64代/最终冠军外层账户，避免把扩大验证选优范围全归因于训练质量。重复冠军只缓存相同参数的精确验证结果。','',
            '训练、验证账户各自从10万元起步；外层账户连续运行，年度之间继承真实资金与持仓。模型读取账户状态，因此状态分布变化、市场环境变化也可能造成跨期落差。“训练提高、验证下降”与过拟合相符，但本轮不能将落差唯一归因于过拟合。','',
            '首轮8代外层账户逐表复现；缓存账本在24个旧冠军上的四表零容差一致。64代和最终路径进行原始特征截断及账户前缀检查；所有新增候选和最终账户均按旧独立审计函数核对现金、股数、费用、T+1及官方涨跌停。','',
            '限制保持：93股全期质量隔离、已观察历史、原基线参数曾使用验证年；每个宿主每年度仍只有一条固定随机流，不能做强统计归因。30%为历史验收门槛。','',
            '过程数据：`search/`保存逐候选参数、账本、审计、逐代中心及随机状态；`extension_decisions.json`保存64代续训决定；`all_locked_choices.json`保存年度验证选择；`diagnostics/`含训练曲线、验证检查点和平台判据。']
    if 'execution_amendment_sha256' in manifest:
        lines+=['','## 执行调度补记','',
                '为减少最后几个长任务的串行等待，中途在已保存的代际检查点恢复，将同代12个独立候选的回放改为8进程并行，再按原候选顺序更新。冻结的网络、引擎、演化公式、种子、预算和筛选规则没有修改；新增调度脚本、恢复检查点与指纹另存 `execution_amendment.json`。切换前已对24个旧候选完成六张表零容差复现，中心、冠军、步长和随机状态也完全一致。', '',
                '评价预算指固定搜索序列中的候选位置数；调度核验及中断时尚未提交整代的候选复算不作为新搜索候选。`intentional_interruption.json`记录中断处已完成但需复算的候选。']
    verification_path=output/'completion_verification.json'
    if verification_path.exists():
        checks=json.loads(verification_path.read_text());assert checks['status']=='passed'
        lines+=['','## 归档核验','',
                f'核心文件指纹 {checks["core_artifact_hashes_checked"]:,} 个全部一致；独立账户审计 {checks["archived_audits_checked"]:,} 份、成交记录 {checks["trade_records_checked"]:,} 条全部核对；96项前缀检查通过。最大现金核对误差 {checks["maximum_reconciliation_errors"]["cash_max_error"]:.3g} 元。中断处40个已完成候选的参数与指标记录复算后指纹一致。']
    lines+=['','## 训练与验证曲线','',
            '图题中的 Outer 年份指外层账户年；2021/2022/2023分别对应训练2019/2019—2020/2019—2021、验证2020/2021/2022。蓝色为基线初始化，橙色为随机初始化；各子图纵轴独立。','',
            f'![训练期财富相对第8代]({output/"training_convergence.png"})','',
            f'![训练冠军在验证年的收益]({output/"training_champion_validation.png"})']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(16,9),sharex=True)
    for iy,year in enumerate(sorted(df.year.unique())):
        for ih,host in enumerate(baseline):
            ax=axes[iy,ih]
            for mode,color in [('baseline','#1565c0'),('random','#ef6c00')]:
                q=df[(df.year==year)&(df.host==host)&(df.initialization==mode)]
                ax.plot(q.generation,np.exp(q.log_wealth-q.loc[q.generation==8,'log_wealth'].iloc[0]),label=mode,color=color)
            ax.axvline(8,color='gray',linestyle=':',linewidth=.8);ax.axvline(64,color='gray',linestyle=':',linewidth=.8)
            ax.set_title('Outer '+str(year)+' '+host,fontsize=9);ax.grid(alpha=.2)
            if ih==0: ax.set_ylabel('Training wealth / generation 8')
            if iy==2: ax.set_xlabel('Generation')
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(output/'training_convergence.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(16,9),sharex=True)
    for iy,year in enumerate(sorted(df.year.unique())):
        for ih,host in enumerate(baseline):
            ax=axes[iy,ih]
            for mode,color in [('baseline','#1565c0'),('random','#ef6c00')]:
                q=df[(df.year==year)&(df.host==host)&(df.initialization==mode)]
                ax.plot(q.generation,q.validation_total*100,label=mode,color=color)
            ax.axvline(8,color='gray',linestyle=':',linewidth=.8);ax.axvline(64,color='gray',linestyle=':',linewidth=.8)
            ax.axhline(0,color='gray',linewidth=.7)
            ax.set_title('Outer '+str(year)+' '+host,fontsize=9);ax.grid(alpha=.2)
            if ih==0: ax.set_ylabel('Validation return of training champion (%)')
            if iy==2: ax.set_xlabel('Generation')
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(output/'training_champion_validation.png',dpi=150);plt.close(fig)
    report=ROOT/'docs/POLICY_CONVERGENCE_RESULTS_20260920.md';report.write_text('\n'.join(lines)+'\n')
    (output/report.name).write_text(report.read_text())
    save_json(output/'report_sources.json',dict(script_sha256=digest(Path(__file__)),manifest_sha256=digest(run/'manifest.json'),report_sha256=digest(report)))
    print(json.dumps(stats,indent=2));print(summary[summary.selector=='validation_selected'].to_string(index=False))


if __name__=='__main__': main()
