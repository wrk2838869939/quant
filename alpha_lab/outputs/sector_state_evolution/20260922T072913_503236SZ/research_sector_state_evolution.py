#!/usr/bin/env python3
"""165个首次上涨回落事件：固定观察日的广度/跌速演变及后续走势。"""
from pathlib import Path
from datetime import datetime,timezone
import json,os,sys,shutil
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from datahub import DataHub
from sector_dataset import SectorDataset,digest
from sector_candidate_quality import assign_clusters,summarize_group
from sector_state_evolution import build_paths,forward_labels


def save_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')


def main():
    config=ROOT/'configs/sector_state_evolution_v1.yaml'
    cfg=yaml.safe_load(config.read_text())
    source,dataset=PROJECT/cfg['events'],PROJECT/cfg['dataset']
    ds=SectorDataset(dataset)
    source_manifest=json.loads((source/'manifest.json').read_text())
    assert source_manifest['dataset_content_id']==ds.manifest['content_id']
    for name,h in source_manifest['artifacts'].items():assert digest(source/name)==h,name
    if os.environ.get('DATAHUB_CONTENT_ID')!=ds.manifest['market_content_id']:raise RuntimeError('请用固定行情版本启动')
    hub=DataHub()
    assert hub.cfg.root.resolve()==(PROJECT/ds.manifest['config']['release']).resolve()
    out=PROJECT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fSZ')
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'manifest.json',dict(status='building',config=cfg))
    print('OUTPUT',out,flush=True)
    panel=pd.read_parquet(dataset/'daily.parquet')
    observations=pd.read_parquet(source/'observations.parquet')
    origins=observations[observations.first_pullback].sort_values(['date','sector_code']).reset_index(drop=True)
    assert origins.event_id.nunique()==len(origins)==165
    calendar=hub.calendar.cn_days(ds.manifest['config']['start'],ds.manifest['config']['end'])
    paths=build_paths(panel,origins,observations,calendar,cfg)
    cut=pd.Timestamp(cfg['prefix_check_date'])
    prefix=build_paths(panel[panel.date<=cut],origins[origins.date<=cut],observations[observations.date<=cut],calendar[calendar<=cut],cfg)
    pd.testing.assert_frame_equal(prefix,paths[paths.date<=cut].reset_index(drop=True))
    assert not paths.duplicated(['event_id','offset']).any()
    # 状态先落盘；未来标签只在以下步骤独立计算。
    origins.to_parquet(out/'origins.parquet',index=False)
    paths.to_parquet(out/'state_paths.parquet',index=False)
    quotes=pd.read_parquet(hub.cfg.root/'datasets/index/cn_index_daily.parquet')
    benchmark=quotes[quotes.ts_code.eq(cfg['benchmark'])].set_index('date').close.reindex(calendar)
    labels=forward_labels(panel,paths,calendar,benchmark,cfg)
    labels.to_parquet(out/'future_outcomes.parquet',index=False)
    old=pd.read_parquet(source/'outcomes.parquet')
    old=old[old.trigger.eq('first_pullback')]
    check=labels[labels.offset.eq(0)].merge(old,on=['event_id','horizon'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(check)==len(old) and (check.complete_new==check.complete_old).all()
    np.testing.assert_allclose(check.close_return_new,check.close_return_old,equal_nan=True,atol=1e-12)
    clusters=assign_clusters(origins.date,calendar,cfg['cluster_gap_sessions']).rename(columns={'date':'origin_date'})
    merged=paths.merge(labels,on=['event_id','offset','date'],validate='one_to_many').merge(clusters,on='origin_date',validate='many_to_one')
    counts=merged.groupby(['offset','horizon','date']).close_return.transform('count')
    avg=merged.groupby(['offset','horizon','date']).close_return.transform('mean')
    merged['within_date_excess']=(merged.close_return-avg).where(counts>=2)
    merged['same_day_peers']=counts
    merged.to_parquet(out/'landmarks_with_outcomes.parquet',index=False)
    origin_clusters=origins.rename(columns={'date':'origin_date'}).merge(clusters,on='origin_date',validate='many_to_one')
    cluster_sizes=origin_clusters.groupby('cluster_id').agg(first=('origin_date','min'),last=('origin_date','max'),events=('event_id','size'),dates=('origin_date','nunique'))
    cluster_sizes.to_csv(out/'origin_cluster_sizes.csv')
    largest=int(cluster_sizes.events.idxmax())
    baselines=[];groups=[];sensitivity=[];contrasts=[];support=[]
    for landmark in cfg['landmarks']:
        p=paths[paths.offset.eq(landmark)]
        support.append(dict(landmark=landmark,origin_events=len(origins),dates_reached=len(p),
                            available=int(p.available.sum()),missing_segment=int((~p.available).sum()),
                            end_of_dataset=len(origins)-len(p),original_episode_active=int(p.episode_active.sum()),
                            original_pullback_still_valid=int(p.still_pullback.sum()),complete_candidate=int(p.complete_candidate.sum())))
        for horizon in cfg['horizons']:
            part=merged[merged.offset.eq(landmark)&merged.horizon.eq(horizon)]
            for metric in ['close_return','market_excess']:
                baselines.append(dict(landmark=landmark,horizon=horizon,metric=metric,**summarize_group(part,metric,cfg,cfg['bootstrap_seed'])))
            for state in cfg['states']:
                sub=part[part.state.eq(state)]
                for metric in ['close_return','market_excess','within_date_excess']:
                    groups.append(dict(landmark=landmark,horizon=horizon,state=state,metric=metric,**summarize_group(sub,metric,cfg,cfg['bootstrap_seed'])))
                if horizon==cfg['primary_horizon']:
                    sensitivity.append(dict(landmark=landmark,state=state,removed_cluster=largest,
                        **summarize_group(sub[sub.cluster_id.ne(largest)],'market_excess',cfg,cfg['bootstrap_seed'])))
            # 只在同一观察日同时有两组的日期比较，避免拿不同市场阶段直接排序。
            pair=part[part.state.isin([cfg['states'][0],cfg['states'][3]])].groupby(['date','cluster_id','state']).market_excess.mean().unstack('state')
            for col in [cfg['states'][0],cfg['states'][3]]:
                if col not in pair:pair[col]=np.nan
            paired=pair.dropna(subset=[cfg['states'][0],cfg['states'][3]]).reset_index()
            paired['contrast']=paired[cfg['states'][3]]-paired[cfg['states'][0]]
            contrasts.append(dict(landmark=landmark,horizon=horizon,**summarize_group(paired,'contrast',cfg,cfg['bootstrap_seed'])))
            paired.to_csv(out/f'matched_dates_day{landmark}_h{horizon}.csv',index=False)
    baseline=pd.DataFrame(baselines);grouped=pd.DataFrame(groups);contrasted=pd.DataFrame(contrasts)
    baseline.to_csv(out/'baseline.csv',index=False);grouped.to_csv(out/'state_groups.csv',index=False)
    contrasted.to_csv(out/'same_date_state_contrasts.csv',index=False)
    pd.DataFrame(support).to_csv(out/'cohort_followup.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'largest_cluster_removed.csv',index=False)
    initial=paths[paths.offset.eq(0)][['event_id','state','available']].rename(columns={'state':'initial_state','available':'initial_available'})
    trans_rows=[]
    transitions=merged.merge(initial,on='event_id',validate='many_to_one')
    before=transitions.initial_state.eq(cfg['states'][3]);after=transitions.state.eq(cfg['states'][3])
    known=transitions.initial_available&transitions.available&transitions.initial_state.ne('状态数据不足')&transitions.state.ne('状态数据不足')
    transitions['transition']=np.select([~known,before&after,~before&after,before&~after],
        ['状态不足','两端均双改善','转入双改善','离开双改善'],default='两端均非双改善')
    transitions.to_parquet(out/'transitions_with_outcomes.parquet',index=False)
    for step in [x for x in cfg['landmarks'] if x>0]:
        snap=paths[paths.offset.eq(step)].merge(initial,on='event_id',validate='one_to_one')
        matrix=pd.crosstab(snap.initial_state,snap.state).reindex(index=cfg['states'],columns=cfg['states'],fill_value=0)
        matrix.to_csv(out/f'transition_counts_0_to_{step}.csv')
        for transition,part in transitions[transitions.offset.eq(step)&transitions.horizon.eq(cfg['primary_horizon'])].groupby('transition'):
            trans_rows.append(dict(landmark=step,transition=transition,**summarize_group(part,'market_excess',cfg,cfg['bootstrap_seed'])))
    pd.DataFrame(trans_rows).to_csv(out/'transition_summary.csv',index=False)
    state_counts=paths[paths.available].groupby(['offset','state']).size().unstack(fill_value=0).reindex(columns=cfg['states'],fill_value=0)
    state_counts.to_csv(out/'state_counts_by_offset.csv')
    cohort_stats=merged[merged.horizon.eq(cfg['primary_horizon'])].groupby('offset').apply(lambda x:pd.Series(dict(
        complete_events=int(x.complete.sum()),complete_dates=x.loc[x.complete,'date'].nunique(),complete_clusters=x.loc[x.complete,'cluster_id'].nunique(),
        multi_dates=x.loc[x.same_day_peers.ge(2),'date'].nunique(),triple_dates=x.loc[x.same_day_peers.ge(3),'date'].nunique())),include_groups=False)
    cohort_stats.to_csv(out/'ranking_support.csv')
    font_manager.fontManager.addfont(str(PROJECT/cfg['font']))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=PROJECT/cfg['font']).get_name(),'axes.unicode_minus':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(21,6),layout='constrained')
    for ax,landmark in zip(axes,cfg['landmarks']):
        sub=grouped[grouped.landmark.eq(landmark)&grouped.horizon.eq(cfg['primary_horizon'])&grouped.metric.eq('market_excess')].set_index('state').reindex(cfg['states'])
        for i,(state,r) in enumerate(sub.iterrows()):
            if r.dates:
                ax.scatter(i,r['mean']*100,s=42,color='#3e7ba3')
                if np.isfinite(r.ci_low):ax.vlines(i,r.ci_low*100,r.ci_high*100,color='#3e7ba3',lw=2)
                ax.annotate(f'{int(r.events)}事件/{int(r.dates)}日/{int(r.clusters)}簇',(i,r['mean']*100),xytext=(0,8),textcoords='offset points',ha='center',fontsize=7)
            else:ax.text(i,0,'无样本',ha='center',fontsize=8,color='#888')
        ax.axhline(0,color='#888',ls='--');ax.grid(axis='y',alpha=.15)
        ax.set_xticks(range(len(sub)),sub.index,rotation=25,ha='right',fontsize=8)
        ax.set(title=f'第{landmark}日观察，从该日再向后20日',ylabel='日期等权相对沪深300均值（百分点）')
    fig.suptitle('首次先涨后落事件的状态演变：固定观察日比较\n竖线为行情簇重采样95%区间；不足5簇不画区间；各观察日对应不同未来区间',fontsize=15)
    fig.savefig(out/'landmark_states.png',dpi=cfg['dpi'],bbox_inches='tight');fig.savefig(out/'landmark_states.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    axes[0].stackplot(state_counts.index,*[state_counts[c] for c in state_counts],labels=cfg['states'],alpha=.8)
    axes[0].set(xlabel='距首次先涨后落的交易日',ylabel='有同段行情的事件数',title='整个起点队列的状态构成')
    axes[0].legend(fontsize=8,loc='lower left')
    mat=pd.read_csv(out/'transition_counts_0_to_5.csv',index_col=0)
    im=axes[1].imshow(mat,cmap='Blues')
    for i in range(len(mat)):
        for j in range(len(mat.columns)):axes[1].text(j,i,str(mat.iloc[i,j]),ha='center',va='center',fontsize=10,color='white' if mat.iloc[i,j]>mat.to_numpy().max()*.6 else '#222')
    axes[1].set_xticks(range(len(mat.columns)),mat.columns,rotation=25,ha='right',fontsize=8)
    axes[1].set_yticks(range(len(mat)),mat.index,fontsize=8)
    axes[1].set(xlabel='第5日状态',ylabel='第0日状态',title='状态迁移计数（事件数，不是独立行情数）')
    fig.savefig(out/'state_transitions.png',dpi=cfg['dpi'],bbox_inches='tight');plt.close(fig)
    headline=dict(origins=len(origins),origin_dates=origins.date.nunique(),origin_clusters=len(cluster_sizes),
        largest_cluster_events=int(cluster_sizes.events.max()),largest_cluster_start=str(cluster_sizes.loc[largest,'first'].date()),largest_cluster_end=str(cluster_sizes.loc[largest,'last'].date()),
        cohort=support,ranking_support=cohort_stats.reset_index().to_dict('records'),paired_contrasts=contrasts)
    save_json(out/'summary.json',headline)
    lines=['# 165个首次上涨回落事件：广度与减速的状态演变','',f'固定源：`{source}`；事件定义与此前一致，只扩大到已有的165个首次结构观察点。','',
        '## 观察与结果的时间顺序','',
        '- 第0日为首次满足历史上行、先上涨至少20%再回落至少15%的结构观察日。第5/10日按全市场交易日定位，不是等到条件满足才计时。',
        '- 广度改善：最近2日上涨占比均值大于此前3日；跌速放缓：最近2日平均负收益幅度≤此前3日的65%。前3日无下跌时减速比无定义，单列为“此前无下跌”；缺失不填零。',
        '- 即使原事件已结束或不再符合回落条件，只要同一连续行情段仍有数据，仍保留观察。跨段/缺口/数据末尾作为不可观察或截尾，不跳到下一个有数据的日子。',
        '- 每个观察日的结果从该日收盘向后计算5/10/20交易日。因此第0日结果和第5日结果并非相同收益区间，不能把它们的差直接解释为等待确认的收益。',
        '- “转入双改善”等迁移标签仅用第0日和观察日状态，从观察日之后计结果。“两端均双改善”不表示中间每天都保持。',
        '- 先同日平均、再日期等权；按起点日间隔≤30交易日传递连接行情簇，整簇重采样2000次，少于5簇不报区间。最大20日结果加最长10日观察使簇间覆盖区间不重叠；簇仍不保证统计独立。',
        '- 相对结果减同期沪深300收益，仍可能含其他风格因素；没有成交模拟、费用、多重检验校正或新样本外验证。','',
        '## 样本','',f'- 165事件、{origins.date.nunique()}个起点日期，连接为{len(cluster_sizes)}个行情簇；最大簇{cluster_sizes.events.max()}事件。',
        f'- 最大簇起点范围：{headline["largest_cluster_start"]}至{headline["largest_cluster_end"]}。','',
        '| 观察日 | 仍有同段行情 | 原事件仍活跃 | 仍满足结构回落 | 当日完整候选 |','|---|---:|---:|---:|---:|']
    for r in support:lines.append(f'| {r["landmark"]} | {r["available"]}/165 | {r["original_episode_active"]} | {r["original_pullback_still_valid"]} | {r["complete_candidate"]} |')
    lines+=['','## 各观察日后20日，相对沪深300','', '| 观察日 | 状态 | 事件/日期/簇 | 日期等权均值 | 95%簇重采样区间 | 剔除最大簇后 |','|---|---|---|---:|---|---:|']
    sens=pd.DataFrame(sensitivity).set_index(['landmark','state'])
    for r in grouped[grouped.horizon.eq(cfg['primary_horizon'])&grouped.metric.eq('market_excess')].itertuples():
        ci=f'{r.ci_low:.2%} 至 {r.ci_high:.2%}' if np.isfinite(r.ci_low) else '不足5簇'
        leave=sens.loc[(r.landmark,r.state),'mean']
        lines.append(f'| {r.landmark} | {r.state} | {r.events}/{r.dates}/{r.clusters} | {r.mean:.2%} | {ci} | {leave:.2%} |')
    lines+=['','## 同一观察日内：双改善减两者均未改善','',
        '仅使用同一天两组均有完整结果的日期；不足日期的组不强行比较。差值可能仍受成员构成影响。','',
        '| 观察日 | 后续日 | 匹配日期/行情簇 | 平均差值 | 95%区间 |','|---|---:|---|---:|---|']
    for r in contrasted.itertuples():
        ci=f'{r.ci_low:.2%} 至 {r.ci_high:.2%}' if np.isfinite(r.ci_low) else '不足5簇'
        lines.append(f'| {r.landmark} | {r.horizon} | {r.dates}/{r.clusters} | {r.mean:.2%} | {ci} |')
    lines+=['','## 从第0日迁移到观察日，再看后20日','', '| 观察日 | 端点迁移 | 事件/日期/簇 | 相对均值 |','|---|---|---|---:|']
    for r in trans_rows:lines.append(f'| {r["landmark"]} | {r["transition"]} | {r["events"]}/{r["dates"]}/{r["clusters"]} | {r["mean"]:.2%} |')
    lines+=['','## 核验与文件','',
        '- 四项测试覆盖形态结束后仍跟踪、缺口不跳日/不跨段、真实可知前缀和无前期下跌状态、未来结果从观察日而非起点开始。',
        '- 真实数据截至2023-06-30截断重算与全量路径逐值一致。第0日495条结果与原事件库逐值复现；唯一键和源文件哈希检查通过。',
        '- origins.parquet、state_paths.parquet仅保留当时状态；future_outcomes.parquet、landmarks_with_outcomes.parquet和transitions_with_outcomes.parquet包含未来结果，不能直接作为预测特征。',
        '- state_groups.csv保留全部状态、窗口、绝对/市场相对/同日相对三种口径；largest_cluster_removed.csv检查大行情依赖。',
        '- same_date_state_contrasts.csv及matched_dates_*.csv保留同日比较；transition_summary.csv保留端点迁移；cohort_followup.csv保留失效和截尾数量。',
        '- 图表：landmark_states.png/pdf、state_transitions.png。','']
    (out/'README.md').write_text('\n'.join(lines))
    save_json(out/'validation.json',dict(real_prefix_exact=True,origin_outcome_rows_reproduced=len(check),unique_path_keys=True,source_hashes=True))
    for p in [config,Path(__file__),ROOT/'src/sector_state_evolution.py',ROOT/'tests/test_sector_state_evolution.py',ROOT/'src/sector_candidate_quality.py']:shutil.copy2(p,out/p.name)
    save_json(out/'manifest.json',dict(status='complete',dataset_content_id=ds.manifest['content_id'],source_manifest_sha256=digest(source/'manifest.json'),config=cfg,
        artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(dict(output=str(out),**headline),ensure_ascii=False,default=str),flush=True)


if __name__=='__main__':main()
