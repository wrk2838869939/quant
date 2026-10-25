#!/usr/bin/env python3
"""固定候选的质量分层描述，保留市场与同日比较及日期聚集敏感性。"""
from pathlib import Path
from datetime import datetime,timezone
import json,os,sys,shutil
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
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
from sector_candidate_quality import build_features,assign_clusters,summarize_group


def save_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')


def leader_gaps(hub,dataset,source,signals,cfg):
    """沿用已保存前日选择的代理，从其余成员中排除代理后单独计收益。"""
    mapping=pd.read_parquet(dataset/'membership_daily.parquet')
    sectors=json.loads((dataset/'sector_codes.json').read_text())
    leaders=pd.read_parquet(source/'leader_snapshots.parquet')
    leaders=leaders.merge(signals[['event_id','date']],on=['event_id','date'],validate='many_to_one')
    selected_dates=pd.DatetimeIndex(sorted(signals.date.unique()))
    positions=mapping.index.get_indexer(selected_dates)
    needed=mapping.index[np.unique(np.r_[positions,positions-1])]
    quotes=pads.dataset(str(hub.cfg.cn_daily),format='parquet',partitioning='hive').to_table(
        columns=['date','ts_code','hfq_close'],filter=pads.field('date').isin(needed.to_pydatetime().tolist())).to_pandas()
    assert not quotes.duplicated(['date','ts_code']).any()
    close=quotes.pivot(index='date',columns='ts_code',values='hfq_close').reindex(index=needed,columns=mapping.columns)
    rows=[]
    for r in leaders.itertuples():
        i=mapping.index.get_loc(r.date)
        previous=mapping.index[i-1]
        sid=sectors.index(r.sector_code)
        members=mapping.loc[r.date].eq(sid)&mapping.loc[previous].eq(sid)
        ret=(close.loc[r.date]/close.loc[previous]-1).where(members & close.loc[r.date].gt(0)&close.loc[previous].gt(0))
        selected=r.selected.split('|') if r.selected else []
        selected_values=ret.reindex(selected).dropna()
        remaining=ret.drop(index=selected,errors='ignore').dropna()
        assert not set(selected_values.index)&set(remaining.index)
        enough=len(selected_values)>=cfg['minimum_leader_observations'] and len(remaining)>=cfg['minimum_remaining_observations']
        mean=float(selected_values.mean())
        np.testing.assert_allclose(mean,r.mean_return,equal_nan=True,atol=1e-12)
        rows.append(dict(event_id=r.event_id,date=r.date,proxy=r.proxy,selected=r.selected,
                         leader_members=len(selected_values),remaining_members=len(remaining),
                         leader_mean=mean,remaining_mean=float(remaining.mean()),
                         gap=mean-float(remaining.mean()) if enough else np.nan))
    return pd.DataFrame(rows)


def main():
    config=ROOT/'configs/sector_candidate_quality_v1.yaml'
    cfg=yaml.safe_load(config.read_text())
    dataset,source=PROJECT/cfg['dataset'],PROJECT/cfg['events']
    ds=SectorDataset(dataset)
    source_manifest=json.loads((source/'manifest.json').read_text())
    assert source_manifest['status']=='complete'
    assert source_manifest['dataset_content_id']==ds.manifest['content_id']
    for name,h in source_manifest['artifacts'].items():assert digest(source/name)==h,name
    if os.environ.get('DATAHUB_CONTENT_ID')!=ds.manifest['market_content_id']:raise RuntimeError('请用固定行情启动')
    hub=DataHub()
    assert hub.cfg.root.resolve()==(PROJECT/ds.manifest['config']['release']).resolve()
    out=PROJECT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'manifest.json',dict(status='building',config=cfg))
    print('OUTPUT',out,flush=True)
    panel=pd.read_parquet(dataset/'daily.parquet')
    obs=pd.read_parquet(source/'observations.parquet')
    signals=obs[obs.first_candidate].copy().sort_values(['date','sector_code']).reset_index(drop=True)
    features=build_features(panel,signals,cfg)
    cut=pd.Timestamp(cfg['prefix_check_date'])
    prefix=build_features(panel[panel.date<=cut],signals[signals.date<=cut],cfg)
    pd.testing.assert_frame_equal(prefix,features[features.date<=cut].reset_index(drop=True))
    assert len(features)==len(signals) and not features.event_id.duplicated().any()
    calendar=hub.calendar.cn_days(ds.manifest['config']['start'],ds.manifest['config']['end'])
    clusters=assign_clusters(features.date,calendar,cfg['cluster_gap_sessions'])
    features=features.merge(clusters,on='date',validate='many_to_one')
    gaps=leader_gaps(hub,dataset,source,signals,cfg)
    gaps.to_parquet(out/'leader_gaps.parquet',index=False)
    names=dict(cfg['feature_names'])
    labels={'price_zone':cfg['price_zones'],**cfg['labels']}
    for proxy,title in [('cap_top3','市值龙头相对强弱'),('momentum60_top3','历史强势代理相对强弱')]:
        key=proxy+'_gap'
        features=features.merge(gaps[gaps.proxy.eq(proxy)][['event_id','gap']].rename(columns={'gap':key}),on='event_id',how='left',validate='one_to_one')
        features[key+'_group']=pd.cut(features[key],bins=cfg['leader_gap_edges'],labels=cfg['leader_gap_labels'],right=False).astype('object').fillna('缺失或超出分组范围')
        names[key]=title;labels[key]=cfg['leader_gap_labels']
    features.to_parquet(out/'candidate_features.parquet',index=False)
    features.to_csv(out/'candidate_features.csv',index=False)
    # 结果在特征完成并保存之后才载入，避免分组依赖后续走势。
    outcomes=pd.read_parquet(source/'outcomes.parquet')
    outcomes=outcomes[outcomes.trigger.eq('first_candidate')].rename(columns={'signal_date':'date'})
    quotes=pd.read_parquet(hub.cfg.root/'datasets/index/cn_index_daily.parquet')
    benchmark=quotes[quotes.ts_code.eq(cfg['benchmark'])].set_index('date').close.reindex(calendar)
    analysis=features.merge(outcomes[['event_id','horizon','complete','close_return','max_close_return','min_close_return','end_date']],on='event_id',validate='one_to_many')
    starts=benchmark.reindex(analysis.date).to_numpy()
    ends=benchmark.reindex(analysis.end_date).to_numpy()
    analysis['market_return']=ends/starts-1
    analysis.loc[~analysis.complete,'market_return']=np.nan
    analysis['market_excess']=analysis.close_return-analysis.market_return
    peers=analysis.groupby(['date','horizon']).close_return.transform('count')
    centers=analysis.groupby(['date','horizon']).close_return.transform('mean')
    analysis['within_date_excess']=(analysis.close_return-centers).where(peers>=2)
    analysis['same_day_complete_peers']=peers
    analysis.to_parquet(out/'analysis_with_outcomes.parquet',index=False)
    zero=analysis.groupby(['date','horizon']).within_date_excess.mean().dropna()
    assert np.allclose(zero,0,atol=1e-12)
    summary_rows=[];baseline=[]
    metrics=['close_return','market_excess','within_date_excess']
    for h in cfg['horizons']:
        part=analysis[analysis.horizon.eq(h)]
        for metric in metrics:
            baseline.append(dict(horizon=h,metric=metric,**summarize_group(part,metric,cfg,cfg['bootstrap_seed'])))
        for feature,feature_labels in labels.items():
            for label in [*feature_labels,'缺失或超出分组范围']:
                group=part[part[feature+'_group'].eq(label)]
                for metric in metrics:
                    stats=summarize_group(group,metric,cfg,cfg['bootstrap_seed'])
                    summary_rows.append(dict(feature=feature,feature_name=names[feature],group=label,horizon=h,metric=metric,**stats))
    grouped=pd.DataFrame(summary_rows)
    grouped.to_csv(out/'group_summary.csv',index=False)
    pd.DataFrame(baseline).to_csv(out/'baseline_by_horizon.csv',index=False)
    dates=features.groupby(['date','cluster_id']).agg(events=('event_id','size')).reset_index()
    dates.to_csv(out/'date_clusters.csv',index=False)
    clusters_summary=dates.groupby('cluster_id').agg(first=('date','min'),last=('date','max'),dates=('date','size'),events=('events','sum'))
    clusters_summary.to_csv(out/'cluster_sizes.csv')
    within=[]
    for h in cfg['horizons']:
        part=analysis[analysis.horizon.eq(h)&analysis.complete]
        for feature in names:
            if feature=='price_zone':continue
            for date,day in part.groupby('date'):
                xy=day[[feature,'market_excess']].dropna()
                if len(xy)>=3 and xy[feature].nunique()>1:
                    within.append(dict(feature=feature,date=date,horizon=h,events=len(xy),spearman=float(xy[feature].corr(xy.market_excess,method='spearman'))))
    pd.DataFrame(within,columns=['feature','date','horizon','events','spearman']).to_csv(out/'within_date_rank_diagnostics.csv',index=False)
    # 最大事件簇整体剔除，观察描述结果是否被一次行情主导；不据此更改分组。
    largest=int(clusters_summary.events.idxmax())
    sensitivity=[]
    part=analysis[analysis.horizon.eq(cfg['primary_horizon'])]
    for feature,feature_labels in labels.items():
        for label in feature_labels:
            subset=part[part[feature+'_group'].eq(label)&part.cluster_id.ne(largest)]
            sensitivity.append(dict(feature=feature,group=label,excluded_cluster=largest,
                **summarize_group(subset,'market_excess',cfg,cfg['bootstrap_seed'])))
    pd.DataFrame(sensitivity).to_csv(out/'largest_cluster_excluded.csv',index=False)
    # 二维切片保留全部格子及样本数，不按表现选取阈值组合。
    interaction=[]
    for zone in cfg['price_zones']:
        for turn in cfg['labels']['turnover_contraction']:
            subset=part[part.price_zone_group.eq(zone)&part.turnover_contraction_group.eq(turn)]
            interaction.append(dict(price_zone=zone,turnover_group=turn,**summarize_group(subset,'market_excess',cfg,cfg['bootstrap_seed'])))
    pd.DataFrame(interaction).to_csv(out/'position_turnover_grid.csv',index=False)
    font_manager.fontManager.addfont(str(PROJECT/cfg['font']))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=PROJECT/cfg['font']).get_name(),'axes.unicode_minus':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,4,figsize=(23,11),layout='constrained')
    for ax,(feature,title) in zip(axes.ravel(),names.items()):
        show=grouped[(grouped.feature==feature)&grouped.horizon.eq(cfg['primary_horizon'])&grouped.metric.eq('market_excess')]
        show=show.set_index('group').reindex(labels[feature])
        for j,(label,r) in enumerate(show.iterrows()):
            if r.dates:
                color='#367da7' if r.dates>=cfg['minimum_display_dates'] else '#999999'
                ax.scatter(j,r['mean']*100,color=color,s=45)
                if np.isfinite(r.ci_low):ax.vlines(j,r.ci_low*100,r.ci_high*100,color=color,lw=2)
                ax.annotate(f'{int(r.events)}事件/{int(r.dates)}日/{int(r.clusters)}簇',(j,r['mean']*100),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7)
            else:ax.text(j,0,'无样本',ha='center',color='#999',fontsize=8)
        ax.set_xticks(range(len(show)),show.index,rotation=18,ha='right',fontsize=8)
        ax.axhline(0,color='#888',ls='--',lw=1)
        ax.set(title=title,ylabel='20日相对沪深300的日期等权均值（百分点）')
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('固定形态候选的质量分层：描述性结果\n竖线为行情簇重采样95%区间（非多重检验校正）；少于5簇不画区间；灰点少于5个触发日期',fontsize=15)
    fig.savefig(out/'quality_groups.png',dpi=cfg['dpi'],bbox_inches='tight');fig.savefig(out/'quality_groups.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    axes[0].bar(clusters_summary.index.astype(str),clusters_summary.events,color='#367da7')
    axes[0].set(xlabel='相邻信号间隔≤20交易日连接成的行情簇',ylabel='候选事件数',title='样本在行情簇上的集中程度')
    hist=analysis[analysis.horizon.eq(cfg['primary_horizon'])].groupby('date')[['close_return','market_excess']].mean()
    axes[1].scatter(hist.close_return*100,hist.market_excess*100,color='#bd8140')
    axes[1].axhline(0,color='#999',ls='--');axes[1].axvline(0,color='#999',ls='--')
    axes[1].set(xlabel='同日候选平均20日涨跌（%）',ylabel='减同期沪深300（百分点）',title='每点一个触发日期')
    fig.savefig(out/'sample_diagnostics.png',dpi=cfg['dpi'],bbox_inches='tight');plt.close(fig)
    pair_date_counts=analysis.groupby('horizon').apply(lambda x:pd.Series(dict(multi_dates=x.loc[x.same_day_complete_peers.ge(2),'date'].nunique(),
                 triple_dates=x.loc[x.same_day_complete_peers.ge(3),'date'].nunique())),include_groups=False)
    pair_date_counts.to_csv(out/'same_day_ranking_support.csv')
    headline=dict(candidates=len(features),trigger_dates=len(dates),clusters=len(clusters_summary),largest_cluster_events=int(clusters_summary.events.max()),
                  largest_cluster_first=str(clusters_summary.loc[largest,'first'].date()),largest_cluster_last=str(clusters_summary.loc[largest,'last'].date()),
                  ranking_support=pair_date_counts.reset_index().to_dict('records'),baseline=baseline)
    save_json(out/'summary.json',headline)
    lines=['# 板块候选质量分层 v1','',f'源事件库：`{source}`。固定69个首次完整候选，不改变形态定义，不搜索阈值，不训练排序模型。','',
        '## 口径','',
        '- 主观察窗口20个交易日，5/10日作为描述性补充。板块信号日收盘到未来同段收盘，无交易执行和成本。',
        '- 绝对结果、减沪深300同期涨跌的相对结果、减同日候选均值的组内结果分别保存。同日仅1个完整候选时，组内结果为空。',
        '- 先在同一触发日内对候选平均，再对日期等权。间隔不超过20个交易日的信号日传递连接成行情簇；推断时整簇重采样2000次，少于5簇不提供区间。簇只供事后推断，不进入特征。',
        '- 分组边界在计算之前写入YAML，保留全部档位及空格，不假定单调。价格区间按本轮平台下沿/上沿/上涨初期参照上沿划分；跌速为峰后日均对数跌速。',
        '- 广度为最近2日上涨占比均值减此前3日均值；换手为最近2日均值除以前3日均值。所用日的已知成员观测覆盖至少80%，否则该特征为空。减速比为最近2日平均负收益幅度除以前3日。',
        '- 龙头代理沿用首次候选日之前已选好的市值/过去60日涨幅前三，与剔除三股后的同板块其余成员比较当日收益，至少2只代理和2只其余成员有效；不重叠、不回填缺失。',
        '- 分组之间往往发生在不同市场阶段。相对沪深300仍可能包含风格差异，不能单凭分组均值排序；区间不是多重检验校正，也不是策略有效性证据。','',
        '## 样本结构','',f'- {len(features)}个候选、{len(dates)}个触发日，按预设规则连成{len(clusters_summary)}个行情簇。',
        f'- 最大簇：{headline["largest_cluster_first"]}至{headline["largest_cluster_last"]}，{headline["largest_cluster_events"]}个候选。',
        '- 同日比较支持情况：'+str(headline['ranking_support'])+'。','',
        '## 各窗口整体观察','','| 后续日 | 口径 | 事件/日期/簇 | 日期等权均值 | 95%簇重采样区间 |','|---|---|---|---:|---|']
    for r in baseline:
        if r['metric']=='within_date_excess':continue
        ci=f'{r["ci_low"]:.2%} 至 {r["ci_high"]:.2%}' if np.isfinite(r['ci_low']) else '样本不足'
        lines.append(f'| {r["horizon"]} | {r["metric"]} | {r["events"]}/{r["dates"]}/{r["clusters"]} | {r["mean"]:.2%} | {ci} |')
    lines+=['','## 20日分组比较','','下表为减沪深300的日期等权结果；各组不是相同日期样本，不能当作同日优先级。','',
        '| 特征 | 分组 | 事件/日期/簇 | 相对均值 | 剔除最大行情簇后 |','|---|---|---|---:|---:|']
    sens=pd.DataFrame(sensitivity).set_index(['feature','group'])
    for r in grouped[grouped.horizon.eq(cfg['primary_horizon'])&grouped.metric.eq('market_excess')&grouped.events.gt(0)].itertuples():
        v=sens.loc[(r.feature,r.group),'mean'] if (r.feature,r.group) in sens.index else np.nan
        lines.append(f'| {r.feature_name} | {r.group} | {r.events}/{r.dates}/{r.clusters} | {r.mean:.2%} | {v:.2%} |')
    lines+=['','## 验证与产物','',
        '- 源板块数据及事件库哈希完整核验；新增特征在物理截断数据下逐值一致。',
        '- 三项单元测试：同日重复不改变日期权重、交易日连接与传递聚簇、历史前缀和未知广度处理。',
        '- 重算代理股收益与原快照一致，并确认与其余成员不重叠；组内超额在每个日期均值为零。',
        '- candidate_features.parquet/csv 与 analysis_with_outcomes.parquet 分离；后者含未来结果，不得作为当时特征。',
        '- group_summary.csv：所有分组、窗口、三种口径与样本数量/区间。',
        '- within_date_rank_diagnostics.csv、same_day_ranking_support.csv：可用于同日排序的日期和逐日期相关诊断。',
        '- largest_cluster_excluded.csv、position_turnover_grid.csv：剔除大簇敏感性及位置×换手二维切片，未挑选最优组合。','']
    (out/'README.md').write_text('\n'.join(lines))
    save_json(out/'validation.json',dict(source_hashes=True,feature_prefix_exact=True,leader_snapshot_reproduction=True,leader_rest_disjoint=True,within_date_zero_mean=True))
    for p in [config,Path(__file__),ROOT/'src/sector_candidate_quality.py',ROOT/'tests/test_sector_candidate_quality.py']:shutil.copy2(p,out/p.name)
    save_json(out/'manifest.json',dict(status='complete',dataset_content_id=ds.manifest['content_id'],source_manifest_sha256=digest(source/'manifest.json'),config=cfg,
        artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(dict(output=str(out),**headline),ensure_ascii=False,default=str),flush=True)


if __name__=='__main__':main()
