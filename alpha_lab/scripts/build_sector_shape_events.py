#!/usr/bin/env python3
"""冻结板块的形态事件库、独立后续走势和人工复核图册。"""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from datahub import DataHub
from sector_dataset import SectorDataset,digest
from sector_shape_events import identify_events,evaluate_outcomes


def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')


def leader_snapshots(hub,dataset,signals,cfg):
    """仅对首次结构/候选观察日附龙头代理，选择严格按前一日成员和数据。"""
    if signals.empty:
        return pd.DataFrame()
    mapping=pd.read_parquet(dataset/'membership_daily.parquet')
    sector_codes=json.loads((dataset/'sector_codes.json').read_text())
    raw=hub.cn.daily(start=str(mapping.index.min().date()),end=str(signals.date.max().date()),fields=['hfq_close','total_mv_yuan'])
    dates=mapping.index[mapping.index<=signals.date.max()]
    close=raw.pivot(index='date',columns='ts_code',values='hfq_close').reindex(index=dates,columns=mapping.columns)
    cap=raw.pivot(index='date',columns='ts_code',values='total_mv_yuan').reindex_like(close)
    del raw
    close=close.where(close>0)
    # 不以主档当前状态筛选；归属矩阵已排除已知非存续日期。
    close=close.where(mapping.reindex(dates).ge(0))
    ret=close.pct_change(fill_method=None)
    mom=(close/close.shift(cfg['leader_momentum_window'])-1).shift(1)
    score_cap=cap.where(cap>0).shift(1)
    records=[]
    for r in signals.itertuples():
        i=dates.get_loc(r.date)
        if not i:continue
        sid=sector_codes.index(r.sector_code)
        old=mapping.iloc[i-1].eq(sid)
        current=mapping.iloc[i].eq(sid)
        for kind,scores in [('cap_top3',score_cap),('momentum60_top3',mom)]:
            line=scores.loc[r.date].where(old).dropna().sort_values(ascending=False,kind='stable')
            selected=line.index[:cfg['leader_count']]
            values=ret.loc[r.date,selected].where(current[selected])
            records.append(dict(event_id=r.event_id,date=r.date,sector_code=r.sector_code,proxy=kind,
                                selected='|'.join(selected),selected_count=len(selected),observed_count=int(values.notna().sum()),
                                mean_return=float(values.mean()),positive_fraction=float(values.dropna().gt(0).mean()),
                                previous_scores='|'.join(f'{v:.6g}' for v in line.iloc[:cfg['leader_count']])))
    return pd.DataFrame(records)


def main():
    config=ROOT/'configs/sector_shape_events_v1.yaml'
    cfg=yaml.safe_load(config.read_text())
    dataset=PROJECT/cfg['dataset']
    ds=SectorDataset(dataset)
    if os.environ.get('DATAHUB_CONTENT_ID')!=ds.manifest['market_content_id']:
        raise RuntimeError('必须使用固定行情启动')
    hub=DataHub()
    out=PROJECT/cfg['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'manifest.json',dict(status='building',config=cfg))
    print('OUTPUT',out,flush=True)
    panel=pd.read_parquet(dataset/'daily.parquet')
    catalog=pd.read_parquet(dataset/'catalog_sw_l2.parquet')
    names={(r.src,r.index_code):r.industry_name for r in catalog.itertuples()}
    observations,episodes=identify_events(panel,cfg)
    print('IDENTIFIED',len(episodes),len(observations),flush=True)
    outcomes=evaluate_outcomes(panel,observations,cfg['outcome_horizons'])
    observations.to_parquet(out/'observations.parquet',index=False)
    episodes.to_parquet(out/'episodes.parquet',index=False)
    outcomes.to_parquet(out/'outcomes.parquet',index=False)
    signals=observations[observations.first_pullback|observations.first_candidate].copy()
    signals['name']=[names.get((r.classification_version,r.sector_code),r.sector_code) for r in signals.itertuples()]
    signals.to_csv(out/'first_observations.csv',index=False)
    assert not observations.duplicated(['event_id','date']).any()
    candidates=observations[observations.candidate]
    assert ((candidates.platform_end<candidates.breakout_date)&(candidates.breakout_date<candidates.peak_date)&(candidates.peak_date<candidates.date)).all()
    assert candidates.trend_at_peak.all()
    assert (candidates.rally_return>=cfg['minimum_rally']).all()
    assert (candidates.drawdown<=-cfg['minimum_pullback']).all()
    for _,part in observations[observations.reference_ready].groupby('event_id'):
        assert part.reference_high.nunique()==1
    cut=pd.Timestamp(cfg['prefix_check_end'])
    short,_=identify_events(panel[panel.date<=cut],cfg)
    expected=observations[observations.date<=cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(short,expected)
    # 人为改变未来路径，确保已生成观察行不受影响。
    shocked=panel.copy()
    shocked.loc[shocked.date>cut,'index_close']*=3
    shock_obs,_=identify_events(shocked,cfg)
    pd.testing.assert_frame_equal(shock_obs[shock_obs.date<=cut].reset_index(drop=True),expected)
    stages=[]
    mask=np.ones(len(observations),bool)
    for label,condition in [('平台突破',np.ones(len(observations),bool)),('上涨达到门槛',observations.rally_confirmed),
        ('峰值时历史上行',observations.trend_at_peak),('上涨后回落',observations.pullback_gate),
        ('回落过程中曾快速下跌',observations.fast_fall_seen),('跌破峰值前支撑',observations.support_broken),
        ('接近同轮平台/上涨初期参照',observations.near_reference),('最近五日跌速放缓',observations.slowed)]:
        mask &= np.asarray(condition)
        part=observations.loc[mask]
        stages.append(dict(stage=label,observation_days=len(part),events=part.event_id.nunique(),sectors=part.sector_code.nunique()))
    pd.DataFrame(stages).to_csv(out/'funnel.csv',index=False)
    outcomes_summary=[]
    for (trigger,horizon),part in outcomes.groupby(['trigger','horizon']):
        complete=part[part.complete]
        outcomes_summary.append(dict(trigger=trigger,horizon=int(horizon),observations=len(part),complete=len(complete),
             positive=int(complete.close_return.gt(0).sum()),negative=int(complete.close_return.lt(0).sum()),
             flat=int(complete.close_return.eq(0).sum()),censored=int((~part.complete).sum()),
             median_close_return=float(complete.close_return.median())))
    pd.DataFrame(outcomes_summary).to_csv(out/'outcome_counts.csv',index=False)
    print('LEADER_SNAPSHOTS',len(signals),flush=True)
    leaders=leader_snapshots(hub,dataset,signals,cfg)
    leaders.to_parquet(out/'leader_snapshots.parquet',index=False)
    font_manager.fontManager.addfont(str(PROJECT/cfg['font']))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=PROJECT/cfg['font']).get_name(),'axes.unicode_minus':False,'pdf.fonttype':42})
    # 图册按信号信息选样：候选优先、首次日期排序，每板块先一例；不使用未来结果选样。
    pool=signals.sort_values(['first_candidate','date','sector_code'],ascending=[False,True,True]).drop_duplicates('event_id')
    chosen=pd.concat([pool.drop_duplicates('sector_code'),pool]).drop_duplicates('event_id').head(cfg['review_max_cases'])
    cases=out/'cases';cases.mkdir()
    pages=[]
    with PdfPages(out/'event_review.pdf') as book:
        for num,r in enumerate(chosen.itertuples(),1):
            full=panel[(panel.sector_code==r.sector_code)&(panel.segment_id==r.segment_id)&panel.index_valid].sort_values('date').reset_index(drop=True)
            where={d:i for i,d in enumerate(full.date)}
            anchor_i=where[r.platform_start];signal_i=where[r.date]
            part=full.iloc[max(0,anchor_i-cfg['review_context_days']):signal_i+cfg['review_followup_days']+1]
            fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[3,1,1]})
            axes[0].plot(part.date,part.index_close,label='限权市值指数',color='#286596')
            base=full[full.date.eq(r.platform_end)].iloc[0]
            axes[0].plot(part.date,part.equal_index_close/base.equal_index_close*base.index_close,label='等权指数（平台末日对齐）',color='#be812f',alpha=.8)
            axes[0].axvspan(r.platform_start,r.platform_end,color='#78a985',alpha=.18,label='本轮平台')
            axes[0].axhspan(r.platform_low,r.reference_high,color='#dba969',alpha=.15,label='本轮价格参照带')
            axes[0].scatter([r.breakout_date,r.peak_date,r.date],[full.loc[where[r.breakout_date],'index_close'],r.peak_price,r.index_close],color=['#30814c','#d0872c','#b02e39'],s=40,zorder=5)
            axes[0].axhline(r.support_at_peak,color='#976bb0',ls=':',label='峰值前支撑')
            axes[0].legend(ncol=3,fontsize=8)
            axes[0].set_ylabel('板块代理点位')
            axes[1].plot(part.date,part.up_share,color='#368467',label='上涨成员占比')
            axes[1].axhline(.5,color='#aaa',ls='--');axes[1].set_ylim(0,1);axes[1].set_ylabel('上涨占比')
            axes[2].plot(part.date,part.turnover_pct,color='#80629d',label='板块聚合换手率')
            axes[2].set_ylabel('换手率（%）')
            for ax in axes:
                ax.axvline(r.date,color='#b02e39',ls='--',lw=1)
                ax.axvspan(r.date,part.date.max(),color='#ccc',alpha=.13)
                ax.grid(alpha=.15)
            title='完整候选' if r.first_candidate else '上涨回落观察（尚未满足全部条件）'
            fig.suptitle(f'{num:02d} {r.name} {r.sector_code} · {r.date:%Y-%m-%d} · {title}\n平台后上涨 {r.rally_return:.1%}，峰后回落 {r.drawdown:.1%}；虚线右侧为独立后续观察，不参与识别',fontsize=13)
            fig.text(.01,.003,f'事件 {r.event_id}；参照为历史价格区间，非基本面估值；成员累计调入/出 {r.added_members_since_platform}/{r.removed_members_since_platform}。',fontsize=8)
            name=f'{num:02d}_{r.event_id}.png'
            fig.savefig(cases/name,dpi=cfg['dpi'],bbox_inches='tight');book.savefig(fig);plt.close(fig)
            pages.append(dict(number=num,event_id=r.event_id,sector_code=r.sector_code,name=r.name,date=str(r.date.date()),kind=title,file='cases/'+name))
    pd.DataFrame(pages).to_csv(out/'review_cases.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    funnel=pd.DataFrame(stages)
    axes[0].barh(funnel.stage,funnel.events,color='#397bac');axes[0].invert_yaxis();axes[0].set_xlabel('曾在同一日满足累计条件的独立事件数')
    for i,v in enumerate(funnel.events):axes[0].text(v+1,i,str(v),va='center',fontsize=9)
    annual=signals[signals.first_candidate].groupby(signals[signals.first_candidate].date.dt.year).event_id.nunique()
    axes[1].bar(annual.index.astype(str),annual.values,color='#cc8750');axes[1].set(xlabel='首次完整候选年份',ylabel='事件数',title='全部候选保留，不按后续涨跌筛选')
    fig.suptitle('平台—上涨—回落事件库：识别覆盖',fontsize=15)
    fig.savefig(out/'event_overview.png',dpi=cfg['dpi'],bbox_inches='tight');plt.close(fig)
    summary=dict(episodes=len(episodes),confirmed_rallies=int(episodes.rally_confirmed.sum()),
        trend_pullback_events=int(observations.loc[observations.pullback_gate,'event_id'].nunique()),
        candidate_events=int(candidates.event_id.nunique()),candidate_days=len(candidates),
        candidate_sectors=int(candidates.sector_code.nunique()),observations=len(observations),leader_snapshots=len(leaders),review_pages=len(pages))
    save_json(out/'summary.json',summary)
    save_json(out/'validation.json',dict(prefix_exact=True,future_shock_prefix_exact=True,
        same_episode_time_order=True,reference_frozen=True,no_cross_segment_outcomes=True,unique_observation_keys=True))
    lines=['# 板块平台—上涨—回落事件库 v1','',f'固定源：`{ds.manifest["dataset_id"]}`。覆盖 {panel.date.min():%Y-%m-%d} 至 {panel.date.max():%Y-%m-%d}。','',
        '## 固定定义','',
        '- 平台：突破前20日，高低收盘范围≤15%、首末涨跌绝对值≤8%；高于平台上沿2%记录突破。',
        '- 上涨：峰值比平台上沿至少高20%，突破到峰值至少5日；峰值当时120日涨幅≥5%，高于60日均线且均线较20日前抬升。',
        '- 回落：峰后2—40日，回撤至少15%；回落段内曾在不超过10日内下跌至少10%。',
        '- 完整候选：同时跌破峰值前20日最低收盘支撑、进入本轮平台至上涨初期5日参照附近（上沿容差5%、平台下沿容差20%），最近2日平均负收益幅度≤此前3日的65%。不要求当前重回均线上方。',
        '- 平台和参照绑定同一事件；上涨初期5日后参照上沿冻结。未确认上涨60日、事件180日或峰后40日超时结束；数据缺口及分类切换结束事件。峰值可随新高更新，当时趋势和支撑同步更新。',
        '- 所有数值是固定的首版语义假设，不是寻优结果；本轮不扫描参数，不用后续收益选择参数。',
        '- 日线高低价边界不参与识别；只使用同一连续段的收盘点位。历史价格参照不是基本面内在价值。市值变化字段为观测值，含成员/股本变化，不能直接当成低估。',
        '- 龙头代理仅对首次上涨回落/完整候选观察日附加；前日市值前三或前日60日涨幅前三，不参与候选条件。换手率、广度也作为观察字段。','',
        '## 规模','',*['- '+k+': '+str(v) for k,v in summary.items()],'',
        '## 独立后续观察（不是账户收益）','','| 观察点 | 后续交易日 | 完整观察 | 收盘上涨/下跌/平盘 | 右截尾或缺口 |','|---|---:|---:|---|---:|']
    for r in outcomes_summary:lines.append(f'| {r["trigger"]} | {r["horizon"]} | {r["complete"]} | {r["positive"]}/{r["negative"]}/{r["flat"]} | {r["censored"]} |')
    lines+=['','后续观察从信号收盘量到同段未来收盘，无选股、无成交、无成本。同板块不同事件或不同板块可能重叠，因此这些不是独立试验或策略胜率。缺口/样本末尾不补零。','',
        '## 文件与核验','','- observations.parquet：逐日截至当时的状态快照，供后续特征研究。',
        '- episodes.parquet：最终事件目录，包含最终峰值、结束原因等事后信息，禁止作为早期信号特征。',
        '- first_observations.csv：所有首次结构观察/完整候选；outcomes.parquet 为隔离的未来结果。',
        '- leader_snapshots.parquet：仅首次观察日的龙头代理、代码、前日排序值及当日收益；未保存的日期不能解释为无龙头。',
        '- event_review.pdf / cases/：按首次候选优先、日期顺序、板块去重选取的图册；不按未来结果筛选，图册并不替代完整事件表。',
        '- 与2023-06-30物理截断重算的状态逐值一致；改变未来点位为3倍也不改写此前状态；断言参考带冻结、时序和唯一键。',
        '- 源行业历史不完整的问题仍然存在，不因本次构建消失；不同分类版本分别按独立指数段处理。','']
    (out/'README.md').write_text('\n'.join(lines))
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>板块事件复核</title><style>body{max-width:1250px;margin:auto;font-family:system-ui}img{width:100%}details{margin:20px 0}</style><h1>板块事件复核</h1><p>虚线右侧为独立后续观察；选样未使用后续收益。</p>'+''.join(f'<details open><summary>{p["number"]} {p["name"]} {p["date"]} {p["kind"]}</summary><img loading="lazy" src="{p["file"]}"></details>' for p in pages))
    for p in [config,Path(__file__),ROOT/'src/sector_shape_events.py',ROOT/'tests/test_sector_shape_events.py']:
        shutil.copy2(p,out/p.name)
    save_json(out/'manifest.json',dict(status='complete',dataset_content_id=ds.manifest['content_id'],config=cfg,
        artifacts={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(dict(output=str(out),**summary),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
