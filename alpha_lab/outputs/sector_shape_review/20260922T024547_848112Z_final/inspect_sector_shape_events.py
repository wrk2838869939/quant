#!/usr/bin/env python3
"""事件时间聚集审计及全部候选图册，不修改原事件库。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sector_dataset import digest


def main():
    run=Path(sys.argv[1]).resolve()
    manifest=json.loads((run/'manifest.json').read_text())
    for name,h in manifest['artifacts'].items():
        assert digest(run/name)==h,name
    cfg=manifest['config']
    dataset=ROOT.parent/cfg['dataset']
    panel=pd.read_parquet(dataset/'daily.parquet')
    obs=pd.read_parquet(run/'observations.parquet')
    selected=obs[obs.first_candidate].sort_values(['date','sector_code']).reset_index(drop=True)
    catalog=pd.read_parquet(dataset/'catalog_sw_l2.parquet')
    names={(r.src,r.index_code):r.industry_name for r in catalog.itertuples()}
    outcomes=pd.read_parquet(run/'outcomes.parquet')
    review=ROOT/'outputs/sector_shape_review'/(run.name+'_final')
    review.mkdir(parents=True,exist_ok=False)
    date_counts=selected.groupby('date').agg(events=('event_id','size'),sectors=('sector_code','nunique'))
    date_counts.to_csv(review/'candidate_date_counts.csv')
    records=[]
    for horizon,part in outcomes[outcomes.trigger.eq('first_candidate')].groupby('horizon'):
        complete=part[part.complete]
        grouped=complete.groupby('signal_date').agg(events=('event_id','size'),mean_return=('close_return','mean'),median_return=('close_return','median'))
        grouped.to_csv(review/f'date_equal_outcomes_{horizon}d.csv')
        records.append(dict(horizon=int(horizon),events=len(complete),dates=len(grouped),
                            event_median=float(complete.close_return.median()),date_mean_return_median=float(grouped.mean_return.median()),
                            positive_dates=int(grouped.mean_return.gt(0).sum()),negative_dates=int(grouped.mean_return.lt(0).sum())))
    pd.DataFrame(records).to_csv(review/'date_equal_summary.csv',index=False)
    font_manager.fontManager.addfont(str(ROOT.parent/cfg['font']))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=ROOT.parent/cfg['font']).get_name(),'axes.unicode_minus':False,'pdf.fonttype':42})
    images=review/'cases';images.mkdir()
    pages=[]
    with PdfPages(review/'all_candidates.pdf') as book:
        for i,r in enumerate(selected.itertuples(),1):
            full=panel[(panel.sector_code==r.sector_code)&panel.segment_id.eq(r.segment_id)&panel.index_valid].sort_values('date').reset_index(drop=True)
            start=full.index[full.date.eq(r.platform_start)][0]
            signal=full.index[full.date.eq(r.date)][0]
            part=full.iloc[max(0,start-cfg['review_context_days']):signal+cfg['review_followup_days']+1]
            base=full[full.date.eq(r.platform_end)].iloc[0]
            name=names[(r.classification_version,r.sector_code)]
            fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[3,1,1]})
            axes[0].plot(part.date,part.index_close,color='#306f9f',label='板块点位')
            axes[0].plot(part.date,part.equal_index_close/base.equal_index_close*base.index_close,color='#b88841',label='等权走势（平台末日对齐）')
            axes[0].axvspan(r.platform_start,r.platform_end,color='#41956e',alpha=.15,label='平台')
            axes[0].axhspan(r.platform_low,r.reference_high,color='#e7c56b',alpha=.20,label='本轮价格参照')
            axes[0].axhline(r.support_at_peak,color='#9973a2',ls=':',label='峰值前支撑')
            axes[0].scatter([r.peak_date,r.date],[r.peak_price,r.index_close],color=['#dba64d','#b94145'],s=35,zorder=4)
            axes[0].legend(ncol=3,fontsize=8)
            axes[0].set_ylabel('指数点位')
            axes[1].plot(part.date,part.up_share,color='#448f71');axes[1].axhline(.5,color='#aaa',ls='--');axes[1].set(ylim=(0,1),ylabel='上涨占比')
            axes[2].plot(part.date,part.turnover_pct,color='#8b6aa4');axes[2].set_ylabel('换手率（%）')
            for ax in axes:
                ax.axvline(r.date,color='#b94145',ls='--')
                ax.axvspan(r.date,part.date.max(),color='#bbb',alpha=.14)
                ax.grid(alpha=.15)
                ax.set_xlim(part.date.min(),part.date.max())
            locator=mdates.AutoDateLocator(minticks=4,maxticks=9)
            axes[-1].xaxis.set_major_locator(locator)
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[-1].tick_params(axis='x',labelrotation=25,labelsize=8)
            fig.suptitle(f'{i:02d}/{len(selected)} {name} {r.sector_code} · {r.date:%Y-%m-%d}\n先涨 {r.rally_return:.1%}，再回落 {r.drawdown:.1%}；灰色区是独立后续观察',fontsize=13)
            fname=f'{i:02d}_{r.event_id}.png'
            fig.savefig(images/fname,dpi=cfg['dpi'],bbox_inches='tight');book.savefig(fig);plt.close(fig)
            pages.append(dict(page=i,event_id=r.event_id,date=str(r.date.date()),name=name,file='cases/'+fname))
            if i%20==0:print('PAGES',i,flush=True)
    pd.DataFrame(pages).to_csv(review/'all_candidate_pages.csv',index=False)
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    ax.bar(np.arange(len(date_counts)),date_counts.events,color='#387caf')
    ax.set_xticks(np.arange(len(date_counts)),[d.strftime('%Y-%m-%d') for d in date_counts.index],rotation=60,ha='right',fontsize=8)
    ax.set(xlabel='首次完整候选日期',ylabel='同日板块事件数',title=f'{len(selected)}个候选集中在{len(date_counts)}个日期；相邻日期也可能属于同一行情')
    fig.savefig(review/'date_concentration.png',dpi=cfg['dpi'],bbox_inches='tight');plt.close(fig)
    lines=['# 事件库复核：全部候选与日期聚集','',f'源事件库：`{run}`，原始文件哈希全部核验通过。','',
        f'{len(selected)}个候选分布在{len(date_counts)}个触发日期。最多一天{date_counts.events.max()}个候选，占全部{date_counts.events.max()/len(selected):.1%}。这不是独立试验数量，日期聚合也不能完全消除相邻行情及跨板块依赖。','',
        '## 事后观察敏感性','','| 后续日数 | 完整事件 | 独立日期数（非独立行情数） | 逐事件收益中位数 | 先同日平均再按日期取中位数 | 同日均值为正/负的日期 |','|---|---:|---:|---:|---:|---|']
    for r in records:lines.append(f'| {r["horizon"]} | {r["events"]} | {r["dates"]} | {r["event_median"]:.2%} | {r["date_mean_return_median"]:.2%} | {r["positive_dates"]}/{r["negative_dates"]} |')
    lines+=['','这些是已观察样本的板块点位走势，不是个股策略收益、胜率或独立样本外技能。先用全部候选图检查形态定义，不按这些结果调参。','',
        'all_candidates.pdf 包含全部候选，按触发日期和板块编号排序，不按后续收益筛选。已显式固定每张图的日期范围，修复旧图日期轴误扩展至1970年前后的显示问题；事件数值没有改动。原图册最早案例较集中，复核时以本完整图册为主。',
        '全套图：cases/；逐页索引：all_candidate_pages.csv；原始候选日计数及分日期结果均保留为CSV。','']
    (review/'README.md').write_text('\n'.join(lines))
    (review/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>全部板块候选</title><style>body{max-width:1200px;margin:auto;font-family:system-ui}img{width:100%}details{margin:15px}</style><h1>全部69个候选复核</h1>'+''.join(f'<details><summary>{p["page"]} {p["date"]} {p["name"]}</summary><img loading="lazy" src="{p["file"]}"></details>' for p in pages))
    (review/Path(__file__).name).write_text(Path(__file__).read_text())
    (review/'manifest.json').write_text(json.dumps(dict(source_manifest_sha256=digest(run/'manifest.json'),source_files_verified=len(manifest['artifacts']),
        script_sha256=digest(Path(__file__)),artifacts={str(p.relative_to(review)):digest(p) for p in review.rglob('*') if p.is_file()}),ensure_ascii=False,indent=2))
    print(json.dumps(dict(review=str(review),events=len(selected),dates=len(date_counts),largest_day_events=int(date_counts.events.max()),outcomes=records),ensure_ascii=False))


if __name__=='__main__':main()
