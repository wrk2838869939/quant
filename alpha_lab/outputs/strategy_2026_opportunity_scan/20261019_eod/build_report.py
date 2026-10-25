"""将既有扫描结果整理成数据表和图表，不改变模型或数据构建程序。"""
from pathlib import Path
import json,hashlib,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
ROOT=Path('/root/autodl-tmp');OUT=Path(__file__).resolve().parent
font=FontProperties(fname=str(ROOT/'fonts/SimHei.ttf'))
plt.rcParams['axes.unicode_minus']=False

def md_table(df):
    if df.empty:return '无。'
    def cell(x):return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(map(cell,r))+' |' for r in df.itertuples(index=False,name=None)])

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    scan=(ROOT/summary['sector_scan']).resolve();dataset=(ROOT/summary['dataset']).resolve()
    watch=pd.read_csv(OUT/'watch_today.csv')
    events=pd.read_csv(OUT/'interval_first_confirmations_asof_model.csv')
    candidates=pd.read_csv(OUT/'stock_candidates_asof.csv')
    update=ROOT/'alpha_lab/outputs/sector_live_update/20261019T143409_401486Z'
    quality=pd.read_csv(update/'daily_quality.csv',dtype={'date':str})
    model=json.loads((OUT/'annual_model_readonly_check.json').read_text())
    dsum=json.loads((dataset/'summary.json').read_text())
    confirmations=pd.read_parquet(scan/'all_confirmations.parquet')
    predictions=pd.read_parquet(ROOT/'alpha_lab/outputs/sector_asof_replay_v1/20261013T072855_087132Z/predictions_BEFORE_EVALUATION.parquet')
    repeats=[]
    for r in watch[watch.confirmed].itertuples():
        prior=confirmations[(confirmations.event_id==r.event_id)&(confirmations.window==r.window)]
        if len(prior):
            first=prior.iloc[0]
            score=predictions[predictions.event_id.eq(first.merged_event_id)]
            if len(score):
                q=score.iloc[0]
                repeats.append({'板块':r.name,'源事件':r.event_id,'本轮峰值日':r.peak_date,'首次确认日':str(first.date.date()),
                    '首次确认年度分数':float(q.score),'年度阈值':float(q.threshold),'首次模型通过':bool(q.selected)})
    repeats=pd.DataFrame(repeats)
    repeats.to_csv(OUT/'reconfirmed_event_audit.csv',index=False)
    # 条件表保留实际布尔值，不把观察/价格触发解读为可以买入。
    today_codes=set(events.loc[events.date.eq('2026-10-19'),'sector_code'])
    watch['new_first_confirmation']=watch.sector_code.isin(today_codes)
    fields=['watch','near_reference','trigger_recent','support_recent','slowed','confirmed','new_first_confirmation']
    labels=['观察形态','当前参照区','近5日价格触发','近5日破支撑','当前减速','当日形态确认','新增首次确认']
    if len(watch):
        matrix=watch[fields].astype(int).to_numpy()
        fig,ax=plt.subplots(figsize=(11,max(3,0.7*len(watch)+1.6)))
        ax.imshow(matrix,cmap=ListedColormap(['#f2dada','#b7dfc6']),vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(len(labels)),labels,fontproperties=font)
        ax.set_yticks(range(len(watch)),watch.name,fontproperties=font)
        for i in range(len(watch)):
            for j in range(len(fields)):
                ax.text(j,i,'通过' if matrix[i,j] else '未过',ha='center',va='center',fontproperties=font)
        ax.set_title('2026-10-19 收盘：形态条件检查（尚须首次事件与年度模型筛选）',fontproperties=font,pad=14)
        fig.tight_layout();fig.savefig(OUT/'gate_status.png',dpi=160);plt.close(fig)
        panel=pd.read_parquet(dataset/'daily.parquet')
        n=len(watch);fig,axs=plt.subplots(n,1,figsize=(11,max(3,2.8*n)),squeeze=False)
        chart_rows=[]
        for ax,r in zip(axs.flat,watch.itertuples()):
            g=panel[(panel.sector_code==r.sector_code)&panel.index_valid].sort_values('date').tail(60).copy()
            g['normalized_close']=g.index_close/g.index_close.iloc[0]*100
            chart_rows.append(g[['date','sector_code','index_close','normalized_close']])
            ax.plot(g.date,g.normalized_close,color='#2e638b',lw=1.7)
            if 'reference_high' in watch.columns and pd.notna(r.reference_high):
                ax.axhline(r.reference_high/g.index_close.iloc[0]*100,color='#bd7735',ls='--',label='当前窗口参照价')
                ax.legend(prop=font,loc='best')
            ax.axvline(pd.Timestamp('2026-10-13'),color='#888',ls=':',alpha=.8)
            ax.set_title(f'{r.name} | 代表窗口 {r.window} 日 | 19日 {r.index_return:+.2%} | 峰值回撤 {r.drawdown:.2%}',fontproperties=font)
            ax.grid(alpha=.2);ax.tick_params(axis='x',labelsize=8)
        fig.suptitle('板块近60个交易日走势：起点=100，灰线为上次扫描日',fontproperties=font,y=1.002)
        fig.tight_layout();fig.savefig(OUT/'watch_prices.png',dpi=150,bbox_inches='tight');plt.close(fig)
        pd.concat(chart_rows).to_csv(OUT/'watch_price_chart_data.csv',index=False)
    rows=[]
    for r in watch.itertuples():
        missing=[label for key,label in [('near_reference','当前参照区'),('trigger_recent','近5日价格触发'),('support_recent','近5日破支撑'),('slowed','当前减速')] if not bool(getattr(r,key))]
        ratio=r.recent_loss_mean/r.previous_loss_mean if r.previous_loss_mean>0 else np.nan
        rows.append({'板块':r.name,'窗口':r.window,'19日涨跌':f'{r.index_return:+.2%}','峰值回撤':f'{r.drawdown:.2%}',
            '价格触发':'是' if r.trigger else '否','形态确认':'是' if r.confirmed else '否','减速比':f'{ratio:.3f}' if pd.notna(ratio) else '不可计算','尚缺条件':'、'.join(missing) if missing else '旧事件再次确认，非新入场事件'})
    table=pd.DataFrame(rows);table.to_csv(OUT/'watch_interpretation.csv',index=False)
    if len(events):
        et=events[['date','sector_name','score','threshold','selected','entry_date','entry_status']].copy()
        et.columns=['首次确认日','板块','年度分数','阈值','模型通过','最早入场日','有效期状态']
    else:et=pd.DataFrame()
    text=f'''# 2026-10-19 收盘机会扫描

**今日结果：{summary['valid_sectors']}个有效申万二级自算板块中，{summary['watch_sectors']}个观察、{summary['trigger_sectors']}个价格触发、{summary['confirmed_today']}个当日形态确认；新增首次确认{summary['new_first_confirmations']}个，年度板块模型通过{summary['annual_sector_pass']}个，个股新入场候选{summary['stock_candidates']}只。**

本次补齐10月14、15、16、19日四个交易日，共22,259条有效日线，其中19日5,564条。扫描依据为19日完整收盘数据。

## 今日板块状态

{md_table(table)}

![条件状态](gate_status.png)

![板块走势](watch_prices.png)

观察、价格触发、形态确认、去重后的首次确认、年度模型通过是不同阶段。只有新的首次确认通过年度板块模型，才启动同日个股筛选。减速比为最近2日平均负收益／此前3日平均负收益，门槛沿用0.65；分母须大于0。

## 酒店餐饮为何不是新买点

酒店餐饮19日重新满足形态确认，但仍是9月8日见峰、9月17日首次确认的同一轮事件。首次确认年度模型分数约-0.009877，低于0.003289阈值；既定策略不会因后续重复确认重新给旧事件开放入场。今天没有新的板块事件进入个股层，因此阻塞点不是个股筛选。

{md_table(repeats)}

与13日相比，观察板块由4个增至7个，价格触发由2个增至7个；医疗服务、房地产服务、焦炭Ⅱ进入当前观察池。不同日期的代表窗口可能变化，不能将不同窗口的参照价直接作连续比较。酒店餐饮是由“缺减速”变成“形态满足，但旧事件不可重复入场”。

## 补扫期间的首次确认

10月14日至19日新增首次确认共{summary['interval_first_confirmations']}个，年度板块模型通过{summary['interval_annual_sector_pass']}个；按各确认日当时的成员及行情形成的历史候选共{summary['interval_stock_candidates']}条。

{md_table(et)}

策略的入场有效期是确认后的一个交易日；14日至16日的信号即使入选，也不能当成19日收盘后新出现的机会。19日确认的候选最早进入10月20日流程。交易日采用本地日历，并核对[上交所2026年休市安排](https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml)。

## 今日个股候选

{md_table(candidates)}

{'今日未进入个股评分：没有通过年度板块模型的新首次确认。候选为0不表示将全市场个股逐只评分后全部淘汰。' if summary['annual_sector_pass']==0 else '个股沿用价格形态池内正预测、最多5只的年度排序规则；趋势与换手不另作硬门槛。候选尚未进行账户权限、资金和次日成交认证。'}

## 数据与模型核验

{md_table(quality)}

- 日线股票—日期无重复；正价格、复权和覆盖检查通过；来源涨跌幅与收盘/前收推导值的最大舍入差约0.00005个百分点。
- 原有更新、全量构建、五窗口扫描入口完成；旧行情前缀、旧板块前缀和既有确认事件一致性检查通过。现有程序、策略参数及4份年度模型权重的哈希未变。
- 年度模型只装载已有权重，无重新训练或调参。板块训练标签最晚2025-11-18，个股最晚2025-11-19；{model['events']}个既有年度板块预测复核最大误差{model['max_abs_score_difference']:.3g}，板块阈值{model['threshold']:.12f}。
- 既有扫描器仍保留旧随机模型评分接口，本报告未采用该评分。本次没有新的首次确认，未执行新事件的板块及个股评分；年度板块模型装载后复核了已有17条预测，酒店餐饮引用同轮首次确认日的年度分数。
- 年度权重是此前研究按历史信息边界生成的模型，并不意味着权重文件在2026年初已经存在。历史成员数据尚未获得完整PIT认证。
- 最新有效板块{dsum['latest_valid_sectors']}个；行业映射覆盖{dsum['latest_assignment']:.2%}，历史成员获取失败记录{dsum['failed_member_requests']}个。范围并非全市场完整行业覆盖。财务、名称历史、历史行业、指数、美股未在此次四接口更新中刷新；名称历史公告最新到2026-09-08，如有个股候选仍须补核之后的名称/ST变化。

## 产物与复现

[摘要](summary.json) · [期间首次确认](interval_first_confirmations_asof_model.csv) · [个股期间诊断](stock_diagnostics_interval.csv) · [今日候选](stock_candidates_asof.csv) · [与13日对比](comparison_20261013_20261019.csv) · [哈希与时序核验](verification.json)

行情快照：`datahub/releases/selection_live_20261019_eod`。板块数据：`{dataset.relative_to(ROOT)}`。扫描产物：`{scan.relative_to(ROOT)}`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python datahub/scripts/frozen_research.py run --release datahub/releases/selection_live_20261019_eod alpha_lab/outputs/strategy_2026_opportunity_scan/20261019_eod/evaluate_existing_models.py --scan {scan.relative_to(ROOT)}
```
'''
    (OUT/'REPORT.md').write_text(text)
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables','fenced_code'])
    except ImportError:
        body='<pre>'+html.escape(text)+'</pre>'
    style='body{font:16px/1.7 sans-serif;max-width:1250px;margin:32px auto;padding:0 24px;color:#203040}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd5dc;padding:8px}th{background:#eef3f6}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f7;padding:14px}a{color:#24638c}'
    (OUT/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>2026-10-19 收盘机会扫描</title><style>'+style+'</style><body>'+body+'</body></html>')
    nav=ROOT/'alpha_lab/docs/current_strategy_20261013/OPPORTUNITY_SCAN_20261019_EOD.md'
    nav.write_text('# 2026-10-19收盘机会扫描\n\n'+text.split('**')[1]+'\n\n[详细报告与图表](../../outputs/strategy_2026_opportunity_scan/20261019_eod/REPORT.html) · [Markdown](../../outputs/strategy_2026_opportunity_scan/20261019_eod/REPORT.md)\n')
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(dict(status='complete',summary=summary,artifacts=artifacts),ensure_ascii=False,indent=2))
    print(text)

if __name__=='__main__':main()
