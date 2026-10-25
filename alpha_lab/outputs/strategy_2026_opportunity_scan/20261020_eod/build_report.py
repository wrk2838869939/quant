"""汇总20日收盘正式扫描及酒店餐饮的独立观察，不改变策略。"""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
import markdown
ROOT=Path('/root/autodl-tmp');OUT=Path(__file__).resolve().parent
HOTEL=ROOT/'alpha_lab/outputs/hotel_stock_review/20261020_eod'
font=FontProperties(fname=str(ROOT/'fonts/SimHei.ttf'));plt.rcParams['axes.unicode_minus']=False
s=json.loads((OUT/'summary.json').read_text());scan=(ROOT/s['sector_scan']).resolve();dataset=(ROOT/s['dataset']).resolve()
w=pd.read_csv(OUT/'watch_today.csv');events=pd.read_csv(OUT/'today_first_confirmations_asof_model.csv');candidates=pd.read_csv(OUT/'stock_candidates_asof.csv')
hotel=pd.read_csv(HOTEL/'all_members_final.csv');old=pd.read_csv(ROOT/'alpha_lab/outputs/hotel_stock_review/20261019_eod/all_members_final.csv')
hs=json.loads((HOTEL/'summary.json').read_text());dq=json.loads((OUT/'data_quality_extra.json').read_text())
update=ROOT/dq['update_run'];quality=pd.read_csv(update/'daily_quality.csv')
model=json.loads((OUT/'annual_model_readonly_check.json').read_text());dsum=json.loads((dataset/'summary.json').read_text())
def table(df):return df.to_markdown(index=False) if len(df) else '无。'
def yn(v):return '是' if bool(v) else '否'
# 当日形态通过并不自动通过首次确认去重。
w['new_first_confirmation']=w.sector_code.isin(events.sector_code)
fields=['near_reference','trigger_recent','support_recent','slowed','confirmed','new_first_confirmation']
labels=['当前参照区','近5日价格触发','近5日破支撑','当前减速','形态确认','新首次确认']
if len(w):
 matrix=w[fields].astype(int).to_numpy();fig,ax=plt.subplots(figsize=(11,max(3,0.65*len(w)+1.5)))
 ax.imshow(matrix,cmap=ListedColormap(['#f0dcdc','#bbdfc7']),vmin=0,vmax=1,aspect='auto')
 ax.set_xticks(range(len(fields)),labels,fontproperties=font);ax.set_yticks(range(len(w)),w.name,fontproperties=font)
 for i in range(len(w)):
  for j in range(len(fields)):ax.text(j,i,'通过' if matrix[i,j] else '未过',ha='center',va='center',fontproperties=font)
 ax.set_title('2026-10-20收盘：板块条件，年度模型另行筛选',fontproperties=font,pad=12)
 fig.tight_layout();fig.savefig(OUT/'gate_status.png',dpi=160);plt.close(fig)
rows=[]
for r in w.itertuples():
 missing=[label for key,label in [('near_reference','当前参照区'),('trigger_recent','近5日价格触发'),('support_recent','近5日破支撑'),('slowed','减速确认')] if not bool(getattr(r,key))]
 rows.append({'板块':r.name,'窗口':r.window,'20日涨跌':f'{r.index_return:+.2%}','峰值回撤':f'{r.drawdown:.2%}','形态确认':yn(r.confirmed),'新增首次确认':yn(r.new_first_confirmation),'当前条件缺口':'、'.join(missing) if missing else '形态齐备，须核对事件去重及年度模型'})
watch_table=pd.DataFrame(rows);watch_table.to_csv(OUT/'watch_interpretation.csv',index=False)
if len(events):
 et=events[['date','sector_name','score','threshold','selected','entry_date']].copy();et.columns=['日期','板块','年度分数','阈值','年度模型通过','最早入场日']
else:et=pd.DataFrame()
tracked=['000721.SZ','605108.SH'];compare=hotel[hotel.ts_code.isin(tracked)].merge(old[['ts_code','close','price','model_score','observation_candidate']],on='ts_code',how='left',suffixes=('_20','_19'))
compare.to_csv(OUT/'hotel_tracking_20261019_20261020.csv',index=False)
ht=compare[['name','ts_code','close_19','close_20','day_return','price_20','model_score_19','model_score_20','observation_candidate_20','reason']].copy()
ht['day_return']=ht.day_return.map(lambda v:f'{v:+.2%}')
for col in ['model_score_19','model_score_20']:ht[col]=ht[col].map(lambda v:f'{v:.6f}')
for col in ['price_20','observation_candidate_20']:ht[col]=ht[col].map(yn)
ht.columns=['股票','代码','19日收盘','20日收盘','20日涨跌','20日价格形态','19日排序分数','20日排序分数','仍在观察名单','说明']
# 本次重点两只的走势图与全体同行评分均保存。
raw=pd.read_parquet(HOTEL/'stock_history.parquet');fig,axs=plt.subplots(1,2,figsize=(12,4.2))
for ax,code in zip(axs,tracked):
 g=raw[raw.ts_code.eq(code)].sort_values('date').tail(40);name=hotel.set_index('ts_code').at[code,'name']
 ax.plot(g.date,g.hfq_close/g.hfq_close.iloc[0]*100,color='#286b97')
 ax.axvline(pd.Timestamp('2026-10-19'),color='#999',ls='--');ax.set_title(f'{name} {code} | 近40日后复权价格，起点=100',fontproperties=font)
 ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25,labelsize=8)
fig.suptitle('截至20日收盘；灰线为上次19日扫描',fontproperties=font);fig.tight_layout();fig.savefig(OUT/'hotel_tracking.png',dpi=160);plt.close(fig)
allstates=pd.read_csv(scan/'sector_states_today.csv');hotelstate=allstates[allstates.sector_code.eq('801219.SI')]
accepted=pd.read_parquet(scan/'all_confirmations.parquet');preds=pd.read_parquet(ROOT/'alpha_lab/outputs/sector_asof_replay_v1/20261013T072855_087132Z/predictions_BEFORE_EVALUATION.parquet')
hotel_note='酒店餐饮今日没有可用的活动形态状态。'
if len(hotelstate):
 r=hotelstate.iloc[0];prior=accepted[(accepted.sector_code==r.sector_code)&(accepted.event_id==r.event_id)&(accepted.window==r.window)]
 hotel_note=f'酒店餐饮20日形态确认：{yn(r.confirmed)}；当日板块涨跌{r.index_return:+.2%}。'
 if len(prior):
  first=prior.iloc[0];q=preds[preds.event_id.eq(first.merged_event_id)]
  hotel_note+=f'当前代表窗口对应首次确认日为{first.date:%Y-%m-%d}。'
  if len(q):hotel_note+=f'该首次事件的年度模型分数{q.score.iloc[0]:.6f}，阈值{q.threshold.iloc[0]:.6f}；后续重复形态不能自动重开入场。'
fullhotel=hotel[['name','ts_code','close','day_return','price','model_score','observation_candidate','reason']].copy()
fullhotel['day_return']=fullhotel.day_return.map(lambda v:f'{v:+.2%}');fullhotel['model_score']=fullhotel.model_score.map(lambda v:f'{v:.6f}' if pd.notna(v) else '未评分')
fullhotel.columns=['名称','代码','20日收盘','20日涨跌','价格形态','排序分数','独立观察入选','原因']
text=f'''# 2026-10-20收盘数据与模型扫描

**正式策略结果：{s['valid_sectors']}个有效板块中，{s['watch_sectors']}个观察、{s['trigger_sectors']}个价格触发、{s['confirmed_today']}个当日形态确认；新增独立首次确认{s['new_first_confirmations']}个，年度板块模型通过{s['annual_sector_pass']}个，正式模型新入场候选{s['stock_candidates']}只。**

20日新增{dq['rows']:,}条有效日线，收盘数据冻结完成。数据工程代码、正式策略参数和年度模型权重均未修改；没有重新训练模型。

## 正式策略扫描

{table(watch_table)}

![板块条件](gate_status.png)

价格触发、当日形态确认和去重后的首次确认分别记录。只有新首次确认通过年度板块模型，才进入正式个股排序。

### 今日新增首次确认及年度评分

{table(et)}

### 正式模型个股候选

{table(candidates)}

{'今日没有通过年度板块模型的新首次事件，因此正式流程未进入个股评分；不能理解为全市场股票逐只评分后被淘汰。' if s['annual_sector_pass']==0 else '正式候选仅代表模型条件通过；名称/ST的时效、账户权限、资金和次日成交条件仍须核验。'}

## 西安饮食与同庆楼的更新

以下是用户指定的酒店餐饮独立观察复核，与正式模型候选分开。即使这里个股条件通过，也不会自动绕过首次事件和板块模型门槛。

{table(ht)}

{hotel_note}

![两只股票走势](hotel_tracking.png)

酒店餐饮全部{hs['members']}只成员重新检查，{hs['qualified']}只通过资格筛选，{hs['price_pass']}只通过个股价格形态，{len(hs['observation_candidates'])}只进入价格形态池内正分数的独立观察名单。年度分数仅用于板块内相对排序，不是预期收益或胜率。名称/ST历史通过原查询封装重新拉取，保存在独立研究目录，未更改冻结行情。

{table(fullhotel)}

## 数据和模型核验

{quality.to_markdown(index=False)}

- 新增日线无重复键；复权因子、市值/流通股本、换手率覆盖均通过既有质量门槛。涨跌幅与收盘/前收推导值最大舍入差{dq['daily_return_rounding_error_pp']:.8f}个百分点。
- 更新产物{dq['update_artifacts_verified']}项哈希核验通过；旧行情前缀、板块历史和原确认事件保持一致。
- 年度板块模型的{model['events']}个历史预测复核最大误差{model['max_abs_score_difference']:.3g}。板块训练标签最晚2025-11-18、个股最晚2025-11-19，未调用旧随机划分模型分数作策略判断。
- 原扫描入口保留旧随机模型接口；本报告使用独立的2026年度评价结果。既有年度权重是历史信息边界复核产物，不意味着文件在2026年初已经存在。
- 行业映射最新覆盖{dsum['latest_assignment']:.2%}，有效板块{dsum['latest_valid_sectors']}个，历史成员请求失败记录{dsum['failed_member_requests']}个；历史成员PIT尚未完整认证。此次更新四个A股日线接口和在市主档，未全量刷新财务、历史名称、历史行业、指数、美股；酒店餐饮名称历史另行核对。
- 原“旧拒绝事件重复确认”的历史诊断未证明补买有效，本次没有放宽这一规则。

## 产物

[摘要](summary.json) · [新首次确认](today_first_confirmations_asof_model.csv) · [正式候选](stock_candidates_asof.csv) · [与19日板块对比](comparison_20261019_20261020.csv) · [两只股票前后对比](hotel_tracking_20261019_20261020.csv) · [代码与模型核验](verification.json)

行情快照：`datahub/releases/selection_live_20261020_eod`。板块数据：`{dataset.relative_to(ROOT)}`。原扫描：`{scan.relative_to(ROOT)}`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python datahub/scripts/frozen_research.py run --release datahub/releases/selection_live_20261020_eod alpha_lab/outputs/strategy_2026_opportunity_scan/20261020_eod/evaluate_existing_models.py --scan {scan.relative_to(ROOT)}
```
'''
(OUT/'REPORT.md').write_text(text)
style='body{font:16px/1.7 sans-serif;max-width:1320px;margin:32px auto;padding:0 24px;color:#263440}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd5dc;padding:7px}th{background:#eef3f6}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f7;padding:12px}a{color:#246b97}'
(OUT/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>2026-10-20收盘机会扫描</title><style>'+style+'</style><body>'+markdown.markdown(text,extensions=['tables','fenced_code'])+'</body></html>')
artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(dict(status='complete',summary=s,hotel_summary=hs,artifacts=artifacts),ensure_ascii=False,indent=2))
ha={str(p.relative_to(HOTEL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HOTEL.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
(HOTEL/'manifest.json').write_text(json.dumps(dict(status='complete',summary=hs,artifacts=ha),ensure_ascii=False,indent=2))
nav=ROOT/'alpha_lab/docs/current_strategy_20261013/OPPORTUNITY_SCAN_20261020_EOD.md'
nav.write_text('# 2026-10-20收盘机会扫描\n\n'+text.split('**')[1]+'\n\n[详细报告与图表](../../outputs/strategy_2026_opportunity_scan/20261020_eod/REPORT.html) · [Markdown](../../outputs/strategy_2026_opportunity_scan/20261020_eod/REPORT.md)\n')
print(json.dumps(s,ensure_ascii=False));print(ht.to_string(index=False));print('COMPLETE',OUT/'REPORT.html')
