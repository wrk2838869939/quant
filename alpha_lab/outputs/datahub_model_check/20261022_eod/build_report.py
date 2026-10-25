"""生成22日行情核验及模型结果报告，不改写行情或策略。"""
from pathlib import Path
import hashlib
import json
import pandas as pd
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path('/root/autodl-tmp')
OUT = Path(__file__).resolve().parent


def js(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


pipeline = js(OUT / 'pipeline.json')
assert pipeline['status'] == 'complete'
update, dataset, scan, model = (Path(pipeline[k]) for k in ['update', 'dataset', 'scan', 'model'])
quality = js(OUT / 'latest_quality.json')
ss = js(scan / 'summary.json')
ms = js(model / 'summary.json')
ds = js(dataset / 'summary.json')
assert quality['asof'] == ss['asof'] == ms['asof'] == '2026-10-22'
assert not ms['annual_sector_gate_enabled'] and not ms['legacy_random_scores_used']
assert ms['record_origin'] == 'current_policy_scan'
protected = js(OUT / 'protected_before.json')
assert all(digest(ROOT / name) == h for name, h in protected.items())
for path in [update, dataset, scan, model]:
    for name, h in js(path / 'manifest.json')['artifacts'].items():
        assert digest(path / name) == h, (path, name)
watch = pd.read_csv(model / 'watch_today.csv')
states = pd.read_csv(scan / 'sector_states_today.csv')
events = pd.read_csv(model / 'first_confirmations.csv')
candidates = pd.read_csv(model / 'stock_candidates.csv')
conditions = {'pullback_gate': '回落门槛未过', 'near_reference': '未到参照区', 'trigger_recent': '近期无触发', 'support_recent': '近期未破支撑', 'slowed': '下跌未减速'}
new_sectors = set(events.sector_code)
watch['状态说明'] = [
    ('新首次确认' if r.sector_code in new_sectors else '旧事件再次满足形态') if r.confirmed else '；'.join(v for k, v in conditions.items() if not bool(getattr(r, k)))
    for r in watch.itertuples()
]
watch.to_csv(OUT / 'watch_with_reasons.csv', index=False)
wt = watch[['name', 'sector_code', 'confirmed', 'drawdown', 'reference_distance', '状态说明']].copy()
wt.columns = ['板块', '代码', '当日形态确认', '距峰值回撤', '距参照位', '状态说明']
for col in ['距峰值回撤', '距参照位']:
    wt[col] = wt[col].map(lambda x: f'{x:.2%}')
wt['当日形态确认'] = wt['当日形态确认'].map({True: '是', False: '否'})
before = js(OUT / 'before_pointers.json')
old = pd.read_csv(ROOT / before['LATEST_SCAN'] / 'sector_states_today.csv')
changes = old[['sector_code', 'name', 'watch', 'confirmed']].merge(states[['sector_code', 'watch', 'confirmed']], on='sector_code', how='outer', suffixes=('_21', '_22'))
changes['name'] = changes.sector_code.map(states.set_index('sector_code')['name']).fillna(changes['name'])
changes = changes[changes.watch_21.ne(changes.watch_22) | changes.confirmed_21.ne(changes.confirmed_22)]
changes.to_csv(OUT / 'state_changes.csv', index=False)
focus = states[states.name.isin(['养殖业', '酒店餐饮'])][['name', 'watch', 'confirmed', 'drawdown', 'pullback_gate', 'near_reference', 'trigger_recent', 'support_recent', 'slowed']].copy()
focus.to_csv(OUT / 'previous_events_status.csv', index=False)
focus_lines = []
for r in focus.itertuples():
    reason = '当日形态通过' if r.confirmed else '；'.join(v for k, v in conditions.items() if not bool(getattr(r, k)))
    focus_lines.append(f'- {r.name}：距峰值回撤{r.drawdown:.2%}，{reason}。')
if len(candidates):
    ct = candidates[['sector_name', 'name', 'ts_code', 'close', 'model_score']].rename(columns={'sector_name':'板块','name':'股票','ts_code':'代码','close':'22日收盘','model_score':'个股分数'}).to_markdown(index=False)
else:
    ct = '无新个股候选。' + ('当日没有新首次事件，个股评分流程按规则跳过，不代表所有个股模型评分失败。' if not len(events) else '首次事件已进入个股流程，但没有股票通过全部现行条件。')
font = FontProperties(fname=str(ROOT / 'fonts/SimHei.ttf'))
fig, ax = plt.subplots(figsize=(10, max(3.6, len(watch) * .48)))
if len(watch):
    plot = watch.sort_values('drawdown')
    ax.barh(range(len(plot)), plot.drawdown * 100, color=['#3684ad' if v else '#9cb8cb' for v in plot.confirmed])
    ax.set_yticks(range(len(plot)), plot.name, fontproperties=font)
    for i, v in enumerate(plot.drawdown):
        ax.text(v * 100 - .2, i, f'{v:.2%}', ha='right', va='center')
    ax.set_xlim(plot.drawdown.min() * 100 - 5, 0)
ax.set_xlabel('距本轮峰值回撤（%）；深色为当日形态确认，未区分新旧事件', fontproperties=font)
ax.set_title('2026-10-22 观察板块', fontproperties=font)
fig.tight_layout()
fig.savefig(OUT / 'watch_drawdown.png', dpi=150)
plt.close(fig)
um = js(update / 'manifest.json')
focus_text = '\n'.join(focus_lines)
report = f'''# 2026年10月22日收盘更新与模型结果

**A股行情已更新至22日，新增{quality['daily_rows']:,}条有效日线；当前策略新首次确认{ms['new_first_confirmations']}个，新个股候选{ms['stock_candidates']}只。对应下一交易日为{ms['earliest_entry_date']}。**

年度板块判断继续暂停。原个股资格、价格形态、正分前5名、首次确认去重及下一交易日有效规则保持原样。

## 数据核验

{pd.read_csv(OUT / 'api_coverage.csv').to_markdown(index=False)}

四接口覆盖全部当日有交易股票，15项核验字段无缺失；重复键、OHLC异常、复权不一致均为0。原始成交量、成交额和收盘价与入库一致。21日及以前的2026行情与旧冻结快照逐行一致。

上市主档{quality['active_master_rows']:,}条，{quality['daily_rows']:,}只有行情、{quality['suspended']}只全日停牌、{quality['stale_delisted']}条已知终止上市但状态滞后记录。详见[逐股原因](missing_daily_reconciled.csv)、[字段覆盖](field_coverage.csv)、[核验汇总](latest_quality.json)。过期主档问题沿用[此前核验]({ROOT}/alpha_lab/outputs/datahub_model_check/20261021T064829_488127Z/REPORT.html)，本次未改写历史主档。

板块数据：{ds['latest_valid_sectors']}个当日有效板块，成员归属覆盖{ds['latest_assignment']:.2%}。美股、基准指数和历史行业资料未在本次同步；行业历史PIT/完整性限制仍在。本次只对现有模型使用的A股数据与既定板块范围进行更新和扫描。

## 当前策略结果

| 环节 | 数量 |
|---|---:|
| 有效板块 | {ss['valid_sectors']} |
| 观察板块 | {ss['watch_sectors']} |
| 触发层板块 | {ss['trigger_sectors']} |
| 当日形态确认 | {ms['shape_confirmed_today']} |
| 去重后新首次确认 | {ms['new_first_confirmations']} |
| 进入个股筛选的板块 | {ms['sectors_entering_stock_filter']} |
| 本次评分股票 | {ms['stocks_scored']} |
| 新个股候选 | {ms['stock_candidates']} |

{wt.to_markdown(index=False)}

![观察板块回撤](watch_drawdown.png)

## 候选与旧事件

{ct}

{focus_text}

养殖业20日旧事件和酒店餐饮9月17日旧事件不会因为形态再次通过而重开入场；只有新的独立首次确认才能进入个股流程。20日养殖业候选的原有效日为21日，不顺延至23日。分数是排序输出，模型候选不等于实际成交。

## 运行记录

本次依次执行原 `update_market.py`、`build_sector_dataset.py`、`scan_sectors.py` 和当前策略 `scan_stocks.py`；仅新增日期配置和本目录的调用/审计/报告脚本。原数据工程代码、模型权重及策略配置哈希未变，数据和模型产物哈希均通过检查。

快照：`{um['release']}`；内容ID：`{um['content_id']}`。

[模型报告]({model}/REPORT.html) · [候选CSV]({model}/stock_candidates.csv) · [首次事件]({model}/first_confirmations.csv) · [完整状态]({scan}/sector_states_today.csv) · [相较21日的变化](state_changes.csv) · [调用参数及返回码](pipeline.json) · [保护文件验证](verification.json)

日志：[更新](update.log)、[构建](build_dataset.log)、[板块扫描](scan_sectors.log)、[当前策略](scan_stocks.log)、[审计](audit.log)。
'''
(OUT / 'REPORT.md').write_text(report)
(OUT / 'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>22日收盘更新与模型</title><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#243647}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5df;padding:7px}th{background:#edf3f7}img{max-width:100%}</style><body>'+markdown.markdown(report,extensions=['tables','fenced_code'])+'</body></html>')
(OUT / 'verification.json').write_text(json.dumps(dict(status='passed', protected_files_unchanged=True, all_pipeline_artifacts_verified=True, annual_sector_gate_enabled=False, model_summary=ms),ensure_ascii=False,indent=2))
(OUT / 'manifest.json').write_text(json.dumps(dict(status='complete', asof='2026-10-22', artifacts={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}),ensure_ascii=False,indent=2))
(OUT.parent / 'LATEST').write_text(str(OUT.relative_to(ROOT))+'\n')
print(json.dumps(ms,ensure_ascii=False));print(OUT/'REPORT.html')
