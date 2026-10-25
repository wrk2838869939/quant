"""汇总21日收盘扫描及22日开盘前报告，只写本次报告目录。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import pandas as pd
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path('/root/autodl-tmp')
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'alpha_lab/experiments/sector_live_v1'


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(name):
    return ROOT / (BASE / name).read_text().strip()


update = pointer('LATEST_UPDATE')
scan = pointer('LATEST_SCAN')
model = pointer('LATEST_CANDIDATES')
um = read_json(update / 'manifest.json')
sm = read_json(scan / 'manifest.json')
mm = read_json(model / 'manifest.json')
dataset = ROOT / sm['dataset']
ds = read_json(dataset / 'summary.json')
summary = read_json(model / 'summary.json')
ss = read_json(scan / 'summary.json')
assert summary['asof'] == ss['asof'] == um['asof'] == '2026-10-21'
assert summary['record_origin'] == 'current_policy_scan'
assert summary['annual_sector_gate_enabled'] is False
assert summary['legacy_random_scores_used'] is False
for path, manifest in [(update, um), (scan, sm), (model, mm), (dataset, read_json(dataset / 'manifest.json'))]:
    for name, h in manifest['artifacts'].items():
        assert digest(path / name) == h, (path, name)
protected = read_json(OUT / 'protected_before.json')
assert all(digest(ROOT / p) == h for p, h in protected.items())
q = read_json(OUT / 'latest_quality.json')
watch = pd.read_csv(model / 'watch_today.csv')
states = pd.read_csv(scan / 'sector_states_today.csv')
breeding = states[states.name.eq('养殖业')].iloc[0]
events = pd.read_parquet(scan / 'all_confirmations.parquet')
new = pd.read_csv(model / 'first_confirmations.csv')
candidates = pd.read_csv(model / 'stock_candidates.csv')
last = events.groupby('sector_code').date.max()
wt = watch[['name', 'sector_code', 'confirmed', 'rally_return', 'drawdown', 'reference_distance']].copy()
wt['最近首次确认'] = wt.sector_code.map(last).fillna('无')
wt['是否21日首次'] = wt.sector_code.isin(new.sector_code)
conditions = {'pullback_gate': '回落条件未过', 'near_reference': '未到参照区', 'trigger_recent': '近期无价格触发', 'support_recent': '近期未破支撑', 'slowed': '下跌尚未减速'}
wt['状态说明'] = [
    ('新首次确认' if row.sector_code in set(new.sector_code) else '旧事件再次满足形态')
    if row.confirmed else '；'.join(label for key, label in conditions.items() if not bool(getattr(row, key)))
    for row in watch.itertuples()
]
wt.to_csv(OUT / 'sector_watch_summary.csv', index=False)
for col in ['rally_return', 'drawdown', 'reference_distance']:
    wt[col] = wt[col].map(lambda v: f'{v:.2%}')
wt = wt.rename(columns={'name': '板块', 'sector_code': '代码', 'confirmed': '21日形态确认', 'rally_return': '本轮涨幅', 'drawdown': '距峰值回撤', 'reference_distance': '距参照位'})
wt['21日形态确认'] = wt['21日形态确认'].map({True: '是', False: '否'})
wt['是否21日首次'] = wt['是否21日首次'].map({True: '是', False: '否'})
followup = pd.read_csv(OUT / 'previous_candidates_followup.csv')
ft = followup[['name', 'ts_code', 'close_20261020', 'close_20261021', 'pct_chg_20261021']].copy()
ft.columns = ['20日候选', '代码', '20日收盘', '21日收盘', '21日涨跌幅']
ft['21日涨跌幅'] = ft['21日涨跌幅'].map(lambda v: f'{v:+.2f}%')
font = FontProperties(fname=str(ROOT / 'fonts/SimHei.ttf'))
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(followup.name, followup.pct_chg_20261021, color=['#b74a48' if v > 0 else '#378966' for v in followup.pct_chg_20261021])
ax.set_xticks(range(len(followup)), followup.name, fontproperties=font)
ax.set_title('20日候选的21日收盘涨跌幅（历史跟踪，非交易收益）', fontproperties=font)
ax.set_ylabel('涨跌幅（%）', fontproperties=font)
ax.axhline(0, color='#777', linewidth=.8)
for i, v in enumerate(followup.pct_chg_20261021):
    ax.text(i, v + (.12 if v >= 0 else -.12), f'{v:+.2f}%', ha='center', va='bottom' if v >= 0 else 'top')
ax.set_ylim(min(-1, followup.pct_chg_20261021.min() - .7), followup.pct_chg_20261021.max() + 1)
fig.tight_layout()
fig.savefig(OUT / 'previous_candidates_followup.png', dpi=150)
plt.close(fig)
if len(candidates):
    ct = candidates[['sector_name', 'name', 'ts_code', 'close', 'model_score']].to_markdown(index=False)
else:
    ct = '**本次没有新个股候选。** 当前策略只对当日去重后的首次确认执行个股筛选；无新事件时，个股评分流程按规则跳过，不是个股评分全部失败。'
headline = f"新首次确认{summary['new_first_confirmations']}个，进入个股筛选{summary['sectors_entering_stock_filter']}个板块，新个股候选{summary['stock_candidates']}只。"
report = f'''# 21日收盘数据与模型结果：22日开盘前

**已将A股数据更新至2026年10月21日收盘，完成冻结、板块构建和当前策略扫描。{headline}**

年度板块模型继续暂停过滤，仅保留诊断；不使用旧随机分组模型评分决定资格。数据更新与数据集构建使用原封装程序，只增加本次日期配置，原工程代码、模型权重及策略规则哈希未变。

## 数据完整性

本次新增1个交易日、{q['daily_rows']:,}条有效股票日线。行情、复权、日度指标和资金流覆盖本次全部有交易股票；关键字段无缺失，重复键、OHLC异常、复权不一致均为0，成交量、成交额与原始数据按既有单位换算后一致。

{pd.read_csv(OUT / 'api_coverage.csv').to_markdown(index=False)}

上市主档5,578条 = 5,568条有效行情 + 8只全日停牌 + 2条此前已发现的过期上市状态（信达、东兴）。停牌名单已通过既有接口按21日重新查询，详见[无行情原因](missing_daily_reconciled.csv)。两家终止上市依据沿用[上次核验报告]({ROOT}/alpha_lab/outputs/datahub_model_check/20261021T064829_488127Z/REPORT.html)。

更新前后20日及以前全部2026日线逐行一致；更新产物哈希和冻结快照校验通过。2026分区实测{read_json(OUT / 'update_verification.json')['new_2026_rows']:,}行。

新板块数据含{ds['latest_valid_sectors']}个有效板块，当日成员归属覆盖{ds['latest_assignment']:.2%}。既有行业历史完整性/PIT限制仍在；美股、基准指数和全仓库名称历史未在本次同步。这些数据不参与本次信号生成，不能把本次A股更新描述成全仓库已完整。

## 模型结果与事件有效期

| 环节 | 21日结果 |
|---|---:|
| 有效板块 | {ss['valid_sectors']} |
| 观察池 | {ss['watch_sectors']} |
| 触发层 | {ss['trigger_sectors']} |
| 当日形态确认 | {summary['shape_confirmed_today']} |
| 去重后新首次确认 | {summary['new_first_confirmations']} |
| 进入个股筛选的板块 | {summary['sectors_entering_stock_filter']} |
| 本次评分个股 | {summary['stocks_scored']} |
| 新候选 | {summary['stock_candidates']} |

{wt.to_markdown(index=False)}

当日满足形态与新的首次确认是两件事。养殖业20日事件的原入场有效日是21日，不顺延至22日；酒店餐饮旧事件也不重开。暂停年度板块判断没有取消事件去重或延长入场窗口。

与20日相比：养殖业、医疗服务退出观察层，航海装备Ⅱ进入观察层。养殖业21日反弹后距本轮峰值回撤为{breeding.drawdown:.2%}，未达到原规则至少10%的回撤要求，当前回落门槛未过。航海装备Ⅱ仍缺近期跌破支撑和下跌减速两个条件。[状态变化明细](state_changes.csv)

{ct}

本次是21日的正式当前策略扫描，记录类型 `current_policy_scan`，对应下一交易日22日；不是按新规则回填旧信号。模型报告不等同于订单或成交，未执行任何交易。

## 20日候选后续跟踪

以下保留20日的候选身份，仅列21日价格变化；不是22日新推荐，也没有按21日涨跌重新排序。涨跌幅采用行情字段，不是实际买入收益或策略回测收益。

{ft.to_markdown(index=False, floatfmt='.2f')}

![历史候选跟踪](previous_candidates_followup.png)

## 产物与复现

- [当前模型报告]({model}/REPORT.html)
- [当前候选CSV]({model}/stock_candidates.csv)
- [首次确认事件]({model}/first_confirmations.csv)
- [完整板块状态]({scan}/sector_states_today.csv)
- [本次字段覆盖](field_coverage.csv)、[更新验证](update_verification.json)、[保护文件核验](verification.json)
- [更新日志](update.log)、[板块构建日志](build_dataset.log)、[板块扫描日志](scan_sectors.log)、[当前策略日志](scan_stocks.log)

行情快照：`{um['release']}`；内容ID：`{um['content_id']}`。

```bash
python alpha_lab/experiments/sector_live_v1/update_market.py --config alpha_lab/experiments/sector_live_v1/config_20261021_eod.yaml
python datahub/scripts/frozen_research.py run --release {um['release']} alpha_lab/scripts/build_sector_dataset.py --config alpha_lab/experiments/sector_live_v1/dataset_config_20261021_eod.yaml
python datahub/scripts/frozen_research.py run --release {um['release']} alpha_lab/experiments/sector_live_v1/scan_sectors.py --dataset {dataset} --config alpha_lab/experiments/sector_live_v1/config_20261021_eod.yaml
python datahub/scripts/frozen_research.py run --release {um['release']} alpha_lab/experiments/sector_live_v1/scan_stocks.py --scan {scan}
```

更新已完成，不应再次执行首条增量下载；复现模型可以直接使用已冻结快照及后续命令。未重新训练模型、修改筛选门槛或回填历史事件。
'''
(OUT / 'REPORT.md').write_text(report)
(OUT / 'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>21日收盘模型报告</title><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#243647}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5df;padding:7px}th{background:#edf3f7}img{max-width:100%}pre{background:#edf3f7;padding:15px;overflow:auto}</style><body>'+markdown.markdown(report,extensions=['tables','fenced_code'])+'</body></html>')
verification = dict(status='passed', protected_files_unchanged=True, artifact_hashes_verified=True, no_annual_gate=True, no_legacy_random_gate=True, model_asof='2026-10-21', model_summary=summary)
(OUT / 'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
(OUT / 'pipeline.json').write_text(json.dumps(dict(update=str(update), dataset=str(dataset), scan=str(scan), model=str(model), finished_at=datetime.now(timezone.utc).isoformat()),ensure_ascii=False,indent=2))
(OUT / 'manifest.json').write_text(json.dumps(dict(status='complete', asof='2026-10-21', artifacts={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}),ensure_ascii=False,indent=2))
(OUT.parent / 'LATEST').write_text(str(OUT.relative_to(ROOT)) + '\n')
print('REPORT', OUT / 'REPORT.html')
print('SUMMARY', json.dumps(summary, ensure_ascii=False))
