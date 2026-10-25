"""固定年度模型，用未来20日相对确认收盘的盈亏天数差评价。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path('/root/autodl-tmp')
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'alpha_lab/outputs/sector_classification_audit/20261021'
RUN = ROOT / 'alpha_lab/outputs/sector_asof_replay_v1/20261013T072855_087132Z'
DATA = ROOT / 'alpha_lab/data/sector_live_datasets/sw_l2_v1_20261020T113446_268600Z'
HORIZON = 20
REPETITIONS = 2000
SEED = 2026102102

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for folder in [RUN, PREVIOUS]:
    m = json.loads((folder / 'manifest.json').read_text())
    for name, value in m['artifacts'].items():
        assert digest(folder / name) == value, name
pred = pd.read_parquet(RUN / 'predictions_BEFORE_EVALUATION.parquet')
old = pd.read_csv(PREVIOUS / 'event_predictions_and_labels.csv')
assert pred.selected.eq(pred.score.ge(pred.threshold)).all()
panel = pd.read_parquet(DATA / 'daily.parquet', columns=['date', 'sector_code', 'index_valid', 'index_close', 'segment_id'])
calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
groups = {code: frame.set_index('date') for code, frame in panel.groupby('sector_code')}
rows, paths = [], []
for event in pred.itertuples():
    i = calendar.get_loc(event.date)
    dates = calendar[i:i + HORIZON + 1]
    path = groups[event.sector_code].reindex(dates)
    mature = len(dates) == HORIZON + 1
    complete = mature and path.index_valid.eq(True).all() and path.segment_id.eq(path.segment_id.iloc[0]).all() and path.index_close.gt(0).all()
    row = dict(event_id=event.event_id, complete=bool(complete), positive_days=np.nan, negative_days=np.nan,
               flat_days=np.nan, profit_days_score=np.nan, terminal_return=np.nan,
               end_date=dates[-1] if mature else pd.NaT,
               missing_reason='' if complete else '未满20交易日' if not mature else '报价缺失或指数连续段中断')
    if complete:
        prices = path.index_close.to_numpy(float)
        returns = prices[1:] / prices[0] - 1
        signs = np.sign(returns)
        pos, neg, flat = int((signs > 0).sum()), int((signs < 0).sum()), int((signs == 0).sum())
        value = (pos - neg) / HORIZON
        assert pos + neg + flat == HORIZON and -1 <= value <= 1
        np.testing.assert_allclose(value, signs.mean(), atol=1e-15)
        row.update(positive_days=pos, negative_days=neg, flat_days=flat, profit_days_score=value, terminal_return=returns[-1])
        paths.extend(dict(event_id=event.event_id, horizon=t, date=d, base_close=prices[0], close=price,
                          return_vs_confirmation=ret, sign=int(sign))
                     for t, (d, price, ret, sign) in enumerate(zip(dates[1:], prices[1:], returns, signs), 1))
    rows.append(row)
data = pred.merge(pd.DataFrame(rows), on='event_id', validate='one_to_one')
data['year'] = data.date.dt.year
compare = data.merge(old[['event_id', 'complete', 'target']], on='event_id', suffixes=('', '_old'), validate='one_to_one')
assert compare.complete.eq(compare.complete_old).all()
np.testing.assert_allclose(compare.loc[compare.complete, 'terminal_return'], compare.loc[compare.complete, 'target'], atol=1e-12)
known = data[data.complete].copy()
date_positions = calendar.get_indexer(pd.DatetimeIndex(sorted(known.date.unique())))
cluster_map = dict(zip(sorted(known.date.unique()), np.cumsum(np.r_[0, np.diff(date_positions) > HORIZON])))
known['cluster'] = known.date.map(cluster_map)
data.to_csv(OUT / 'event_metrics.csv', index=False)
pd.DataFrame(paths).to_csv(OUT / 'daily_paths.csv', index=False)
data[~data.complete].to_csv(OUT / 'excluded.csv', index=False)

def metrics(frame):
    """事件等权，对比通过、拒绝及未经过模型筛选的全池。"""
    passed, rejected = frame[frame.selected], frame[~frame.selected]
    return dict(n=len(frame), passed_n=len(passed), rejected_n=len(rejected),
                all_mean=frame.profit_days_score.mean(), passed_mean=passed.profit_days_score.mean(),
                rejected_mean=rejected.profit_days_score.mean(),
                difference=passed.profit_days_score.mean() - rejected.profit_days_score.mean(),
                lift_vs_all=passed.profit_days_score.mean() - frame.profit_days_score.mean(),
                passed_median=passed.profit_days_score.median(), rejected_median=rejected.profit_days_score.median(),
                passed_positive_days=passed.positive_days.mean(), rejected_positive_days=rejected.positive_days.mean(),
                passed_positive_fraction=passed.profit_days_score.gt(0).mean(),
                rejected_positive_fraction=rejected.profit_days_score.gt(0).mean(),
                passed_terminal_mean=passed.terminal_return.mean(), rejected_terminal_mean=rejected.terminal_return.mean())

periods = [('2019–2026', known), ('2019–2025', known[known.year.lt(2026)])] + [(str(y), g) for y, g in known.groupby('year')]
stats = pd.DataFrame([dict(period=label, **metrics(frame)) for label, frame in periods])
stats.to_csv(OUT / 'metrics_by_year.csv', index=False)

# 不把同日板块当成独立试验；整体和历史期分别整簇重采样。
rng = np.random.default_rng(SEED)
intervals = []
for label, frame in periods[:2]:
    cluster_frames = [g for _, g in frame.groupby('cluster')]
    samples = []
    for _ in range(REPETITIONS):
        chosen = rng.integers(0, len(cluster_frames), len(cluster_frames))
        samples.append(metrics(pd.concat([cluster_frames[i] for i in chosen], ignore_index=True)))
    boot = pd.DataFrame(samples)
    point = metrics(frame)
    for key in ['passed_mean', 'rejected_mean', 'difference', 'lift_vs_all']:
        finite = boot[key].dropna()
        intervals.append(dict(period=label, metric=key, estimate=point[key], low=finite.quantile(.025),
                              high=finite.quantile(.975), clusters=len(cluster_frames), replicates=len(finite)))
intervals = pd.DataFrame(intervals)
intervals.to_csv(OUT / 'cluster_intervals.csv', index=False)

# 未知标签仅给[-1,1]的数学边界，不用部分路径或零值填补。
selected_total = int(data.selected.sum()); rejected_total = int((~data.selected).sum())
missing_pass = int(data[~data.complete].selected.sum()); missing_reject = int((~data[~data.complete].selected).sum())
passed_sum = known.loc[known.selected, 'profit_days_score'].sum()
rejected_sum = known.loc[~known.selected, 'profit_days_score'].sum()
pl, ph = (passed_sum - missing_pass) / selected_total, (passed_sum + missing_pass) / selected_total
rl, rh = (rejected_sum - missing_reject) / rejected_total, (rejected_sum + missing_reject) / rejected_total
bounds = pd.DataFrame([dict(metric='passed_mean', low=pl, high=ph), dict(metric='rejected_mean', low=rl, high=rh),
                       dict(metric='difference', low=pl-rh, high=ph-rl)])
bounds.to_csv(OUT / 'missing_bounds.csv', index=False)

font = FontProperties(fname=str(ROOT / 'fonts/SimHei.ttf'))
plt.rcParams['axes.unicode_minus'] = False
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
years = stats.iloc[2:]
x = np.arange(len(years))
axs[0].bar(x-.18, years.passed_mean, width=.36, color='#357b72', label='模型通过')
axs[0].bar(x+.18, years.rejected_mean, width=.36, color='#be8059', label='模型拒绝')
axs[0].set_xticks(x, [f'{r.period}\n{r.passed_n}/{r.rejected_n}' for r in years.itertuples()])
axs[0].set_ylim(-1,1); axs[0].axhline(0,color='#555',lw=.8)
axs[0].legend(prop=font); axs[0].set_title('逐年平均指标；年份下方为通过/拒绝样本数',fontproperties=font)
axs[0].grid(axis='y',alpha=.2)
hist_data = [known.loc[known.selected, 'profit_days_score'], known.loc[~known.selected, 'profit_days_score']]
axs[1].hist(hist_data, bins=np.linspace(-1.05,1.05,22), weights=[np.ones(len(g))/len(g) for g in hist_data],
            label=['模型通过', '模型拒绝'], color=['#357b72','#be8059'])
axs[1].set_title('全部完整事件：指标分布（组内占比）', fontproperties=font)
axs[1].legend(prop=font); axs[1].grid(axis='y',alpha=.2)
fig.tight_layout(); fig.savefig(OUT / 'profit_days.png', dpi=160); plt.close(fig)

show = stats[['period','n','passed_n','rejected_n','passed_mean','rejected_mean','difference','passed_positive_days','rejected_positive_days']].copy()
for col in ['passed_mean','rejected_mean','difference']:show[col]=show[col].map(lambda x:f'{x:+.3f}')
for col in ['passed_positive_days','rejected_positive_days']:show[col]=show[col].map(lambda x:f'{x:.2f}')
show.columns=['期间','完整事件','通过','拒绝','通过组均值','拒绝组均值','通过减拒绝','通过组平均盈利天数','拒绝组平均盈利天数']
v = stats.iloc[0]
endpoint = stats[['period','passed_positive_fraction','rejected_positive_fraction','passed_terminal_mean','rejected_terminal_mean']].copy()
for col in endpoint.columns[1:]:endpoint[col]=endpoint[col].map(lambda x:f'{x:.2%}')
endpoint.columns=['期间','通过组指标>0占比','拒绝组指标>0占比','通过组20日终点平均收益','拒绝组20日终点平均收益']
ci_show=intervals.copy()
for col in ['estimate','low','high']:ci_show[col]=ci_show[col].map(lambda x:f'{x:+.3f}')
text=f'''# 年度板块模型：未来20日盈亏天数差评价

评价日期2026-10-21，行情截至2026-10-20。用户已确认：正负收益相对确认日收盘计算，而非每天相对前一天涨跌。本次更换评价指标，沿用已保存年度模型分数和原有门槛，不重新训练，不修改数据构建与正式策略。

## 指标定义

令P0为首次确认日板块收盘，Pt为之后第t个交易日的板块收盘（t=1…20）。

`指标 S = [Σ 1(Pt/P0−1 > 0) − Σ 1(Pt/P0−1 < 0)] / 20`

确认当天不计入；平价日记0，分母固定20。范围[-1,1]，15天盈利、5天亏损得到0.5；没有平价日时，S=2×盈利天数占比−1。0意味着盈亏天数相等，并不意味着终点收益为0。

这个指标衡量相对信号日收盘的盈利持续性，能区分“途中大多盈利”和“仅终点盈利”。它不度量盈亏幅度：19天微赚、最后一天大跌仍可能得到0.9，因此保留终点收益对照。该收盘价为观察基准，不是假定能够在信号生成后按该价成交。

## 原模型在新指标下的表现

{show.to_markdown(index=False)}

![逐年均值与分布](profit_days.png)

事件等权，完整标签仍为226条（通过100、拒绝126）。通过组指标均值{v.passed_mean:+.3f}，拒绝组{v.rejected_mean:+.3f}，差值{v.difference:+.3f}。相对未经过模型筛选的完整事件池，通过组提升{v.lift_vs_all:+.3f}。平均盈利天数分别为{v.passed_positive_days:.2f}和{v.rejected_positive_days:.2f}天，差{v.passed_positive_days-v.rejected_positive_days:+.2f}天。

## 与终点收益并列

{endpoint.to_markdown(index=False)}

指标>0表示盈利天数多于亏损天数。此处均为未扣费用的自算板块指数观察，不是交易胜率或账户收益。

## 不确定性与缺失标签

同日及临近事件相关，按超过20交易日的日期间隔分行情簇，整簇重采样2000次。以下为描述性95%区间：

{ci_show.to_markdown(index=False)}

原预测256条，30条不能评价：29条报价路径或连续段不完整，1条酒店餐饮未满20交易日。2025年37条中25条缺完整标签，不能以剩余12条代表全年。未知路径不取部分天数、不缩分母、不填零。假设未知完整指标可在[-1,1]中任取，完整256条的数学边界如下（不是置信区间）：

{bounds.round(4).to_markdown(index=False)}

2026年仅16条成熟事件；当年较好表现不等于稳定性已验证。整体差异还可能受到各年度通过比例与市场环境影响，应同时查看逐年表。

## 评价与重新训练的区别

原模型学习的是20日终点收益，本次回答的是“这个旧模型是否也能筛出盈利持续性更好的事件”。本次没有把新指标改成训练标签。若以后用它训练新模型，需要重新执行年度时间推进、历史内层选参和完整20日标签成熟隔离；不能据这份全历史结果调整门槛后，再把同一批结果当独立测试。

此次口径是看到旧评价后的新增研究定义，需保留这一研究历史。模型文件仍是10月13日按年初信息边界重建，供应商历史版本和成员公告时点的缺口仍在。10月20日养殖业尚未成熟，不纳入该评价。

## 核验与文件

全部完整事件终点收益与上一份评价逐值一致，完整/缺失集合完全一致；逐事件正/负/平天数之和均为20。原年度研究与前次评价清单哈希通过核对，预测通过集合未改变。

[逐事件指标](event_metrics.csv) · [逐日路径与符号](daily_paths.csv) · [逐年数据](metrics_by_year.csv) · [区间](cluster_intervals.csv) · [缺失样本](excluded.csv)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python alpha_lab/outputs/sector_profit_days_audit/20261021/evaluate.py
```
'''
(OUT/'REPORT.md').write_text(text)
style='body{font:16px/1.7 sans-serif;max-width:1320px;margin:30px auto;padding:0 24px}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccd5dc;padding:7px}img{max-width:100%}pre{white-space:pre-wrap}'
(OUT/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>板块盈利天数评价</title><style>'+style+'</style><body>'+markdown.markdown(text,extensions=['tables','fenced_code'])+'</body></html>')
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',scored=len(data),complete=len(known),paths=len(paths),terminal_returns_matched=True,completeness_unchanged=True,no_retraining=True,no_threshold_changes=True,benchmark='confirmation_close',horizon=HORIZON),indent=2))
(OUT/'manifest.json').write_text(json.dumps(dict(status='complete',artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}),indent=2))
print(show.to_string(index=False));print(intervals.to_string(index=False));print(endpoint.to_string(index=False))
