"""固定既有年度预测，复核板块20日标签并补充分类指标。"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path('/root/autodl-tmp')
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'alpha_lab/outputs/sector_asof_replay_v1/20261013T072855_087132Z'
DATA = ROOT / 'alpha_lab/data/sector_live_datasets/sw_l2_v1_20261020T113446_268600Z'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((RUN / 'manifest.json').read_text())
for name, value in manifest['artifacts'].items():
    assert digest(RUN / name) == value, name
pred = pd.read_parquet(RUN / 'predictions_BEFORE_EVALUATION.parquet')
original = pd.read_parquet(RUN / 'research_inputs_WITH_LABELS.parquet')
assert not pred.event_id.duplicated().any()
assert pred.selected.eq(pred.score.ge(pred.threshold)).all()
panel = pd.read_parquet(DATA / 'daily.parquet', columns=['date', 'sector_code', 'index_valid', 'index_close', 'segment_id'])
calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
groups = {code: frame.set_index('date') for code, frame in panel.groupby('sector_code')}
rows = []
for r in pred.itertuples():
    dates = calendar[calendar.get_loc(r.date):calendar.get_loc(r.date) + 21]
    path = groups[r.sector_code].reindex(dates)
    mature = len(dates) == 21
    complete = mature and path.index_valid.eq(True).all() and path.segment_id.eq(path.segment_id.iloc[0]).all() and path.index_close.gt(0).all()
    rows.append(dict(event_id=r.event_id, complete=bool(complete), mature=mature,
                     end_date=dates[-1] if mature else pd.NaT,
                     target=path.index_close.iloc[-1] / path.index_close.iloc[0] - 1 if complete else np.nan,
                     missing_reason='' if complete else ('尚未满20交易日' if not mature else '报价缺失或指数连续段中断')))
labels = pd.DataFrame(rows)
comparison = labels.merge(original[['event_id', 'complete', 'target']], on='event_id', suffixes=('_current', '_old'), validate='one_to_one')
old_complete = comparison[comparison.complete_old]
assert old_complete.complete_current.all()
np.testing.assert_allclose(old_complete.target_current, old_complete.target_old, atol=1e-12, rtol=1e-10)
data = pred.merge(labels, on='event_id', validate='one_to_one')
data['year'] = data.date.dt.year
data.to_csv(OUT / 'event_predictions_and_labels.csv', index=False)
data[~data.complete].to_csv(OUT / 'excluded_labels.csv', index=False)
known = data[data.complete].copy()
known['actual_positive'] = known.target.gt(0)

# 只检查训练信息边界，不重新拟合模型。
fits = json.loads((RUN / 'fits.json').read_text())
for f in fits:
    if f['status'] != 'complete':
        continue
    assert pd.Timestamp(f['train_last_label']) < pd.Timestamp(f['cutoff']) < pd.Timestamp(f['start'])
    assert pd.Timestamp(f['fit_last_label']) < pd.Timestamp(f['inner_cutoff']) < pd.Timestamp(f['validation_start'])
    assert not set(f['train_ids']) & set(pred[pred.date.dt.year.eq(f['year'])].event_id)

def metrics(frame):
    """事件等权；分母为零时不伪造零值。"""
    actual, chosen = frame.actual_positive, frame.selected
    tp = int((actual & chosen).sum()); fp = int((~actual & chosen).sum())
    tn = int((~actual & ~chosen).sum()); fn = int((actual & ~chosen).sum())
    def ratio(a, b):
        return a / b if b else np.nan
    precision = ratio(tp, tp + fp); recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return dict(n=len(frame), passed=int(chosen.sum()), actual_positive=int(actual.sum()), TP=tp, FP=fp, TN=tn, FN=fn,
                accuracy=ratio(tp + tn, len(frame)), precision=precision, recall=recall,
                specificity=specificity, balanced_accuracy=(recall + specificity) / 2,
                f1=ratio(2 * tp, 2 * tp + fp + fn), base_positive_rate=actual.mean(),
                always_reject_accuracy=1 - actual.mean(), precision_lift=precision - actual.mean(),
                rejected_positive_rate=ratio(fn, fn + tn))

stats = []
for name, group in [('2019–2026合计', known), ('2019–2025', known[known.year.lt(2026)])] + [(str(y), g) for y, g in known.groupby('year')]:
    stats.append(dict(period=name, **metrics(group), total_scored=len(data) if name=='2019–2026合计' else len(data[data.year.lt(2026)]) if name=='2019–2025' else len(data[data.year.eq(int(name))])))
results = pd.DataFrame(stats)
results['excluded'] = results.total_scored - results.n
results.to_csv(OUT / 'metrics.csv', index=False)

# 用相隔超过20交易日的事件日期划分行情簇，保留簇内相关性。
unique_dates = pd.DatetimeIndex(sorted(known.date.unique()))
locations = calendar.get_indexer(unique_dates)
clusters = np.cumsum(np.r_[0, np.diff(locations) > 20])
known['cluster'] = known.date.map(dict(zip(unique_dates, clusters)))
known.to_csv(OUT / 'complete_events_with_clusters.csv', index=False)
cluster_frames = [g for _, g in known.groupby('cluster')]
rng = np.random.default_rng(20261021)
bootstrap = []
for _ in range(2000):
    sample = pd.concat([cluster_frames[i] for i in rng.integers(0, len(cluster_frames), size=len(cluster_frames))], ignore_index=True)
    bootstrap.append(metrics(sample))
boot = pd.DataFrame(bootstrap)
ci = []
for metric in ['accuracy', 'precision', 'recall', 'balanced_accuracy', 'precision_lift']:
    valid = boot[metric].dropna()
    ci.append(dict(metric=metric, estimate=results.iloc[0][metric], low=valid.quantile(.025), high=valid.quantile(.975), clusters=len(cluster_frames), finite_replicates=len(valid)))
ci = pd.DataFrame(ci)
ci.to_csv(OUT / 'cluster_intervals.csv', index=False)

font = FontProperties(fname=str(ROOT / 'fonts/SimHei.ttf'))
plt.rcParams['axes.unicode_minus'] = False
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
yearly = results.iloc[2:]
for key, title in [('precision', '精确率：选中后上涨比例'), ('recall', '召回率：上涨事件被选中比例'), ('base_positive_rate', '原事件池上涨比例')]:
    axs[0].plot(yearly.period, yearly[key] * 100, marker='o', label=title)
axs[0].set_ylim(0, 105); axs[0].grid(alpha=.2); axs[0].legend(prop=font, fontsize=9)
axs[0].set_title('逐年分类表现（%）', fontproperties=font)
v = results.iloc[0]
missing = data[~data.complete]
missing_pass = int(missing.selected.sum())
missing_reject = len(missing) - missing_pass
bounds = pd.DataFrame([
    dict(metric='accuracy', low=(v.TP + v.TN) / len(data), high=(v.TP + v.TN + len(missing)) / len(data)),
    dict(metric='precision', low=v.TP / (v.passed + missing_pass), high=(v.TP + missing_pass) / (v.passed + missing_pass)),
    dict(metric='recall', low=v.TP / (v.actual_positive + missing_reject), high=(v.TP + missing_pass) / (v.actual_positive + missing_pass)),
])
bounds.to_csv(OUT / 'missing_label_bounds.csv', index=False)
boundshow = bounds.copy()
for col in ['low', 'high']:
    boundshow[col] = boundshow[col].map(lambda x: f'{x:.2%}')
matrix = np.array([[v.TP, v.FP], [v.FN, v.TN]], dtype=int)
axs[1].imshow(matrix, cmap='Blues')
for (i, j), value in np.ndenumerate(matrix):
    axs[1].text(j, i, str(value), ha='center', va='center', fontsize=19)
axs[1].set_xticks([0, 1], ['实际上涨', '实际不涨'], fontproperties=font)
axs[1].set_yticks([0, 1], ['模型通过', '模型拒绝'], fontproperties=font)
axs[1].set_title('完整标签事件混淆矩阵', fontproperties=font)
fig.tight_layout(); fig.savefig(OUT / 'classification.png', dpi=160); plt.close(fig)

show = results[['period', 'n', 'excluded', 'passed', 'accuracy', 'precision', 'recall', 'base_positive_rate', 'balanced_accuracy']].copy()
for col in ['accuracy', 'precision', 'recall', 'base_positive_rate', 'balanced_accuracy']:
    show[col] = show[col].map(lambda x: f'{x:.2%}')
show.columns = ['期间', '完整标签', '未评价', '模型通过', '准确率', '精确率', '召回率', '原池上涨率', '平衡准确率']
cishow = ci.copy()
for col in ['estimate', 'low', 'high']:
    cishow[col] = cishow[col].map(lambda x: f'{x:.2%}')
text = f'''# 年度板块模型：准确率与召回率补充核验

核验日期2026-10-21；行情截至2026-10-20。既有年度模型报告没有提供这组分类指标，本次用保存的逐年预测补算，不训练、不调门槛、不修改数据工程。

## 评价定义

- 样本：原年度研究保存的256个板块首次确认事件；每年使用只含此前成熟标签的模型。没有把未形成形态的板块日纳入负样本，所以召回率仅覆盖该形态事件池，不代表捕获全市场所有上涨。
- 预测正类：保存的分数≥该年原有阈值（selected）。
- 实际正类：确认日收盘至20交易日后板块收盘收益严格>0；收益等于0归负类。这里是自算板块指数的价格涨跌，没有扣费用，不是可成交收益。
- 准确率=(TP+TN)/全部；精确率=TP/(TP+FP)，即通过事件的上涨比例；召回率=TP/(TP+FN)，即实际上涨事件中被通过的比例；平衡准确率为正负两类召回率均值。
- “实际上涨”是本次事后新增的诊断口径。原模型做收益回归，训练时没有直接优化这些分类指标。

## 结果

{show.to_markdown(index=False)}

![逐年表现和混淆矩阵](classification.png)

合计TP={int(v.TP)}、FP={int(v.FP)}、TN={int(v.TN)}、FN={int(v.FN)}。始终预测上涨的准确率为原池上涨率{v.base_positive_rate:.2%}；始终预测不涨的准确率为{v.always_reject_accuracy:.2%}。因此单看准确率可能误导，需同时观察精确率、召回率、平衡准确率和基准。

模型拒绝事件中，仍有{v.rejected_positive_rate:.2%}后来上涨；拒绝不等于确定会跌。分类指标也不能判断平均盈亏比、成交成本和账户收益。

## 缺失标签的影响

30条未评价样本中，29条因报价缺失或指数连续段中断，1条为尚未成熟的2026-09-17酒店餐饮。尤其2025年37条中25条不可评价，12条完整样本不能代表全年。完整样本过滤可能产生非随机缺失偏差。

未评价事件中模型通过{missing_pass}条、拒绝{missing_reject}条。保留原预测不变，将未知实际涨跌作所有可能赋值，可得到完整256条样本各项指标的数学边界（不是置信区间；各指标端点不要求同时实现）：

{boundshow.to_markdown(index=False)}

即使缺失标签全部按有利于模型的情况补齐，通过事件的上涨比例上界仍为{bounds.loc[bounds.metric.eq('precision'), 'high'].iloc[0]:.2%}。这不修复标签缺失，只说明缺失样本对本组分类结论的最大可能影响。

## 相关性与不确定性

事件等权；同日、临近日期不同板块不是独立样本。按相隔超过20个交易日划分{len(cluster_frames)}个行情簇，整簇重采样2000次，给出描述性95%分位区间：

{cishow.to_markdown(index=False)}

precision_lift为通过事件上涨率减原池上涨率，单位为比例点。区间用于说明样本不确定性，不修复研究家族事后开发偏差。跨年度合计还会混入各年市场和模型保留比例的变化，不能替代逐年比较。

## 数据与时间边界

- 用10月20日板块数据重新计算全部256条目标路径；此前完整标签逐值一致。尚未满20交易日、缺报价或跨连续段的样本单独列出，不按亏损或零收益处理。
- 本报告只评价原保存的256条预测；10月20日养殖业是后续扫描新增事件，尚未成熟，未加入分类统计。
- 2026年模型最晚训练标签为2025-11-18；模型文件实际在10月13日按历史信息边界重建。历史供应商版本与行业公告时点仍有缺口，规则家族本身也由事后研究确定；不能称完全独立的实盘前瞻测试。
- 原研究清单全部产物哈希核对通过；各年训练标签截止与当年事件不相交检查通过；不改动任何模型权重、门槛或正式策略。

## 明细与复现

[逐事件预测和标签](event_predictions_and_labels.csv) · [未评价样本及原因](excluded_labels.csv) · [指标CSV](metrics.csv) · [区间CSV](cluster_intervals.csv)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python alpha_lab/outputs/sector_classification_audit/20261021/evaluate.py
```
'''
(OUT / 'REPORT.md').write_text(text)
style = 'body{font:16px/1.7 sans-serif;max-width:1250px;margin:30px auto;padding:0 22px}table{border-collapse:collapse}td,th{border:1px solid #ccd5dc;padding:7px}img{max-width:100%}pre{white-space:pre-wrap}'
(OUT / 'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>年度板块模型分类核验</title><style>' + style + '</style><body>' + markdown.markdown(text, extensions=['tables', 'fenced_code']) + '</body></html>')
(OUT / 'verification.json').write_text(json.dumps(dict(status='passed', original_artifacts_verified=len(manifest['artifacts']), scored=len(data), complete=len(known), excluded=len(data)-len(known), old_complete_labels_matched=len(old_complete), latest_data=str(calendar[-1].date()), no_retraining=True), indent=2))
(OUT / 'manifest.json').write_text(json.dumps(dict(status='complete', source_run=str(RUN), dataset=str(DATA), artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}), indent=2))
print(show.to_string(index=False));print(ci.to_string(index=False));print(data[~data.complete][['event_id','date','selected','missing_reason']].to_string(index=False))
