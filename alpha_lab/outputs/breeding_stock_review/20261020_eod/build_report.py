"""养殖业首次形态确认后的独立个股观察报告，只读取既有模型结果。"""
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
import yaml
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path('/root/autodl-tmp')
OUT = Path(__file__).resolve().parent
MAIN = ROOT / 'alpha_lab/outputs/strategy_2026_opportunity_scan/20261020_eod'
summary = json.loads((OUT / 'summary.json').read_text())
allrows = pd.read_csv(OUT / 'all_members_final.csv')
selected = pd.read_csv(OUT / 'observation_candidates.csv')
event = pd.read_csv(MAIN / 'today_first_confirmations_asof_model.csv').query("sector_code == '801017.SI'").iloc[0]
plan = yaml.safe_load((ROOT / 'alpha_lab/configs/strategy_2026_sector_stock_plan.yaml').read_text())
stock_run = ROOT / plan['source']['stock_run']
font = FontProperties(fname=str(ROOT / 'fonts/SimHei.ttf'))
plt.rcParams['axes.unicode_minus'] = False

def table(frame):
    """展示数值保留适当精度，原始精度另存 CSV。"""
    if frame.empty:
        return '无。'
    return frame.to_markdown(index=False)

def display(frame, details=False):
    cols = ['name', 'ts_code', 'close', 'day_return', 'ret5', 'ret20', 'volatility', 'model_score']
    labels = ['股票', '代码', '收盘价', '当日涨跌', '5日涨跌', '20日涨跌', '20日日波动率', '排序分数']
    if details:
        cols += ['qualified_fresh', 'price', 'reason']
        labels += ['资格通过', '价格形态通过', '说明']
    result = frame[cols].copy()
    for col in ['day_return', 'ret5', 'ret20', 'volatility']:
        result[col] = result[col].map(lambda v: f'{v:.2%}')
    result['model_score'] = result.model_score.map(lambda v: f'{v:.6f}' if pd.notna(v) else '未评分')
    result.columns = labels
    return result

# 保留每个股票的可解释贡献：与原预测点积严格一致。
features = pd.read_parquet(OUT / 'cross_section_features.parquet')
weights = np.load(stock_run / 'models/2026.npz')
names = weights['features'].tolist()
contributions = features[['ts_code']].copy()
for i, col in enumerate(names):
    contributions[col] = features[col] * weights['coef'][i]
contributions['sum'] = contributions[names].sum(axis=1)
scores = allrows.set_index('ts_code').reindex(contributions.ts_code).model_score.to_numpy()
np.testing.assert_allclose(contributions['sum'], scores, atol=1e-12)
contributions.to_csv(OUT / 'score_contributions.csv', index=False)

ranked = allrows[allrows.qualified_fresh].sort_values('model_score')
fig, ax = plt.subplots(figsize=(11, 8))
colors = ['#2a8060' if r.observation_candidate else '#dfad54' if r.price else '#b4bec7' for r in ranked.itertuples()]
ax.barh(range(len(ranked)), ranked.model_score, color=colors)
ax.set_yticks(range(len(ranked)), ranked.name, fontproperties=font)
ax.axvline(0, color='#444', lw=.8)
ax.set_title('养殖业个股排序：绿=观察入选，黄=形态通过，灰=形态未过', fontproperties=font)
ax.set_xlabel('2026年度固定模型分数；不是胜率或收益承诺', fontproperties=font)
ax.grid(axis='x', alpha=.2)
fig.tight_layout()
fig.savefig(OUT / 'stock_scores.png', dpi=160)
plt.close(fig)

raw = pd.read_parquet(OUT / 'stock_history.parquet')
if len(selected):
    fig, axs = plt.subplots(int(np.ceil(len(selected) / 2)), 2, figsize=(12, 3.1 * int(np.ceil(len(selected) / 2))), squeeze=False)
    for ax, r in zip(axs.flat, selected.itertuples()):
        g = raw[raw.ts_code.eq(r.ts_code)].sort_values('date').tail(40)
        ax.plot(g.date, g.hfq_close / g.hfq_close.iloc[0] * 100, color='#2a8060')
        ax.set_title(f'{r.name} {r.ts_code} | 分数 {r.model_score:.6f}', fontproperties=font)
        ax.grid(alpha=.2)
        ax.tick_params(axis='x', rotation=25, labelsize=8)
    for ax in list(axs.flat)[len(selected):]:
        ax.set_visible(False)
    fig.suptitle('独立观察股票：近40交易日后复权价格，起点=100；截至10月20日', fontproperties=font)
    fig.tight_layout()
    fig.savefig(OUT / 'candidate_prices.png', dpi=160)
    plt.close(fig)

shape = allrows[['name', 'ts_code', 'early_return', 'late_return', 'slowdown', 'trend', 'expansion', 'contraction']].copy()
for col in ['early_return', 'late_return']:
    shape[col] = shape[col].map(lambda v: f'{v:.2%}')
shape.columns = ['股票', '代码', '5日窗口前3日涨跌', '近2日涨跌', '下跌减速比', '趋势条件', '换手扩张比', '换手收缩比']

text = f'''# 养殖业个股观察报告：2026-10-20收盘

**养殖业已完成首次形态确认，但年度板块模型未通过。此前正式扫描在板块层停止，未生成养殖业个股报告；本报告补充个股独立观察，正式策略候选仍为0。**

## 板块确认与模型判断

| 项目 | 结果 |
|---|---|
| 板块 | 养殖业 801017.SI |
| 首次确认日 | 2026-10-20，代表窗口 {int(event.window)} |
| 峰值日期 | {event.peak_date} |
| 峰值后回撤 | {event.drawdown:.2%} |
| 当日板块涨跌 | {event.index_return:.2%} |
| 年度模型分数 | {event.score:.9f} |
| 原有通过阈值 | {event.threshold:.9f} |
| 年度板块模型通过 | 否 |

近5日价格触发、近5日支撑条件及当前减速等形态条件已经齐备。“形态确认”与“年度模型通过”是两个关口，不能互换。该事件的次日窗口是10月21日，但本次没有通过板块关口的入场信号。

## 个股独立观察结果

本次覆盖{summary['members']}只板块成员，{summary['qualified']}只通过资格核对，{summary['price_pass']}只通过价格形态；在形态池内按原有正分数、前5名规则，得到{len(selected)}只独立观察股票。

{table(display(selected))}

这些分数仅用于当日板块内的相对排序。该个股模型原先基于板块模型通过的历史事件研究；将它用于板块模型拒绝的事件，排序有效性尚未验证。表中顺序不是新的买入建议。

![全部成员排序](stock_scores.png)

{'![观察股票走势](candidate_prices.png)' if len(selected) else ''}

## 全部成员及未入选原因

{table(display(allrows.sort_values('model_score', ascending=False, na_position='last'), True))}

## 原有个股筛选口径

1. 资格：截至20日已上市且未退市、名称/ST筛查通过、当日价格和成交量为正。本次重新查询全体成员名称历史，并按公告日和生效日截断到20日。
2. 价格形态：最近5日窗口的前3日累计跌幅至少1%；近2日平均负收益幅度/前3日平均负收益幅度不超过0.65；近2日累计涨幅不超过2%；当日跌幅不超过2%。减速比只计算负收益幅度，不是两个区间累计收益之比。
3. 截面排序：沿用2026年固定权重，使用强弱、波动、换手扩张/收缩、减速和趋势/形态等既有特征；在资格池内排名、去均值，然后在价格形态通过的正分数股票中取前5名。没有重新训练或调参。
4. 趋势、换手扩张和换手收缩参与评分与诊断，但当前方案不将其设为独立硬门槛。价格形态通过也不代表已经止跌。

{table(shape.round(6))}

## 数据、时序与核验

- 行情快照：`datahub/releases/selection_live_20261020_eod`，只读取截至20日的数据。
- 板块模型训练标签最晚2025-11-18；个股模型训练标签最晚2025-11-19。使用既有年度权重，不使用扫描器附带的旧随机划分模型分数。
- 最新名称历史核对完成，资格与原快照是否一致：{summary['qualification_unchanged']}。成员归属沿用20日冻结板块数据；历史行业成员PIT尚未完整认证。
- 原数据工程代码、策略方案和模型权重哈希未改变；模型贡献求和与评分一致。报告不调整正式扫描结果，也未提交交易指令。
- 京基智农等股票可能在较大跌幅后仍满足减速条件，请结合表内5日/20日涨跌和日波动率理解排名；模型高分不等于低风险。

## 可复现文件

[全部成员明细 CSV](all_members_final.csv) · [独立观察名单 CSV](observation_candidates.csv) · [评分贡献 CSV](score_contributions.csv) · [核验记录](verification.json) · [20日原扫描报告](../../strategy_2026_opportunity_scan/20261020_eod/REPORT.html)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python datahub/scripts/frozen_research.py run --release datahub/releases/selection_live_20261020_eod alpha_lab/outputs/breeding_stock_review/20261020_eod/review.py --scan alpha_lab/outputs/sector_live_scan/20261020T113957_683276Z
python alpha_lab/outputs/breeding_stock_review/20261020_eod/fetch_names.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python datahub/scripts/frozen_research.py run --release datahub/releases/selection_live_20261020_eod alpha_lab/outputs/breeding_stock_review/20261020_eod/finalize.py --scan alpha_lab/outputs/sector_live_scan/20261020T113957_683276Z
python alpha_lab/outputs/breeding_stock_review/20261020_eod/build_report.py
```
'''
(OUT / 'REPORT.md').write_text(text)
style = 'body{font:16px/1.7 sans-serif;max-width:1380px;margin:32px auto;padding:0 24px;color:#263440}table{border-collapse:collapse;font-size:14px;width:100%}td,th{border:1px solid #ccd5dc;padding:7px}th{background:#eef3f6}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f7;padding:12px}a{color:#246b97}'
(OUT / 'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>养殖业个股观察20261020</title><style>' + style + '</style><body>' + markdown.markdown(text, extensions=['tables', 'fenced_code']) + '</body></html>')
artifacts = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
(OUT / 'manifest.json').write_text(json.dumps(dict(status='complete', summary=summary, artifacts=artifacts), ensure_ascii=False, indent=2))
nav = ROOT / 'alpha_lab/docs/current_strategy_20261013/BREEDING_STOCK_REVIEW_20261020.md'
nav.write_text('# 养殖业个股独立观察：2026-10-20\n\n形态首次确认，年度板块模型未通过；正式候选为0。\n\n[详细报告与图表](../../outputs/breeding_stock_review/20261020_eod/REPORT.html) · [全部成员CSV](../../outputs/breeding_stock_review/20261020_eod/all_members_final.csv)\n')
print(table(display(selected)))
print(OUT / 'REPORT.html')
