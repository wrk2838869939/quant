"""收益预测和风险预测的年度误差改善，不展示为策略收益。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from research_exit_history import ROOT, save, digest

run = Path(sys.argv[1]).resolve()
out = run / 'figures'
out.mkdir()
font = ROOT.parent / 'fonts/SimHei.ttf'
fontManager.addfont(str(font))
plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus'] = False
yearly = pd.read_csv(run / 'analysis/yearly_metrics.csv')
controls = pd.read_csv(run / 'controls/metrics.csv')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ret = yearly[yearly.cohort.eq('shadow') & yearly.target.eq('ret') & yearly.horizon.eq(60)]
axes[0].bar(ret.year.astype(str), ret.r2 * 100, color=np.where(ret.r2 > 0, '#258577', '#c35e50'))
axes[0].set_title('60日终点收益：九特征模型')
for model, label, color in [('nine_features', '九特征', '#b7665f'), ('volatility_drawdown', '波动率＋回撤（事后对照）', '#287fad')]:
    g = controls[controls.cohort.eq('shadow') & controls.model.eq(model) & ~controls.period.str.contains('_')]
    axes[1].plot(g.period, g.r2 * 100, marker='o', label=label, color=color)
axes[1].set_title('60日最大浮亏：复杂模型与简单对照')
axes[1].legend(fontsize=9)
for ax in axes:
    ax.axhline(0, color='gray', lw=.8)
    ax.grid(axis='y', alpha=.2)
    ax.set_ylabel('相对历史均值的预测误差改善（%）')
    ax.tick_params(axis='x', rotation=45)
fig.suptitle('历史候选退出事件｜逐年只用成熟标签训练｜不是组合收益\n2018—2026研究样本；2026仅包含完整后续行情的事件')
fig.tight_layout()
fig.savefig(out / 'annual_prediction_evidence.png', dpi=160)
fig.savefig(out / 'annual_prediction_evidence.pdf')
plt.close(fig)
save(out / 'completion.json', dict(status='complete', source_sha256=digest(__file__),
    artifacts={p.name: digest(p) for p in out.iterdir()}))
