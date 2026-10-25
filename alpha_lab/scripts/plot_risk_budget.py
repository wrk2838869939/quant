"""展示同起点真实资金账本净值、回撤与现金，而非预测代理。"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from research_risk_budget import ROOT, save, digest

run = Path(sys.argv[1]).resolve()
out = run / 'figures'
out.mkdir()
font = ROOT.parent / 'fonts/SimHei.ttf'
fontManager.addfont(str(font))
plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for policy, label, color in [('baseline', '原基线', '#666666'), ('uniform', '统一降仓', '#c58b35'),
                             ('full_risk', '新旧仓风险折扣', '#217eab')]:
    d = pd.read_csv(run / 'execution' / f'cold_calendar_01_{policy}' / 'daily.csv', index_col='date', parse_dates=True)
    axes[0].plot(d.index, d.equity/50000, label=label, color=color)
    axes[1].plot(d.index, (d.equity/d.equity.cummax().clip(lower=50000)-1)*100, color=color)
    axes[2].plot(d.index, d.cash/d.equity*100, color=color)
axes[0].set_ylabel('账户净值')
axes[1].set_ylabel('账户回撤（%）')
axes[2].set_ylabel('现金比例（%）')
axes[0].legend(loc='upper left')
for ax in axes:
    ax.grid(alpha=.2)
    ax.axvspan(pd.Timestamp('2026-06-01'), pd.Timestamp('2026-06-30'), color='#da8678', alpha=.15)
fig.suptitle('5万元｜2024年空仓启动｜原月初复核｜最多5股、100股整数买单\n已计费用及模型分红税；阴影为2026年6月；已观察历史回放，不是实盘')
fig.tight_layout()
fig.savefig(out / 'monthly_account_risk_budget.png', dpi=150)
fig.savefig(out / 'monthly_account_risk_budget.pdf')
plt.close(fig)
save(out / 'completion.json', dict(status='complete', source_sha256=digest(__file__),
    artifacts={p.name: digest(p) for p in out.iterdir()}))
