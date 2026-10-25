"""跨起点恢复/缓冲的配对收益与回撤变化。"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from research_risk_recovery import ROOT, digest, save

run = Path(sys.argv[1]).resolve()
out = run / 'figures'
out.mkdir()
font = ROOT.parent / 'fonts/SimHei.ttf'
fontManager.addfont(str(font))
plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus'] = False
data = pd.read_csv(run / 'analysis/paired.csv')
fig, axes = plt.subplots(1,2, figsize=(11,5))
for ax, window, title in zip(axes, ['continued','cold'], ['2022连续账户','2024空仓启动']):
    for policy, label, color in [('restore','恢复旧仓','#c96d42'),('grace','整手缓冲','#267eab'),('combined','两者组合','#57966a')]:
        g = data[data.window.eq(window) & data.policy.eq(policy) & data.stress.eq('main')]
        ax.scatter(g.total_delta*100, g.mdd_delta*100, label=label, color=color, s=55)
    ax.axhline(0, color='gray', lw=.8)
    ax.axvline(0, color='gray', lw=.8)
    ax.grid(alpha=.2)
    ax.set_title(title)
    ax.set_xlabel('累计收益变化（百分点）')
    ax.set_ylabel('最大回撤改善（百分点，正值更好）')
    ax.legend(fontsize=9)
fig.suptitle('5万元｜相对上轮风险仓位版｜每个点代表一个复核日历\n右上象限才是收益和回撤同时改善；全部为已观察历史')
fig.tight_layout()
fig.savefig(out / 'paired_recovery_tradeoffs.png', dpi=160)
fig.savefig(out / 'paired_recovery_tradeoffs.pdf')
plt.close(fig)
save(out / 'completion.json', dict(status='complete', source_sha256=digest(__file__),
    artifacts={p.name:digest(p) for p in out.iterdir()}))
