"""展示所有匹配日历的收益/回撤变化；不画不可交易的平均净值。"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
from research_profit_defense import digest,save
run=Path(sys.argv[1]).resolve();out=run/'figures';out.mkdir(exist_ok=False)
d=pd.read_csv(run/'analysis/paired_vs_baseline.csv')
fontManager.addfont('/root/autodl-tmp/fonts/SimHei.ttf');plt.rcParams['font.family']=FontProperties(fname='/root/autodl-tmp/fonts/SimHei.ttf').get_name();plt.rcParams['axes.unicode_minus']=False
names={'duration':'交易日持续证据','wait':'退出后等待下次复核','recover':'等待＋恢复优先','lot_buffer':'整手缓冲'}
fig,axes=plt.subplots(2,2,figsize=(12,9))
for ax,(policy,title) in zip(axes.flat,names.items()):
    for window,color,label in [('continued','#2166ac','2022连续'),('cold','#d6604d','2024启动')]:
        g=d[d.policy.eq(policy)&d.window.eq(window)]
        ax.scatter(g.ann_delta*100,g.mdd_delta*100,c=color,label=label,s=35,alpha=.85)
        for r in g.itertuples():
            text=('C' if r.schedule.startswith('calendar') else 'R')+str(int(r.schedule[-2:]))
            ax.annotate(text,(r.ann_delta*100,r.mdd_delta*100),xytext=(3,3),textcoords='offset points',fontsize=8,color=color)
    ax.axhline(0,c='gray',lw=.8);ax.axvline(0,c='gray',lw=.8);ax.grid(alpha=.15)
    ax.set_title(title);ax.set_xlabel('年化收益变化（百分点）');ax.set_ylabel('最大回撤改善（百分点，向上较好）');ax.legend(fontsize=9)
fig.suptitle('相对同日历基线：全部16个起点/窗口｜右上方为收益与回撤同时改善\nC：月内日期；R：20交易日偏移｜2026-09-08截止')
fig.tight_layout();fig.savefig(out/'paired_mechanisms.png',dpi=150);fig.savefig(out/'paired_mechanisms.pdf');plt.close(fig)
save(out/'completion.json',dict(status='complete',source_sha256=digest(__file__),artifacts={p.name:digest(p) for p in out.iterdir()}))
