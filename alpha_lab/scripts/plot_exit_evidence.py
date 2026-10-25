"""固定时序预测与未来标记收益散点；明确不是组合收益。"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
from research_profit_defense import save,digest
run=Path(sys.argv[1]).resolve();out=run/'figures';out.mkdir(exist_ok=False)
p=pd.read_csv(run/'models/predictions.csv');metrics=pd.read_csv(run/'models/metrics.csv')
fontManager.addfont('/root/autodl-tmp/fonts/SimHei.ttf');plt.rcParams['font.family']=FontProperties(fname='/root/autodl-tmp/fonts/SimHei.ttf').get_name();plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,h in zip(axes,[20,60]):
    g=p[p.cohort.eq('exit')&p.horizon.eq(h)&p.actual.notna()];m=metrics[metrics.cohort.eq('exit')&metrics.horizon.eq(h)].iloc[0]
    for year,color in [(2024,'#3575ad'),(2025,'#dd8e35'),(2026,'#2c9a78')]:
        s=g[g.year.eq(year)];ax.scatter(s.prediction*100,s.actual*100,label=f'{year}（{len(s)}事件）',c=color,s=42,alpha=.85)
    ax.axhline(0,c='gray',lw=.8);ax.axvline(0,c='gray',lw=.8);ax.grid(alpha=.2)
    ax.set_xlabel('事前预测标记收益（%）');ax.set_ylabel('实际后续标记收益（%）')
    ax.set_title(f'{h}交易日｜n={int(m.n)}，R2={m.r2:.3f}，秩相关={m.ic:.3f}');ax.legend(fontsize=9)
fig.suptitle('退出事件的历史顺序检验｜不重叠主样本｜非组合或可成交收益\n60日完整标签仅覆盖至2026-06-08的退出决策')
fig.tight_layout();fig.savefig(out/'exit_predictions.png',dpi=160);fig.savefig(out/'exit_predictions.pdf');plt.close(fig)
save(out/'completion.json',dict(status='complete',source_sha256=digest(__file__),artifacts={f.name:digest(f) for f in out.iterdir()}))
