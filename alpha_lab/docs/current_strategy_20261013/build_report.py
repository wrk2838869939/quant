"""从核验后的统计生成图表，并将技术报告排版为HTML与PDF。"""
from pathlib import Path
import re,json,hashlib,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import markdown
import pymupdf
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
font=ROOT/'fonts/SimHei.ttf'
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':12})
# 流程示意只描述当前已接入组件。
fig,ax=plt.subplots(figsize=(10,4.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
items=[(.02,.58,'1  数据与冻结','日线、复权、基本面接口\n质量检查 → 独立快照'),(.36,.58,'2  板块指数','历史成员与昨日市值权重\n报价覆盖和连续段约束'),(.70,.58,'3  事件状态机','平台 → 上涨 → 回落\n观察 → 触发 → 首次确认'),(.70,.08,'4  模糊神经网络','13维确认日特征\n两轮冻结模型与阈值'),(.36,.08,'5  个股联合检查','历史趋势＋五日量价\n近期模型信号＋当前确认'),(.02,.08,'6  输出诊断和候选','板块图表、逐项原因\n账户执行尚未接入')]
for x,y,title,body in items:
 ax.add_patch(FancyBboxPatch((x,y),.28,.29,boxstyle='round,pad=0.012',facecolor='#eef4f8',edgecolor='#50758b',linewidth=1.2))
 ax.text(x+.14,y+.225,title,ha='center',va='center',fontsize=13,color='#15354a',weight='bold')
 ax.text(x+.14,y+.105,body,ha='center',va='center',fontsize=11,linespacing=1.6)
for a,b in [((.30,.725),(.35,.725)),((.64,.725),(.69,.725)),((.84,.57),(.84,.39)),((.69,.225),(.65,.225)),((.35,.225),(.31,.225))]:
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.8,'color':'#50758b'})
fig.savefig(HERE/'workflow.png',dpi=180,bbox_inches='tight');plt.close(fig)
samples=pd.read_csv(HERE/'verified_test_samples.csv')
fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
for ax,h in zip(axes,[3,10,20]):
 g=samples[(samples.horizon==h)&samples.complete];base=g.end_return.to_numpy()*100;chosen=g[g.selected].end_return.to_numpy()*100
 edges=np.linspace(min(base.min(),0),max(base.max(),0),22)
 ax.hist(base,bins=edges,density=True,color='#91a3b0',alpha=.55,label=f'全部测试事件 n={len(base)}')
 ax.hist(chosen,bins=edges,density=True,histtype='step',linewidth=1.8,color='#c86828',label=f'FNN筛选事件 n={len(chosen)}')
 ax.axvline(0,color='#555',linewidth=.7,ls=':');ax.set(title=f'{h}个交易日',xlabel='期末板块收益（%）',ylabel='概率密度')
 ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
fig.savefig(HERE/'return_histograms.png',dpi=180);plt.close(fig)
text=(HERE/'TECHNICAL_REPORT.md').read_text()
body=markdown.markdown(text,extensions=['tables','fenced_code'])
# 为图表预留完整版面，避免排版器把图片缩小到页尾剩余高度。
body=body.replace('<h2>2. 当前处理流程</h2>', '<h2 style="page-break-before:always">2. 当前处理流程</h2>')
body=body.replace('<p><img alt="测试事件收益直方图"', '<p style="page-break-before:always"><img alt="测试事件收益直方图"')
# 长本地路径允许在分隔符后换行；不修改可复制的Markdown源文件。
def wrap_code(m):
 t=m.group(1)
 return '<code>'+t.replace('/','/&#8203;').replace('_','_&#8203;')+'</code>'
body=re.sub(r'<code>(.*?)</code>',wrap_code,body,flags=re.S)
css='''
@font-face {font-family: ReportCN; src: url(SimHei.ttf);}
body {font-family: ReportCN,sans-serif; font-size: 10pt; line-height: 1.5; color:#202b32;}
h1 {font-size:22pt; line-height:1.3; color:#163c52; margin-bottom:16pt;}
h2 {font-size:15pt; color:#163c52; margin-top:20pt; margin-bottom:9pt; page-break-after:avoid;}
h3 {font-size:11.5pt; color:#31586b; margin-top:13pt; page-break-after:avoid;}
p {margin-top:5pt; margin-bottom:8pt;}
table {border-collapse:collapse; width:100%; font-size:8.8pt; margin:10pt 0;}
th {background-color:#e8eff3; font-weight:bold;}
td,th {border:0.5pt solid #bccbd3; padding:5pt; vertical-align:top;}
code {font-family:ReportCN,sans-serif; font-size:8.5pt; word-wrap:break-word;}
pre {white-space:pre-wrap; background-color:#f2f5f7; padding:8pt; font-size:8.5pt;}
img {width:100%; max-width:100%;}
a {color:#245f85;}
'''
(HERE/'TECHNICAL_REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>当前策略技术报告</title><style>'+css.replace('url(SimHei.ttf)','url(../../../fonts/SimHei.ttf)')+'</style><body>'+body+'</body></html>')
archive=pymupdf.Archive(str(HERE));archive.add(str(ROOT/'fonts'))
story=pymupdf.Story(html=body,user_css=css,archive=archive)
tmp=HERE/'report_layout.pdf'
writer=pymupdf.DocumentWriter(str(tmp))
page=pymupdf.paper_rect('a4');content=pymupdf.Rect(42,42,page.width-42,page.height-42)
count=0
while True:
 dev=writer.begin_page(page);more,filled=story.place(content);story.draw(dev);writer.end_page();count+=1
 if not more:break
 if count>60:raise RuntimeError('排版页数异常')
writer.close()
doc=pymupdf.open(tmp)
for i,p in enumerate(doc):
 p.insert_text((42,24),'CURRENT STRATEGY | 2026-10-13',fontsize=7,color=(.3,.4,.46))
 p.insert_text((page.width-75,page.height-22),f'{i+1} / {len(doc)}',fontsize=8,color=(.3,.4,.46))
doc.set_metadata({'title':'板块左侧事件筛选与模糊神经网络策略技术报告','subject':'当前代码、冻结模型与复核统计；行情截至2026-10-12','author':'alpha_lab'})
doc.save(HERE/'TECHNICAL_REPORT.pdf',garbage=4,deflate=True)
print('PDF_PAGES',len(doc),'PDF_BYTES',(HERE/'TECHNICAL_REPORT.pdf').stat().st_size)
texts=[p.get_text() for p in doc]
assert all(len(t)>60 for t in texts)
assert '当前策略' in ''.join(texts) or '板块左侧' in ''.join(texts)
(HERE/'pdf_text_check.txt').write_text('\n\n'.join(texts))
# 导出全报告缩略图以检查排版。
from PIL import Image
cols=3;thumbw=330;thumbh=470
canvas=Image.new('RGB',(cols*thumbw,((len(doc)+cols-1)//cols)*thumbh),'#d8dfe4')
for i,p in enumerate(doc):
 pix=p.get_pixmap(matrix=pymupdf.Matrix(.55,.55),alpha=False)
 im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((thumbw-10,thumbh-10))
 canvas.paste(im,((i%cols)*thumbw+5,(i//cols)*thumbh+5))
canvas.save(HERE/'report_preview.png')
doc.close();tmp.unlink()
manifest={'status':'complete','report_date':'2026-10-13','market_asof':'2026-10-12','historical_cutoff':'2026-09-08','model_replay_audit':'alpha_lab/outputs/sector_fuzzy_random_audit/20261013T054725_005561Z/audit.json','pages':count,'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.name!='manifest.json'}}
(HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
