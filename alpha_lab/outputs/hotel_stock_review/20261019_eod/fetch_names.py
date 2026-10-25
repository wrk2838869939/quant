"""只调用既有查询封装，将本次名称历史核对写入研究输出目录。"""
from pathlib import Path
import sys,importlib.util,json,yaml
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path('/root/autodl-tmp');OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('existing_update',ROOT/'alpha_lab/experiments/sector_live_v1/update_market.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cfg=yaml.safe_load((ROOT/'alpha_lab/experiments/sector_live_v1/config_20261019_eod.yaml').read_text())
codes=['000428.SZ','000721.SZ','002186.SZ','002306.SZ','301073.SZ','600258.SH','600754.SH','601007.SH','605108.SH']
raw=OUT/'fresh_names';raw.mkdir(exist_ok=True)
def fetch(code):
 p=raw/f'{code}.parquet'
 if p.exists():return m.pd.read_parquet(p)
 d=m.query('namechange',cfg,ts_code=code);assert not d.empty and d.ts_code.eq(code).all()
 d.to_parquet(p,index=False);return d
parts=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 jobs={pool.submit(fetch,c):c for c in codes}
 for f in as_completed(jobs):
  d=f.result();parts.append(d);print('NAME_HISTORY',jobs[f],len(d),flush=True)
m.pd.concat(parts,ignore_index=True).to_parquet(OUT/'fresh_namechange.parquet',index=False)
print('COMPLETE',flush=True)
