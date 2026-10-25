"""独立把新归一化公告时间回查原始发布记录，不导入策略。"""
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path('/root/autodl-tmp/alpha_lab');O=Path(__file__).resolve().parent;I=ROOT/'outputs/causal_execution/20260913T130810_374654Z/inputs'
a=pd.read_parquet(I/'raw_actions.parquet');a=a[a.div_proc.eq('实施')].copy()
for c in ['ex_date','imp_ann_date','record_date']:a[c]=pd.to_datetime(a[c],errors='coerce')
k=a.groupby(['ts_code','ex_date']).imp_ann_date.max().rename('expected_known')
n=pd.read_parquet(I/'normalized_actions.parquet').set_index(['ts_code','ex_date']).join(k)
assert n.expected_known.notna().all() and n.known_date.eq(n.expected_known).all() and n.known_date.le(n.record_date).all()
r=pd.read_parquet(ROOT/'data/raw/namechange_all.parquet');r=r[r.name.str.contains('ST',case=False,na=False)].copy()
for c,d in [('start_date','start'),('end_date','end'),('ann_date','known_date')]:r[d]=pd.to_datetime(r[c],errors='coerce')
st=pd.read_parquet(I/'st.parquet');pd.testing.assert_frame_equal(st.reset_index(drop=True),r[['ts_code','start','end','known_date']].reset_index(drop=True),check_exact=True)
assert st.known_date.notna().all() and st.known_date.le(st.start).all()
out=dict(status='complete',action_known_dates_exact=len(n),ST_known_dates_exact=len(st),no_late_implementation_or_ST_announcements=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(O/'known_date_checks.json').write_text(json.dumps(out,indent=2));print(out)
