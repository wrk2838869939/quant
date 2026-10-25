from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import sys,json
import pandas as pd
r=Path('/root/autodl-tmp/alpha_lab');sys.path.insert(0,str(r/'src'))
out=r/'outputs/monthly_manual_current/index_probe';out.mkdir(exist_ok=False);rows=[]
for api,params in [('index_daily',dict(ts_code='000300.SH',trade_date='20260912')),('index_daily',dict(ts_code='000300.SH',trade_date='20260914')),('index_daily',dict(ts_code='000300.SH',trade_date='20260911')),('index_dailybasic',dict(ts_code='000300.SH',trade_date='20260914')),('daily',dict(trade_date='20260912'))]:
 item=dict(api=api,params=params,requested_at=pd.Timestamp.now(tz='UTC').isoformat())
 try:
  with open('/dev/null','w') as sink,redirect_stdout(sink),redirect_stderr(sink):
   from ts_client import get_pro
   frame=get_pro().query(api,**params)
  item.update(rows=len(frame),status='downloaded',received_at=pd.Timestamp.now(tz='UTC').isoformat());frame.to_parquet(out/f'{len(rows):02}_{api}.parquet',index=False)
  cols=[c for c in ['ts_code','trade_date','close','pre_close','pct_chg'] if c in frame];print(api,params,frame[cols].head(3).to_dict('records'),flush=True)
 except Exception as exc:item.update(status='failed',error_type=type(exc).__name__)
 rows.append(item)
 (out/'receipt.json').write_text(json.dumps(rows,indent=2))
