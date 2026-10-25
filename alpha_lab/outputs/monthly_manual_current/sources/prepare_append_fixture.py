from pathlib import Path
import json,sys
import pandas as pd
r=Path('/root/autodl-tmp/alpha_lab');sys.path.insert(0,str(r/'src'))
from execution_market_feed import file_hash
root=r/'outputs/monthly_manual_current';source=root/'observation_verified';out=root/'append_prefix_fixture';out.mkdir(exist_ok=False)
for name,datecol in [('feature_panel.parquet','date'),('st.parquet','start')]:
 x=pd.read_parquet(source/name);x=x[x[datecol]<=pd.Timestamp('2026-09-11')];x.to_parquet(out/name,index=False)
(out/'manifest.json').write_text(json.dumps(dict(historical_fixture=True,real_account=False,output_sha256={p.name:file_hash(p) for p in out.iterdir()}),indent=2))
