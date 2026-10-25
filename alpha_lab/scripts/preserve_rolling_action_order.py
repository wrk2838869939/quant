"""新增股票改变不稳定同日排序；保留原行动相对顺序并严格重跑原账户。"""
from pathlib import Path
import json
import shutil
import sys
import pandas as pd
import yaml
from research_rolling_review import ROOT,digest,save
run=Path(sys.argv[1]);cfg=yaml.safe_load((run/'config.yaml').read_text());parent=ROOT/cfg['parent']
attempt=run/'attempts/action_order';attempt.mkdir(parents=True,exist_ok=False)
for window in cfg['windows']:
    folder=run/'execution'/f'{window}_calendar_01_hybrid'
    for name in ['daily','trades','holdings']:
        pd.testing.assert_frame_equal(pd.read_csv(folder/f'{name}.csv'),pd.read_csv(parent/'execution'/f'{window}_hybrid'/f'{name}.csv'),check_exact=True)
    a=pd.read_csv(folder/'events.csv');b=pd.read_csv(parent/'execution'/f'{window}_hybrid/events.csv')
    changed=(a.fillna('__NA__')!=b.fillna('__NA__')).any(axis=1)
    assert changed.sum()==4
    assert set(a.loc[changed,'kind'])=={'dividend_paid','dividend_receivable'}
    keys=['date','kind','ts_code']
    pd.testing.assert_frame_equal(a[changed].sort_values(keys).reset_index(drop=True),b[changed].sort_values(keys).reset_index(drop=True),check_exact=True)
shutil.move(str(run/'execution'),str(attempt/'execution'))
for name in ['continued.log','cold.log','execution_input_completion.json']:
    shutil.copy2(run/name,attempt/name)
p=run/'inputs/normalized_actions.parquet';shutil.copy2(p,attempt/p.name)
a=pd.read_parquet(p);old=pd.read_parquet(parent/'inputs/normalized_actions.parquet')
oldkeys=set(zip(old.ts_code,old.ex_date));new=a[[key not in oldkeys for key in zip(a.ts_code,a.ex_date)]]
ordered=pd.concat([old,new.sort_values(['ex_date','ts_code'])],ignore_index=True).sort_values('ex_date',kind='stable').reset_index(drop=True)
keys=['ts_code','ex_date']
pd.testing.assert_frame_equal(ordered.sort_values(keys).reset_index(drop=True),a.sort_values(keys).reset_index(drop=True),check_exact=True)
ordered.to_parquet(p,index=False)
receipt=json.loads((run/'execution_input_completion.json').read_text());receipt['same_day_original_order_preserved']=True
receipt['hashes']['inputs/normalized_actions.parquet']=digest(p)
save(run/'execution_input_completion.json',receipt)
shutil.copy2(__file__,attempt/Path(__file__).name)
save(attempt/'completion.json',dict(status='input_order_only_repaired',economic_values_exact=True,account_equity_trades_holdings_exact=True,event_differences_per_account=4,hashes={str(p.relative_to(attempt)):digest(p) for p in attempt.rglob('*') if p.is_file()}))
print('Economic values unchanged; original within-day action order restored. Both reference accounts will rerun strictly.')
