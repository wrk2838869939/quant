"""以实际事件核验缓冲边界、恢复资格及真实退出先后。"""
import sys
from pathlib import Path
import pandas as pd
from research_path_mechanisms import ROOT,verify,save,digest
run=Path(sys.argv[1]).resolve();cfg,m=verify(run);parent=ROOT/cfg['parent'];checks=[]
for case in m['cases']:
    f=run/'execution'/case['name'];e=pd.read_csv(f/'events.csv',parse_dates=['date']);t=pd.read_csv(f/'trades.csv',parse_dates=['date']);holds=pd.read_csv(f/'holdings.csv',parse_dates=['date'])
    grace=e[e.kind.eq('lot_cap_grace')]
    if len(grace):
        assert case['policy']=='lot_buffer'
        assert grace.required_shares.gt(0).all() and grace.required_shares.lt(100).all() and grace.stock_weight.le(.35).all()
    memory=e[e.kind.eq('recovery_slot_wait')]
    for row in memory.itertuples():
        sold=t[t.date.eq(row.date)&t.ts_code.eq(row.ts_code)&t.side.eq('sell')];assert len(sold)
        assert holds[holds.date.eq(row.date)&holds.ts_code.eq(row.ts_code)].empty
        orders=e[e.kind.eq('order_planned')&e.order_id.isin(sold.order_id)];assert orders.reason.eq('hold_false').all()
    recovery=e[e.kind.eq('recovery_candidate')]
    if len(recovery):
        assert case['policy']=='recover'
        sig=pd.read_parquet(parent/'schedules'/case['schedule']/'signals.parquet')
        for row in recovery.itertuples():
            assert pd.notna(sig.at[pd.Timestamp(row.signal_date),row.ts_code])
            assert len(memory[memory.ts_code.eq(row.ts_code)&memory.date.le(row.date)])
    checks.append(dict(name=case['name'],lot_graces=len(grace),filled_exits_remembered=len(memory),eligible_recovery_events=len(recovery)))
save(run/'mechanism_event_verification.json',dict(status='passed',checks=checks,source_sha256=digest(__file__)))
print('Verified mechanism events for',len(checks),'accounts')
