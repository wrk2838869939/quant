"""为滚动日历新增候选核验公司行动；只补本轮输入，保持已封存来源不变。"""
from pathlib import Path
import sys
import shutil
import requests
import pandas as pd
import yaml
from research_rolling_review import ROOT, verify_manifest, digest, save, normalize_actions

run=Path(sys.argv[1]);verify_manifest(run)
cfg=yaml.safe_load((run/'config.yaml').read_text());inp=ROOT/cfg['parent']/'inputs'
assert not (run/'execution').exists(), '必须在账户结果产生前冻结补充证据'
evidence=run/'evidence';evidence.mkdir(exist_ok=False)
url='https://pdf.dfcfw.com/pdf/H2_AN202410301640607006_1.pdf'
r=requests.get(url,timeout=60);r.raise_for_status();assert r.content.startswith(b'%PDF')
(evidence/'002352_20241031_2024-098.pdf').write_bytes(r.content)
overlay=dict(components=[dict(ts_code='002352.SZ',ex_date='2024-11-07',component_count=2,cash_div_tax=1.4,stk_div=0.,evidence=url,check='发行人公告2024-098：中期0.4元+特别1.0元；登记11月6日，除息及支付11月7日。')],reference_adjustments=[dict(ts_code='002352.SZ',ann_date='2024-10-31',ex_date='2024-11-07',total_shares=4816186983,participating_shares=4795415625,cash_div_tax=1.4,stk_div=0.,evidence=url,check='回购20771358股不参与；参考扣减1.3939620元/股，实际权益1.4元/股。')])
(evidence/'corporate_action_overlay.yaml').write_text(yaml.safe_dump(overlay,allow_unicode=True,sort_keys=False))
shutil.copy2(__file__,evidence/Path(__file__).name)
shutil.copy2(run/'data.log',evidence/'initial_data_attempt.log')
save(evidence/'frozen.json',dict(status='verified_before_accounts',hashes={p.name:digest(p) for p in evidence.iterdir()}))
codes=pd.read_csv(run/'execution_codes.csv').ts_code.tolist()
cache=ROOT/'data/raw/execution_audit/20220101_20260908'
market=pd.read_parquet(ROOT/'data/panel/daily_panel.parquet',columns=['date','ts_code','open','high','low','close','pre_close','adj_factor','vol_share'],filters=[('date','>=',pd.Timestamp(cfg['start'])),('date','<=',pd.Timestamp(cfg['end'])),('ts_code','in',codes)])
limits=pd.concat([pd.read_parquet(cache/'stk_limit'/f'{c}.parquet') for c in codes],ignore_index=True);limits['date']=pd.to_datetime(limits.trade_date)
assert not limits.duplicated(['date','ts_code']).any()
cover=market.merge(limits[['date','ts_code','up_limit','down_limit']],on=['date','ts_code'],how='left',validate='one_to_one')
assert not cover[['up_limit','down_limit']].isna().any(axis=None)
actions=pd.concat([pd.read_parquet(cache/'dividend'/f'{c}.parquet') for c in codes],ignore_index=True)
ac=yaml.safe_load((inp/'corporate_action_components.yaml').read_text())
for key,values in overlay.items():ac.setdefault(key,[]).extend(values)
normalized=normalize_actions(actions,ac['components'],ac.get('reference_adjustments',[]),ac.get('source_corrections',[]))
imp=actions[actions.div_proc.eq('实施')].copy()
for col in ['imp_ann_date','ex_date']:imp[col]=pd.to_datetime(imp[col])
assert imp.imp_ann_date.notna().all()
known=imp.groupby(['ts_code','ex_date']).imp_ann_date.max().rename('known_date')
normalized=normalized.merge(known,on=['ts_code','ex_date'],how='left',validate='one_to_one')
assert normalized.known_date.notna().all() and normalized.known_date.le(normalized.record_date).all()
old=pd.read_parquet(inp/'market.parquet').set_index(['date','ts_code']).sort_index()
pd.testing.assert_frame_equal(market.set_index(['date','ts_code']).sort_index().loc[old.index,old.columns],old,check_exact=True)
old_actions=pd.read_parquet(inp/'normalized_actions.parquet').set_index(['ts_code','ex_date']).sort_index()
pd.testing.assert_frame_equal(normalized.set_index(['ts_code','ex_date']).sort_index().loc[old_actions.index,old_actions.columns],old_actions,check_exact=True)
for name,frame in [('market',market),('limits',limits),('raw_actions',actions),('normalized_actions',normalized)]:
    target=run/'inputs'/f'{name}.parquet';assert not target.exists();frame.to_parquet(target,index=False)
for name in ['basics.parquet','st.parquet','execution_config.yaml','symmetric_phase_research.yaml']:shutil.copy2(inp/name,run/'inputs'/name)
(run/'inputs/corporate_action_components.yaml').write_text(yaml.safe_dump(ac,allow_unicode=True,sort_keys=False))
save(run/'execution_input_completion.json',dict(status='complete',codes=len(codes),rows=len(market),old_market_exact=True,old_actions_exact=True,missing_limits=0,actions=len(normalized),evidence_frozen_sha256=digest(evidence/'frozen.json'),hashes={str(p.relative_to(run)):digest(p) for p in (run/'inputs').iterdir()}))
print('Execution inputs ready:',len(codes),'codes',len(normalized),'actions',flush=True)
