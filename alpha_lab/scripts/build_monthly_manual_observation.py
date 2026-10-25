"""重建最新候选收盘包；没有账户快照时仅做观察，不创建账户或订单。"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from execution_metadata import inspect_metadata
from execution_market_feed import file_hash
from shadow_inputs import build_bundle
from shadow_planner import canonical,fingerprint,ShadowError
from monthly_risk_live import read_index_close,reviews


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot',type=Path);p.add_argument('--index-stage',type=Path)
    p.add_argument('--previous-observation',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/monthly_risk_manual.yaml');a=p.parse_args()
    index_stage=a.index_stage or a.stage/'index'
    a.output.mkdir(parents=True,exist_ok=False)
    cfgpath=a.config;cfg=yaml.safe_load(cfgpath.read_text())
    metadata,data=inspect_metadata(a.stage/'metadata');date=metadata['price_date']
    old=ROOT/'outputs/recent_low_turnover/20260913T090336_282870Z/data_repair/repaired_feature_projection.parquet'
    same_previous_config=False
    if a.previous_observation:
        previous=json.loads((a.previous_observation/'manifest.json').read_text())
        for name,digest in previous['output_sha256'].items():
            if file_hash(a.previous_observation/name)!=digest:raise ShadowError('前一收盘观察包发生变化')
        old=a.previous_observation/'feature_panel.parquet'
        previous_bundle=a.previous_observation/'bundle.json'
        same_previous_config=(previous_bundle.exists() and
            json.loads(previous_bundle.read_text()).get('strategy_config')==fingerprint(cfg))
    inputs=[old,cfgpath,ROOT/'configs/models/monthly_market_risk.json',a.stage/'metadata/receipt.json',index_stage/'index_close.parquet',index_stage/'receipt.json']
    raw=pd.read_parquet(old)
    stages=([ROOT/'outputs/live_ingestion/20260914_market_stage'] if not a.previous_observation else [])+[a.stage/'market_stage']
    market=[]
    for stage in stages:
        package=json.loads((stage/'validated/package.json').read_text())
        for name,digest in package['artifacts'].items():
            if file_hash(stage/'validated'/name)!=digest:raise ShadowError('价格增量散列改变')
        f=stage/'validated/feature_increment.parquet';m=stage/'validated/market_increment.parquet'
        inputs.extend([f,m,stage/'validated/package.json']);increment=pd.read_parquet(f)
        if raw.date.max()>=increment.date.min():raise ShadowError('增量与已有历史重叠')
        raw=pd.concat([raw,increment],ignore_index=True);market.append(pd.read_parquet(m))
    # 原面板2021-12-13整日缺失；仅在新观察投影显式补入，旧封存数据不改。
    repair=ROOT/'outputs/forward_risk/repair_20211213';rp=repair/'projection.parquet';bp=ROOT/'data/raw/by_date/daily_basic/20211213.parquet'
    inputs.extend([rp,bp]);patch=pd.read_parquet(rp);basic=pd.read_parquet(bp)
    patch=patch.merge(basic[['ts_code','total_mv','turnover_rate']],on='ts_code',how='left',validate='one_to_one')
    patch['total_mv_yuan']=patch.total_mv*10000
    if raw.date.eq('2021-12-13').any():
        known=raw.loc[raw.date.eq('2021-12-13')].sort_values('ts_code').reset_index(drop=True)
        pd.testing.assert_frame_equal(known,patch[raw.columns].sort_values('ts_code').reset_index(drop=True),check_dtype=False)
    else:
        raw=pd.concat([raw,patch[raw.columns]],ignore_index=True).sort_values(['date','ts_code'])
    idxpath=index_stage/'index_close.parquet';index=read_index_close(idxpath)
    if not pd.DatetimeIndex(sorted(raw.date.unique())).equals(index.index):raise ShadowError('全市场历史与指数交易日不一致')
    calendar=dict(source=metadata['calendar_basis'],sessions=sorted(set([str(d.date()) for d in index.index]+data['sessions'])))
    base=ROOT/'outputs/selection_stability/20260914T070228_976007Z'
    # 旧观察ST文件仅覆盖旧复核日；日期平移时须重建历史，不能将缺行当作非ST。
    stpath=a.previous_observation/'st.parquet' if same_previous_config else base/'inputs/st.parquet'
    oldst=pd.read_parquet(stpath);inputs.append(stpath)
    # ST最后一条结束日不是数据覆盖截止日，长期生效记录的end可能为空。
    if same_previous_config:
        historical_st_through=pd.Timestamp(json.loads(previous_bundle.read_text())['asof'])
    else:
        inputs.append(base/'config.yaml')
        historical_st_through=pd.Timestamp(yaml.safe_load((base/'config.yaml').read_text())['end'])
    strows=[]
    for day in sorted(set(reviews(calendar,cfg,date))|{pd.Timestamp(date)}):
        iso=str(day.date())
        if iso in data['st']:
            codes=data['st'][iso].ts_code.tolist()
        elif day>historical_st_through:
            raise ShadowError('近期复核缺全市场ST快照')
        else:
            codes=oldst.loc[oldst.known_date.le(day)&oldst.start.le(day)&(oldst.end.isna()|oldst.end.ge(day)),'ts_code'].tolist()
        strows.extend(dict(ts_code=c,start=day,end=day,known_date=day) for c in codes)
    st=pd.DataFrame(strows,columns=['ts_code','start','end','known_date'])
    snapshot=json.loads(a.snapshot.read_text()) if a.snapshot else None
    if snapshot and snapshot['asof']!=date:raise ShadowError('账户核对日与数据日不同')
    receipt=dict(asof=date,complete=True,source='verified EOD increments and timestamped metadata',downloaded_at=max(metadata['first_available_at'],json.loads((index_stage/'receipt.json').read_text())['downloaded_at']),
        expected_rows=int(raw.date.eq(date).sum()),market_index=json.loads((index_stage/'receipt.json').read_text()))
    modelpath=ROOT/'outputs/shadow_execution/20260914T054517_770961Z/model_2026.json';inputs.append(modelpath)
    model=json.loads(modelpath.read_text())
    b=build_bundle(raw,pd.concat(market),data['basics'],st,data['actions'],date,calendar,cfg,model,snapshot['positions'] if snapshot else [],receipt,index)
    if not set(b['signals'])|set(snapshot['positions'] if snapshot else []) <= set(metadata['action_scope']):
        (a.output/'candidate_scope_request.json').write_text(canonical(dict(signal_date=b['signal_date'],signals=b['signals'],scope_only=True)))
        raise ShadowError('最新候选需要补查公司行动；可用candidate_scope_request.json重新获取元数据')
    for name,obj in [('bundle.json',b),('calendar.json',calendar),('metadata_report.json',metadata),('receipt.json',receipt)]:
        (a.output/name).write_text(canonical(obj))
    # 保存可直接交给build-bundle的规范文件，便于另一次独立复算。
    raw.to_parquet(a.output/'feature_panel.parquet',index=False)
    pd.concat(market).to_parquet(a.output/'market.parquet',index=False)
    data['basics'].to_parquet(a.output/'basics.parquet',index=False);st.to_parquet(a.output/'st.parquet',index=False);data['actions'].to_parquet(a.output/'actions.parquet',index=False)
    manifest=dict(asof=date,generated_at=pd.Timestamp.now(tz='UTC').isoformat(),real_account_supplied=snapshot is not None,
        strategy_config=fingerprint(cfg),schedule=cfg['schedule'],
        snapshot_sha256=fingerprint(snapshot) if snapshot else None,bundle_sha256=fingerprint(b),historical_backfill=False,
        input_sha256={str(x):file_hash(x) for x in inputs},output_sha256={x.name:file_hash(x) for x in a.output.iterdir()},
        research_history_gap_repaired='2021-12-13',history_patch_rows=len(patch),missing_patch_basics=int(patch.turnover_rate.isna().sum()),
        source_vintage_externally_certified=False,live_orders=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(canonical(dict(asof=date,signal_date=b['signal_date'],is_review=b['is_review'],candidates=len(b['signals']),latest_risk_probability=b['market_risk_history'][-1]['probability'],account_supplied=snapshot is not None,metadata_blockers=metadata['blockers'])))


if __name__=='__main__':main()
