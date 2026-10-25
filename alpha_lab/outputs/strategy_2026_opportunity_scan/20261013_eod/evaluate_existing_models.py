"""今日扫描产物的策略层评价：只调用既有特征和已保存权重，不构建数据。"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path('/root/autodl-tmp')
OUT = Path(__file__).resolve().parent
PLAN = yaml.safe_load((ROOT/'alpha_lab/configs/strategy_2026_sector_stock_plan.yaml').read_text())
SECTOR = ROOT/PLAN['source']['sector_run']
STOCK = ROOT/PLAN['source']['stock_run']
SCAN = ROOT/'alpha_lab/outputs/sector_live_scan/20261013T155427_428380Z'
sys.path[:0] = [str(SECTOR/'sources'),str(ROOT/'alpha_lab/src')]
from model import FuzzyNetwork, features_at_events, predict_network
from sector_dataset import SectorDataset, digest
from sector_asof_replay import announced_normal_names
from stock_selection_asof import cross_section_features, select_positive_top
from sector_entry_window import trend_at_date
from left_side_rules import stock_shape
from datahub import DataHub


def score_sector(features):
    """直接装载参数，完全不调用训练、聚类或选参。"""
    cfg=yaml.safe_load((SECTOR/'sources/config.yaml').read_text())
    values=[]
    for seed in [17,43,101]:
        saved=torch.load(SECTOR/'models'/f'2026_{seed}.pt',map_location='cpu',weights_only=False)
        assert pd.Timestamp(saved['audit']['train_last_label']) < pd.Timestamp('2026-01-01')
        net=FuzzyNetwork.__new__(FuzzyNetwork);torch.nn.Module.__init__(net)
        for name,value in saved['state_dict'].items():
            net.register_parameter(name,torch.nn.Parameter(torch.empty_like(value)))
        net.width_floor=cfg['fnn']['width_floor']
        net.load_state_dict(saved['state_dict'],strict=True);net.eval()
        x=features[cfg['features']].to_numpy(float)
        x=np.clip((np.where(np.isfinite(x),x,saved['median'])-saved['mean'])/saved['scale'],
                  -cfg['feature_clip'],cfg['feature_clip']).astype('float32')
        values.append(predict_network(net,x)*cfg['target_scale'])
    return np.mean(values,axis=0),saved['audit']['threshold']


def main():
    torch.set_num_threads(1)
    sm=json.loads((SCAN/'manifest.json').read_text());assert sm['status']=='complete'
    for name,expected in sm['artifacts'].items():assert digest(SCAN/name)==expected
    for run in [SECTOR,STOCK]:
        m=json.loads((run/'manifest.json').read_text())
        for name,expected in m['artifacts'].items():assert digest(run/name)==expected
    ds=SectorDataset(sm['dataset'])
    panel=pd.read_parquet(ds.root/'daily.parquet')
    asof=panel.date.max();assert asof==pd.Timestamp('2026-10-13')
    hub=DataHub();assert hub.cfg.root.resolve()==(ROOT/'datahub/releases/selection_live_20261013_eod').resolve()
    catalog=pd.read_parquet(ds.root/'catalog_sw_l2.parquet')
    names=catalog[catalog.src.eq('SW2021')].set_index('index_code').industry_name.to_dict()
    events=pd.read_parquet(SCAN/'all_confirmations.parquet')
    latest=events[events.date.eq(asof)].copy()
    latest['source_event_id']=latest.event_id
    latest['event_id']=latest.merged_event_id
    latest['sector_name']=latest.sector_code.map(names)
    latest['score']=np.nan;latest['threshold']=np.nan;latest['selected']=False
    if len(latest):
        mcfg=yaml.safe_load((SECTOR/'sources/config.yaml').read_text())
        features=features_at_events(latest,panel,mcfg)
        features.to_parquet(OUT/'today_sector_features.parquet',index=False)
        latest['score'],latest['threshold']=score_sector(features)
        latest['selected']=latest.score.ge(latest.threshold)
    latest.to_csv(OUT/'today_first_confirmations_asof_model.csv',index=False)
    selected=latest[latest.selected]
    columns=['event_id','date','sector_code','ts_code','qualified','trend','price','strength','volatility','expansion','contraction','slowdown']
    decisions=pd.DataFrame(columns=columns+['model_score','ridge_price5'])
    if len(selected):
        scfg=yaml.safe_load((STOCK/'config.yaml').read_text())
        rules=yaml.safe_load((SECTOR/'sources/left_side_trend_v2.yaml').read_text())
        mapping=pd.read_parquet(ds.root/'membership_daily.parquet').loc[asof]
        sector_codes=json.loads((ds.root/'sector_codes.json').read_text())
        members={r.sector_code:mapping.index[mapping.eq(sector_codes.index(r.sector_code))].tolist() for r in selected.itertuples()}
        codes=sorted({c for value in members.values() for c in value})
        dates=hub.calendar.cn_days(end=str(asof.date()))[-400:]
        raw=hub.cn.daily(codes=codes,start=str(dates[0].date()),end=str(asof.date()),fields=['hfq_close','turnover_rate','close','vol_share'])
        wide={f:raw.pivot(index='date',columns='ts_code',values=f).reindex(index=dates,columns=codes) for f in ['hfq_close','turnover_rate','close','vol_share']}
        shapes=stock_shape(wide,rules['stock'],rules['stock'])
        ret=wide['hfq_close'].pct_change(fill_method=None)
        p=rules['stock']
        price=(shapes['early_return'].le(-p['minimum_early_fall']) & shapes['slowdown'].le(p['slowdown_ratio']) & shapes['late_return'].le(p['maximum_recent_gain']) & ret.ge(-p['maximum_current_daily_fall'])).loc[asof]
        trend=trend_at_date(wide['hfq_close'],asof,rules['trend'])
        strength=(wide['hfq_close']/wide['hfq_close'].shift(60)-1).loc[asof]
        volatility=ret.rolling(20).std().loc[asof]
        nh=pd.read_parquet(hub.cfg.meta_dir/'namechange.parquet')
        for key in ['start_date','end_date','ann_date']:nh[key]=pd.to_datetime(nh[key],format='%Y%m%d',errors='coerce')
        normal=announced_normal_names(nh,asof,codes)
        master=pd.read_parquet(hub.cfg.meta_dir/'stock_basic.parquet').set_index('ts_code')
        listed=pd.to_datetime(master.list_date,format='%Y%m%d',errors='coerce').reindex(codes)
        delisted=pd.to_datetime(master.delist_date,format='%Y%m%d',errors='coerce').reindex(codes)
        qualified=normal & listed.le(asof) & (delisted.isna()|delisted.gt(asof)) & wide['close'].loc[asof].gt(0) & wide['vol_share'].loc[asof].gt(0)
        rows=[]
        for e in selected.itertuples():
            for c in members[e.sector_code]:
                rows.append(dict(event_id=e.event_id,date=asof,sector_code=e.sector_code,ts_code=c,qualified=bool(qualified[c]),
                                 trend=bool(trend[c]),price=bool(price[c]),strength=strength[c],volatility=volatility[c],
                                 **{f:shapes[f].at[asof,c] for f in ['expansion','contraction','slowdown']}))
        all_rows=pd.DataFrame(rows);all_rows.to_csv(OUT/'all_member_qualification.csv',index=False)
        decisions=all_rows[all_rows.qualified].copy().reset_index(drop=True)
        feats,names_=cross_section_features(decisions,scfg['model'])
        weights=np.load(STOCK/'models/2026.npz');assert names_==weights['features'].tolist()
        decisions['model_score']=feats[names_].to_numpy()@weights['coef']
        decisions['ridge_price5']=False
        decisions.loc[select_positive_top(decisions,'model_score',scfg),'ridge_price5']=True
    decisions.to_csv(OUT/'stock_diagnostics_asof.csv',index=False)
    candidates=decisions[decisions.ridge_price5.eq(True)]
    candidates.to_csv(OUT/'stock_candidates_asof.csv',index=False)
    states=pd.read_csv(SCAN/'watch_today.csv')
    states.to_csv(OUT/'watch_today.csv',index=False)
    before=pd.read_csv(ROOT/PLAN['source']['state_scan']/'watch_today.csv')
    compare=states[['sector_code','name','window','index_return','drawdown','trigger','confirmed','slowed','support_recent','near_reference']].merge(
        before[['sector_code','window','drawdown','trigger','confirmed','slowed','support_recent','near_reference']],on='sector_code',how='outer',suffixes=('_today','_previous'))
    compare.to_csv(OUT/'comparison_20261012_20261013.csv',index=False)
    entry=(hub.calendar.cn_days(start=str(asof.date()),end='2026-10-31'))[1]
    summary=dict(asof=str(asof.date()),earliest_entry_date=str(entry.date()),valid_sectors=sm['summary']['valid_sectors'],watch_sectors=len(states),
                 trigger_sectors=int(states.trigger.sum()),confirmed_today=int(states.confirmed.sum()),new_first_confirmations=len(latest),
                 annual_sector_pass=len(selected),qualified_stocks_scored=len(decisions),stock_candidates=len(candidates),
                 model_year=2026,sector_model_last_training_label='2025-11-18',stock_model_last_training_label='2025-11-19',
                 no_retraining=True,no_data_engineering_code_changes=True,scope='既有申万二级自算板块',execution_certified=False,
                 legacy_random_model_scores_used=False,sector_scan=str(SCAN),dataset=str(ds.root))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    checks=json.loads((ROOT/'alpha_lab/outputs/sector_live_update/20261013T123214_941319Z/engineering_changes_withdrawn/existing_pipeline_before_rescan.json').read_text())
    for name,h in checks.items():assert digest(ROOT/name)==h,name
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',data_engineering_hashes_unchanged=checks,
        dataset_validated=True,original_scanner_historical_prefix_unchanged=sm['summary']['previous_confirmations_unchanged']),ensure_ascii=False,indent=2))
    artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(dict(status='complete',summary=summary,artifacts=artifacts),ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    print(states[['name','window','index_return','drawdown','trigger','confirmed','slowed','support_recent','near_reference']].to_string(index=False),flush=True)


if __name__=='__main__':main()
