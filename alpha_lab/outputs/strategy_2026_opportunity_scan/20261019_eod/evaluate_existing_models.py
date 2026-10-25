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
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--scan',type=Path,required=True)
SCAN=parser.parse_args().scan.resolve()
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


def stock_at_date(selected, asof, ds, hub):
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
        all_rows=pd.DataFrame(rows);all_rows.to_csv(OUT/f'all_member_qualification_{asof:%Y%m%d}.csv',index=False)
        decisions=all_rows[all_rows.qualified].copy().reset_index(drop=True)
        if decisions.empty:
            decisions['model_score']=pd.Series(dtype=float)
            decisions['ridge_price5']=pd.Series(dtype=bool)
            return decisions
        feats,names_=cross_section_features(decisions,scfg['model'])
        weights=np.load(STOCK/'models/2026.npz');assert names_==weights['features'].tolist()
        decisions['model_score']=feats[names_].to_numpy()@weights['coef']
        decisions['ridge_price5']=False
        decisions.loc[select_positive_top(decisions,'model_score',scfg),'ridge_price5']=True
    return decisions

def main():
    torch.set_num_threads(1)
    sm=json.loads((SCAN/'manifest.json').read_text());assert sm['status']=='complete'
    for run in [SCAN,SECTOR,STOCK]:
        m=json.loads((run/'manifest.json').read_text())
        for name,h in m['artifacts'].items():assert digest(run/name)==h,(run,name)
    ds=SectorDataset(sm['dataset']);panel=pd.read_parquet(ds.root/'daily.parquet')
    asof=panel.date.max();assert asof==pd.Timestamp('2026-10-19')
    hub=DataHub();assert hub.cfg.root.resolve()==(ROOT/'datahub/releases/selection_live_20261019_eod').resolve()
    catalog=pd.read_parquet(ds.root/'catalog_sw_l2.parquet')
    names=catalog[catalog.src.eq('SW2021')].set_index('index_code').industry_name.to_dict()
    events=pd.read_parquet(SCAN/'all_confirmations.parquet')
    recent=events[events.date.gt('2026-10-13') & events.date.le(asof)].copy()
    recent['source_event_id']=recent.event_id;recent['event_id']=recent.merged_event_id
    recent['sector_name']=recent.sector_code.map(names)
    recent['score']=np.nan;recent['threshold']=np.nan;recent['selected']=False
    mcfg=yaml.safe_load((SECTOR/'sources/config.yaml').read_text())
    # 用原有 2026 年预测复核只读装载；白名单读取特征，不读收益标签。
    original=pd.read_parquet(SECTOR/'predictions_BEFORE_EVALUATION.parquet')
    original=original[pd.to_datetime(original.model_start).eq(pd.Timestamp('2026-01-01'))]
    original_features=pd.read_parquet(SECTOR/'research_inputs_WITH_LABELS.parquet',
        columns=['event_id','date','sector_code']+mcfg['features']).set_index('event_id').loc[original.event_id].reset_index()
    values,threshold=score_sector(original_features)
    np.testing.assert_allclose(values,original.score.to_numpy(),rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(threshold,original.threshold.to_numpy(),rtol=0,atol=1e-12)
    fits=json.loads((STOCK/'fits.json').read_text())
    stock_fit=next(f for f in fits if 'start' in f and pd.Timestamp(f['start'])==pd.Timestamp('2026-01-01'))
    assert pd.Timestamp(stock_fit['train_last_label']) < pd.Timestamp('2026-01-01')
    (OUT/'annual_model_readonly_check.json').write_text(json.dumps(dict(status='passed',year=2026,
        events=len(original),max_abs_score_difference=float(np.max(np.abs(values-original.score.to_numpy()))),
        threshold=float(threshold),stock_last_label=stock_fit['train_last_label'],stock_alpha=stock_fit['alpha'],
        no_retraining=True),ensure_ascii=False,indent=2))
    if len(recent):
        features=features_at_events(recent,panel,mcfg)
        features.to_parquet(OUT/'interval_sector_features.parquet',index=False)
        recent['score'],recent['threshold']=score_sector(features)
        recent['selected']=recent.score.ge(recent.threshold)
        # 逐日截断重算，核验补扫历史事件没有读取之后行情。
        for date, group in recent.groupby('date',sort=True):
            truncated=features_at_events(group,panel[panel.date.le(date)],mcfg)
            scores,_=score_sector(truncated)
            np.testing.assert_allclose(scores,group.score.to_numpy(),rtol=1e-6,atol=1e-8)
    calendar=hub.calendar.cn_days(end='2026-10-31')
    recent['entry_date']=[calendar[calendar.searchsorted(d,side='right')] for d in recent.date]
    recent['entry_status']=['next_session' if d==asof else 'historical_entry_window_closed' for d in recent.date]
    recent.to_csv(OUT/'interval_first_confirmations_asof_model.csv',index=False)
    today=recent[recent.date.eq(asof)].copy()
    today.to_csv(OUT/'today_first_confirmations_asof_model.csv',index=False)
    selected=recent[recent.selected]
    pieces=[]
    for date, group in selected.groupby('date',sort=True):
        piece=stock_at_date(group,date,ds,hub)
        pieces.append(piece)
    if pieces:
        decisions=pd.concat(pieces,ignore_index=True)
    else:
        decisions=stock_at_date(selected.iloc[:0],asof,ds,hub)
    decisions['entry_status']=['next_session' if d==asof else 'historical_entry_window_closed' for d in decisions.date]
    decisions.to_csv(OUT/'stock_diagnostics_interval.csv',index=False)
    candidates=decisions[decisions.ridge_price5.eq(True)].copy()
    candidates.to_csv(OUT/'stock_candidates_interval.csv',index=False)
    today_decisions=decisions[decisions.date.eq(asof)]
    today_candidates=candidates[candidates.date.eq(asof)]
    today_decisions.to_csv(OUT/'stock_diagnostics_asof.csv',index=False)
    today_candidates.to_csv(OUT/'stock_candidates_asof.csv',index=False)
    states=pd.read_csv(SCAN/'watch_today.csv');states.to_csv(OUT/'watch_today.csv',index=False)
    before=pd.read_csv(ROOT/'alpha_lab/outputs/sector_live_scan/20261013T155427_428380Z/watch_today.csv')
    fields=['sector_code','window','drawdown','trigger','confirmed','slowed','support_recent','near_reference']
    compare=states[fields+['name','index_return']].merge(before[fields+['name']],on='sector_code',how='outer',suffixes=('_today','_previous'))
    compare.to_csv(OUT/'comparison_20261013_20261019.csv',index=False)
    checks={}
    for filename in ['pipeline_hashes_before.json','model_hashes_before.json']:
        saved=json.loads((OUT/filename).read_text())
        for name,h in saved.items():assert digest(ROOT/name)==h,name
        checks[filename]=saved
    summary=dict(asof=str(asof.date()),earliest_entry_date=str(calendar[calendar.searchsorted(asof,side='right')].date()),
        valid_sectors=sm['summary']['valid_sectors'],watch_sectors=len(states),trigger_sectors=int(states.trigger.sum()),
        confirmed_today=int(states.confirmed.sum()),new_first_confirmations=len(today),annual_sector_pass=int(today.selected.sum()),
        qualified_stocks_scored=len(today_decisions),stock_candidates=len(today_candidates),
        interval_first_confirmations=len(recent),interval_annual_sector_pass=len(selected),interval_stock_candidates=len(candidates),
        model_year=2026,sector_model_last_training_label='2025-11-18',stock_model_last_training_label='2025-11-19',
        no_retraining=True,no_data_engineering_code_changes=True,scope='既有申万二级自算板块',execution_certified=False,
        legacy_random_model_scores_used=False,sector_scan=str(SCAN),dataset=str(ds.root))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',hashes_unchanged=checks,
        dataset_validated=True,original_scanner_historical_prefix_unchanged=sm['summary']['previous_confirmations_unchanged'],
        historical_events_truncated_recomputed_count=len(recent),stock_event_dates_scored=int(selected.date.nunique()),
        stock_features_cut_at_each_confirmation_date='not_applicable_no_new_events' if selected.empty else True),ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    print(recent[['date','sector_name','score','threshold','selected','entry_status']].to_string(index=False),flush=True)
    print(states[['name','window','index_return','drawdown','trigger','confirmed','slowed','support_recent','near_reference']].to_string(index=False),flush=True)

if __name__=='__main__':main()
