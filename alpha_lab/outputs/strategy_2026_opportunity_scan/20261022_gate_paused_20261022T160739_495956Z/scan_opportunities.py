#!/usr/bin/env python3
"""当前机会扫描：首次形态确认直接筛股，年度板块评分按策略开关作诊断。"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import yaml
import markdown
ROOT=Path('/root/autodl-tmp')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'alpha_lab/outputs/sector_asof_replay_v1/20261013T072855_087132Z/sources'),str(ROOT/'alpha_lab/src')]
from model import FuzzyNetwork, features_at_events, predict_network
from sector_dataset import SectorDataset, digest
from sector_asof_replay import announced_normal_names
from stock_selection_asof import cross_section_features, select_positive_top
from sector_entry_window import trend_at_date
from left_side_rules import stock_shape
from datahub import DataHub
from update_market import query
PLAN=None
SECTOR=None
STOCK=None
OUT=None


def current_events(events, asof):
    """只返回去重后的当日首次事件，重复确认和旧事件不重开。"""
    rows=events[pd.to_datetime(events.date).eq(pd.Timestamp(asof))].copy()
    if 'accepted' in rows:
        rows=rows[rows.accepted.eq(True)]
    if rows.merged_event_id.duplicated().any():
        raise ValueError('首次确认存在重复事件ID')
    rows['source_event_id']=rows.event_id
    rows['event_id']=rows.merged_event_id
    return rows


def sector_stock_eligibility(events, gate_enabled):
    """暂停时不使用评分决定资格；启用时未知评分拒绝。"""
    if not isinstance(gate_enabled,bool):
        raise TypeError('annual_sector_gate_enabled必须是布尔值')
    if not gate_enabled:
        return pd.Series(True,index=events.index,dtype=bool)
    valid=np.isfinite(events['score']) & np.isfinite(events['threshold'])
    return valid & events['score'].ge(events['threshold'])


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str))


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


def stock_at_date(selected, asof, ds, hub, name_history):
    columns=['event_id','date','sector_code','ts_code','qualified','trend','price','strength','volatility','expansion','contraction','slowdown']
    decisions=pd.DataFrame(columns=columns+['model_score','ridge_price5'])
    if len(selected):
        scfg=yaml.safe_load((STOCK/'config.yaml').read_text())
        scfg['top_n']=PLAN['signal']['top_n']
        scfg['model']['minimum_predicted_excess']=PLAN['signal']['minimum_stock_score_exclusive']
        rules=yaml.safe_load((ROOT/PLAN['source']['stock_rules']).read_text())
        mapping=pd.read_parquet(ds.root/'membership_daily.parquet').loc[asof]
        sector_codes=json.loads((ds.root/'sector_codes.json').read_text())
        members={r.sector_code:mapping.index[mapping.eq(sector_codes.index(r.sector_code))].tolist() for r in selected.itertuples()}
        codes=sorted({c for value in members.values() for c in value})
        dates=hub.calendar.cn_days(end=str(asof.date()))[-PLAN['stock_history_sessions']:]
        raw=hub.cn.daily(codes=codes,start=str(dates[0].date()),end=str(asof.date()),fields=['hfq_close','turnover_rate','close','vol_share'])
        wide={f:raw.pivot(index='date',columns='ts_code',values=f).reindex(index=dates,columns=codes) for f in ['hfq_close','turnover_rate','close','vol_share']}
        shapes=stock_shape(wide,rules['stock'],rules['stock'])
        ret=wide['hfq_close'].pct_change(fill_method=None)
        p=rules['stock']
        price=(shapes['early_return'].le(-p['minimum_early_fall']) & shapes['slowdown'].le(p['slowdown_ratio']) & shapes['late_return'].le(p['maximum_recent_gain']) & ret.ge(-p['maximum_current_daily_fall'])).loc[asof]
        trend=trend_at_date(wide['hfq_close'],asof,rules['trend'])
        strength=(wide['hfq_close']/wide['hfq_close'].shift(60)-1).loc[asof]
        volatility=ret.rolling(20).std().loc[asof]
        nh=name_history.copy()
        for key in ['start_date','end_date','ann_date']:nh[key]=pd.to_datetime(nh[key],format='%Y%m%d',errors='coerce')
        normal=announced_normal_names(nh,asof,codes)
        master=pd.read_parquet(hub.cfg.meta_dir/'stock_basic.parquet').set_index('ts_code')
        listed=pd.to_datetime(master.list_date,format='%Y%m%d',errors='coerce').reindex(codes)
        delisted=pd.to_datetime(master.delist_date,format='%Y%m%d',errors='coerce').reindex(codes)
        qualified=normal & listed.le(asof) & (delisted.isna()|delisted.gt(asof)) & wide['close'].loc[asof].gt(0) & wide['vol_share'].loc[asof].gt(0)
        rows=[]
        for e in selected.itertuples():
            for c in members[e.sector_code]:
                rows.append(dict(event_id=e.event_id,date=asof,sector_code=e.sector_code,ts_code=c,name=master.at[c,'name'],close=wide['close'].at[asof,c],day_return=ret.at[asof,c],early_return=shapes['early_return'].at[asof,c],late_return=shapes['late_return'].at[asof,c],qualified=bool(qualified[c]),
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


def main(argv=None):
    """只读已完成行情/形态产物，按当前策略配置运行个股筛选。"""
    global PLAN, SECTOR, STOCK, OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--scan',type=Path,required=True)
    parser.add_argument('--policy',type=Path,default=ROOT/'alpha_lab/configs/strategy_2026_opportunity_scan.yaml')
    parser.add_argument('--config',type=Path,help='可选市场配置，只核对与扫描行情一致')
    parser.add_argument('--name-history',type=Path,help='复核时可指定已拉取的名称历史；默认重新查询')
    parser.add_argument('--review-policy',action='store_true',help='按新策略复核生效日以前行情，不回填历史信号')
    args=parser.parse_args(argv)
    PLAN=yaml.safe_load(args.policy.read_text())
    SECTOR=ROOT/PLAN['source']['sector_run'];STOCK=ROOT/PLAN['source']['stock_run']
    scan=args.scan.resolve();manifest=json.loads((scan/'manifest.json').read_text())
    assert manifest['status']=='complete'
    for name,h in manifest['artifacts'].items():assert digest(scan/name)==h,name
    market=manifest['config'];asof=pd.Timestamp(market['asof'])
    assert asof.year==PLAN['signal']['model_year']==2026,'当前入口仅配置2026权重，不能跨年误用'
    if asof<pd.Timestamp(PLAN['effective_from']) and not args.review_policy:
        parser.error('行情早于策略生效日；历史复核请显式加 --review-policy')
    if args.config:
        supplied=yaml.safe_load(args.config.read_text())
        assert supplied['asof']==market['asof'] and supplied['release']==market['release']
    hub=DataHub();assert hub.cfg.root.resolve()==(ROOT/market['release']).resolve()
    ds=SectorDataset(manifest['dataset'])
    panel=pd.read_parquet(ds.root/'daily.parquet');assert panel.date.max()==asof
    assert ds.manifest['config']['release']==market['release']
    gate_enabled=PLAN['signal']['annual_sector_gate_enabled']
    if not isinstance(gate_enabled,bool):raise TypeError('年度板块开关必须为布尔值')
    # 保持既有个股规则，未支持的策略变体明确拒绝，避免配置看似生效。
    assert PLAN['signal']['decision']=='first_confirmation_close'
    assert PLAN['signal']['same_event_required'] and PLAN['signal']['entry_valid_sessions']==1
    assert PLAN['signal']['require_price_shape']
    assert not any(PLAN['signal'][key] for key in ['require_stock_trend','require_turnover_expansion','require_turnover_contraction'])
    protected=[ROOT/'alpha_lab/experiments/sector_live_v1/update_market.py',
        ROOT/'alpha_lab/scripts/build_sector_dataset.py',ROOT/'alpha_lab/src/sector_dataset.py',
        ROOT/'alpha_lab/experiments/sector_live_v1/scan_sectors.py',
        ROOT/'alpha_lab/configs/strategy_2026_sector_stock_plan.yaml',
        *list((SECTOR/'models').glob('2026_*.pt')),STOCK/'models/2026.npz']
    before={str(p.relative_to(ROOT)):digest(p) for p in protected}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    OUT=ROOT/PLAN['output_root']/f'{asof:%Y%m%d}_gate_paused_{stamp}'
    OUT.mkdir(parents=True);save_json(OUT/'manifest.json',dict(status='building'))
    shutil.copy2(__file__,OUT/'scan_opportunities.py');shutil.copy2(args.policy,OUT/'policy.yaml')
    torch.set_num_threads(1)
    all_events=pd.read_parquet(scan/'all_confirmations.parquet')
    events=current_events(all_events,asof)
    catalog=pd.read_parquet(ds.root/'catalog_sw_l2.parquet')
    names=catalog[catalog.src.eq('SW2021')].set_index('index_code').industry_name.to_dict()
    events['sector_name']=events.sector_code.map(names)
    events['score']=np.nan;events['threshold']=np.nan
    diagnostic_status='not_needed_no_events'
    if len(events) and (gate_enabled or PLAN['signal']['annual_sector_diagnostics']):
        try:
            m=json.loads((SECTOR/'manifest.json').read_text())
            for name,h in m['artifacts'].items():assert digest(SECTOR/name)==h,name
            model_cfg=yaml.safe_load((SECTOR/'sources/config.yaml').read_text())
            features=features_at_events(events,panel[panel.date.le(asof)],model_cfg)
            events['score'],events['threshold']=score_sector(features)
            features.to_parquet(OUT/'sector_features.parquet',index=False)
            diagnostic_status='available_reference_only' if not gate_enabled else 'available_filter_enabled'
        except Exception as error:
            if gate_enabled:raise
            diagnostic_status=f'unavailable_nonblocking:{type(error).__name__}'
    events['annual_model_pass']=events.score.notna()&events.threshold.notna()&events.score.ge(events.threshold)
    events['stock_entry_eligible']=sector_stock_eligibility(events,gate_enabled)
    events['annual_sector_gate_enabled']=gate_enabled
    events['gate_status']='ANNUAL_SECTOR_GATE_PAUSED' if not gate_enabled else 'ANNUAL_SECTOR_GATE_ENABLED'
    events.to_csv(OUT/'first_confirmations.csv',index=False)
    selected=events[events.stock_entry_eligible].copy()
    next_days=hub.calendar.cn_days(start=str((asof+pd.Timedelta(days=1)).date()),end=str((asof+pd.Timedelta(days=40)).date()))
    next_date=str(next_days[0].date()) if len(next_days) else None
    if len(selected):
        m=json.loads((STOCK/'manifest.json').read_text())
        for name,h in m['artifacts'].items():assert digest(STOCK/name)==h,name
        fits=json.loads((STOCK/'fits.json').read_text())
        fit=next(f for f in fits if 'start' in f and pd.Timestamp(f['start'])==pd.Timestamp('2026-01-01'))
        assert pd.Timestamp(fit['train_last_label'])<pd.Timestamp('2026-01-01')
        mapping=pd.read_parquet(ds.root/'membership_daily.parquet').loc[asof]
        sectors=json.loads((ds.root/'sector_codes.json').read_text())
        codes=sorted({c for sector in selected.sector_code for c in mapping.index[mapping.eq(sectors.index(sector))]})
        if args.name_history:
            nh=pd.read_parquet(args.name_history)
            name_source=dict(mode='provided_review_cache',path=str(args.name_history.resolve()),sha256=digest(args.name_history))
        else:
            parts=[]
            with ThreadPoolExecutor(max_workers=market['workers']) as executor:
                jobs={executor.submit(query,'namechange',market,ts_code=code):code for code in codes}
                for job in as_completed(jobs):
                    frame=job.result()
                    assert not frame.empty and frame.ts_code.eq(jobs[job]).all(),jobs[job]
                    parts.append(frame)
            nh=pd.concat(parts,ignore_index=True)
            name_source=dict(mode='fresh_existing_query_wrapper',queried_at=datetime.now(timezone.utc).isoformat())
        assert set(codes).issubset(set(nh.ts_code)),'成员名称历史不完整'
        nh.to_parquet(OUT/'namechange_for_review.parquet',index=False);save_json(OUT/'name_history_source.json',name_source)
    else:
        nh=pd.DataFrame()
    decisions=stock_at_date(selected,asof,ds,hub,nh)
    if len(decisions):
        decisions['sector_name']=decisions.sector_code.map(names)
        decisions['earliest_entry_date']=next_date
        decisions['record_origin']='policy_change_review' if args.review_policy else 'current_policy_scan'
        decisions['annual_sector_gate_enabled']=gate_enabled
        decisions['execution_certified']=False
        decisions['reason']=np.select([~decisions.price,decisions.model_score.le(0),~decisions.ridge_price5],
            ['价格形态未过','个股分数非正','未进入前5名'],default='当前策略候选')
    decisions.to_csv(OUT/'stock_diagnostics.csv',index=False)
    candidates=decisions[decisions.ridge_price5.eq(True)].copy()
    if len(candidates):candidates=candidates.sort_values(['sector_code','model_score','ts_code'],ascending=[True,False,True])
    candidates.to_csv(OUT/'stock_candidates.csv',index=False)
    watch=pd.read_csv(scan/'watch_today.csv');watch.to_csv(OUT/'watch_today.csv',index=False)
    summary=dict(strategy_id=PLAN['strategy_id'],effective_from=PLAN['effective_from'],asof=str(asof.date()),
        record_origin='policy_change_review' if args.review_policy else 'current_policy_scan',
        annual_sector_gate_enabled=gate_enabled,annual_diagnostic_status=diagnostic_status,
        shape_confirmed_today=int(watch.confirmed.sum()),new_first_confirmations=len(events),
        annual_model_pass_reference=int(events.annual_model_pass.sum()),sectors_entering_stock_filter=len(selected),
        stocks_scored=len(decisions),stock_candidates=len(candidates),earliest_entry_date=next_date,
        execution_certified=False,auto_orders=False,legacy_random_scores_used=False)
    save_json(OUT/'summary.json',summary)
    table=events[['sector_name','date','score','threshold','annual_model_pass','stock_entry_eligible']].to_markdown(index=False) if len(events) else '当日没有新的首次形态确认。'
    stock_table=candidates[['sector_name','name','ts_code','close','model_score']].to_markdown(index=False) if len(candidates) else '无。'
    gate_text='已暂停，评分只作参考，不影响个股入口' if not gate_enabled else '已启用'
    report=f'''# 当前策略机会扫描：{asof:%Y-%m-%d}

策略版本：{PLAN['strategy_id']}；生效日期：{PLAN['effective_from']}。年度板块过滤：**{gate_text}**。

记录性质：{summary['record_origin']}。{'这是按新规则对旧收盘数据的复核，不改写原历史信号，不表示当前仍能按旧价格成交。' if args.review_policy else '按当前有效规则生成收盘候选。'}

## 板块到个股

当日形态确认{summary['shape_confirmed_today']}个，去重后的新首次确认{len(events)}个，进入个股筛选{len(selected)}个，合格个股评分{len(decisions)}只，候选{len(candidates)}只。

{table}

若当日没有新的首次确认，不能因旧事件再次满足形态就重开入场。年度板块分数的“通过/不通过”只在过滤开关启用时影响资格；当前诊断状态：{diagnostic_status}。

## 候选

{stock_table}

原信号的下一交易日为{next_date}。候选仍需核对当日执行条件；本程序不下单，也不把候选当已成交。历史复核不恢复已经结束的执行窗口。

个股保留资格/名称ST、价格减速形态、原2026年度个股排序正分数前5名规则。趋势、放量、缩量仍按原方案参与诊断或评分，不新增硬门槛。年度个股模型权重未改。

暂停板块过滤会扩大个股模型的应用事件范围，这部分完整交易效果尚未验证；本次只执行已确认的策略调整，不引用旧收益证明扩展有效。

[全部合格个股及原因](stock_diagnostics.csv) · [候选CSV](stock_candidates.csv) · [首次事件与参考评分](first_confirmations.csv) · [策略配置](policy.yaml) · [核验](verification.json)
'''
    (OUT/'REPORT.md').write_text(report)
    (OUT/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:30px auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}</style><body>'+markdown.markdown(report,extensions=['tables'])+'</body></html>')
    for name,h in before.items():assert digest(ROOT/name)==h,name
    save_json(OUT/'verification.json',dict(status='passed',protected_original_hashes_unchanged=True,
        source_scan_manifest_verified=True,old_events_not_reopened=True,stock_weights_unchanged=True,
        annual_gate_enabled=gate_enabled,no_orders=True))
    save_json(OUT/'manifest.json',dict(status='complete',summary=summary,sector_scan=str(scan),
        artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}))
    pointer='LATEST_POLICY_REVIEW' if args.review_policy else 'LATEST_CANDIDATES'
    (HERE/pointer).write_text(str(OUT.relative_to(ROOT))+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    print(stock_table,flush=True);print('COMPLETE',OUT,flush=True)


if __name__=='__main__':
    main()
