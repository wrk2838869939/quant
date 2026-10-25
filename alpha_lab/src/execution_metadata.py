"""检查实际元数据内容与获取时刻；空响应不是无风险证明。"""
import json
from pathlib import Path

import pandas as pd

from execution_market_feed import file_hash
from shadow_planner import ShadowError, validate_calendar, fingerprint


def normalize_dividends(frame, active_from=None):
    """实施日程只能引用实施公告；预案公告不能为后来字段提供可见性。"""
    required={'ts_code','div_proc','imp_ann_date','record_date','ex_date','pay_date','div_listdate','cash_div_tax','stk_div'}
    if not required.issubset(frame):raise ShadowError('分红缺少实施字段')
    exact_duplicates=int(frame.duplicated().sum())
    x=frame.drop_duplicates().copy()
    # 预案日期不是实施日程的可见日期；其版本差异留在原件，不择一回填。
    if 'ann_date' in x:x=x.drop(columns='ann_date')
    proposal_variants=int(x.duplicated().sum())
    x=x.drop_duplicates()
    for col in ['imp_ann_date','record_date','ex_date','pay_date','div_listdate']:
        x[col]=pd.to_datetime(x[col].replace('',None),format='%Y%m%d',errors='raise')
    implemented=x.div_proc.eq('实施')
    bad=implemented & (x.imp_ann_date.isna()|x.ex_date.isna()|x.record_date.isna()|
        x.imp_ann_date.gt(x.record_date)|x.record_date.ge(x.ex_date))
    accepted=x.loc[implemented & ~bad].copy()
    for col in ['cash_div_tax','stk_div']:
        accepted[col]=pd.to_numeric(accepted[col],errors='raise')
        if accepted[col].isna().any() or (~accepted[col].between(0,1e6)).any():
            raise ShadowError('实施分红金额或送转比例缺失/非法')
    historical_conflicts=accepted.iloc[:0].copy()
    duplicate=accepted.duplicated(['ts_code','ex_date'],keep=False)
    if duplicate.any():
        if active_from is None:
            raise ShadowError('同股同除权日有多条实施方案，不自动取最后一条')
        cutoff=pd.Timestamp(active_from)
        conflicts=accepted.loc[duplicate]
        # 实盘元数据不承担旧账户的历史分红结算；已结束的歧义保留原件并隔离，绝不自动合计。
        settled=conflicts.ex_date.lt(cutoff)&conflicts.record_date.lt(cutoff)&conflicts.imp_ann_date.lt(cutoff)
        settled &= ((conflicts.cash_div_tax.eq(0)) | conflicts.pay_date.lt(cutoff))
        settled &= ((conflicts.stk_div.eq(0)) | conflicts.div_listdate.lt(cutoff))
        if not settled.all():
            raise ShadowError('当前或未来同股同除权日有多条实施方案，须核对')
        historical_conflicts=conflicts.assign(quarantine_reason='historical_multiple_implemented_records')
        accepted=accepted.loc[~duplicate].copy()
    accepted['known_date']=accepted.imp_ann_date
    accepted['availability_basis']='implementation_announcement_date_without_intraday_time'
    accepted.attrs['exact_duplicate_rows_removed']=exact_duplicates
    accepted.attrs['proposal_only_variants_removed']=proposal_variants
    return accepted,pd.concat([x.loc[bad].copy(),historical_conflicts],ignore_index=True)


def inspect_metadata(stage):
    """返回各项检查和原件，不生成无凭据的verified_through声明。"""
    stage=Path(stage);receipt=json.loads((stage/'receipt.json').read_text())
    loaded=[];errors=[]
    for item in receipt['requests']:
        if item['status']!='downloaded':
            errors.append(item['api']+':'+item['status']);continue
        path=stage/item['file']
        if file_hash(path)!=item['sha256']:raise ShadowError('元数据原件哈希改变')
        requested=pd.Timestamp(item['requested_at']);received=pd.Timestamp(item['received_at'])
        if requested.tzinfo is None or received.tzinfo is None or received<requested:
            raise ShadowError('元数据获取时刻缺时区或倒退')
        frame=pd.read_parquet(path)
        if len(frame)!=item['rows'] or len(frame)>=item['row_cap']:
            raise ShadowError('元数据行数不一致或可能截断')
        loaded.append((item,frame))
    def get(api,**params):
        matched=[frame for item,frame in loaded if item['api']==api and all(item['params'].get(k)==v for k,v in params.items())]
        if len(matched)!=1:raise ShadowError(api+'请求缺失或重复')
        return matched[0]
    basics=[]
    for status in ['L','D','P','G','UN']:
        d=get('stock_basic',list_status=status)
        if not {'ts_code','list_date','delist_date','list_status'}.issubset(d):raise ShadowError('证券主表缺字段')
        if not d.list_status.eq(status).all():raise ShadowError('证券状态与查询不一致')
        if status in ['L','D'] and d.empty:errors.append('security_master_'+status+'_empty')
        basics.append(d)
    basics=pd.concat(basics,ignore_index=True)
    if basics.ts_code.duplicated().any():raise ShadowError('跨状态证券代码重复')
    for col in ['list_date','delist_date']:
        basics[col]=pd.to_datetime(basics[col].replace('',None),format='%Y%m%d',errors='raise')
    calendar=get('trade_cal',exchange='SSE').copy()
    if calendar.empty or calendar.cal_date.duplicated().any():raise ShadowError('沪市日历为空或日期重复')
    calendar=calendar.sort_values('cal_date')
    if not calendar.is_open.astype(str).isin(['0','1']).all():raise ShadowError('日历开市标识非法')
    dates=pd.to_datetime(calendar.cal_date,format='%Y%m%d')
    if not dates.reset_index(drop=True).equals(pd.Series(pd.date_range(dates.min(),dates.max()))):
        raise ShadowError('日历自然日不完整')
    shenzhen=get('trade_cal',exchange='SZSE')
    if shenzhen.empty:
        # Tushare官方说明A股三所日历相同；显式记录共用映射，不捏造深市返回。
        calendar_basis='SSE_shared_A_share_calendar_per_provider_documentation'
    else:
        other=shenzhen.sort_values('cal_date')
        if calendar[['cal_date','is_open']].astype(str).reset_index(drop=True).equals(other[['cal_date','is_open']].astype(str).reset_index(drop=True)):
            calendar_basis='SSE_SZSE_agree'
        else:raise ShadowError('沪深日历冲突')
    sessions=pd.to_datetime(calendar.loc[calendar.is_open.astype(str).eq('1'),'cal_date'],format='%Y%m%d').dt.strftime('%Y-%m-%d').tolist()
    date=receipt['price_date'];observed=receipt['observation_date']
    next_date=validate_calendar(dict(source=calendar_basis,sessions=sessions),date)
    st={};st_counts={}
    for day in sorted({receipt['signal_date'],date,observed}):
        d=get('stock_st',trade_date=day.replace('-',''))
        if d.empty or not {'ts_code','trade_date'}.issubset(d):
            errors.append('stock_st_unavailable_'+day);st_counts[day]=None;continue
        if d.ts_code.duplicated().any() or not d.trade_date.astype(str).eq(day.replace('-','')).all():
            raise ShadowError('ST股票主键或日期不符')
        st[day]=d;st_counts[day]=len(d)
    limits=get('stk_limit',trade_date=observed.replace('-',''))
    if limits.empty:errors.append('execution_limits_unavailable_'+observed)
    elif (limits.ts_code.duplicated().any() or not limits.trade_date.astype(str).eq(observed.replace('-','')).all()
          or (limits.down_limit<=0).any() or (limits.up_limit<limits.down_limit).any()):
        raise ShadowError('当日涨跌停价格主键、日期或区间非法')
    suspensions=get('suspend_d',trade_date=observed.replace('-',''))
    if not {'ts_code','trade_date','suspend_type','suspend_timing'}.issubset(suspensions):
        errors.append('execution_suspension_schema_unavailable_'+observed)
    elif len(suspensions) and not suspensions.trade_date.astype(str).eq(observed.replace('-','')).all():
        raise ShadowError('停牌查询日期不一致')
    actions=[];quarantine=[];duplicate_counts={};proposal_variants=0
    for code in receipt['scope']:
        d=get('dividend',ts_code=code)
        if len(d) and not d.ts_code.eq(code).all():raise ShadowError('分红跨代码串行')
        accepted,bad=normalize_dividends(d,active_from=date)
        duplicate_counts[code]=accepted.attrs['exact_duplicate_rows_removed']
        proposal_variants+=accepted.attrs['proposal_only_variants_removed']
        actions.append(accepted);quarantine.append(bad)
    actions=pd.concat(actions,ignore_index=True);quarantine=pd.concat(quarantine,ignore_index=True)
    if len(quarantine.loc[quarantine.ex_date.ge(pd.Timestamp(date)) | quarantine.ex_date.isna()]):
        errors.append('unresolved_implementation_dates')
    for day in [receipt['signal_date'],date]:
        if day not in st:errors.append('signal_st_snapshot_missing_'+day)
    relevant=get('dividend',ex_date=observed.replace('-',''))
    names=get('namechange')
    latest=max(item['received_at'] for item,frame in loaded)
    report=dict(schema_version=1,price_date=date,observation_date=observed,next_session=next_date,
        first_available_at=latest,receipt_sha256=file_hash(stage/'receipt.json'),
        security_rows=len(basics),calendar_basis=calendar_basis,st_counts=st_counts,
        action_scope=receipt['scope'],action_rows=len(actions),quarantined_actions=len(quarantine),
        exact_duplicate_rows_removed=sum(duplicate_counts.values()),duplicate_counts=duplicate_counts,
        proposal_only_variants_removed=proposal_variants,
        execution_day_dividend_rows=len(relevant),namechange_rows=len(names),
        current_market_status_complete=not errors,blockers=errors,
        account_scope_supplied=receipt['account_scope_supplied'],historical_vintage_certified=False,
        price_date_backfill_only=pd.Timestamp(latest).tz_convert('Asia/Shanghai').strftime('%Y-%m-%d')>date,
        current_inputs_are_provider_snapshots=True,all_corporate_actions_covered=False)
    return report,dict(basics=basics,calendar=calendar,sessions=sessions,st=st,actions=actions,quarantine=quarantine)


def guard_plan(bundle, snapshot, calendar, metadata, decision_at, provenance):
    """供新操作入口使用；真实时间过窗、回补伪前向、账户范围不齐均拒绝。"""
    now=pd.Timestamp(decision_at)
    if now.tzinfo is None:raise ShadowError('执行检查需要真实带时区时刻')
    local=now.tz_convert('Asia/Shanghai')
    execution=validate_calendar(calendar,bundle['asof'])
    if execution!=metadata['next_session']:raise ShadowError('执行日期与已核验元数据日历不一致')
    if local>=pd.Timestamp(execution,tz='Asia/Shanghai')+pd.Timedelta(hours=9,minutes=25):
        raise ShadowError('已错过下一交易日开盘执行窗口，不能补发历史订单')
    if local<pd.Timestamp(bundle['asof'],tz='Asia/Shanghai')+pd.Timedelta(hours=21):
        raise ShadowError('当前入口需等待决策日21点后元数据更新')
    if pd.Timestamp(metadata['first_available_at'])>now:raise ShadowError('使用了决策时刻之后才获取的元数据')
    if metadata['price_date']!=bundle['asof'] or snapshot['asof']!=bundle['asof']:
        raise ShadowError('元数据、行情与账户决策日期不一致')
    if metadata['blockers']:raise ShadowError('执行元数据不可用：'+','.join(metadata['blockers']))
    if not metadata['account_scope_supplied'] or not set(snapshot['positions']).issubset(metadata['action_scope']):
        raise ShadowError('公司行动查询未覆盖真实账户范围')
    if not set(bundle['signals']).issubset(metadata['action_scope']):
        raise ShadowError('候选范围发生变化，须补查新增候选')
    if provenance.get('historical_backfill') is not False:
        raise ShadowError('回补观察包不能作为真实前向计划')
    if provenance.get('bundle_sha256')!=fingerprint(bundle) or provenance.get('snapshot_sha256')!=fingerprint(snapshot):
        raise ShadowError('账户/信号不是本次核验的输入')
    if not metadata.get('all_corporate_actions_covered'):
        raise ShadowError('分红接口不覆盖配股、合并等全部公司行动，缺补充核验')
    return True
