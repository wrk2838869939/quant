"""执行准备数据检查与单位转换；原始接口缓存不可静默去重或补零。"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from shadow_planner import ShadowError, iso_date


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def normalize_day(daily, basic, adj, date):
    """采用已有Tushare字段契约：手→股、千元→元、万元→元。"""
    iso_date(date);day=date.replace('-','')
    required={
        'daily':{'ts_code','trade_date','open','high','low','close','pre_close','vol','amount'},
        'basic':{'ts_code','trade_date','close','turnover_rate','total_mv'},
        'adj':{'ts_code','trade_date','adj_factor'}}
    frames={}
    for name,frame in [('daily',daily),('basic',basic),('adj',adj)]:
        if frame.empty or not required[name].issubset(frame):raise ShadowError(name+'缺必需字段或为空')
        frame=frame.copy()
        if frame[['ts_code','trade_date']].isna().any().any() or not frame.trade_date.astype(str).eq(day).all():
            raise ShadowError(name+'包含错误交易日期或空主键')
        if not frame.ts_code.str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all() or frame.ts_code.duplicated().any():
            raise ShadowError(name+'证券代码格式或重复主键异常')
        for col in required[name]-{'ts_code','trade_date'}:
            frame[col]=pd.to_numeric(frame[col],errors='raise')
            if not np.isfinite(frame[col]).all():raise ShadowError(name+'/'+col+'含缺值或非有限值')
        frames[name]=frame
    daily,basic,adj=frames['daily'],frames['basic'],frames['adj']
    if set(daily.ts_code)!=set(basic.ts_code):raise ShadowError('日线与daily_basic股票集合不同，不取交集掩盖缺数')
    if not set(daily.ts_code).issubset(set(adj.ts_code)):raise ShadowError('实际报价股票缺复权因子')
    if (daily[['open','high','low','close','pre_close']]<=0).any().any():raise ShadowError('原始价格须为正')
    if (daily[['vol','amount']]<0).any().any() or (basic[['turnover_rate','total_mv']]<0).any().any():
        raise ShadowError('成交、换手或市值为负')
    if (daily.low>daily[['open','close']].min(axis=1)+1e-8).any() or (daily.high<daily[['open','close']].max(axis=1)-1e-8).any():
        raise ShadowError('OHLC关系不成立')
    if (adj.adj_factor<=0).any():raise ShadowError('复权因子须为正')
    x=daily.merge(basic[['ts_code','close','turnover_rate','total_mv']],on='ts_code',validate='one_to_one',suffixes=('','_basic'))
    if (x.close-x.close_basic).abs().gt(.005).any():raise ShadowError('两个接口的收盘价不一致')
    x=x.merge(adj[['ts_code','adj_factor']],on='ts_code',validate='one_to_one')
    x['date']=pd.Timestamp(date)
    x['hfq_close']=x.close*x.adj_factor
    x['vol_share']=x.vol*100.
    x['amount_yuan']=x.amount*1000.
    x['total_mv_yuan']=x.total_mv*10000.
    feature=x[['date','ts_code','hfq_close','total_mv_yuan','turnover_rate']].sort_values('ts_code').reset_index(drop=True)
    market=x[['date','ts_code','open','high','low','close','pre_close','adj_factor','vol_share','amount_yuan']].sort_values('ts_code').reset_index(drop=True)
    return feature,market,dict(date=date,rows=len(x),adj_only_codes=sorted(set(adj.ts_code)-set(daily.ts_code)),
        checks=['unique_keys','same_date','daily_basic_code_set','adjustment_coverage','close_agreement','finite_units','ohlc'])


def require_current_metadata(package, asof):
    """价格包完整不等于可产生交易计划，慢变量必须另有覆盖回执。"""
    iso_date(asof)
    if package.get('asof')!=asof or not package.get('price_checks_passed'):
        raise ShadowError('价格包日期或核验状态不符')
    missing=[]
    for key in ['security_master','st_history','corporate_actions','trading_calendar']:
        item=package.get('metadata',{}).get(key,{})
        if not item.get('source') or item.get('verified_through','')<asof or not item.get('path'):
            missing.append(key);continue
        if file_hash(item['path'])!=item.get('sha256'):raise ShadowError(key+'文件哈希改变')
    if missing:raise ShadowError('尚缺截至决策日的元数据覆盖回执：'+','.join(missing))
    return True
