"""固定规则基线与候选分数插件的接口；不做参数搜索或未来年度延用。"""
from __future__ import annotations
import re
from copy import deepcopy
import numpy as np
import pandas as pd
from rule_search import RuleSignals,decision_dates,config_id


def validate_entry(entry, years):
    """年度参数必须覆盖评价范围，且保持锁定ID与历史内层截止关系。"""
    schedule=entry['annual_parameters']
    if any(int(year) not in schedule for year in years):
        raise ValueError('缺少年度锁定参数；禁止向新年份自动延用最后一组参数')
    for year in years:
        item=schedule[int(year)];p=item['params']
        if item['selection_id']!=config_id(p):
            raise ValueError('参数与锁定ID不一致')
        if p['family']!=entry['source_family'] or not 1<=p['top_n']<=5:
            raise ValueError('基线家族或小资金持股数无效')
        if pd.Timestamp(item['inner_end'])>=pd.Timestamp(f'{year}-01-01'):
            raise ValueError('参数选择窗口进入评价年度')


def validate_candidate(config,registry):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',config['candidate_id']):
        raise ValueError('候选ID只能使用字母、数字与下划线')
    if config['candidate_id'] in registry['baselines']:
        raise ValueError('候选ID不可覆盖基线ID')
    if config['parent_baseline'] not in registry['baselines']:
        raise ValueError('候选必须指定已登记父基线')
    for field in ['plugin_module','plugin_function']:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',config[field]):
            raise ValueError('插件使用单个模块名及函数名')


def build_signals(entry,wide,eligible,calendar,start,end,transform=None,params=None):
    """逐年度生成基线分数；插件只改变分数，资金、执行、频率和持股数由外部锁定。"""
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    dates=eligible.index[(eligible.index>=start)&(eligible.index<=end)]
    if len(dates)==0:
        raise ValueError('评价区间没有交易日')
    validate_entry(entry,range(start.year,end.year+1))
    parts=[];schedules=[]
    for year in sorted(set(dates.year)):
        days=dates[dates.year==year]
        p=entry['annual_parameters'][int(year)]['params']
        cutoff=days[-1]
        past={k:v.loc[:cutoff].copy() for k,v in wide.items()}
        allowed=eligible.loc[:cutoff].copy()
        score=RuleSignals(past,allowed,calendar).score(p).loc[days]
        if transform is not None:
            context=dict(wide=past,eligible=allowed,calendar=calendar.copy(),year=int(year),baseline_params=deepcopy(p))
            transformed=transform(score.copy(),context=context,params=deepcopy(params or {}))
            if not isinstance(transformed,pd.DataFrame) or not transformed.index.equals(score.index) or not transformed.columns.equals(score.columns):
                raise ValueError('插件必须返回同日期、同股票列的DataFrame')
            values=transformed.to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError('候选分数不能含无穷值')
            if transformed.where(~eligible.reindex_like(score)).notna().any().any():
                raise ValueError('候选试图给当日不合格股票打分')
            score=transformed
        decision=decision_dates(days,p['frequency'],calendar)
        parts.append(score.loc[decision])
        schedules.append(pd.Series(p['top_n'],index=decision,name='top_n'))
    return pd.concat(parts).sort_index(),pd.concat(schedules).sort_index()
