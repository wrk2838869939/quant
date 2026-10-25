"""首次建仓与后续日历隔离：仅拼接已封存当日信号，不修改交易引擎。"""
import pandas as pd


def startup_matrices(base, shifted, discounts, boot_date, start, end, policy):
    """首次复核不增加后续持有失败计数；后续hold严格保留原日历矩阵。"""
    boot_date,start,end=map(pd.Timestamp,[boot_date,start,end])
    if not start<=boot_date<end:raise ValueError('首次复核必须在共同窗口内且留有执行日')
    if policy not in ['wait','stale_once','fresh_once','shifted_calendar']:raise ValueError('未知启动政策')
    if boot_date not in shifted['signals'].index:raise ValueError('启动日期无当日完整截面信号')
    next_month=start.to_period('M')+1
    keep=base['signals'].index[(base['signals'].index.to_period('M')>=next_month)&(base['signals'].index<=end)]
    # 所有组资本从同一日可用；wait直到下月原复核日才有第一次信号。
    result={key:value.loc[keep].copy() for key,value in base.items()}
    risk=discounts.reindex(index=keep,columns=base['signals'].columns).copy()
    source_date=None
    if policy=='shifted_calendar':
        keep=shifted['signals'].index[(shifted['signals'].index>=boot_date)&(shifted['signals'].index<=end)]
        result={key:value.loc[keep].copy() for key,value in shifted.items()}
        risk=discounts.reindex(index=keep,columns=base['signals'].columns).copy()
        source_date=boot_date
    elif policy!='wait':
        source=base if policy=='stale_once' else shifted
        eligible=source['signals'].index[source['signals'].index<=boot_date]
        if not len(eligible):raise ValueError('没有启动日前可用的旧信号')
        source_date=eligible[-1]
        if policy=='fresh_once' and source_date!=boot_date:raise ValueError('新信号日期不匹配')
        for key in ['signals','forecasts']:
            result[key].loc[boot_date]=source[key].loc[source_date]
        # 当天收盘仍是空账户，新增行不参与任何旧仓退出判断。
        result['hold'].loc[boot_date]=1.
        risk.loc[boot_date]=discounts.loc[source_date].reindex(risk.columns)
    result={key:value.sort_index() for key,value in result.items()};risk=risk.sort_index()
    if policy in ['fresh_once','stale_once','wait']:
        normal=result['signals'].index[result['signals'].index!=boot_date]
        for key in result:pd.testing.assert_frame_equal(result[key].loc[normal],base[key].loc[normal],check_exact=True)
    return result,risk,dict(policy=policy,boot_date=str(boot_date.date()),source_date=None if source_date is None else str(source_date.date()),
        following_calendar='shifted' if policy=='shifted_calendar' else 'unchanged_month_start',
        future_source_used=bool(source_date is not None and source_date>boot_date))
