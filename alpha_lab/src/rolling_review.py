"""因果决策日历：固定月内日期及等间隔交易日复核。"""
import numpy as np
import pandas as pd


def decision_dates(calendar, kind, parameter, interval=20, anchor=None):
    """只需已发生的交易日即可决定是否复核，截断未来不改变过去。

    rolling的anchor固定于全研究起点；现金启动窗口不重置历史复核计数。
    月内日期为自然日阈值，休市后取当月第一个达到阈值的交易日。
    """
    calendar=pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('交易日历必须唯一递增')
    if kind=='calendar':
        if type(parameter) is not int or not 1<=parameter<=28:
            raise ValueError('自然日阈值须在1到28之间')
        eligible=calendar[calendar.day>=parameter]
        return eligible[~eligible.to_period('M').duplicated()]
    if kind=='rolling':
        if type(interval) is not int or interval<1 or type(parameter) is not int or not 0<=parameter<interval:
            raise ValueError('滚动间隔或初始偏移非法')
        if anchor is None:
            raise ValueError('滚动日历必须指定固定锚点')
        eligible=calendar[calendar>=pd.Timestamp(anchor)]
        return eligible[np.arange(len(eligible))%interval==parameter]
    raise ValueError('未知日历类型')


def schedule_specs(config):
    """完整对照矩阵，所有参数来自登记配置。"""
    return ([dict(name=f'calendar_{d:02d}',kind='calendar',parameter=d)
             for d in config['calendar_days']]+
            [dict(name=f'rolling_{o:02d}',kind='rolling',parameter=o)
             for o in config['rolling_offsets']])
