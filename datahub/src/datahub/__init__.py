"""datahub: A股(主板/创业板/科创板/北交所)+美股 结构化数据集查询接口。

    from datahub import DataHub
    hub = DataHub()
    hub.cn.daily('000001.SZ', '2024-01-01', '2024-06-30')
    hub.cn.features(codes, specs=['ma20', 'turnover', 'macd'], n_jobs=4)
    hub.us.daily('BABA')
"""
from .config import HubConfig, find_hub_root
from .hub import DataHub, CNView, USView, IndexView, MetaView
from .calendar_view import CalendarView
from . import indicators

__version__ = "1.0.0"
__all__ = ["DataHub", "HubConfig", "find_hub_root",
           "CNView", "USView", "IndexView", "MetaView", "CalendarView", "indicators"]
