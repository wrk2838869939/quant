#!/usr/bin/env python3
"""Phase 0 前置: 探测 Tushare 接口能力 (历史深度/权限/单次行数上限)"""
import os
import time, warnings
import pandas as pd
import tushare as ts
from tushare.pro import client as _c

warnings.filterwarnings("ignore")
PROXY = "http://jiaoch.site"
TOKEN = os.environ.get("TUSHARE_TOKEN", "")
_c.DataApi._DataApi__http_url = PROXY
pro = ts.pro_api(TOKEN)

def probe(name, **kw):
    t0 = time.time()
    try:
        d = getattr(pro, name)(**kw)
        el = time.time() - t0
        if d is None or len(d) == 0:
            print(f"[EMPTY ] {name} {kw} -> 0 rows ({el:.1f}s)")
        else:
            key = 'trade_date' if 'trade_date' in d.columns else ('ann_date' if 'ann_date' in d.columns else d.columns[1] if len(d.columns)>1 else d.columns[0])
            print(f"[OK    ] {name} -> {len(d)} rows, {el:.1f}s | cols={list(d.columns)[:8]}")
            if key in d.columns:
                print(f"         {key}: {d[key].min()} ~ {d[key].max()}")
        return d
    except Exception as e:
        print(f"[FAIL  ] {name} {kw} -> {str(e)[:120]}")
        return None

# 1. 股票全集 (含退市)
d = probe("stock_basic", exchange="", list_status="D", fields="ts_code,name,list_date,delist_date,market")
d2 = probe("stock_basic", exchange="", list_status="L", fields="ts_code,name,list_date")
d3 = probe("stock_basic", exchange="", list_status="P", fields="ts_code,name,list_date")
# 2. 日线: 按日期拉全市场 (2014年深处)
probe("daily", trade_date="20140102")
probe("daily", trade_date="20240102")
# 3. 复权因子 / daily_basic / moneyflow 按日期
probe("adj_factor", trade_date="20140102")
probe("daily_basic", trade_date="20140102")
probe("moneyflow", trade_date="20140102")
# 4. 低频风险数据
probe("fina_indicator", ts_code="000001.SZ", start_date="20140101", end_date="20160101", fields="ts_code,ann_date,end_date,roe,debt_to_assets")
probe("pledge_stat", ts_code="000001.SZ", start_date="20180101", end_date="20200101")
probe("pledge_detail", ts_code="000001.SZ", start_date="20190101", end_date="20190601")
probe("share_float", ts_code="000001.SZ", start_date="20140101", end_date="20200101")
probe("namechange", ts_code="000001.SZ")
probe("index_daily", ts_code="000300.SH", start_date="20140101", end_date="20140201")
probe("top_list", trade_date="20240102")       # 龙虎榜
probe("top_inst", trade_date="20240102")
probe("suspend_d", trade_date="20240102")      # 停牌
probe("stk_factor", trade_date="20240102")     # 每日技术面因子(可能无权限)
