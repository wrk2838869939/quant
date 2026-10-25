"""Tushare 数据访问统一封装 (jiaoch.site 代理)。

所有 fetch 脚本共用: 限流 / 重试 / 断点续传工具。
"""
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import tushare as ts
from tushare.pro import client as _c

warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/alpha_lab")
RAW = ROOT / "data/raw"

TOKEN_FILE = Path(os.environ.get('TUSHARE_TOKEN_FILE', str(Path.home() / '.config/alpha_lab/tushare_token')))
TOKEN = os.environ.get("TUSHARE_TOKEN", "")
if not os.environ.get('TUSHARE_TOKEN') and TOKEN_FILE.exists():
    TOKEN = TOKEN_FILE.read_text().strip()
PROXY = os.environ.get("TUSHARE_PROXY", "http://jiaoch.site")

# 代理限流 2 并发; 单线程内再留间隔
CALL_INTERVAL = 0.35


def get_pro():
    _c.DataApi._DataApi__http_url = PROXY
    return ts.pro_api(TOKEN)


_pro = None


def pro():
    global _pro
    if _pro is None:
        _pro = get_pro()
    return _pro


_last_call = 0.0


def fetch(api: str, retries: int = 4, **kw) -> pd.DataFrame | None:
    """带限流间隔与指数退避重试的 API 调用。权限类错误直接返回 None。"""
    global _last_call
    for r in range(retries):
        wait = CALL_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            d = getattr(pro(), api)(**kw)
            return d
        except Exception as e:
            msg = str(e)
            if "权限" in msg or "积分" in msg:
                print(f"[PERM ] {api} {kw} -> {msg[:80]}")
                return None
            if "无效的字段" in msg or "参数错误" in msg:  # 永久失败, 不重试
                print(f"[BADARG] {api} {kw} -> {msg[:100]}")
                return None
            if r == retries - 1:
                print(f"[FAIL ] {api} {kw} -> {msg[:100]}")
                return None
            time.sleep(1.5 * (r + 1))
    return None


def save_day(df: pd.DataFrame, api: str, trade_date: str):
    """按日落盘到 data/raw/by_date/{api}/{yyyymmdd}.parquet"""
    d = RAW / "by_date" / api
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"{trade_date}.parquet", index=False)


def done_days(api: str) -> set:
    d = RAW / "by_date" / api
    if not d.exists():
        return set()
    return {p.stem for p in d.glob("*.parquet")}
