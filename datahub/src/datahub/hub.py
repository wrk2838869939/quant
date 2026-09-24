"""DataHub 门面: A股/美股/指数/元数据/日历 统一查询接口。

快速上手:
    from datahub import DataHub
    hub = DataHub()

    # 1) 原始日线 (长表)
    df = hub.cn.daily('000001.SZ', '2024-01-01', '2024-06-30')

    # 2) 进阶指标 (均线/换手率/MACD/ATR...), 默认后复权价, 无前视
    f = hub.cn.features(['600519.SH', '300750.SZ'], '2024-01-01', '2024-12-31',
                        specs=['ma20', 'ma60', 'turnover', 'macd', 'atr14'],
                        n_jobs=4)                      # 进程级并行
    ma20 = hub.cn.ma(['600519.SH'], 20)                # 宽表快捷方式

    # 3) 美股 / 全球指数
    hub.us.daily(['BABA', 'SPX'], '2024-01-01', '2024-06-30')
    hub.us.ma('SPX', 50)

    # 4) 元数据 / PIT 行业 / 日历
    hub.meta.sector_of('600519.SH', '2020-06-01')
    hub.calendar.cn_days('2024-01-01', '2024-12-31')
"""
from __future__ import annotations

import pandas as pd

from .calendar_view import CalendarView
from .config import HubConfig
from . import indicators as ind
from . import io_utils as iou


# ---------------------------------------------------------------- 通用工具
def _pivot(df: pd.DataFrame, value: str, code_col: str) -> pd.DataFrame:
    w = df.pivot_table(index="date", columns=code_col, values=value, dropna=False)
    return w.sort_index()


def _as_list(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


# ---------------------------------------------------------------- A股
class CNView:
    """A股日线视图 (沪主板/深主板/创业板/科创板/北交所, 含退市股)。"""

    def __init__(self, hub):
        self._hub = hub

    # ---------- 原始数据 ----------
    def daily(self, codes=None, start=None, end=None, fields=None, board=None,
              adjust: str | None = None) -> pd.DataFrame:
        """日线长表。codes: str/list/None(全市场); adjust='hfq' 时 OHLC 用后复权。

        fields 常用: open high low close pre_close pct_chg vol_share amount_yuan
                    turnover_rate turnover_rate_f volume_ratio pe_ttm pb circ_mv_yuan ...
        """
        return iou.read_cn_daily(self._hub.cfg, codes=codes, start=start, end=end,
                                 fields=fields, board=board, adjust=adjust)

    def pivot(self, field: str, codes=None, start=None, end=None, board=None) -> pd.DataFrame:
        """单列宽表: index=date, columns=ts_code。"""
        df = self.daily(codes, start, end, fields=[field], board=board)
        return _pivot(df, field, "ts_code")

    def close(self, codes=None, start=None, end=None, adjust=None, board=None):
        return self.pivot("close", codes, start, end, board) if adjust != "hfq" \
            else self.pivot("hfq_close", codes, start, end, board)

    def volume(self, codes=None, start=None, end=None, board=None):
        """成交量 (股)。"""
        return self.pivot("vol_share", codes, start, end, board)

    def amount(self, codes=None, start=None, end=None, board=None):
        """成交额 (元)。"""
        return self.pivot("amount_yuan", codes, start, end, board)

    def turnover(self, codes=None, start=None, end=None, free: bool = False, board=None):
        """换手率 (%)。free=True 用自由流通换手 turnover_rate_f。"""
        return self.pivot("turnover_rate_f" if free else "turnover_rate",
                          codes, start, end, board)

    def snapshot(self, date, fields=None, board=None) -> pd.DataFrame:
        """某交易日全市场截面。"""
        d = pd.Timestamp(date).normalize()
        return self.daily(None, d, d, fields=fields, board=board)

    def codes(self, board=None, status: str = "L") -> list[str]:
        """股票列表 (默认在市); board: sh_main/sz_main/chinext/star/bse。"""
        return self._hub.meta.stock_codes(board=board, status=status)

    # ---------- 进阶指标 ----------
    def features(self, codes=None, start=None, end=None, specs: list[str] | None = None,
                 price: str = "hfq_close",
                 n_jobs: int = 1, board=None, cache: bool = False,
                 refresh: bool = False) -> pd.DataFrame:
        """批量计算进阶指标, 返回长表 (ts_code, date, 特征列...)。

        specs 例: ma20 ema12 bias20 ret5 rv20 rsi14 atr14 macd boll20_2
                 turnover turnover_ma5 turnover_std10 amount_ma20 vratio5 amihud20
                 dd20 high20 low20 overnight intraday upper_shadow lower_shadow
                 mf_net_ratio lg_net_ratio elg_net_ratio sm_net_ratio
        price: 计算用价格, 默认 hfq_close (后复权, 除权无假跳空); 看盘口径传 'close'。
        所有滚动窗口 min_periods=窗口长度 (严格, 无前视), 窗口未满为 NaN。
        n_jobs>1 时按代码分块走进程池 (每个 worker 自行读数)。
        cache=True 时结果按参数哈希落盘缓存 (数据集重建后自动失效)。
        """
        specs = specs or ["ma20"]
        codes_l = _as_list(codes)
        if codes_l is None:
            # 以实际行情为全集，避免最新主档遗漏历史改码股票。
            codes_l = self.daily(start=start, end=end, fields=[], board=board)["ts_code"].unique().tolist()
        if price not in ind.PRICE_FIELDS:
            raise ValueError(f"price 须为 {ind.PRICE_FIELDS} 之一")

        def _compute() -> pd.DataFrame:
            from .parallel import parallel_cn_features
            return parallel_cn_features(
                codes_l, specs, price, start, end,
                root=str(self._hub.cfg.root),
                n_jobs=n_jobs, board=board)

        if cache:
            version = f"{self._hub.data_version()}|{self._hub.cn_max_date()}"
            payload = dict(codes=sorted(codes_l), specs=list(specs), price=price,
                           start=str(start), end=str(end), board=board)
            return iou.load_or_compute("cnfeat", payload, _compute, self._hub.cfg,
                                       version, refresh=refresh)
        return _compute()

    def ma(self, codes=None, window: int = 20, start=None, end=None,
           price: str = "hfq_close", n_jobs: int = 1, board=None) -> pd.DataFrame:
        """N日均线, 宽表 (index=date, columns=ts_code)。"""
        f = self.features(codes, start, end, specs=[f"ma{window}"], price=price,
                          n_jobs=n_jobs, board=board)
        return _pivot(f, f"ma{window}", "ts_code")

    def ema(self, codes=None, window: int = 12, start=None, end=None,
            price: str = "hfq_close", n_jobs: int = 1, board=None) -> pd.DataFrame:
        f = self.features(codes, start, end, specs=[f"ema{window}"], price=price,
                          n_jobs=n_jobs, board=board)
        return _pivot(f, f"ema{window}", "ts_code")

    def macd(self, codes=None, start=None, end=None, price: str = "hfq_close",
             n_jobs: int = 1, board=None) -> dict[str, pd.DataFrame]:
        """{'dif','dea','hist'} 三个宽表。"""
        f = self.features(codes, start, end, specs=["macd"], price=price,
                          n_jobs=n_jobs, board=board)
        return {k: _pivot(f, f"macd_{k}", "ts_code") for k in ("dif", "dea", "hist")}

    def turnover_ma(self, codes=None, window: int = 5, start=None, end=None,
                    n_jobs: int = 1, board=None) -> pd.DataFrame:
        f = self.features(codes, start, end, specs=[f"turnover_ma{window}"],
                          n_jobs=n_jobs, board=board)
        return _pivot(f, f"turnover_ma{window}", "ts_code")

    # ---------- 信息 ----------
    def available_features(self) -> list[str]:
        return sorted({*ind.SPECS.keys(), *ind.BASE_FIELDS})

    def info(self) -> dict:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        files = sorted(self._hub.cfg.cn_daily.glob("year=*/data.parquet"))
        rows = 0
        for p in files:
            rows += pq.read_metadata(p).num_rows
        ds = iou.cn_dataset(self._hub.cfg)
        t = ds.to_table(columns=["date"])
        lo = pd.Timestamp(pc.min(t["date"]).as_py())
        hi = pd.Timestamp(pc.max(t["date"]).as_py())
        n_codes = ds.to_table(columns=["ts_code"]).to_pandas()["ts_code"].nunique()
        return dict(rows=rows, codes=int(n_codes), start=lo, end=hi,
                    size_mb=round(iou.parquet_dir_size_mb(self._hub.cfg.cn_daily), 1))


# ---------------------------------------------------------------- 美股
class USView:
    """美股/全球指数/利率日线视图 (qfq 口径)。"""

    BASE = ["open", "high", "low", "close", "volume"]

    def __init__(self, hub):
        self._hub = hub

    def daily(self, tickers=None, start=None, end=None, fields=None,
              asset_type=None) -> pd.DataFrame:
        """日线长表。asset_type: stock/etf/index/index_hk/index_jp/index_uk/rate。"""
        return iou.read_us_daily(self._hub.cfg, tickers=tickers, start=start, end=end,
                                 fields=fields, asset_type=asset_type)

    def close(self, tickers=None, start=None, end=None, asset_type=None):
        df = self.daily(tickers, start, end, fields=["close"], asset_type=asset_type)
        return _pivot(df, "close", "ticker")

    def tickers(self, asset_type=None) -> list[str]:
        m = self._hub.meta.us_meta()
        if asset_type is not None:
            m = m[m["asset_type"].isin(_as_list(asset_type))]
        return m["ticker"].tolist()

    def features(self, tickers=None, start=None, end=None, specs: list[str] | None = None,
                 price: str = "close", n_jobs: int = 1, cache: bool = False,
                 refresh: bool = False) -> pd.DataFrame:
        """美股特征 (价格/量类指标; 无换手率/资金流)。"""
        specs = specs or ["ma20"]
        tickers_l = _as_list(tickers)
        if tickers_l is None:
            tickers_l = self.tickers()
        # 可用性校验: 美股只有 OHLCV
        base_avail = set(self.BASE)
        for s in specs:
            req = [f for f in ind.required_fields(s, price) if f not in ("date", "ts_code")]
            bad = [f for f in req if f not in base_avail]
            if bad:
                raise ValueError(f"美股不支持特征 {s}: 缺基础字段 {bad} "
                                 f"(换手率/资金流/估值类指标仅 A股)")
        df = iou.read_us_daily(self._hub.cfg, tickers=tickers_l, start=None, end=end)

        def _compute() -> pd.DataFrame:
            if df.empty:
                return ind.compute_features(df.rename(columns={"ticker": "ts_code"}), specs,
                                            price=price).rename(columns={"ts_code": "ticker"})
            parts = []
            for t, g in df.groupby("ticker", sort=False):
                g = g.rename(columns={"ticker": "ts_code"})
                parts.append(ind.compute_features(g, specs, price=price))
            out = pd.concat(parts, ignore_index=True).rename(columns={"ts_code": "ticker"})
            if start is not None:
                out = out[out["date"] >= pd.Timestamp(start)]
            return out.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)

        if cache:
            version = f"{self._hub.data_version()}"
            payload = dict(mkt="us", tickers=sorted(tickers_l), specs=list(specs),
                           price=price, start=str(start), end=str(end))
            return iou.load_or_compute("usfeat", payload, _compute, self._hub.cfg,
                                       version, refresh=refresh)
        return _compute()

    def ma(self, tickers=None, window: int = 50, start=None, end=None) -> pd.DataFrame:
        f = self.features(tickers, start, end, specs=[f"ma{window}"])
        return _pivot(f, f"ma{window}", "ticker")

    def info(self) -> dict:
        m = self._hub.meta.us_meta()
        return dict(tickers=len(m), by_type=m.groupby("asset_type")["rows"].sum().to_dict(),
                    start=m["start"].min(), end=m["end"].max())


# ---------------------------------------------------------------- 指数
class IndexView:
    """A股指数日线 (单位与个股一致)。"""

    def __init__(self, hub):
        self._hub = hub

    def daily(self, ts_code: str = "000300.SH", start=None, end=None,
              fields=None) -> pd.DataFrame:
        df = pd.read_parquet(self._hub.cfg.cn_index)
        df = df[df["ts_code"] == ts_code]
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        if fields:
            df = df[["ts_code", "date"] + [f for f in fields if f not in ("ts_code", "date")]]
        return df.sort_values("date").reset_index(drop=True)

    def list(self) -> pd.DataFrame:
        return pd.read_parquet(self._hub.cfg.datasets / "index/cn_index_meta.parquet")

    def close(self, ts_code: str = "000300.SH", start=None, end=None) -> pd.Series:
        df = self.daily(ts_code, start, end, fields=["close"])
        return df.set_index("date")["close"]


# ---------------------------------------------------------------- 元数据
class MetaView:
    """股票主档 / 改名ST史 / 申万L1 PIT成分。"""

    def __init__(self, hub):
        self._hub = hub

    def _load(self, name: str) -> pd.DataFrame:
        cache = getattr(self, "_cache", None)
        if cache is None:
            cache = self._cache = {}
        if name not in cache:
            cache[name] = pd.read_parquet(self._hub.cfg.meta_dir / f"{name}.parquet")
        return cache[name]

    def stocks(self, board=None, status=None) -> pd.DataFrame:
        df = self._load("stock_basic")
        if board is not None:
            df = df[df["board"].isin(_as_list(board))]
        if status is not None:
            df = df[df["list_status"] == status]
        return df.reset_index(drop=True)

    def stock_codes(self, board=None, status: str | None = "L") -> list[str]:
        return self.stocks(board=board, status=status)["ts_code"].tolist()

    def stock(self, ts_code: str) -> pd.Series:
        df = self._load("stock_basic")
        hit = df[df["ts_code"] == ts_code]
        if hit.empty:
            raise KeyError(f"未找到 {ts_code}")
        return hit.iloc[0]

    def search(self, keyword: str) -> pd.DataFrame:
        return self._load("stock_basic")[
            self._load("stock_basic")["name"].str.contains(keyword, na=False)]

    def name_history(self, ts_code: str) -> pd.DataFrame:
        df = self._load("namechange")
        return df[df["ts_code"] == ts_code].sort_values("start_date").reset_index(drop=True)

    def name_at(self, ts_code: str, date=None) -> str | None:
        d = (pd.Timestamp(date).strftime("%Y%m%d") if date is not None
             else pd.Timestamp.now().strftime("%Y%m%d"))
        h = self.name_history(ts_code)
        h = h[(h["start_date"] <= d) & (h["end_date"].isna() | (h["end_date"] >= d))]
        return None if h.empty else h.iloc[-1]["name"]

    def is_st(self, ts_code: str, date=None) -> bool | None:
        """历史名称缺失时返回 None（未知），不能当作已确认非 ST。"""
        n = self.name_at(ts_code, date)
        return None if not n else "ST" in n.upper()

    # ---- 申万 L1 (PIT) ----
    def sectors(self) -> pd.DataFrame:
        return self._load("sw_l1_classify")

    def sector_of(self, ts_code: str, date=None) -> str | None:
        """股票在某交易日的申万L1行业名 (point-in-time, in/out 区间判定)。"""
        d = (pd.Timestamp(date).strftime("%Y%m%d") if date is not None
             else "20991230")
        m = self._load("sw_l1_member_hist")
        hit = m[(m["con_code"] == ts_code) & (m["in_date"] <= d)
                & (m["out_date"] > d)]
        if hit.empty:
            return None
        if hit["index_code"].nunique() > 1:
            raise ValueError(f"{ts_code}@{d} 历史行业区间冲突，须核验源数据")
        cls = self._load("sw_l1_classify").set_index("index_code")["industry_name"]
        return cls.get(hit.iloc[-1]["index_code"])

    def sector_members(self, index_code: str | None = None, industry_name: str | None = None,
                       date=None) -> list[str]:
        """某行业在某交易日的成分股。"""
        d = (pd.Timestamp(date).strftime("%Y%m%d") if date is not None else "20991230")
        m = self._load("sw_l1_member_hist")
        if index_code is None and industry_name is not None:
            cls = self._load("sw_l1_classify")
            hit = cls[cls["industry_name"] == industry_name]
            if hit.empty:
                raise KeyError(f"未知行业: {industry_name}")
            index_code = hit.iloc[0]["index_code"]
        active = m[(m["in_date"] <= d) & (m["out_date"] > d)]
        mem = active[active["index_code"] == index_code]
        conflicts = active.groupby("con_code")["index_code"].nunique()
        if mem["con_code"].isin(conflicts[conflicts > 1].index).any():
            raise ValueError(f"行业 {index_code}@{d} 存在成分区间冲突，须核验源数据")
        return mem["con_code"].drop_duplicates().tolist()

    def us_meta(self) -> pd.DataFrame:
        return self._load("us_meta")


# ---------------------------------------------------------------- 门面
class DataHub:
    """统一数据入口。root 缺省自动发现 (环境变量 DATAHUB_ROOT 可覆盖)。"""

    def __init__(self, root=None):
        self.cfg = HubConfig(root)
        if not (self.cfg.cn_daily).is_dir():
            raise FileNotFoundError(
                f"未找到数据集目录 {self.cfg.cn_daily} — 请先运行 scripts/build_dataset.py "
                f"或用 DataHub(root=...) 指定 datahub 目录")
        self.cn = CNView(self)
        self.us = USView(self)
        self.index = IndexView(self)
        self.meta = MetaView(self)
        self.calendar = CalendarView(self)
        self._us_frame: pd.DataFrame | None = None  # 美股小, 整表驻留

    # ---- 版本/缓存 ----
    def data_version(self) -> str:
        mp = self.cfg.root / "build_manifest.json"
        if mp.exists():
            import json
            return json.loads(mp.read_text()).get("built_at", "unknown")
        return "no-manifest"

    def cn_max_date(self):
        return iou.cn_max_date(self.cfg)

    def clear_cache(self):
        iou.clear_cache(self.cfg)
        print(f"缓存已清空: {self.cfg.cache_dir}")

    def info(self) -> dict:
        try:
            cn = self.cn.info()
        except Exception as e:  # 数据集未建好时也能看其余部分
            cn = {"error": str(e)}
        return dict(root=str(self.cfg.root), data_version=self.data_version(),
                    cn=cn, us=self.us.info(),
                    indexes=self.index.list()["ts_code"].nunique(),
                    stocks=len(self.meta.stocks(status=None)))
