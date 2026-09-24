"""指标库: 均线/换手率/波动率/资金流等进阶数据, 逐股向量化计算。

设计约束:
  - 无前视: 所有滚动统计只用 ≤t 的数据 (量比分母显式 shift(1))
  - 计算一律基于统一单位列 (元/股), 禁用原始单位
  - 默认价格字段 hfq_close (后复权, 除权日无假跳空); 研究/看盘可传 price='close'
  - min_periods 默认=窗口长度 (严格), 前段为 NaN

用法 (一般经 DataHub 调用, 也可直接用本模块):
    from datahub.indicators import compute_features
    feat = compute_features(df_one_code, ['ma20', 'macd', 'turnover_ma5'])
"""
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 基础字段
BASE_FIELDS = [
    "date", "ts_code",
    "open", "high", "low", "close", "pre_close",
    "hfq_open", "hfq_high", "hfq_low", "hfq_close",
    "vol_share", "amount_yuan",
    "turnover_rate", "turnover_rate_f", "volume_ratio",
    "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm",
    "total_share", "float_share", "free_share", "total_mv_yuan", "circ_mv_yuan",
    "buy_sm_amount_yuan", "sell_sm_amount_yuan", "buy_md_amount_yuan", "sell_md_amount_yuan",
    "buy_lg_amount_yuan", "sell_lg_amount_yuan", "buy_elg_amount_yuan", "sell_elg_amount_yuan",
    "net_mf_amount_yuan",
]

PRICE_FIELDS = ("close", "hfq_close", "open", "hfq_open", "high", "low",
                "hfq_high", "hfq_low", "amount_yuan", "vol_share", "turnover_rate")


# ---------------------------------------------------------------- 指标实现
def _sma(s: pd.Series, w: int, min_periods: int | None = None) -> pd.Series:
    return s.rolling(w, min_periods=min_periods or w).mean()


def _std(s: pd.Series, w: int, min_periods: int | None = None) -> pd.Series:
    return s.rolling(w, min_periods=min_periods or w).std()


def _wilder(s: pd.Series, w: int, min_periods: int | None = None) -> pd.Series:
    """Wilder 平滑 (RSI/ATR 标准), 等价 ewm(alpha=1/w, adjust=False)。"""
    return s.ewm(alpha=1.0 / w, adjust=False, min_periods=min_periods or w).mean()


def i_ma(df, price, w, min_periods=None):
    return _sma(df[price], w, min_periods)


def i_ema(df, price, w, min_periods=None):
    return df[price].ewm(span=w, adjust=False, min_periods=min_periods or w).mean()


def i_bias(df, price, w, min_periods=None):
    return df[price] / _sma(df[price], w, min_periods) - 1.0


def i_ret(df, price, n, fill_na=True):
    r = df[price].pct_change(n, fill_method=None)
    if fill_na:
        r = r.replace([np.inf, -np.inf], np.nan)
    return r


def i_rv(df, price, w, min_periods=None):
    """年化滚动波动率 = 日收益std × sqrt(250)。"""
    logret = np.log(df[price] / df[price].shift(1))
    return _std(logret, w, min_periods) * np.sqrt(250.0)


def i_rsi(df, price, w, min_periods=None):
    delta = df[price].diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    au = _wilder(up, w, min_periods)
    ad = _wilder(dn, w, min_periods)
    return 100.0 * au / (au + ad)


def i_atr(df, price, w, min_periods=None):
    """真实波幅均值 (Wilder)。price 参数决定用原始价还是后复权价 (内部一致即可)。"""
    h, l = df[price.replace("close", "high")], df[price.replace("close", "low")]
    pc = df[price].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return _wilder(tr, w, min_periods)


def i_macd(df, price):
    """(dif, dea, hist×2) A股惯例。"""
    dif = df[price].ewm(span=12, adjust=False).mean() - df[price].ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": (dif - dea) * 2.0})


def i_boll(df, price, w, nstd, min_periods=None):
    mid = _sma(df[price], w, min_periods)
    sd = _std(df[price], w, min_periods)
    return pd.DataFrame({"mid": mid, "up": mid + nstd * sd, "low": mid - nstd * sd})


def i_overnight(df):
    """隔夜收益 = open/pre_close - 1 (pre_close 已是除权调整口径)。"""
    return df["open"] / df["pre_close"] - 1.0


def i_intraday(df):
    """日内收益 = close/open - 1 (同日比值, 与复权无关)。"""
    return df["close"] / df["open"] - 1.0


def i_upper_shadow(df):
    """上影长度 = (high - max(open,close)) / close (排雷「无上影」同口径)。"""
    return (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]


def i_lower_shadow(df):
    return (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]


def i_turnover_ma(df, w, min_periods=None):
    return _sma(df["turnover_rate"], w, min_periods)


def i_turnover_std(df, w, min_periods=None):
    return _std(df["turnover_rate"], w, min_periods)


def i_amount_ma(df, w, min_periods=None):
    return _sma(df["amount_yuan"], w, min_periods)


def i_vratio(df, w, min_periods=None):
    """量比(自定义) = 当日量 / 过去w日均量(不含当日, shift(1) 防当日稀释)。"""
    return df["vol_share"] / _sma(df["vol_share"], w, min_periods).shift(1)


def i_amihud(df, price, w, min_periods=None):
    """Amihud 非流动性 = mean(|日收益| / 成交额) × 1e9。"""
    ret = df[price].pct_change(fill_method=None).abs()
    illiq = ret / df["amount_yuan"] * 1e9
    return _sma(illiq, w, min_periods)


def i_dd(df, price, w, min_periods=None):
    """距过去w日最高收盘的回撤 (≤0)。"""
    return df[price] / df[price].rolling(w, min_periods=min_periods or w).max() - 1.0


def i_high(df, price, w, min_periods=None):
    return df[price.replace("close", "high")].rolling(w, min_periods=min_periods or w).max()


def i_low(df, price, w, min_periods=None):
    return df[price.replace("close", "low")].rolling(w, min_periods=min_periods or w).min()


def i_vol_ma(df, w, min_periods=None):
    return _sma(df["vol_share"], w, min_periods)


def _mf_ratio(df, buy, sell):
    amt = df["amount_yuan"]
    return (df[buy] - df[sell]) / amt


# ---------------------------------------------------------------- 规格注册
# 每项: (解析正则, 需要的基础字段(与 price 相关的占位 {p}), 输出列后缀, 计算函数)
SPECS = {
    "ma":     (re.compile(r"^ma(\d+)$"),                    ["{p}"],                    [""],      lambda df, p, m: i_ma(df, p, *m)),
    "ema":    (re.compile(r"^ema(\d+)$"),                   ["{p}"],                    [""],      lambda df, p, m: i_ema(df, p, *m)),
    "bias":   (re.compile(r"^bias(\d+)$"),                  ["{p}"],                    [""],      lambda df, p, m: i_bias(df, p, *m)),
    "ret":    (re.compile(r"^ret(\d+)$"),                   ["{p}"],                    [""],      lambda df, p, m: i_ret(df, p, *m)),
    "rv":     (re.compile(r"^(?:rv|volatility)(\d+)$"),     ["{p}"],                    [""],      lambda df, p, m: i_rv(df, p, *m)),
    "rsi":    (re.compile(r"^rsi(\d+)$"),                   ["{p}"],                    [""],      lambda df, p, m: i_rsi(df, p, *m)),
    "atr":    (re.compile(r"^atr(\d+)$"),                   ["{p}", "{p:close>high}", "{p:close>low}"], [""], lambda df, p, m: i_atr(df, p, *m)),
    "macd":   (re.compile(r"^macd$"),                       ["{p}"],                    ["_dif", "_dea", "_hist"], lambda df, p, m: i_macd(df, p)),
    "boll":   (re.compile(r"^boll(\d+)_(\d+(?:\.\d+)?)$"),  ["{p}"],                    ["_mid", "_up", "_low"],   lambda df, p, m: i_boll(df, p, *m)),
    "overnight":     (re.compile(r"^overnight$"),      ["open", "pre_close"],      [""], lambda df, p, m: i_overnight(df)),
    "intraday":      (re.compile(r"^intraday$"),       ["open", "close"],          [""], lambda df, p, m: i_intraday(df)),
    "upper_shadow":  (re.compile(r"^upper_shadow$"),   ["open", "high", "low", "close"], [""], lambda df, p, m: i_upper_shadow(df)),
    "lower_shadow":  (re.compile(r"^lower_shadow$"),   ["open", "high", "low", "close"], [""], lambda df, p, m: i_lower_shadow(df)),
    "turnover":      (re.compile(r"^turnover$"),       ["turnover_rate"],          [""], lambda df, p, m: df["turnover_rate"]),
    "turnover_f":    (re.compile(r"^turnover_f$"),     ["turnover_rate_f"],        [""], lambda df, p, m: df["turnover_rate_f"]),
    "turnover_ma":   (re.compile(r"^turnover_ma(\d+)$"), ["turnover_rate"],        [""], lambda df, p, m: i_turnover_ma(df, *m)),
    "turnover_std":  (re.compile(r"^turnover_std(\d+)$"), ["turnover_rate"],       [""], lambda df, p, m: i_turnover_std(df, *m)),
    "amount_ma":     (re.compile(r"^amount_ma(\d+)$"), ["amount_yuan"],            [""], lambda df, p, m: i_amount_ma(df, *m)),
    "vol_ma":        (re.compile(r"^vol_ma(\d+)$"),    ["vol_share"],              [""], lambda df, p, m: i_vol_ma(df, *m)),
    "vratio":        (re.compile(r"^vratio(\d+)$"),    ["vol_share"],              [""], lambda df, p, m: i_vratio(df, *m)),
    "amihud":        (re.compile(r"^amihud(\d+)$"),    ["{p}", "amount_yuan"],     [""], lambda df, p, m: i_amihud(df, p, *m)),
    "dd":            (re.compile(r"^dd(\d+)$"),        ["{p}"],                    [""], lambda df, p, m: i_dd(df, p, *m)),
    "high":          (re.compile(r"^high(\d+)$"),      ["{p:close>high}"],         [""], lambda df, p, m: i_high(df, p, *m)),
    "low":           (re.compile(r"^low(\d+)$"),       ["{p:close>low}"],          [""], lambda df, p, m: i_low(df, p, *m)),
    "mf_net_ratio":  (re.compile(r"^mf_net_ratio$"),   ["net_mf_amount_yuan", "amount_yuan"], [""], lambda df, p, m: df["net_mf_amount_yuan"] / df["amount_yuan"]),
    "lg_net_ratio":  (re.compile(r"^lg_net_ratio$"),   ["buy_lg_amount_yuan", "sell_lg_amount_yuan", "amount_yuan"], [""], lambda df, p, m: _mf_ratio(df, "buy_lg_amount_yuan", "sell_lg_amount_yuan")),
    "elg_net_ratio": (re.compile(r"^elg_net_ratio$"),  ["buy_elg_amount_yuan", "sell_elg_amount_yuan", "amount_yuan"], [""], lambda df, p, m: _mf_ratio(df, "buy_elg_amount_yuan", "sell_elg_amount_yuan")),
    "sm_net_ratio":  (re.compile(r"^sm_net_ratio$"),   ["buy_sm_amount_yuan", "sell_sm_amount_yuan", "amount_yuan"], [""], lambda df, p, m: _mf_ratio(df, "buy_sm_amount_yuan", "sell_sm_amount_yuan")),
}


class FeatureError(ValueError):
    pass


def parse_spec(name: str):
    """'ma20' → ('ma', (20,)); 'boll20_2' → ('boll', (20, 2.0)); 未知→None。"""
    for key, (pat, _need, _outs, _fn) in SPECS.items():
        m = pat.match(name)
        if m:
            args = tuple(float(g) if "." in g else int(g) for g in m.groups())
            return key, args
    return None


def required_fields(spec_name: str, price: str = "hfq_close") -> list[str]:
    """某特征需要预加载的基础列 (date/ts_code 恒带)。"""
    key, args = parse_spec(spec_name) or (None, None)
    if key is None:
        if spec_name in BASE_FIELDS:  # 基础列直通
            return [spec_name]
        raise FeatureError(f"未知特征: {spec_name}")
    _pat, need, _outs, _fn = SPECS[key]
    if key in ("atr", "high", "low") and price not in ("close", "hfq_close"):
        raise FeatureError(f"{spec_name} 需要 price='close' 或 'hfq_close'，避免混用 OHLC")
    out = []
    for t in need:
        if t == "{p}":
            out.append(price)
        elif t.startswith("{p:"):  # 如 {p:close>high}: price 换成对应 high/low 字段
            out.append(price.replace("close", t.split(">")[1].rstrip("}")))
        else:
            out.append(t)
    return out


def output_columns(spec_name: str) -> list[str]:
    key, _ = parse_spec(spec_name) or (None, None)
    if key is None:
        if spec_name in BASE_FIELDS:
            return [spec_name]
        raise FeatureError(f"未知特征: {spec_name}")
    outs = SPECS[key][2]
    if outs == [""]:
        return [spec_name]
    return [f"{spec_name}{o}" for o in outs]


def compute_features(df: pd.DataFrame, specs: list[str],
                     price: str = "hfq_close") -> pd.DataFrame:
    """对单只股票 (按日期升序) 计算一组特征, 返回 (date, ts_code, 特征列...)。

    df 需包含各特征 required_fields 所需列; 基础列直通规格直接原样带出。
    所有滚动窗口 min_periods=窗口长度 (严格口径, 前段 NaN, 无前视)。
    """
    if df.empty:
        cols = ["date", "ts_code"] + [c for s in specs for c in output_columns(s)]
        return pd.DataFrame(columns=cols)
    out = df[["date", "ts_code"]].copy()
    for name in specs:
        required_fields(name, price)
        key, args = parse_spec(name) or (None, None)
        if key is None:
            if name in BASE_FIELDS and name in df.columns:
                out[name] = df[name].values
                continue
            raise FeatureError(f"未知特征: {name}")
        fn = SPECS[key][3]
        res = fn(df, price, args)
        if isinstance(res, pd.DataFrame):
            for suffix_col, col in zip(SPECS[key][2], res.columns):
                out[f"{name}{suffix_col}"] = res[col].values
        else:
            out[name] = res.values
    return out


def expand_required_fields(specs: list[str], price: str = "hfq_close") -> list[str]:
    """多特征所需基础列并集 (恒含 date/ts_code, 有序去重)。"""
    need: list[str] = ["date", "ts_code"]
    for s in specs:
        for f in required_fields(s, price):
            if f not in need:
                need.append(f)
    return need


def max_lookback(specs: list[str]) -> int:
    """特征所需的最大历史窗口 (交易日), 用于读取起点前移。"""
    lb = 1
    for s in specs:
        key, args = parse_spec(s) or (None, None)
        if key is None:
            continue
        if key in ("macd",):
            lb = max(lb, 60)
        elif args:
            # boll/rsi/atr/amihud 等 ewm 类给 3 倍预热; 纯 rolling 给窗口长度
            lb = max(lb, int(max(args)) * (3 if key in ("rsi", "atr") else 1))
    return lb
